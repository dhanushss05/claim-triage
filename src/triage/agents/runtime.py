"""Shared agent runtime: the run recorder (events + metrics + cost) and the retriever ports."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import httpx

from ..llm_client import LLMPort, LLMResponse
from ..metrics import AGENT_STEP_LATENCY, LLM_COST, LLM_TOKENS, TOOL_CALLS
from ..resilience import NonRetryable, call_with_retry


class Recorder:
    """Appends run events (to Redis via RunStore, or to memory for in-process evaluation) and keeps per-run cost."""

    def __init__(self, run_id: str, agent: str, store=None):
        self.run_id = run_id
        self.agent = agent
        self.store = store
        self.events: list[dict[str, Any]] = []
        self.cost_by_step: dict[str, dict[str, float]] = {}
        self.step_ms: dict[str, float] = {}

    async def event(self, type_: str, **data: Any) -> None:
        ev = {"type": type_, "agent": self.agent, **data}
        self.events.append(ev)
        if self.store is not None:
            await self.store.append_event(self.run_id, ev)

    async def llm(self, step: str, prompt_tag: str, resp: Optional[LLMResponse], error: str = "") -> None:
        if resp is not None:
            LLM_TOKENS.labels(self.agent, step, "prompt").inc(resp.prompt_tokens)
            LLM_TOKENS.labels(self.agent, step, "completion").inc(resp.completion_tokens)
            LLM_COST.labels(self.agent, step).inc(resp.cost_usd)
            agg = self.cost_by_step.setdefault(step, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            agg["calls"] += 1
            agg["prompt_tokens"] += resp.prompt_tokens
            agg["completion_tokens"] += resp.completion_tokens
            agg["cost_usd"] += resp.cost_usd
        await self.event("llm_call", step=step, prompt_version=prompt_tag,
                         prompt_tokens=resp.prompt_tokens if resp else 0,
                         completion_tokens=resp.completion_tokens if resp else 0,
                         cost_usd=round(resp.cost_usd, 8) if resp else 0.0,
                         latency_ms=round(resp.latency_ms, 1) if resp else None,
                         retries=resp.retries if resp else None, model=resp.model if resp else "",
                         provider=resp.provider if resp else "", error=error,
                         response_excerpt=(resp.content[:600] if resp else ""))

    def step_timer(self, step: str) -> "_StepTimer":
        return _StepTimer(self, step)

    def cost_summary(self) -> dict[str, Any]:
        total = {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "calls": 0}
        for v in self.cost_by_step.values():
            for k in total:
                total[k] += v[k]
        total["cost_usd"] = round(total["cost_usd"], 8)
        return {"total": total, "by_step": self.cost_by_step}


class _StepTimer:
    def __init__(self, rec: Recorder, step: str):
        self.rec, self.step = rec, step

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        AGENT_STEP_LATENCY.labels(self.rec.agent, self.step).observe(dt)
        self.rec.step_ms[self.step] = self.rec.step_ms.get(self.step, 0.0) + dt * 1000
        self.ms = dt * 1000
        return False


class RetrieverPort(Protocol):
    async def search(self, query: str, k: int = 5, mode: str = "hybrid", doc_id: Optional[str] = None) -> list[dict]: ...
    async def clause(self, clause_id: str) -> Optional[dict]: ...
    async def related(self, clause_id: str, relations: Optional[list[str]] = None) -> list[dict]: ...


class HttpRetriever:
    def __init__(self, base_url: str, timeout_s: float, max_retries: int, base_delay_s: float, agent: str):
        self.base = base_url.rstrip("/")
        self.timeout_s, self.max_retries, self.base_delay_s, self.agent = timeout_s, max_retries, base_delay_s, agent
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))
        self.run_id = ""

    async def _get(self, method: str, path: str, **kw) -> Any:
        async def op():
            r = await self.client.request(method, self.base + path, headers={"X-Correlation-Id": self.run_id}, **kw)
            if r.status_code == 404:
                return None
            if 400 <= r.status_code < 500:
                raise NonRetryable(f"retriever {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        return await call_with_retry(op, timeout_s=self.timeout_s, max_retries=self.max_retries,
                                     base_delay_s=self.base_delay_s, agent=self.agent, target="retriever")

    async def search(self, query, k=5, mode="hybrid", doc_id=None):
        data = await self._get("POST", "/search", json={"query": query, "k": k, "mode": mode, "doc_id": doc_id})
        return (data or {}).get("passages", [])

    async def clause(self, clause_id):
        return await self._get("GET", f"/clause/{clause_id}")

    async def related(self, clause_id, relations=None):
        data = await self._get("GET", f"/related/{clause_id}", params={"relations": ",".join(relations or [])})
        return (data or {}).get("passages", [])


class LocalRetriever:
    """In-process retriever over a ClauseIndex (used by the evaluation CLI and tests)."""

    def __init__(self, index):
        self.index = index

    async def search(self, query, k=5, mode="hybrid", doc_id=None):
        return self.index.search(query, k=k, mode=mode, doc_id=doc_id)[0]

    async def clause(self, clause_id):
        return self.index.passage(clause_id, relation="lookup")

    async def related(self, clause_id, relations=None):
        return self.index.related(clause_id, set(relations) if relations else None)


@dataclass
class AgentDeps:
    llm: Optional[LLMPort]
    retriever: RetrieverPort
    prompts_dir: str
    prompt_versions: dict[str, str] = field(default_factory=dict)


def tool_metric(agent: str, tool: str, ok: bool) -> None:
    TOOL_CALLS.labels(agent, tool, "ok" if ok else "error").inc()
