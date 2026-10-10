"""Client for the LLM gateway. The provider sits behind the gateway, so it can be changed by configuration.

Per-call timeout, bounded retry with exponential backoff, and no retry when the gateway reports the circuit open
or the budget exhausted: those raise ProviderUnavailable and the caller takes its fallback path.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import httpx

from .resilience import NonRetryable, RetryStats, call_with_retry


class ProviderUnavailable(Exception):
    """The model cannot be used for this call (circuit open, budget exhausted, or retries exhausted)."""


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    model: str = ""
    provider: str = ""
    retries: int = 0
    parsed: Optional[dict[str, Any]] = field(default=None)


class LLMPort(Protocol):
    async def chat(self, *, messages: list[dict[str, str]], prompt_id: str, prompt_version: str, run_id: str,
                   agent: str, step: str, max_tokens: int = 600, json_mode: bool = True) -> LLMResponse: ...


def parse_json(text: str) -> Optional[dict[str, Any]]:
    """Extract the first JSON object from a model response (tolerates code fences and prose)."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


class GatewayLLM:
    def __init__(self, base_url: str, timeout_s: float, max_retries: int, base_delay_s: float):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s))

    async def chat(self, *, messages, prompt_id, prompt_version, run_id, agent, step, max_tokens=600,
                   json_mode=True) -> LLMResponse:
        body = {"messages": messages, "prompt_id": prompt_id, "prompt_version": prompt_version, "run_id": run_id,
                "agent": agent, "step": step, "max_tokens": max_tokens, "json_mode": json_mode}
        stats = RetryStats()

        async def op() -> LLMResponse:
            t0 = time.perf_counter()
            resp = await self.client.post(f"{self.base_url}/v1/chat", json=body,
                                          headers={"X-Run-Id": run_id, "X-Correlation-Id": run_id})
            if resp.status_code == 503:
                detail = resp.json().get("detail", {}) if resp.headers.get("content-type", "").startswith("application/json") else {}
                code = detail.get("code") if isinstance(detail, dict) else None
                if code in ("circuit_open", "budget_exhausted"):
                    raise NonRetryable(code)
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise NonRetryable(f"gateway rejected request: {resp.status_code} {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            return LLMResponse(content=data["content"], prompt_tokens=data["usage"]["prompt_tokens"],
                               completion_tokens=data["usage"]["completion_tokens"], cost_usd=data["cost_usd"],
                               latency_ms=(time.perf_counter() - t0) * 1000, model=data.get("model", ""),
                               provider=data.get("provider", ""))

        try:
            out = await call_with_retry(op, timeout_s=self.timeout_s, max_retries=self.max_retries,
                                        base_delay_s=self.base_delay_s, agent=agent, target="llm", stats=stats)
        except NonRetryable as exc:
            raise ProviderUnavailable(str(exc)) from exc
        except Exception as exc:  # retries exhausted
            raise ProviderUnavailable(f"llm call failed after {stats.attempts} attempts: {type(exc).__name__}") from exc
        out.retries = stats.retries
        out.parsed = parse_json(out.content) if json_mode else None
        return out
