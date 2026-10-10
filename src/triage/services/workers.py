"""Queue workers for the two agents.

    python -m triage.services.workers intake        # intake agent (classification, extraction, guardrails)
    python -m triage.services.workers entitlement   # entitlement agent (LangGraph research/verify loop)

Hand-off: API --(runs:intake)--> intake agent --(runs:entitlement, IntakeHandoff)--> entitlement agent --> result.
Failure mid-run: the message stays pending; another replica reclaims it after CLAIM_IDLE_MS and re-runs the step
(steps are idempotent: they only append events and overwrite their own outputs). After MAX_DELIVERIES the run is
closed as failed + escalated. A run that exceeds RUN_DEADLINE_S is closed the same way. Neither is ever returned
as a confident answer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import signal
import socket
import sys
import time
from typing import Any

import httpx

from ..agents.entitlement import build_graph, initial_state
from ..agents.intake import run_intake
from ..agents.runtime import AgentDeps, HttpRetriever, Recorder
from ..config import get_settings
from ..llm_client import GatewayLLM
from ..logging_setup import bind, setup_logging
from ..metrics import (AGENT_STEPS, AGENT_TASK_LATENCY, AGENT_TASKS, AGENT_TOOL_CALLS, LIVE_EVAL, LIVE_EVAL_HIST,
                       RUN_E2E, RUNS_FINISHED, WORKER_INFLIGHT, serve_metrics)
from ..prompts import load_prompt
from ..quality import score_run
from ..queue import STREAM_ENTITLEMENT, STREAM_INTAKE, StreamConsumer, enqueue
from ..store import RunStore, connect
from ..tracing import attached, carrier_from_current, setup_tracing, tracer

logger = logging.getLogger("worker")
s = get_settings()


def prompt_versions() -> dict[str, str]:
    return {pid: s.prompt_version(pid) for pid in ("extraction", "planner", "determination")}


def prompt_tags() -> dict[str, str]:
    return {pid: load_prompt(s.prompts_dir, pid, v).tag for pid, v in prompt_versions().items()}


async def heartbeat(r, agent: str) -> None:
    key = f"worker:{agent}:{socket.gethostname()}"
    while True:
        try:
            await r.set(key, str(time.time()), ex=20)
        except Exception:
            pass
        await asyncio.sleep(5)


async def close_failed(store: RunStore, run_id: str, reason: str) -> None:
    """Close a run that could not complete: failed status, escalated, never a confident answer."""
    run = await store.get_run(run_id) or {}
    result = {"run_id": run_id, "status": "failed", "route": "escalate_to_human", "citations": [],
              "justification": "", "entitlement": {"compensation_inr": 0, "computation": "", "refund_due": False,
                                                   "facilities": [], "downgrade_refund_inr": 0},
              "escalation": {"required": True, "reasons": [f"Run did not complete: {reason}"],
                             "path": ["Customer relations officer"]},
              "verification": {"passed": False, "attempts": 0, "checks": []}, "degraded": True, "confidence": "low",
              "basis": "incomplete run", "guardrail_flags": [], "prompt_versions": prompt_tags(),
              "disclaimer": "This run did not complete. No recommendation is made; a customer relations officer must review the claim."}
    await store.set_fields(run_id, status="failed", result=result, finished_at=time.time())
    await store.append_event(run_id, {"type": "run_failed", "reason": reason})
    RUNS_FINISHED.labels("failed", "escalate_to_human", "true").inc()
    if run.get("client_id"):
        await store.inflight_decr(run["client_id"])


async def intake_main() -> None:
    r = connect(s.redis_url)
    store = RunStore(r, s.data_dir)
    llm = GatewayLLM(s.llm_gateway_url, s.llm_call_timeout_s, s.llm_max_retries, s.retry_base_delay_s)
    deps = AgentDeps(llm=llm, retriever=None, prompts_dir=s.prompts_dir, prompt_versions=prompt_versions())

    async def handle(run_id: str, payload: dict[str, Any], carrier: dict[str, str]) -> None:
        bind(run_id=run_id, correlation_id=run_id, agent="intake")
        with attached(carrier), tracer().start_as_current_span("intake_agent.run", attributes={"run_id": run_id}):
            WORKER_INFLIGHT.labels("intake").inc()
            t0 = time.perf_counter()
            try:
                run = await store.get_run(run_id)
                if not run or run.get("status") in ("completed", "escalated", "failed"):
                    return
                await store.set_fields(run_id, status="intake")
                rec = Recorder(run_id, "intake", store)
                await rec.event("agent_start", worker=socket.gethostname())
                handoff = await run_intake(run_id, run["claim"], deps, rec)
                await store.set_fields(run_id, status="researching", handoff=handoff,
                                       intake_ms=round((time.perf_counter() - t0) * 1000, 1),
                                       intake_cost=rec.cost_summary())
                await rec.event("handoff", to="entitlement", category=handoff["category"],
                                guardrail_flags=handoff["guardrail_flags"], step_ms=rec.step_ms)
                await enqueue(r, STREAM_ENTITLEMENT, run_id, handoff, carrier_from_current())
                AGENT_TASKS.labels("intake", "success").inc()
            except Exception:
                AGENT_TASKS.labels("intake", "error").inc()
                raise
            finally:
                AGENT_TASK_LATENCY.labels("intake").observe(time.perf_counter() - t0)
                WORKER_INFLIGHT.labels("intake").dec()

    async def on_dead(run_id: str, payload: dict[str, Any], reason: str) -> None:
        AGENT_TASKS.labels("intake", "dead_letter").inc()
        await close_failed(store, run_id, f"intake agent: {reason}")

    asyncio.create_task(heartbeat(r, "intake"))
    await StreamConsumer(r, STREAM_INTAKE, "intake-agents", handle, on_dead, s.worker_concurrency).run()


async def entitlement_main() -> None:
    r = connect(s.redis_url)
    store = RunStore(r, s.data_dir)
    llm = GatewayLLM(s.llm_gateway_url, s.llm_call_timeout_s, s.llm_max_retries, s.retry_base_delay_s)
    retriever = HttpRetriever(s.retriever_url, s.tool_timeout_s, 2, s.retry_base_delay_s, "entitlement")
    deps = AgentDeps(llm=llm, retriever=retriever, prompts_dir=s.prompts_dir, prompt_versions=prompt_versions())
    graph = build_graph(deps, s)
    callbacks = httpx.AsyncClient(timeout=5)

    async def deliver_callback(url: str, result: dict[str, Any]) -> None:
        for attempt in range(3):
            try:
                resp = await callbacks.post(url, json=result)
                if resp.status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2 ** attempt)

    async def handle(run_id: str, handoff: dict[str, Any], carrier: dict[str, str]) -> None:
        bind(run_id=run_id, correlation_id=run_id, agent="entitlement")
        retriever.run_id = run_id
        with attached(carrier), tracer().start_as_current_span("entitlement_agent.run", attributes={"run_id": run_id}):
            WORKER_INFLIGHT.labels("entitlement").inc()
            t0 = time.perf_counter()
            try:
                run = await store.get_run(run_id)
                if not run or run.get("status") in ("completed", "escalated", "failed"):
                    return
                submitted = float(run.get("submitted_ts") or time.time())
                remaining = s.run_deadline_s - (time.time() - submitted)
                if remaining <= 0:
                    await close_failed(store, run_id, f"deadline of {s.run_deadline_s:.0f}s exceeded before research started")
                    AGENT_TASKS.labels("entitlement", "deadline").inc()
                    return
                rec = Recorder(run_id, "entitlement", store)
                await rec.event("agent_start", worker=socket.gethostname(), handoff_category=handoff.get("category"))
                state = initial_state(run_id, handoff)
                try:
                    final = await asyncio.wait_for(
                        graph.ainvoke(state, config={"configurable": {"rec": rec}, "recursion_limit": 100}),
                        timeout=min(remaining, s.agent_max_wall_s + 15))
                except asyncio.TimeoutError:
                    await close_failed(store, run_id, "deadline exceeded during research")
                    AGENT_TASKS.labels("entitlement", "deadline").inc()
                    return
                result = final["result"]
                intake_cost = run.get("intake_cost")
                if isinstance(intake_cost, str):
                    intake_cost = json.loads(intake_cost)
                ent_cost = rec.cost_summary()
                by_step = {**(intake_cost or {}).get("by_step", {}), **ent_cost["by_step"]}
                total = {k: (intake_cost or {}).get("total", {}).get(k, 0) + ent_cost["total"][k]
                         for k in ("prompt_tokens", "completion_tokens", "cost_usd", "calls")}
                result["cost"] = {"total": total, "by_step": by_step}
                result["prompt_versions"] = {**(handoff.get("prompt_versions") or {}), **prompt_tags()}
                e2e = time.time() - submitted
                result["timings_ms"] = {"end_to_end": round(e2e * 1000, 1), "intake": float(run.get("intake_ms") or 0),
                                        "entitlement": round((time.perf_counter() - t0) * 1000, 1),
                                        **{f"step:{k}": round(v, 1) for k, v in rec.step_ms.items()}}
                await store.set_fields(run_id, status=result["status"], result=result, finished_at=time.time())
                await rec.event("run_complete", route=result["route"], end_to_end_ms=result["timings_ms"]["end_to_end"],
                                cost=result["cost"])
                RUN_E2E.labels(result["route"]).observe(e2e)
                RUNS_FINISHED.labels(result["status"], result["route"], str(result["degraded"]).lower()).inc()
                AGENT_STEPS.labels("entitlement").observe(final.get("steps", 0))
                AGENT_TOOL_CALLS.labels("entitlement").observe(final.get("tool_calls", 0))
                AGENT_TASKS.labels("entitlement", "success" if result["verification"]["passed"] else "escalated_unverified").inc()
                if run.get("cache_key") and result["status"] != "failed" and not result["degraded"]:
                    await store.cache_set(run["cache_key"], {"result": result, "source_run": run_id}, s.run_cache_ttl_s)
                if random.random() < s.live_eval_sample_rate:
                    scores = score_run(result, final["evidence"], final["facts"])
                    for m, v in scores.items():
                        LIVE_EVAL_HIST.labels(m).observe(v)
                        LIVE_EVAL.labels(m).set(v)
                    await store.add_quality_sample({"run_id": run_id, "ts": time.time(), **scores})
                    await rec.event("live_eval", **scores)
                if run.get("client_id"):
                    await store.inflight_decr(run["client_id"])
                cb = (run.get("claim") or {}).get("callback_url")
                if cb:
                    asyncio.create_task(deliver_callback(cb, result))
            except Exception:
                AGENT_TASKS.labels("entitlement", "error").inc()
                raise
            finally:
                AGENT_TASK_LATENCY.labels("entitlement").observe(time.perf_counter() - t0)
                WORKER_INFLIGHT.labels("entitlement").dec()

    async def on_dead(run_id: str, payload: dict[str, Any], reason: str) -> None:
        AGENT_TASKS.labels("entitlement", "dead_letter").inc()
        await close_failed(store, run_id, f"entitlement agent: {reason}")

    asyncio.create_task(heartbeat(r, "entitlement"))
    await StreamConsumer(r, STREAM_ENTITLEMENT, "entitlement-agents", handle, on_dead, s.worker_concurrency).run()


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which not in ("intake", "entitlement"):
        print("usage: python -m triage.services.workers intake|entitlement")
        sys.exit(2)
    name = f"{which}-agent"
    setup_logging(name, s.log_level)
    setup_tracing(name, s.otel_endpoint)
    serve_metrics(int(__import__("os").environ.get("METRICS_PORT", "9100")))
    loop = asyncio.new_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, loop.stop)
    loop.run_until_complete(intake_main() if which == "intake" else entitlement_main())


if __name__ == "__main__":
    main()
