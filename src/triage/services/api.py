"""Public API (FastAPI).

POST /v1/claims                 validate -> quarantine rejects (422) | rate-limit (429) | enqueue -> 202 {run_id}
GET  /v1/runs/{run_id}          status and, when finished, the RunResult
GET  /v1/runs/{run_id}/trace    full reconstruction: every event from every service, in order
POST /v1/runs/{run_id}/feedback satisfactory / unsatisfactory with a reason
GET  /v1/feedback/summary       aggregated feedback
GET  /v1/quality                live-sampled quality scores over time
GET  /v1/quarantine             recent rejected records with field-level errors
GET  /v1/contracts/{name}       published JSON Schemas
GET  /health, /ready, /metrics; GET / serves the web client.

Why asynchronous with polling: a run takes tens of seconds (several model calls and tool calls), longer than is
safe to hold an HTTP request through proxies and browsers. Submission returns a run id immediately; a Redis Streams
queue sits between submission and execution so bursts queue instead of failing, and workers scale independently.
Polling is the default because it works for every client (including the browser client and Locust) with no
inbound connectivity; callers that can receive webhooks may pass callback_url and get the result POSTed.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from ..config import get_settings
from ..contracts import ClaimRequest, Feedback, RunResult, cross_field_errors, schema_errors
from ..logging_setup import run_id_var
from ..metrics import CLAIMS_ACCEPTED, CLAIMS_REJECTED, FEEDBACK, RATE_LIMITED, RUN_CACHE
from ..prompts import load_prompt
from ..queue import STREAM_INTAKE, enqueue
from ..store import RateLimiter, RunStore, connect, new_run_id, run_cache_key
from ..tracing import carrier_from_current, current_trace_id, tracer
from .http_common import make_app

s = get_settings()
app = make_app("api", description="Air passenger compensation claim triage. Recommendations only; an officer decides.")
r = connect(s.redis_url)
store = RunStore(r, s.data_dir)
limiter = RateLimiter(r, s.rate_limit_per_minute, s.rate_limit_burst)
STATIC = os.path.join(os.path.dirname(__file__), "static")
CONTRACTS_DIR = os.environ.get("CONTRACTS_DIR", "/app/contracts")
JAEGER_UI = os.environ.get("JAEGER_UI_URL", "")


def _client_id(request: Request) -> str:
    return request.headers.get("X-Client-Id") or (request.client.host if request.client else "anonymous")


def _versions_tag() -> str:
    tags = [load_prompt(s.prompts_dir, p, s.prompt_version(p)).tag for p in ("extraction", "planner", "determination")]
    return "|".join(tags) + f"|planner={s.planner_mode}|det={s.determination_mode}"


@app.get("/health")
async def health():
    return {"status": "ok", "uptime_s": round(time.time() - app.state.started, 1)}


@app.get("/ready")
async def ready():
    await r.ping()
    workers = {}
    for agent in ("intake", "entitlement"):
        keys = [k async for k in r.scan_iter(f"worker:{agent}:*")]
        workers[agent] = len(keys)
    ok = all(workers.values())
    return JSONResponse({"ready": ok, "workers": workers}, status_code=200 if ok else 503)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.post("/v1/claims", status_code=202)
async def submit(request: Request):
    client = _client_id(request)
    cid = request.headers.get("X-Correlation-Id", "")
    allowed, retry_after = await limiter.allow(client)
    if not allowed:
        RATE_LIMITED.labels("rate").inc()
        return JSONResponse({"error": "rate_limited", "retry_after_s": round(retry_after, 2)}, status_code=429,
                            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))})
    try:
        payload: Any = await request.json()
    except Exception:
        payload = (await request.body()).decode("utf-8", "replace")[:5000]
        qid = await store.quarantine(payload, [{"field": "$", "code": "invalid_json", "message": "body is not valid JSON"}], client, cid)
        CLAIMS_REJECTED.labels("invalid_json").inc()
        return JSONResponse({"error": "invalid_request", "quarantine_id": qid, "run_started": False,
                             "errors": [{"field": "$", "code": "invalid_json", "message": "body is not valid JSON"}]}, status_code=422)
    try:
        claim = ClaimRequest.model_validate(payload)
        errors = cross_field_errors(claim)
        reason = "cross_field"
    except ValidationError as exc:
        errors, reason = schema_errors(exc), "schema"
    if errors:
        errs = [e.model_dump() for e in errors]
        qid = await store.quarantine(payload, errs, client, cid)
        CLAIMS_REJECTED.labels(reason).inc()
        return JSONResponse({"error": "invalid_request", "quarantine_id": qid, "run_started": False, "errors": errs},
                            status_code=422)
    if await store.inflight_incr(client) > s.max_inflight_per_client:
        await store.inflight_decr(client)
        RATE_LIMITED.labels("inflight").inc()
        return JSONResponse({"error": "too_many_inflight_runs", "limit": s.max_inflight_per_client}, status_code=429,
                            headers={"Retry-After": "5"})
    data = claim.model_dump(mode="json")
    run_id = new_run_id()
    run_id_var.set(run_id)
    key = run_cache_key(data, _versions_tag()) if s.run_cache_enabled else ""
    with tracer().start_as_current_span("api.submit", attributes={"run_id": run_id, "client_id": client}):
        trace_id = current_trace_id()
        await store.create_run(run_id, data, client, cid or run_id, key, trace_id)
        await store.mark_recent(run_id)
        await store.append_event(run_id, {"type": "submitted", "agent": "api", "client_id": client,
                                          "trace_id": trace_id, "client_reference": claim.client_reference})
        cached = await store.cache_get(key) if key else None
        if cached:
            RUN_CACHE.labels("hit").inc()
            result = dict(cached["result"], run_id=run_id, cached_from=cached["source_run"])
            result["cost"] = {"total": {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "calls": 0},
                              "by_step": {}, "note": "served from whole-run cache"}
            await store.set_fields(run_id, status=result["status"], result=result, finished_at=time.time())
            await store.append_event(run_id, {"type": "cache_hit", "agent": "api", "source_run": cached["source_run"]})
            await store.inflight_decr(client)
            CLAIMS_ACCEPTED.inc()
            return _accepted(run_id, result["status"], cached=True)
        if key:
            RUN_CACHE.labels("miss").inc()
        await enqueue(r, STREAM_INTAKE, run_id, {"run_id": run_id}, carrier_from_current())
    CLAIMS_ACCEPTED.inc()
    return _accepted(run_id, "queued")


def _accepted(run_id: str, status: str, cached: bool = False) -> JSONResponse:
    return JSONResponse({"run_id": run_id, "status": status, "status_url": f"/v1/runs/{run_id}",
                         "trace_url": f"/v1/runs/{run_id}/trace", "cached": cached}, status_code=202)


@app.get("/v1/runs/{run_id}")
async def get_run(run_id: str):
    run = await store.get_run(run_id)
    if not run:
        return JSONResponse({"error": "not_found"}, status_code=404)
    out = {"run_id": run_id, "status": run["status"], "submitted_at": run["submitted_at"],
           "updated_at": run.get("updated_at")}
    if run["status"] in ("completed", "escalated", "failed"):
        out["result"] = run.get("result")
    if JAEGER_UI and run.get("trace_id"):
        out["jaeger_trace_url"] = f"{JAEGER_UI.rstrip('/')}/trace/{run['trace_id']}"
    return out


@app.get("/v1/runs/{run_id}/trace")
async def get_trace(run_id: str):
    run = await store.get_run(run_id)
    if not run:
        return JSONResponse({"error": "not_found"}, status_code=404)
    events = await store.events(run_id)
    return {"run_id": run_id, "status": run["status"], "trace_id": run.get("trace_id"),
            "claim": run.get("claim"), "handoff": run.get("handoff"), "result": run.get("result"),
            "summary": _summarise(events), "events": events}


def _summarise(events: list[dict[str, Any]]) -> dict[str, Any]:
    tools = [e for e in events if e["type"] == "tool_call"]
    llm = [e for e in events if e["type"] == "llm_call"]
    ver = [e for e in events if e["type"] == "verification"]
    return {"tool_calls": len(tools), "queries": [e["args"] for e in tools if e["tool"] == "search_corpus"],
            "llm_calls": len(llm), "prompt_versions": sorted({e["prompt_version"] for e in llm}),
            "cost_usd": round(sum(e.get("cost_usd") or 0 for e in llm), 8),
            "verification": [{"attempt": e.get("attempt"), "passed": e["passed"], "source": e.get("candidate_source")} for e in ver]}


@app.post("/v1/runs/{run_id}/feedback", status_code=201)
async def feedback(run_id: str, request: Request):
    if not await store.get_run(run_id):
        return JSONResponse({"error": "not_found"}, status_code=404)
    try:
        fb = Feedback.model_validate(await request.json())
    except (ValidationError, json.JSONDecodeError) as exc:
        errs = [e.model_dump() for e in schema_errors(exc)] if isinstance(exc, ValidationError) else [{"field": "$", "code": "invalid_json", "message": "invalid JSON"}]
        return JSONResponse({"error": "invalid_request", "errors": errs}, status_code=422)
    await store.add_feedback(run_id, fb.model_dump())
    await store.append_event(run_id, {"type": "feedback", "agent": "api", **fb.model_dump()})
    FEEDBACK.labels(fb.verdict).inc()
    return {"stored": True}


@app.get("/v1/feedback/summary")
async def feedback_summary():
    return await store.feedback_summary()


@app.get("/v1/quality")
async def quality(limit: int = 200):
    samples = await store.quality_samples(limit)
    agg = {}
    for m in ("groundedness", "context_relevance", "faithfulness", "hallucination"):
        vals = [x[m] for x in samples if m in x]
        agg[m] = round(sum(vals) / len(vals), 4) if vals else None
    return {"n": len(samples), "mean": agg, "samples": samples}


@app.get("/v1/quarantine")
async def quarantine(limit: int = 50):
    return {"count": int(await r.get("quarantine:count") or 0), "records": await store.list_quarantine(limit)}


@app.get("/v1/contracts/{name}")
async def contract(name: str):
    if name == "claim":
        return ClaimRequest.model_json_schema()
    if name == "result":
        return RunResult.model_json_schema()
    if name == "feedback":
        return Feedback.model_json_schema()
    path = os.path.join(CONTRACTS_DIR, os.path.basename(name))
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "not_found"}, status_code=404)
