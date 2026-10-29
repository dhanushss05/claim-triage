"""LLM gateway: the one interface to the model provider.

* Provider is chosen by configuration (LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY): any OpenAI-compatible
  chat-completions endpoint - the bundled stub, Google Gemini's OpenAI-compatible endpoint, Groq, OpenRouter, or a
  local Ollama. Changing provider is a .env change.
* Timeout on every provider call (LLM_PROVIDER_TIMEOUT_S). The gateway does not retry; callers retry with backoff,
  so retries are not multiplied across layers.
* Circuit breaker on the provider, with its state in Redis so every replica and the dashboard see the same state:
  CLOSED -> (BREAKER_FAILURE_THRESHOLD consecutive failures) -> OPEN for BREAKER_OPEN_SECONDS (calls fail fast with
  503 circuit_open, callers degrade) -> HALF_OPEN (one trial call) -> CLOSED on success / OPEN on failure.
* Cost recorded at call time from token usage and configured prices; a daily budget refuses calls once spent.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings
from ..metrics import BREAKER_STATE, BUDGET_SPENT, LLM_LATENCY, LLM_REQUESTS
from ..store import connect
from .http_common import make_app

logger = logging.getLogger("llm-gateway")
app = make_app("llm-gateway")
s = get_settings()
PROVIDER, BASE_URL, MODEL, API_KEY = s.llm_provider, s.llm_base_url.rstrip("/"), s.llm_model, s.llm_api_key
r = connect(s.redis_url)
client = httpx.AsyncClient(timeout=httpx.Timeout(s.llm_provider_timeout_s))
STATE_CODE = {"closed": 0, "half_open": 1, "open": 2}

# Pacing to the provider's quota, shared by every gateway replica: each call reserves the next free slot
# (60 / LLM_MAX_RPM seconds apart). A caller that would wait longer than LLM_RATE_WAIT_MAX_S gets 503 and the
# agent takes its fallback path, instead of the provider answering 429 and opening the breaker.
_PACE = r.register_script("""
local now = tonumber(ARGV[1]); local gap = tonumber(ARGV[2]); local maxwait = tonumber(ARGV[3])
local slot = math.max(now, tonumber(redis.call('GET', KEYS[1]) or '0'))
if slot - now > maxwait then return tostring(-1) end
redis.call('SET', KEYS[1], tostring(slot + gap), 'EX', 3600)
return tostring(slot - now)
""")


async def pace() -> bool:
    if s.llm_max_rpm <= 0:
        return True
    wait = float(await _PACE(keys=["llm:pace"], args=[time.time(), 60.0 / s.llm_max_rpm, s.llm_rate_wait_max_s]))
    if wait < 0:
        return False
    if wait > 0:
        await asyncio.sleep(wait)
    return True


class ChatRequest(BaseModel):
    messages: list[dict]
    prompt_id: str
    prompt_version: str
    run_id: str = ""
    agent: str = ""
    step: str = ""
    max_tokens: int = Field(default=600, ge=1, le=4000)
    json_mode: bool = True


class Breaker:
    KEY = "breaker:llm"

    async def state(self) -> str:
        data = await r.hgetall(self.KEY)
        st = data.get("state", "closed")
        if st == "closed":
            BREAKER_STATE.set(0)
            return "closed"
        if st == "open" and time.time() < float(data.get("opened_at", 0)) + s.breaker_open_seconds:
            BREAKER_STATE.set(2)
            return "open"
        # cool-down elapsed (or a half-open trial was lost): exactly one caller wins the trial slot
        if await r.set(self.KEY + ":trial", "1", nx=True, ex=s.breaker_open_seconds):
            await r.hset(self.KEY, "state", "half_open")
            BREAKER_STATE.set(1)
            return "half_open_trial"
        BREAKER_STATE.set(1 if st == "half_open" else 2)
        return "open"

    async def success(self) -> None:
        await r.hset(self.KEY, mapping={"state": "closed", "failures": 0})
        await r.delete(self.KEY + ":trial")
        BREAKER_STATE.set(0)

    async def failure(self, reason: str) -> None:
        failures = await r.hincrby(self.KEY, "failures", 1)
        st = (await r.hget(self.KEY, "state")) or "closed"
        if st == "half_open" or failures >= s.breaker_failure_threshold:
            await r.hset(self.KEY, mapping={"state": "open", "opened_at": time.time(), "last_reason": reason})
            await r.delete(self.KEY + ":trial")
            BREAKER_STATE.set(2)
            logger.warning("circuit opened", extra={"fields": {"reason": reason, "failures": failures}})

    async def info(self) -> dict:
        data = await r.hgetall(self.KEY)
        return {"state": data.get("state", "closed"), "failures": int(data.get("failures", 0)),
                "opened_at": data.get("opened_at"), "last_reason": data.get("last_reason"),
                "threshold": s.breaker_failure_threshold, "open_seconds": s.breaker_open_seconds}


breaker = Breaker()


def _budget_key() -> str:
    return "llm:spend:" + datetime.now(timezone.utc).strftime("%Y%m%d")


@app.get("/health")
async def health():
    await r.ping()
    return {"status": "ok", "provider": PROVIDER, "breaker": (await breaker.info())["state"]}


@app.get("/admin/breaker")
async def breaker_info():
    return await breaker.info()


@app.post("/admin/breaker/reset")
async def breaker_reset():
    await breaker.success()
    return await breaker.info()


@app.post("/v1/chat")
async def chat(req: ChatRequest):
    st = await breaker.state()
    if st == "open":
        LLM_REQUESTS.labels(PROVIDER, "circuit_open").inc()
        raise HTTPException(503, {"code": "circuit_open", "message": "provider circuit is open; use fallback"})
    spent = float(await r.get(_budget_key()) or 0)
    BUDGET_SPENT.set(spent)
    if spent >= s.llm_daily_budget_usd:
        LLM_REQUESTS.labels(PROVIDER, "budget_exhausted").inc()
        raise HTTPException(503, {"code": "budget_exhausted", "message": "daily LLM budget reached"})
    if not await pace():
        LLM_REQUESTS.labels(PROVIDER, "paced_out").inc()
        raise HTTPException(503, {"code": "rate_paced", "message": "provider request rate at its configured limit"})
    body = {"model": MODEL, "messages": req.messages, "max_tokens": req.max_tokens, "temperature": 0}
    if req.json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"X-Prompt-Id": req.prompt_id, "X-Run-Id": req.run_id}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{BASE_URL}/chat/completions", json=body, headers=headers)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise httpx.HTTPStatusError(f"provider {resp.status_code}", request=resp.request, response=resp)
        if resp.status_code >= 400:
            logger.warning("provider_rejected", extra={"fields": {"status": resp.status_code,
                                                              "provider_body": resp.text[:300]}})
            LLM_REQUESTS.labels(PROVIDER, "client_error").inc()
            raise HTTPException(502, {"code": "provider_rejected", "message": resp.text[:300]})
        data = resp.json()
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        detail = ""
        if isinstance(exc, httpx.HTTPStatusError):
            detail = exc.response.text[:300]
        logger.warning("provider_error", extra={"fields": {"error": type(exc).__name__, "detail": str(exc)[:200],
                                                           "provider_body": detail}})
        await breaker.failure(type(exc).__name__)
        LLM_REQUESTS.labels(PROVIDER, "error").inc()
        raise HTTPException(503, {"code": "provider_error", "message": f"{type(exc).__name__}: {str(exc)[:200]}"})
    latency = time.perf_counter() - t0
    await breaker.success()
    usage = data.get("usage") or {}
    pt, ct = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    cost = pt / 1e6 * s.llm_price_prompt_per_1m + ct / 1e6 * s.llm_price_completion_per_1m
    await r.incrbyfloat(_budget_key(), cost)
    await r.expire(_budget_key(), 3 * 86400)
    LLM_REQUESTS.labels(PROVIDER, "ok").inc()
    LLM_LATENCY.labels(PROVIDER, req.step).observe(latency)
    content: Optional[str] = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return {"content": content, "usage": {"prompt_tokens": pt, "completion_tokens": ct}, "cost_usd": cost,
            "model": MODEL, "provider": PROVIDER, "latency_ms": round(latency * 1000, 1)}
