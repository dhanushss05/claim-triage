"""LLM stub service: an OpenAI-compatible /v1/chat/completions that returns fixed responses after a realistic delay.

Used for load testing (D6) so the test measures this system, not a provider's rate limits, and for running the
whole system with no API key. Responses are fixed per prompt id (X-Prompt-Id header):

  extraction     all-null JSON  -> the intake agent relies on its rule extractor
  planner        {"action": "finish"} -> the planner's checklist guard drives research
  determination  a deliberately naive answer that finds the cancellation amount and stops (pays INR 10,000 under
                 para 3.3.2). The verifier must catch it; this is how self-verification is demonstrated.

Fault injection for the circuit-breaker demo: POST /admin/mode {"mode": "ok" | "error" | "timeout" | "slow"}.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import time

from fastapi import HTTPException, Request
from pydantic import BaseModel

from ..stub_responses import STUB_RESPONSES
from .http_common import make_app

app = make_app("llm-stub")
MEAN_MS = float(os.environ.get("STUB_LATENCY_MS", "800"))
JITTER = float(os.environ.get("STUB_LATENCY_JITTER", "0.35"))
STATE = {"mode": os.environ.get("STUB_MODE", "ok")}

FIXED = STUB_RESPONSES


class Mode(BaseModel):
    mode: str


@app.get("/health")
async def health():
    return {"status": "ok", "mode": STATE["mode"]}


@app.get("/admin/mode")
async def get_mode():
    return STATE


@app.post("/admin/mode")
async def set_mode(m: Mode):
    if m.mode not in ("ok", "error", "timeout", "slow"):
        raise HTTPException(400, "mode must be ok|error|timeout|slow")
    STATE["mode"] = m.mode
    return STATE


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    prompt_id = request.headers.get("X-Prompt-Id", "")
    mode = STATE["mode"]
    delay = max(0.05, random.gauss(MEAN_MS, MEAN_MS * JITTER) / 1000)
    if mode == "error":
        await asyncio.sleep(0.05)
        raise HTTPException(503, "stub provider unavailable (fault injection)")
    if mode == "timeout":
        await asyncio.sleep(120)
    if mode == "slow":
        delay *= 6
    await asyncio.sleep(delay)
    content = json.dumps(FIXED.get(prompt_id, {"note": "stub"}))
    prompt_chars = sum(len(m.get("content", "")) for m in body.get("messages", []))
    return {"id": f"stub-{int(time.time() * 1000)}", "object": "chat.completion", "model": body.get("model", "stub"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": max(1, prompt_chars // 4), "completion_tokens": max(1, len(content) // 4),
                      "total_tokens": max(1, prompt_chars // 4) + max(1, len(content) // 4)}}
