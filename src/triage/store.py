"""Redis-backed run store, event log, quarantine, feedback, caches and per-client rate limiting.

Run reconstruction: every step of every service appends a JSON event to `run:{id}:events`. GET
/v1/runs/{id}/trace returns them in order - tool calls, queries, passages, prompt versions, per-step latency and
cost, and the verification outcome. Redis runs with append-only persistence (see docker-compose.yml).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis

RUN_TTL_S = 7 * 86400


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_id() -> str:
    return "run_" + uuid.uuid4().hex[:20]


def connect(url: str) -> aioredis.Redis:
    return aioredis.from_url(url, decode_responses=True, health_check_interval=15)


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass  # Redis remains the system of record; the file copy is a convenience for inspection


class RunStore:
    def __init__(self, r: aioredis.Redis, data_dir: str = "/data"):
        self.r = r
        self.data_dir = data_dir

    # --- runs ---------------------------------------------------------------------------------------------
    async def create_run(self, run_id: str, claim: dict[str, Any], client_id: str, correlation_id: str,
                         cache_key: str = "", trace_id: str = "") -> None:
        key = f"run:{run_id}"
        now = time.time()
        await self.r.hset(key, mapping={
            "run_id": run_id, "status": "queued", "client_id": client_id, "correlation_id": correlation_id,
            "submitted_at": utcnow_iso(), "submitted_ts": f"{now:.6f}", "claim": json.dumps(claim, default=str),
            "cache_key": cache_key, "trace_id": trace_id, "updated_at": utcnow_iso(),
        })
        await self.r.expire(key, RUN_TTL_S)

    async def set_fields(self, run_id: str, **fields: Any) -> None:
        mapping = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)) for k, v in fields.items()}
        mapping["updated_at"] = utcnow_iso()
        await self.r.hset(f"run:{run_id}", mapping=mapping)

    async def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        data = await self.r.hgetall(f"run:{run_id}")
        if not data:
            return None
        for k in ("claim", "result", "handoff"):
            if data.get(k):
                try:
                    data[k] = json.loads(data[k])
                except json.JSONDecodeError:
                    pass
        return data

    async def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        event = {"ts": utcnow_iso(), **event}
        key = f"run:{run_id}:events"
        await self.r.rpush(key, json.dumps(event, default=str))
        await self.r.expire(key, RUN_TTL_S)

    async def events(self, run_id: str) -> list[dict[str, Any]]:
        raw = await self.r.lrange(f"run:{run_id}:events", 0, -1)
        return [json.loads(x) for x in raw]

    async def recent_runs(self, limit: int = 50) -> list[str]:
        return await self.r.lrange("runs:recent", 0, limit - 1)

    async def mark_recent(self, run_id: str) -> None:
        await self.r.lpush("runs:recent", run_id)
        await self.r.ltrim("runs:recent", 0, 999)

    # --- quarantine ------------------------------------------------------------------------------------------
    async def quarantine(self, payload: Any, errors: list[dict[str, Any]], client_id: str, correlation_id: str) -> str:
        qid = "q_" + uuid.uuid4().hex[:16]
        record = {"quarantine_id": qid, "ts": utcnow_iso(), "client_id": client_id, "correlation_id": correlation_id,
                  "errors": errors, "payload": payload}
        await self.r.lpush("quarantine", json.dumps(record, default=str))
        await self.r.ltrim("quarantine", 0, 4999)
        await self.r.incr("quarantine:count")
        _append_jsonl(os.path.join(self.data_dir, "quarantine", "quarantine.jsonl"), record)
        return qid

    async def list_quarantine(self, limit: int = 50) -> list[dict[str, Any]]:
        return [json.loads(x) for x in await self.r.lrange("quarantine", 0, limit - 1)]

    # --- feedback -----------------------------------------------------------------------------------------
    async def add_feedback(self, run_id: str, fb: dict[str, Any]) -> None:
        record = {"run_id": run_id, "ts": utcnow_iso(), **fb}
        await self.r.rpush(f"feedback:{run_id}", json.dumps(record))
        await self.r.lpush("feedback:all", json.dumps(record))
        await self.r.hincrby("feedback:agg", fb["verdict"], 1)
        _append_jsonl(os.path.join(self.data_dir, "feedback", "feedback.jsonl"), record)

    async def feedback_summary(self) -> dict[str, Any]:
        agg = {k: int(v) for k, v in (await self.r.hgetall("feedback:agg")).items()}
        total = sum(agg.values())
        recent = [json.loads(x) for x in await self.r.lrange("feedback:all", 0, 19)]
        reasons: dict[str, int] = {}
        for x in [json.loads(x) for x in await self.r.lrange("feedback:all", 0, 499)]:
            if x["verdict"] == "unsatisfactory":
                key = x["reason"].strip().lower()[:60]
                reasons[key] = reasons.get(key, 0) + 1
        return {"total": total, "counts": agg,
                "satisfaction_rate": (agg.get("satisfactory", 0) / total) if total else None,
                "top_unsatisfactory_reasons": sorted(reasons.items(), key=lambda kv: -kv[1])[:10],
                "recent": recent}

    # --- live quality samples ----------------------------------------------------------------------------------
    async def add_quality_sample(self, sample: dict[str, Any]) -> None:
        await self.r.lpush("quality:samples", json.dumps(sample))
        await self.r.ltrim("quality:samples", 0, 1999)

    async def quality_samples(self, limit: int = 200) -> list[dict[str, Any]]:
        return [json.loads(x) for x in await self.r.lrange("quality:samples", 0, limit - 1)]

    # --- whole-run cache -------------------------------------------------------------------------------------
    async def cache_get(self, key: str) -> Optional[dict[str, Any]]:
        raw = await self.r.get(f"runcache:{key}")
        return json.loads(raw) if raw else None

    async def cache_set(self, key: str, value: dict[str, Any], ttl: int) -> None:
        await self.r.set(f"runcache:{key}", json.dumps(value, default=str), ex=ttl)

    # --- in-flight accounting per client -------------------------------------------------------------------------
    async def inflight_incr(self, client_id: str) -> int:
        key = f"inflight:{client_id}"
        n = await self.r.incr(key)
        await self.r.expire(key, 3600)
        return int(n)

    async def inflight_decr(self, client_id: str) -> None:
        key = f"inflight:{client_id}"
        n = await self.r.decr(key)
        if int(n) < 0:
            await self.r.set(key, 0, ex=3600)


def run_cache_key(claim: dict[str, Any], versions: str) -> str:
    """Normalised claim hash. Excludes fields that do not affect the outcome (reference, callback, contact)."""
    c = json.loads(json.dumps(claim, default=str))
    c.pop("client_reference", None)
    c.pop("callback_url", None)
    pax = c.get("passenger", {})
    for k in ("full_name", "email", "phone"):
        pax.pop(k, None)
    c.get("booking", {}).pop("pnr", None)
    c["narrative"] = " ".join(str(c.get("narrative", "")).lower().split())
    blob = json.dumps(c, sort_keys=True) + "|" + versions
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


# Token bucket in one atomic Lua script: refill at `rate` tokens/sec up to `burst`, take one.
_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1]) or burst
local ts = tonumber(state[2]) or now
tokens = math.min(burst, tokens + (now - ts) * rate)
local allowed = 0
local retry_after = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  retry_after = (1 - tokens) / rate
end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, 3600)
return {allowed, tostring(retry_after)}
"""


class RateLimiter:
    def __init__(self, r: aioredis.Redis, per_minute: int, burst: int):
        self.r = r
        self.rate = per_minute / 60.0
        self.burst = burst
        self._script = r.register_script(_BUCKET_LUA)

    async def allow(self, client_id: str) -> tuple[bool, float]:
        allowed, retry_after = await self._script(keys=[f"ratelimit:{client_id}"],
                                                  args=[self.rate, self.burst, time.time()])
        return bool(int(allowed)), float(retry_after)
