"""Redis Streams work queue with consumer groups.

Why Streams (not a bare list): a message stays in the group's pending list until the worker acknowledges it. If a
worker dies mid-run, another replica reclaims the message with XAUTOCLAIM after CLAIM_IDLE_MS and re-runs it. After
MAX_DELIVERIES attempts the message is moved to the dead-letter stream and the run is closed as failed and
escalated, never returned as a confident answer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from typing import Any, Awaitable, Callable

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from .metrics import QUEUE_DEPTH

logger = logging.getLogger(__name__)

STREAM_INTAKE = "runs:intake"
STREAM_ENTITLEMENT = "runs:entitlement"
STREAM_DEAD = "runs:dead"
CLAIM_IDLE_MS = int(os.environ.get("CLAIM_IDLE_MS", "150000"))
MAX_DELIVERIES = int(os.environ.get("MAX_DELIVERIES", "3"))

Handler = Callable[[str, dict[str, Any], dict[str, str]], Awaitable[None]]
DeadHandler = Callable[[str, dict[str, Any], str], Awaitable[None]]


async def enqueue(r: aioredis.Redis, stream: str, run_id: str, payload: dict[str, Any], carrier: dict[str, str]) -> str:
    return await r.xadd(stream, {"run_id": run_id, "payload": json.dumps(payload, default=str),
                                 "carrier": json.dumps(carrier)}, maxlen=100_000, approximate=True)


async def ensure_group(r: aioredis.Redis, stream: str, group: str) -> None:
    try:
        await r.xgroup_create(stream, group, id="0", mkstream=True)
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


class StreamConsumer:
    def __init__(self, r: aioredis.Redis, stream: str, group: str, handler: Handler, on_dead: DeadHandler,
                 concurrency: int = 4):
        self.r = r
        self.stream = stream
        self.group = group
        self.handler = handler
        self.on_dead = on_dead
        self.consumer = f"{socket.gethostname()}-{os.getpid()}"
        self.sem = asyncio.Semaphore(concurrency)
        self.concurrency = concurrency
        self._tasks: set[asyncio.Task] = set()
        self._stop = asyncio.Event()

    async def _deliveries(self, msg_id: str) -> int:
        info = await self.r.xpending_range(self.stream, self.group, min=msg_id, max=msg_id, count=1)
        return int(info[0]["times_delivered"]) if info else 1

    async def _process(self, msg_id: str, fields: dict[str, str]) -> None:
        try:
            run_id = fields.get("run_id", "")
            payload = json.loads(fields.get("payload", "{}"))
            carrier = json.loads(fields.get("carrier", "{}"))
            delivered = await self._deliveries(msg_id)
            if delivered > MAX_DELIVERIES:
                await self.r.xadd(STREAM_DEAD, {**fields, "reason": f"exceeded {MAX_DELIVERIES} deliveries",
                                                "stream": self.stream})
                await self.on_dead(run_id, payload, f"worker failed {MAX_DELIVERIES} times")
            else:
                await self.handler(run_id, payload, carrier)
            await self.r.xack(self.stream, self.group, msg_id)
        except Exception:  # leave un-acked: another replica reclaims it after CLAIM_IDLE_MS
            logger.exception("handler failed; message left pending for reclaim", extra={"fields": {"msg_id": msg_id}})
        finally:
            self.sem.release()

    async def _spawn(self, msg_id: str, fields: dict[str, str]) -> None:
        await self.sem.acquire()
        task = asyncio.create_task(self._process(msg_id, fields))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def run(self) -> None:
        await ensure_group(self.r, self.stream, self.group)
        logger.info("consumer started", extra={"fields": {"stream": self.stream, "consumer": self.consumer}})
        reclaim_every = 10
        loops = 0
        while not self._stop.is_set():
            try:
                loops += 1
                if loops % reclaim_every == 1:
                    _, claimed, _ = await self.r.xautoclaim(self.stream, self.group, self.consumer,
                                                           min_idle_time=CLAIM_IDLE_MS, start_id="0-0", count=5)
                    for msg_id, fields in claimed:
                        if fields:
                            logger.warning("reclaimed stale message", extra={"fields": {"msg_id": msg_id}})
                            await self._spawn(msg_id, fields)
                    QUEUE_DEPTH.labels(self.stream).set(await self._lag())
                free = self.sem._value  # noqa: SLF001 - read-only peek to size the batch
                if free <= 0:
                    await asyncio.sleep(0.05)
                    continue
                resp = await self.r.xreadgroup(self.group, self.consumer, {self.stream: ">"}, count=free, block=1000)
                for _, messages in resp or []:
                    for msg_id, fields in messages:
                        await self._spawn(msg_id, fields)
            except (aioredis.ConnectionError, aioredis.TimeoutError) as exc:
                logger.warning("redis unavailable, backing off: %s", exc)
                await asyncio.sleep(2)

    async def _lag(self) -> int:
        try:
            groups = await self.r.xinfo_groups(self.stream)
            for g in groups:
                if g["name"] == self.group:
                    return int(g.get("lag") or 0) + int(g.get("pending") or 0)
        except Exception:
            pass
        return 0

    def stop(self) -> None:
        self._stop.set()
