"""Timeouts and bounded retry with exponential backoff and full jitter."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from .metrics import CALLS, RETRIES

T = TypeVar("T")


class NonRetryable(Exception):
    """Raised by an operation to stop retrying immediately (e.g. circuit open, 4xx)."""


@dataclass
class RetryStats:
    attempts: int = 0
    retries: int = 0


async def call_with_retry(op: Callable[[], Awaitable[T]], *, timeout_s: float, max_retries: int, base_delay_s: float,
                          agent: str, target: str, stats: RetryStats | None = None,
                          max_delay_s: float = 8.0) -> T:
    """Run `op` with a per-attempt timeout; retry transient failures up to `max_retries` times."""
    stats = stats or RetryStats()
    CALLS.labels(agent, target).inc()
    last: BaseException | None = None
    for attempt in range(max_retries + 1):
        stats.attempts += 1
        try:
            return await asyncio.wait_for(op(), timeout=timeout_s)
        except NonRetryable:
            raise
        except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 - transient by default
            last = exc
            if attempt == max_retries:
                break
            stats.retries += 1
            RETRIES.labels(agent, target).inc()
            delay = random.uniform(0, min(max_delay_s, base_delay_s * (2 ** attempt)))
            await asyncio.sleep(delay)
    assert last is not None
    raise last
