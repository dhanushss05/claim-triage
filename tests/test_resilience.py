"""Bounded retry with backoff (triage.resilience)."""
import asyncio

import pytest

from triage import resilience
from triage.resilience import RetryStats, call_with_retry


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def fast_sleep(_delay):
        return None
    monkeypatch.setattr(resilience.asyncio, "sleep", fast_sleep)


def _flaky(failures: int, value="ok"):
    calls = {"n": 0}

    async def op():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise ConnectionError(f"transient {calls['n']}")
        return value
    return op, calls


def test_succeeds_after_transient_failures_and_counts_retries():
    op, calls = _flaky(2)
    stats = RetryStats()
    out = asyncio.run(call_with_retry(op, timeout_s=1, max_retries=3, base_delay_s=0.01,
                                      agent="test", target="unit", stats=stats))
    assert out == "ok" and calls["n"] == 3
    assert stats.attempts == 3 and stats.retries == 2


def test_first_try_success_has_no_retries():
    op, _ = _flaky(0)
    stats = RetryStats()
    asyncio.run(call_with_retry(op, timeout_s=1, max_retries=3, base_delay_s=0.01, agent="test", target="unit",
                                stats=stats))
    assert (stats.attempts, stats.retries) == (1, 0)
