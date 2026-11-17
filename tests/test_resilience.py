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


def test_non_retryable_stops_immediately():
    calls = {"n": 0}

    async def op():
        calls["n"] += 1
        raise resilience.NonRetryable("circuit open")

    with pytest.raises(resilience.NonRetryable):
        asyncio.run(call_with_retry(op, timeout_s=1, max_retries=5, base_delay_s=0.01, agent="test", target="unit"))
    assert calls["n"] == 1


def test_exhausted_retries_raise_the_last_error():
    op, calls = _flaky(10)
    with pytest.raises(ConnectionError, match="transient 3"):
        asyncio.run(call_with_retry(op, timeout_s=1, max_retries=2, base_delay_s=0.01, agent="test", target="unit"))
    assert calls["n"] == 3


def test_per_attempt_timeout_is_retried_then_raised():
    stats = RetryStats()

    async def slow():
        await asyncio.Event().wait()   # never finishes; only the timeout ends it

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(call_with_retry(slow, timeout_s=0.01, max_retries=1, base_delay_s=0.01, agent="test",
                                    target="unit", stats=stats))
    assert stats.attempts == 2
