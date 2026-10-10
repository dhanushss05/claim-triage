"""Fixed responses returned by the LLM stub (HTTP service) and by the in-process stub used in offline evaluation."""
from __future__ import annotations

import json
import time

from .llm_client import LLMResponse, parse_json

STUB_RESPONSES = {
    "extraction": {"category": None, "notice_minutes": None, "informed_at_airport": None, "alternate_offered": None,
                   "alternate_accepted": None, "alternate_delay_minutes": None, "block_time_minutes": None,
                   "delay_minutes": None, "contact_details_provided": None, "airline_stated_cause": None,
                   "cause_disputed": None, "refund_requested": None},
    "planner": {"action": "finish", "args": {}, "reason": "stub planner"},
    "determination": {
        "route": "pay_as_computed", "compensation_inr": 10000, "downgrade_refund_inr": 0, "refund_due": True,
        "facilities": [], "computation": "cancellation compensation for block time over two hours",
        "citations": [{"clause_id": "CAR:3.3.2",
                       "quote": "INR 10,000 or the booked one-way basic fare plus airline fuel charge, whichever is less"}],
        "justification": "The flight was cancelled, so cancellation compensation of INR 10,000 is payable under para 3.3.2.",
        "missing_evidence": [], "escalate_reasons": []},
}




class InProcessStubLLM:
    """Zero-latency stand-in for the gateway+stub pair (offline evaluation and tests)."""

    def __init__(self, price_prompt_per_1m: float = 0.0, price_completion_per_1m: float = 0.0):
        self.pp, self.pc = price_prompt_per_1m, price_completion_per_1m

    async def chat(self, *, messages, prompt_id, prompt_version, run_id, agent, step, max_tokens=600, json_mode=True):
        content = json.dumps(STUB_RESPONSES.get(prompt_id, {"note": "stub"}))
        pt = max(1, sum(len(m["content"]) for m in messages) // 4)
        ct = max(1, len(content) // 4)
        return LLMResponse(content=content, prompt_tokens=pt, completion_tokens=ct,
                           cost_usd=pt / 1e6 * self.pp + ct / 1e6 * self.pc, latency_ms=0.0, model="stub-inprocess",
                           provider="stub", parsed=parse_json(content) if json_mode else None)
