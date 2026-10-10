"""Deterministic entitlement engine.

It encodes the decision structure of the CAR (which paragraph applies, which conditions and exemptions can remove
an entitlement) but not the provisions' authority: every component it emits names the clause it rests on and the
phrases that clause must contain. The verifier accepts a component only if that clause was actually retrieved by
the agent and its text contains those phrases. The engine is therefore both
  * the fallback reasoner when the model is unavailable (degraded mode), and
  * the cross-check that catches a model answer that found the amounts and stopped.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

import yaml

RULES_PATH = os.path.join(os.path.dirname(__file__), "car_rules.yaml")
TWO_WEEKS = 14 * 24 * 60
DAY = 24 * 60


@lru_cache(maxsize=1)
def rules() -> dict[str, Any]:
    with open(os.environ.get("CAR_RULES_PATH", RULES_PATH), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def clause(key: str) -> dict[str, Any]:
    return rules()["clauses"][key]


@dataclass
class Component:
    key: str
    statement: str
    clause_id: str
    terms: list[str]
    kind: str  # basis | condition | exemption | amount | facility | escalation | evidence


@dataclass
class Position:
    category: Optional[str]
    route: str = "escalate_to_human"
    compensation_inr: float = 0.0
    computation: str = ""
    refund_due: bool = False
    downgrade_refund_inr: float = 0.0
    facilities: list[str] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    escalate_reasons: list[str] = field(default_factory=list)
    required_clauses: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, key: str, statement: str, kind: str) -> None:
        c = clause(key)
        self.components.append(Component(key, statement, c["id"], list(c["terms"]), kind))

    def cited_clauses(self) -> list[str]:
        seen: list[str] = []
        for c in self.components:
            if c.clause_id not in seen:
                seen.append(c.clause_id)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category, "route": self.route, "compensation_inr": self.compensation_inr,
            "computation": self.computation, "refund_due": self.refund_due,
            "downgrade_refund_inr": self.downgrade_refund_inr, "facilities": self.facilities,
            "components": [c.__dict__ for c in self.components], "missing_facts": self.missing_facts,
            "escalate_reasons": self.escalate_reasons, "required_clauses": self.required_clauses, "notes": self.notes,
        }


def _hm(minutes: Optional[int]) -> str:
    if minutes is None:
        return "an unknown time"
    if minutes < 60:
        return f"{minutes} minutes"
    h, m = divmod(int(minutes), 60)
    if h >= 48 and not m:
        return f"{h // 24} days" if h % 24 == 0 else f"{h} hours"
    return f"{h} hour{'s' if h != 1 else ''}" + (f" {m} minutes" if m else "")


def _inr(x: float) -> str:
    return f"INR {x:,.0f}"


def classify_cause(cause: Optional[str]) -> str:
    if not cause:
        return "none"
    c = cause.lower()
    cfg = rules()["causes"]
    if any(t in c for t in cfg["internal"]):
        return "internal"
    if any(t in c for t in cfg["ambiguous"]):
        return "ambiguous"
    if any(t in c for t in cfg["listed"]):
        return "listed"
    return "unlisted"


def facilities_for_wait(p: Position, block: Optional[int], wait: Optional[int], sched_hour: Optional[int],
                        label: str = "waiting") -> None:
    """CAR 3.4.1 (meals by block-time band) and 3.4.3 (hotel)."""
    if wait is None:
        return
    if block is None:
        threshold = None if 120 <= wait < 240 else (120 if wait >= 240 else 9999)
    elif block <= 150:
        threshold = 120
    elif block <= 300:
        threshold = 180
    else:
        threshold = 240
    if threshold is None:
        p.facilities.append("meals and refreshments (subject to block time; confirm block time)")
        p.add("meals", f"Meals and refreshments are provided free of charge in relation to the waiting time; with "
                       f"{_hm(wait)} of {label} the entitlement depends on the block time band in para 3.4.1.", "facility")
        p.missing_facts.append("block time of the flight (to confirm meals and refreshments under para 3.4.1)")
    elif wait >= threshold:
        p.facilities.append("meals and refreshments while waiting")
        p.add("meals", f"Meals and refreshments are provided free of charge in relation to the waiting time: "
                       f"{_hm(wait)} of {label} meets the {_hm(threshold)} threshold in para 3.4.1.", "facility")
    night = sched_hour is not None and (sched_hour >= 20 or sched_hour < 3)
    if wait > DAY or (night and wait > 360):
        p.facilities.append("hotel accommodation including transfer")
        p.add("hotel", "Hotel accommodation, including transfer, is provided free of charge where the delay is more "
                       "than 24 hours, or more than 6 hours for a departure between 2000 and 0300 hours.", "facility")


def evaluate(f: dict[str, Any], guardrail_flags: Optional[list[str]] = None) -> Position:
    cat = f.get("category")
    p = Position(category=cat)
    p.required_clauses = list(rules()["required_by_category"].get(cat or "", []))

    if "prompt_injection" in (guardrail_flags or []):
        p.escalate_reasons.append("Prompt-injection attempt detected in the claim text; the narrative was withheld "
                                  "from the model and the claim needs a human review.")
    if f.get("baggage_claim") or cat == "baggage":
        p.route = "escalate_to_human"
        p.escalate_reasons.append("Outside the corpus: baggage claims are not covered by the CAR (para 3.1.3).")
        p.add("excluded_claims", "Claims relating to baggage are not covered by this CAR and are governed by the "
                                 "Carriage by Air Act, 1972.", "escalation")
        p.required_clauses = ["CAR:3.1.3"]
        return p
    if f.get("carrier_type") == "foreign":
        p.route = "escalate_to_human"
        p.escalate_reasons.append("Outside the corpus: the flight was operated by a foreign carrier, which the CAR "
                                  "does not cover (para 3.1.1).")
        p.add("applicability", "Flights operated by foreign air carriers are not covered by this CAR.", "escalation")
        p.required_clauses = ["CAR:3.1.1"]
        return p
    if cat not in ("cancellation", "denied_boarding", "delay", "downgrading"):
        p.route = "request_further_evidence"
        p.missing_facts.append("what happened to the flight: denied boarding, cancellation, delay or downgrading")
        p.add("categories", "Cancellation means the non-operation of a flight which was previously planned; the "
                            "claim must state which kind of disruption occurred.", "evidence")
        return _finish(p)

    {"cancellation": _cancellation, "denied_boarding": _denied_boarding, "delay": _delay,
     "downgrading": _downgrading}[cat](f, p)
    return _finish(p)


def _finish(p: Position) -> Position:
    if p.escalate_reasons:
        p.route = "escalate_to_human"
    elif p.route not in ("pay_as_computed", "decline", "request_further_evidence"):
        p.route = "request_further_evidence" if p.missing_facts else "escalate_to_human"
    if p.route != "escalate_to_human":
        p.add("grievance", "Where a claim is declined, the provision under which it was declined is stated; the "
                           "passenger may escalate the grievance through the AirSewa portal.", "condition")
    return p


def _cause_gate(f: dict[str, Any], p: Position, exemption_key: Optional[str]) -> bool:
    """Returns True if an extraordinary-circumstances exemption applies (compensation not payable)."""
    cause = f.get("airline_stated_cause")
    kind = classify_cause(cause)
    if f.get("cause_disputed"):
        p.escalate_reasons.append(f"Cause attribution disputed: the airline cites '{cause}' and the passenger "
                                  "disputes it. The burden of establishing extraordinary circumstances rests with "
                                  "the airline (para 3.6.2).")
        p.add("ec_general_test", "The burden of establishing extraordinary circumstances rests with the airline; "
                                 "circumstances are assessed against the general test.", "escalation")
        return False
    if kind == "listed":
        if exemption_key:
            p.add(exemption_key, f"The airline is not liable to pay compensation where the disruption is caused by "
                                 f"extraordinary circumstances; the stated cause is '{cause}'.", "exemption")
        p.add("ec_list", "Extraordinary circumstances beyond the control of the airline include weather conditions "
                         "incompatible with the operation of the flight, air traffic management decisions, strikes "
                         "and security risks.", "exemption")
        return True
    if kind in ("ambiguous", "unlisted"):
        p.escalate_reasons.append(f"Cause '{cause}' is not clearly within the circumstances listed in para 3.6.1 "
                                  "and needs the general test in para 3.6.2 applied by a human.")
        p.add("ec_general_test", "Circumstances not listed are assessed against the general test on the facts of "
                                 "each case; the burden of establishing extraordinary circumstances rests with the "
                                 "airline.", "escalation")
    return False


def _fare(f: dict[str, Any], p: Position) -> Optional[float]:
    fare = f.get("basic_fare_inr")
    if fare is None:
        p.missing_facts.append("booked one-way basic fare and airline fuel charge (from the ticket)")
        return None
    fuel = f.get("fuel_charge_inr")
    if fuel is None:
        p.notes.append("Airline fuel charge not stated; computed on basic fare only.")
        fuel = 0.0
    return float(fare) + float(fuel)


def _cancellation(f: dict[str, Any], p: Position) -> None:
    notice = f.get("notice_minutes")
    at_airport = bool(f.get("informed_at_airport"))
    if notice is None and at_airport:
        notice = 0
    alt_accepted = f.get("alternate_accepted")
    alt_delay = f.get("alternate_delay_minutes")
    if notice is None:
        p.missing_facts.append("when the passenger was informed of the cancellation")
        p.add("cx_under_two_weeks", "The airline shall inform passengers less than two weeks or at least two weeks before "
                                    "the booked scheduled departure time; what is owed depends on when it informed them.",
              "evidence")
        p.route = "request_further_evidence"
        return
    if notice >= TWO_WEEKS:
        p.add("cx_two_weeks", f"The passenger was informed {_hm(notice)} before departure, at least two weeks ahead; "
                              "the airline arranges an alternate flight or a refund, and no compensation is payable.",
              "basis")
        p.route, p.refund_due = "decline", not alt_accepted
        return
    if notice >= DAY:
        p.add("cx_under_two_weeks", f"The passenger was informed {_hm(notice)} before departure, less than two "
                                    "weeks and at least 24 hours ahead; the airline offers an alternate flight "
                                    "departing within two hours or a refund, and no compensation is payable.", "basis")
        if alt_delay is not None and alt_delay > 120 and alt_accepted is not True:
            p.notes.append("The alternate flight offered departed more than two hours after the original; a refund "
                           "must be available as acceptable to the passenger.")
        p.route, p.refund_due = "decline", not alt_accepted
        return
    when = f"{_hm(notice)} before departure" if notice else "on arrival at the airport"
    if at_airport and notice:
        when += ", at the airport"
    p.add("cx_scope_3_3_2", f"The passenger was informed {when}, less than 24 hours before departure, so the airline "
                            "shall provide either an alternate flight or compensation, in addition to a full refund.",
          "basis")
    if f.get("contact_details_provided") is False:
        p.add("cx_contact", "The passenger did not provide contact details at booking; the airline shall not be "
                            "liable to pay compensation, and the refund of the ticket remains payable.", "exemption")
        p.route, p.refund_due = "decline", True
        return
    exempt = _cause_gate(f, p, "cx_extraordinary")
    wait = alt_delay
    block = f.get("block_time_minutes")
    if exempt:
        p.route, p.refund_due = "decline", not alt_accepted
        if alt_accepted:
            facilities_for_wait(p, block, wait, f.get("scheduled_departure_local_hour"), "waiting for the alternate flight")
        return
    if p.escalate_reasons:
        return
    if alt_accepted is True:
        p.add("cx_alt_accepted", "Where the passenger accepts the alternate flight, no compensation is payable; the "
                                 "passenger travelled on the alternate flight.", "condition")
        p.add("cx_facilities_ref", "The airline shall provide facilities in accordance with para 3.4.1 to passengers "
                                   "awaiting the alternate flight.", "condition")
        p.add("def_alternate", "A passenger who travels on the alternate flight is deemed to have accepted it.",
              "condition")
        p.route = "decline"
        facilities_for_wait(p, block, wait, f.get("scheduled_departure_local_hour"), "waiting for the alternate flight")
        if wait is None:
            p.facilities.append("meals and refreshments while waiting (confirm waiting time)")
            p.add("meals", "Meals and refreshments are provided in relation to the waiting time for the alternate "
                           "flight.", "facility")
        return
    if alt_accepted is None and not f.get("refund_requested"):
        p.missing_facts.append("whether the passenger accepted the alternate flight or took a refund")
        p.route = "request_further_evidence"
        return
    if block is None:
        p.missing_facts.append("block time of the booked flight")
    total = _fare(f, p)
    if block is None or total is None:
        p.add("cx_cap_over_2h", "Compensation is set by block time: INR 5,000, 7,500 or 10,000, or the booked one-way "
                                "basic fare plus airline fuel charge, whichever is less.", "evidence")
        p.route = "request_further_evidence"
        return
    key, cap = ("cx_cap_1h", 5000) if block <= 60 else ("cx_cap_2h", 7500) if block <= 120 else ("cx_cap_over_2h", 10000)
    amount = min(cap, total)
    p.compensation_inr, p.refund_due, p.route = amount, True, "pay_as_computed"
    p.computation = f"min({_inr(cap)}, basic fare + fuel charge {_inr(total)}) = {_inr(amount)}"
    p.add(key, f"For a block time of {_hm(block)} compensation is {_inr(cap)} or the booked one-way basic fare plus "
               f"airline fuel charge, whichever is less: {p.computation}, in addition to a full refund.", "amount")
    p.add("def_block_time", "Block time is the published time from first movement for take-off until the aircraft "
                            "comes to rest.", "condition")


def _denied_boarding(f: dict[str, Any], p: Position) -> None:
    if f.get("volunteered"):
        p.add("db_volunteers", "The passenger volunteered to surrender the reservation; volunteers are not entitled "
                               "to compensation under para 3.2.2.", "exemption")
        p.route = "decline"
        return
    if f.get("presented_on_time") is False:
        p.add("db_exceptions", "The airline shall not be liable to pay compensation where the passenger did not "
                               "present himself for check-in in time.", "exemption")
        p.add("checkin", "Passengers must present themselves for check-in not later than one hour before departure.",
              "condition")
        p.route = "decline"
        return
    if _cause_gate(f, p, None) or p.escalate_reasons:
        if not p.escalate_reasons:
            p.route = "decline"
        return
    if f.get("presented_on_time") is None:
        p.notes.append("Subject to confirmation that the passenger presented for check-in on time (para 3.1.2).")
    alt_accepted, alt_delay = f.get("alternate_accepted"), f.get("alternate_delay_minutes")
    not_opted = alt_accepted is False or (f.get("refund_requested") and alt_accepted is not True)
    if not not_opted and alt_delay is None:
        p.missing_facts.append("departure time of the alternate flight, or whether the passenger declined it")
        p.add("db_within_24h", "Compensation for denied boarding depends on when the alternate flight departs.", "evidence")
        p.route = "request_further_evidence"
        return
    if not not_opted and alt_delay <= 60:
        p.add("db_within_1h", f"The alternate flight departed {_hm(alt_delay)} after the booked time; no compensation "
                              "shall be payable for an alternate flight within one hour.", "basis")
        p.route = "decline"
        return
    total = _fare(f, p)
    if total is None:
        p.add("db_within_24h", "Compensation is a percentage of the booked one-way basic fare plus airline fuel charge.",
              "evidence")
        p.route = "request_further_evidence"
        return
    if not_opted:
        key, pct, cap = "db_not_opted", 4.0, 20000
        desc = "The passenger did not opt for the alternate flight: full refund plus 400%"
        p.refund_due = True
    elif alt_delay <= DAY:
        key, pct, cap, desc = "db_within_24h", 2.0, 10000, f"Alternate flight {_hm(alt_delay)} later (within 24 hours): 200%"
    else:
        key, pct, cap, desc = "db_beyond_24h", 4.0, 20000, f"Alternate flight {_hm(alt_delay)} later (more than 24 hours): 400%"
    amount = min(cap, pct * total)
    p.compensation_inr, p.route = amount, "pay_as_computed"
    p.computation = f"min({int(pct * 100)}% x {_inr(total)}, {_inr(cap)}) = {_inr(amount)}"
    p.add(key, f"{desc} of the booked one-way basic fare plus airline fuel charge, subject to a maximum of "
               f"{_inr(cap)}: {p.computation}.", "amount")


def _delay(f: dict[str, Any], p: Position) -> None:
    delay = f.get("delay_minutes")
    if delay is None:
        p.missing_facts.append("length of the delay")
        p.add("meals", "Facilities depend on the waiting time and the block time of the flight.", "evidence")
        p.route = "request_further_evidence"
        return
    p.add("delay_no_comp", f"The flight was delayed by {_hm(delay)}. No monetary compensation is payable for delay "
                           "under this CAR; facilities are payable irrespective of the cause of the delay.", "basis")
    facilities_for_wait(p, f.get("block_time_minutes"), delay, f.get("scheduled_departure_local_hour"), "delay")
    notice = f.get("notice_minutes")
    if delay > 360 and notice is not None and notice >= DAY:
        p.facilities.append("alternate flight within 6 hours or full refund")
        p.add("delay_alt_refund", "Where the delay is more than 6 hours and the passenger was informed 24 hours ahead, "
                                  "the airline offers an alternate flight or a full refund.", "facility")
    if f.get("requested_remedy") == "facilities_reimbursement" and p.facilities:
        p.missing_facts.append("receipts for meals or accommodation the passenger paid for")
        p.route = "request_further_evidence"
    else:
        p.route = "decline"


def _downgrading(f: dict[str, Any], p: Position) -> None:
    if f.get("downgrade_requested"):
        p.add("dg_requested", "The downgrade was requested by the passenger, so the downgrade refund does not apply.",
              "exemption")
        p.route = "decline"
        return
    sector = f.get("sector")
    total = f.get("total_ticket_cost_inr")
    if sector == "international":
        dist = f.get("sector_distance_km")
        p.escalate_reasons.append("Provisions conflict: for an international downgrade the CAR (para 3.5.1) sets a "
                                  "refund of 30-75% of the ticket cost by distance, while the airline charter (para "
                                  "6.2) offers the fare difference. The CAR is the regulation and normally prevails, "
                                  "but the conflict needs a human decision.")
        if dist is not None:
            key, pct = ("dg_intl_1500", 0.30) if dist <= 1500 else ("dg_intl_3500", 0.50) if dist <= 3500 else ("dg_intl_long", 0.75)
            p.add(key, f"For an international sector of {dist} km the CAR refund is {int(pct * 100)}% of the ticket "
                       "cost for that sector, including taxes.", "escalation")
            if total is not None:
                p.downgrade_refund_inr = round(pct * float(total), 2)
                p.computation = f"CAR reference: {int(pct * 100)}% x {_inr(float(total))} = {_inr(p.downgrade_refund_inr)}"
        else:
            p.add("dg_intl_3500", "For international sectors the CAR refund is 30%, 50% or 75% of the ticket cost by "
                                  "sector distance.", "escalation")
        p.add("charter_dg_intl", "The airline charter offers, for international sectors, the difference between the "
                                 "fare paid and the fare of the class travelled.", "escalation")
        p.required_clauses.append("CHARTER:6.2")
        return
    if sector is None:
        p.missing_facts.append("whether the sector was domestic or international")
    if total is None:
        p.missing_facts.append("cost of the ticket for that sector, including taxes")
    if sector is None or total is None:
        p.add("dg_domestic", "The downgrade refund is a percentage of the ticket cost for that sector, including "
                             "taxes.", "evidence")
        p.route = "request_further_evidence"
        return
    amount = round(0.75 * float(total), 2)
    p.downgrade_refund_inr, p.route = amount, "pay_as_computed"
    p.computation = f"75% x {_inr(float(total))} = {_inr(amount)}"
    p.add("dg_domestic", f"For domestic sectors the airline refunds 75% of the cost of the ticket for that sector, "
                         f"including taxes: {p.computation}.", "amount")


def justification(p: Position) -> str:
    order = {"basis": 0, "condition": 1, "exemption": 2, "amount": 3, "facility": 4, "evidence": 5, "escalation": 6}
    parts = []
    for c in sorted(p.components, key=lambda c: order.get(c.kind, 9)):
        if c.key == "grievance":
            continue
        parts.append(f"{c.statement} [{c.clause_id}]")
    # Evidence requests and escalation reasons are returned in their own result fields, not in the justification.
    return " ".join(parts)
