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


def delay_facilities(p: Position, block: Optional[int], delay: Optional[int], sched_hour: Optional[int]) -> None:
    """CAR 3.4.1 (meals by block-time band, through 3.8.1 (a)) and 3.4.3 (hotel, through 3.8.1 (b))."""
    if delay is None:
        return
    if block is None:
        band = None if 120 <= delay < 240 else (("delay_band_c", 240) if delay >= 240 else ("delay_band_a", 9999))
    elif block <= 150:
        band = ("delay_band_a", 120)
    elif block <= 300:
        band = ("delay_band_b", 180)
    else:
        band = ("delay_band_c", 240)
    if band is None:
        p.facilities.append("meals and refreshments (subject to block time; confirm block time)")
        p.add("delay_band_b", f"Facilities under para 3.8.1 (a) are due after 2, 3 or 4 hours of delay depending on "
                              f"the block time; with {_hm(delay)} of delay the block time decides.", "facility")
        p.missing_facts.append("block time of the flight (to confirm meals and refreshments under para 3.4.1)")
    elif delay >= band[1]:
        p.facilities.append("meals and refreshments while waiting")
        p.add(band[0], f"The passenger checked in on time and the delay of {_hm(delay)} meets the {_hm(band[1])} "
                       "threshold for this block time in para 3.4.1.", "facility")
        p.add("meals", "Meals and refreshments in relation to waiting time are offered free of charge.", "facility")
    night = sched_hour is not None and (sched_hour >= 20 or sched_hour < 3)
    if delay > DAY or (night and delay > 360):
        p.facilities.append("hotel accommodation including transfers")
        p.add("hotel", "Where the total delay is more than 24 hours, or more than 6 hours for a flight scheduled to "
                       "depart between 2000 and 0300 hours, the passenger is offered the facility in para 3.8.1 (b).",
              "facility")
        p.add("hotel_facility", "Hotel accommodation when necessary, including transfers, is offered free of charge.",
              "facility")


def evaluate(f: dict[str, Any], guardrail_flags: Optional[list[str]] = None) -> Position:
    cat = f.get("category")
    p = Position(category=cat)
    p.required_clauses = list(rules()["required_by_category"].get(cat or "", []))

    if "prompt_injection" in (guardrail_flags or []):
        p.escalate_reasons.append("Prompt-injection attempt detected in the claim text; the narrative was withheld "
                                  "from the model and the claim needs a human review.")
    if f.get("baggage_claim") or cat == "baggage":
        p.route = "escalate_to_human"
        p.escalate_reasons.append("Outside the corpus: the CAR covers facilities for denied boarding, cancellation and "
                                  "delays; baggage claims are not covered by any document in the corpus.")
        p.add("scope", "The subject of the CAR is facilities to passengers due to denied boarding, cancellation of "
                       "flights and delays in flights.", "escalation")
        p.required_clauses = ["CAR:subject"]
        return p
    if f.get("carrier_type") == "foreign":
        p.route = "escalate_to_human"
        p.escalate_reasons.append("Outside the corpus: for a foreign carrier, compensation is as contained in the "
                                  "regulations of its country of origin or as in paras 3.2.2/3.3.2/3.5.1 (para 3.6.1). "
                                  "The country-of-origin regulations are not in the corpus, so a human must decide.")
        p.add("foreign_carriers", "For foreign carriers the compensation is as contained in the regulations of their "
                                  "country of origin or as given in paras 3.2.2, 3.3.2 or 3.5.1 of the CAR.", "escalation")
        p.required_clauses = ["CAR:3.6.1"]
        return p
    if cat not in ("cancellation", "denied_boarding", "delay", "downgrading"):
        p.route = "request_further_evidence"
        p.missing_facts.append("what happened to the flight: denied boarding, cancellation, delay or downgrading")
        p.add("def_cancellation", "Cancellation means non-operation of a flight which was previously planned; the "
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
                           "passenger may file a grievance on the Air Sewa app or portal.", "condition")
    return p


def _cause_gate(f: dict[str, Any], p: Position, exemption_key: Optional[str]) -> bool:
    """Returns True if an extraordinary-circumstances exemption (paras 1.4 and 1.5) applies."""
    cause = f.get("airline_stated_cause")
    kind = classify_cause(cause)
    if f.get("cause_disputed"):
        p.escalate_reasons.append(f"Cause attribution disputed: the airline cites '{cause}' and the passenger "
                                  "disputes it. Whether the circumstance was beyond the airline's control and could "
                                  "not have been avoided (paras 1.4 and 1.5) needs a human decision.")
        p.add("ec_general_test", "Extraordinary circumstances are those which could not have been avoided even if all "
                                 "reasonable measures had been taken by the airline.", "escalation")
        return False
    if kind == "listed":
        if exemption_key:
            p.add(exemption_key, f"The exemption for extraordinary circumstances applies; the stated cause is "
                                 f"'{cause}'.", "exemption")
        c = cause.lower()
        if any(t in c for t in ("weather", "fog", "storm", "rain", "meteorolog", "air traffic", "atc", "airspace",
                                "security")):
            p.add("ec_atc_weather", "Airlines are not liable to pay compensation for cancellations and delays clearly "
                                    "attributable to Air Traffic Control, meteorological conditions or security "
                                    "risks.", "exemption")
        else:
            p.add("ec_list", "Extraordinary circumstances include political instability, natural disaster, civil war, "
                             "insurrection or riot, flood, explosion, government regulation, and strikes and labour "
                             "disputes.", "exemption")
        return True
    if kind in ("ambiguous", "unlisted"):
        p.escalate_reasons.append(f"Cause '{cause}' is not clearly within the circumstances listed in paras 1.4 and "
                                  "1.5; the general test in para 1.4 (whether it could have been avoided even if all "
                                  "reasonable measures had been taken) must be applied by a human.")
        p.add("ec_general_test", "Extraordinary circumstances are those which could not have been avoided even if all "
                                 "reasonable measures had been taken by the airline.", "escalation")
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


def _waiting_facilities(p: Position, f: dict[str, Any], key: str) -> None:
    """Facilities at the airport while waiting for the alternate flight (3.3.2 or 3.3.3, through 3.8.1 (a))."""
    if f.get("informed_at_airport"):
        p.facilities.append("meals and refreshments while waiting")
        p.add(key, "The passenger had already reported at the airport, so the airline provides facilities under "
                   "para 3.8.1 (a) while they wait for the alternate flight.", "facility")
        p.add("meals", "Meals and refreshments in relation to waiting time are offered free of charge.", "facility")
    else:
        p.notes.append("Facilities while waiting for the alternate flight apply if the passenger had already reported "
                       "at the airport (para 3.3.2).")


def _cancellation(f: dict[str, Any], p: Position) -> None:
    notice = f.get("notice_minutes")
    at_airport = bool(f.get("informed_at_airport"))
    if notice is None and at_airport:
        notice = 0
    alt_accepted = f.get("alternate_accepted")
    if notice is None:
        p.missing_facts.append("when the passenger was informed of the cancellation")
        p.add("cx_scope_3_3_2", "Para 3.3.2 applies to passengers who have not been informed as per the provisions "
                                "contained in para 3.3.1, so when the passenger was informed is needed.", "evidence")
        p.route = "request_further_evidence"
        return
    if notice >= DAY:
        key = "cx_two_weeks" if notice >= TWO_WEEKS else "cx_under_two_weeks"
        span = "at least two weeks" if notice >= TWO_WEEKS else "less than two weeks but at least 24 hours"
        p.add(key, f"The passenger was informed {_hm(notice)} before departure ({span} ahead); the airline offers an "
                   "alternate flight or a refund, as acceptable to the passenger.", "basis")
        p.add("cx_scope_3_3_2", "Compensation under para 3.3.2 is for passengers who have not been informed as para "
                                "3.3.1 requires, so no compensation is payable.", "basis")
        p.route, p.refund_due = "decline", not alt_accepted
        return
    when = f"{_hm(notice)} before departure" if notice else "on arrival at the airport"
    if at_airport and notice:
        when += ", at the airport"
    p.add("cx_scope_3_3_2", f"The passenger was informed {when}, less than 24 hours ahead, so para 3.3.2 applies.",
          "basis")
    if f.get("contact_details_provided") is False:
        p.add("cx_contact", "No financial compensation is payable to a passenger who did not provide adequate contact "
                            "information at booking; the ticket is refunded or alternate travel arranged.", "exemption")
        p.route, p.refund_due = "decline", not alt_accepted
        if alt_accepted:
            _waiting_facilities(p, f, "cx_contact")
        return
    exempt = _cause_gate(f, p, "cx_extraordinary")
    block = f.get("block_time_minutes")
    if exempt:
        p.route, p.refund_due = "decline", not alt_accepted
        if alt_accepted:
            _waiting_facilities(p, f, "cx_facilities_ref")
        return
    if p.escalate_reasons:
        return
    if alt_accepted is True:
        p.add("cx_either_or", "The airline shall either provide an alternate flight acceptable to the passenger or "
                              "compensation in addition to a full refund; the passenger accepted and took the "
                              "alternate flight, so no compensation is payable.", "basis")
        p.route = "decline"
        _waiting_facilities(p, f, "cx_facilities_ref")
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
    p.add("cx_either_or", "The passenger did not take an alternate flight, so compensation is payable in addition to "
                          "the full refund of the ticket.", "basis")
    p.add(key, f"For a block time of {_hm(block)} compensation is {_inr(cap)} or the booked one-way basic fare plus "
               f"airline fuel charge, whichever is less: {p.computation}.", "amount")
    p.add("def_block_time", "Block time is the total time from the moment the aircraft first moves for take-off until "
                            "it finally comes to rest.", "condition")


def _denied_boarding(f: dict[str, Any], p: Position) -> None:
    if f.get("volunteered"):
        p.add("db_volunteers", "The passenger volunteered to give up the seat, in exchange for benefits the airline "
                               "offers at its own discretion.", "exemption")
        p.add("db_against_will", "Compensation under para 3.2.2 is for passengers denied boarding against their "
                                 "will.", "exemption")
        p.route = "decline"
        return
    if f.get("presented_on_time") is False:
        p.add("def_denied_boarding", "Denied boarding is a refusal to carry a passenger who presented for boarding "
                                     "within the time specified by the airline; this passenger did not, so para "
                                     "3.2.2 does not apply.", "exemption")
        p.route = "decline"
        return
    if f.get("presented_on_time") is None:
        p.notes.append("Subject to confirmation that the passenger presented for boarding within the time specified "
                       "by the airline (para 2.6).")
    alt_accepted, alt_delay = f.get("alternate_accepted"), f.get("alternate_delay_minutes")
    not_opted = alt_accepted is False or (f.get("refund_requested") and alt_accepted is not True)
    if not not_opted and alt_delay is None:
        p.missing_facts.append("departure time of the alternate flight, or whether the passenger declined it")
        p.add("db_within_24h", "Compensation for denied boarding depends on when the alternate flight departs.", "evidence")
        p.route = "request_further_evidence"
        return
    p.add("db_against_will", "The passenger was denied boarding against their will.", "basis")
    if not not_opted and alt_delay <= 60:
        p.add("db_within_1h", f"The alternate flight departed {_hm(alt_delay)} after the original time; the airline is "
                              "not liable for any compensation when the alternate flight departs within one hour.",
              "basis")
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
        p.add("delay_facilities", "Facilities for a delay depend on its length and the block time of the flight.",
              "evidence")
        p.route = "request_further_evidence"
        return
    p.add("delay_facilities", f"For a delay of {_hm(delay)} the airline shall provide "
                              "facilities in accordance with para 3.8.1 (a) if the passenger has checked in on time; "
                              "the delay paragraphs provide facilities, not compensation.", "basis")
    if _cause_gate(f, p, "delay_extraordinary"):
        p.route = "decline"
        return
    if p.escalate_reasons:
        return
    delay_facilities(p, f.get("block_time_minutes"), delay, f.get("scheduled_departure_local_hour"))
    if delay > 360 and f.get("sector") != "international":
        p.facilities.append("alternate flight within 6 hours or full refund")
        p.add("delay_alt_refund", "Where a domestic flight is delayed for more than 6 hours, the airline offers an "
                                  "alternate flight within 6 hours or a full refund.", "facility")
    if f.get("requested_remedy") == "facilities_reimbursement" and p.facilities:
        p.missing_facts.append("receipts for meals or accommodation the passenger paid for")
        p.route = "request_further_evidence"
    else:
        p.route = "decline"


def _downgrading(f: dict[str, Any], p: Position) -> None:
    if f.get("downgrade_requested"):
        p.add("dg_involuntary", "The airline reimburses the passenger where the passenger is downgraded "
                                "involuntarily and is carried in a class lower than that for which the ticket is "
                                "purchased; this passenger requested the downgrade.", "exemption")
        p.route = "decline"
        return
    sector = f.get("sector")
    total = f.get("total_ticket_cost_inr")
    if sector == "international":
        dist = f.get("sector_distance_km")
        p.escalate_reasons.append("Provisions conflict: for an international downgrade the CAR (para 3.5.1) sets a "
                                  "reimbursement of 30-75% of the cost of the ticket by distance, while the airline "
                                  "charter (para 6.2) offers the fare difference. The CAR is the regulation and "
                                  "normally prevails, but the conflict needs a human decision.")
        if dist is not None:
            key, pct = ("dg_intl_1500", 0.30) if dist <= 1500 else ("dg_intl_3500", 0.50) if dist <= 3500 else ("dg_intl_long", 0.75)
            p.add(key, f"For an international flight of {dist} km the CAR reimbursement is {int(pct * 100)}% of the "
                       "cost of the ticket, including taxes.", "escalation")
            if total is not None:
                p.downgrade_refund_inr = round(pct * float(total), 2)
                p.computation = f"CAR reference: {int(pct * 100)}% x {_inr(float(total))} = {_inr(p.downgrade_refund_inr)}"
        else:
            p.add("dg_intl_3500", "For international sectors the CAR reimbursement is 30%, 50% or 75% of the cost of "
                                  "the ticket by flight distance.", "escalation")
        p.add("charter_dg_intl", "The airline charter offers, for international sectors, the difference between the "
                                 "fare paid and the fare of the class travelled.", "escalation")
        p.required_clauses.append("CHARTER:6.2")
        return
    if sector is None:
        p.missing_facts.append("whether the sector was domestic or international")
    if total is None:
        p.missing_facts.append("cost of the ticket, including taxes")
    if sector is None or total is None:
        p.add("dg_domestic", "The downgrade reimbursement is a percentage of the cost of the ticket, including taxes.",
              "evidence")
        p.route = "request_further_evidence"
        return
    amount = round(0.75 * float(total), 2)
    p.downgrade_refund_inr, p.route = amount, "pay_as_computed"
    p.computation = f"75% x {_inr(float(total))} = {_inr(amount)}"
    p.add("dg_involuntary", "The passenger was downgraded involuntarily to a lower class.", "basis")
    p.add("dg_domestic", f"For a domestic sector the airline reimburses 75% of the cost of the ticket, including "
                         f"taxes: {p.computation}.", "amount")


def justification(p: Position) -> str:
    order = {"basis": 0, "condition": 1, "exemption": 2, "amount": 3, "facility": 4, "evidence": 5, "escalation": 6}
    parts = []
    for c in sorted(p.components, key=lambda c: order.get(c.kind, 9)):
        if c.key == "grievance":
            continue
        parts.append(f"{c.statement} [{c.clause_id}]")
    # Evidence requests and escalation reasons are returned in their own result fields, not in the justification.
    return " ".join(parts)
