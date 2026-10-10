"""Disruption classification and entity extraction.

Two extractors feed one merge:
  * the model (intake agent, prompt `extraction`) reads the narrative and returns JSON;
  * a deterministic rule extractor (below) always runs. It is the fallback path when the model is unavailable,
    and a cross-check when it is not.
Merge precedence per field: validated structured field > model > rules. Disagreements on decisive entities are
recorded as conflicts and passed on to the entitlement agent.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "fortyfive": 45, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100,
}
_NUM = r"(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty-five|forty|fifty|sixty|ninety)"
_UNIT = r"(minutes?|mins?|hours?|hrs?|days?|weeks?)"
UNIT_MIN = {"min": 1, "minute": 1, "hour": 60, "hr": 60, "day": 1440, "week": 10080}

DECISIVE = ("category", "notice_minutes", "alternate_accepted", "block_time_minutes", "contact_details_provided",
            "delay_minutes", "alternate_delay_minutes")


def _num(tok: str) -> float:
    tok = tok.lower()
    return float(tok) if re.match(r"^\d", tok) else float(NUMBER_WORDS.get(tok, 0))


def _minutes(num: str, unit: str) -> int:
    u = unit.lower().rstrip("s")
    return int(round(_num(num) * UNIT_MIN.get(u, 1)))


def _first_duration(text: str) -> Optional[int]:
    t = text.lower()
    if re.search(r"half an hour", t):
        return 30
    m = re.search(r"(one|1) and a half hours?", t)
    if m:
        return 90
    m = re.search(_NUM + r"\s*" + _UNIT, t)
    return _minutes(m.group(1), m.group(2)) if m else None


CATEGORY_PATTERNS = {
    "denied_boarding": r"denied (me )?boarding|boarding was denied|offload(ed)?|bumped|over-?book(ed|ing)|not allowed to board|refused (me )?boarding|did not let me board|didn't let me board",
    "downgrading": r"downgrad|moved (me )?(down )?to (economy|premium economy)|seated (me )?in (economy|a lower class)|lower class than",
    "cancellation": r"cancel+ed|cancel+ation|cancel+ing",
    "delay": r"delay(ed)?|late by|hours? late\b|departed late|left late",
    "baggage": r"baggage|luggage|suitcase|my bags? (was|were|got) (lost|damaged|delayed)",
}


def classify(text: str) -> tuple[str, float, list[str]]:
    t = text.lower()
    hits = [cat for cat, rx in CATEGORY_PATTERNS.items() if re.search(rx, t)]
    if not hits:
        return "unknown", 0.0, []
    for cat in ("denied_boarding", "downgrading", "cancellation", "delay", "baggage"):
        if cat in hits:
            if cat == "cancellation" and re.search(r"not (been )?cancel+ed|wasn't cancel+ed", t):
                continue
            conf = 0.75 if len(hits) == 1 else 0.6
            return cat, conf, hits
    return hits[0], 0.5, hits


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def extract_rules(text: str) -> dict[str, Any]:
    t = " ".join(text.split())
    low = t.lower()
    out: dict[str, Any] = {}
    cat, conf, hits = classify(t)
    out["category"] = cat if cat != "unknown" else None
    out["_category_confidence"] = conf
    out["_category_hits"] = hits

    # notice ---------------------------------------------------------------------------------------------------
    for s in sentences(t):
        sl = s.lower()
        if re.search(r"\b(told|informed|notified|intimated|message|sms|e-?mail|called|call|announced|learned|found out|came to know)\b", sl):
            if re.search(r"at the (airport|gate|counter)|already at the airport|on arrival at the airport|when i (reached|arrived)", sl):
                out["informed_at_airport"] = True
            m = re.search(_NUM + r"\s*" + _UNIT + r"\s*(before|prior to|ahead of|in advance)", sl)
            if m:
                out["notice_minutes"] = _minutes(m.group(1), m.group(2))
            elif re.search(r"\b(the )?(day|night) before\b", sl):
                out["notice_minutes"] = 720 if "night" in sl else 1200
            elif re.search(r"\bno (prior )?(notice|information|intimation|message)\b", sl):
                out["notice_minutes"] = 0
            if "notice_minutes" in out or out.get("informed_at_airport"):
                break
    if re.search(r"already at the airport|on arrival at the airport|when i reached the airport|at the check-?in counter|"
                 r"cancel+ed (at|when i (reached|got to)) the airport|cancel+ed at the gate|at the airport[^.]{0,40}cancel+ed", low):
        out["informed_at_airport"] = True
    if "notice_minutes" not in out:
        m = re.search(_NUM + r"\s*" + _UNIT + r"\s*(before|prior|in advance)", low)
        if m and re.search(r"cancel|inform|told|notif", low):
            out["notice_minutes"] = _minutes(m.group(1), m.group(2))

    # alternate flight -------------------------------------------------------------------------------------------
    alt_rx = (r"(put|rebooked|re-booked|moved|transferred|accommodated|shifted|offered|gave|booked|placed)\s+(me\s+|us\s+)?"
              r"(on|onto|to|in|a seat on)?\s*(a|an|another|the next|the|their next|a later)?\s*"
              r"(alternate|alternative|different|later|next|replacement|other)?\s*(flight|seat|option)")
    if (re.search(alt_rx, low) or re.search(r"alternate flight|alternative flight", low)
            or re.search(r"\b(rebooked|re-booked|put|moved|transferred|accommodated|shifted) (me|us) (on|onto|to)\b", low)):
        out["alternate_offered"] = True
        for s in sentences(t):
            sl = s.lower()
            if re.search(r"flight|later|next", sl):
                m = re.search(_NUM + r"\s*" + _UNIT + r"\s*later", sl)
                if m:
                    out["alternate_delay_minutes"] = _minutes(m.group(1), m.group(2))
                    break
                if re.search(r"next day|following day|next morning|following morning|day after", sl):
                    out["alternate_delay_minutes"] = 1500 if re.search(r"more than (24|twenty-four) hours|over a day", sl) else 1200
                    break
        if re.search(r"which i (took|accepted|boarded)|i (took|accepted|boarded|flew on|travelled on|traveled on) (it|that|the|their|this)|took the (alternate|alternative|later|next|new) flight|i had no choice but to take it", low):
            out["alternate_accepted"] = True
        if re.search(r"(declined|refused|rejected|did not take|didn't take|did not accept|didn't accept)\b", low):
            out["alternate_accepted"] = False
    if re.search(r"no (alternate|alternative|other) flight|did not offer (me )?(any|an) (alternate|alternative)|didn't offer (me )?(any|an) (alternate|alternative)", low):
        out["alternate_offered"] = False
    if re.search(r"(asked for|want|took|requested|claim(ed)?) (a |my |the )?(full )?refund|refund of (my|the) ticket", low):
        out["refund_requested"] = True
        if out.get("alternate_offered") and "alternate_accepted" not in out:
            out["alternate_accepted"] = False

    # block time --------------------------------------------------------------------------------------------------
    m = re.search(r"block time (of |was |is )?" + _NUM + r"\s*" + _UNIT, low)
    if not m:
        m = re.search(_NUM + r"[\s-]*" + r"(minutes?|mins?|hours?|hrs?)" + r"[\s-]*(long\s+)?(flight|journey|sector|trip)", low)
        if m:
            out["block_time_minutes"] = _minutes(m.group(1), m.group(2))
    else:
        out["block_time_minutes"] = _minutes(m.group(2), m.group(3))
    m = re.search(r"(\d+)\s*h(?:ours?|rs?)?\s*(\d+)\s*m(?:in(?:ute)?s?)?\s*(long\s+)?(flight|block)", low)
    if m:
        out["block_time_minutes"] = int(m.group(1)) * 60 + int(m.group(2))

    # delay --------------------------------------------------------------------------------------------------------
    m = re.search(r"delayed (by |for )?(about |around |nearly |over |more than )?" + _NUM + r"\s*" + _UNIT, low)
    if not m:
        m = re.search(r"(departed|left|took off) (about |around |nearly |over |more than )?" + _NUM + r"\s*" + _UNIT + r"\s*late", low)
    if m:
        out["delay_minutes"] = _minutes(m.group(3), m.group(4))
    else:
        m = re.search(_NUM + r"\s*" + _UNIT + r"\s*(delay|late)\b", low)
        if m:
            out["delay_minutes"] = _minutes(m.group(1), m.group(2))

    # contact details ------------------------------------------------------------------------------------------------
    if re.search(r"(did not|didn't|never|forgot to) (give|provide|share|enter|add|update) (my |any |a )?(phone|mobile|contact|e-?mail|number)", low):
        out["contact_details_provided"] = False
    elif re.search(r"(gave|provided|registered|entered|shared) (my )?(phone|mobile|contact|e-?mail|number)", low):
        out["contact_details_provided"] = True

    # cause -------------------------------------------------------------------------------------------------------
    m = re.search(r"(due to|because of|citing|reason given was|they said it was|blamed( it on)?|attributed to)\s+(the\s+)?([a-z][a-z \-]{2,40}?)(\.|,| but| and|$)", low)
    if m:
        out["airline_stated_cause"] = m.group(4).strip()
    if re.search(r"(but|however|although)[^.]{0,100}(other (airlines|flights|carriers) (were )?(flying|operating|operated|took off|departed)|weather was (fine|clear|perfect|good)|not true|lying|lie\b|false|made up|actually)", low):
        out["cause_disputed"] = True

    # denied boarding specifics -----------------------------------------------------------------------------------------
    if re.search(r"\b(i |we )?volunteer(ed)?\b", low):
        out["volunteered"] = not re.search(r"(did not|didn't|never|refused to) volunteer|no(t)? (a )?volunteer", low)
    if re.search(r"checked in on time|reached (the )?airport (well )?(in time|early|on time|\d+ hours? (before|early))|arrived (\d+ hours? )?early", low):
        out["presented_on_time"] = True
    if re.search(r"(arrived|reached)( at)?( the)?( airport)? late|missed (the )?check-?in|after (the )?check-?in (had )?closed|check-?in was closed", low):
        out["presented_on_time"] = False

    # downgrade --------------------------------------------------------------------------------------------------
    m = re.search(r"(business|first|premium economy|economy)( class)? (ticket|seat|booking|fare)", low)
    if m and out.get("category") == "downgrading":
        out["booked_class"] = m.group(1).replace(" ", "_")
    m = re.search(r"(moved|downgraded|seated|put) (me |us )?(down )?(to|in|into) (economy|premium economy)", low)
    if m:
        out["flown_class"] = m.group(5).replace(" ", "_")
    if re.search(r"i (asked|requested) (to be|for a) (moved|downgrade)", low):
        out["downgrade_requested"] = True

    if re.search(CATEGORY_PATTERNS["baggage"], low):
        out["baggage_claim"] = True
    return out


def structured_facts(claim: dict[str, Any]) -> dict[str, Any]:
    """Facts that the validated contract fields state directly."""
    f, b, d, p = claim["flight"], claim["booking"], claim.get("disruption") or {}, claim["passenger"]
    sched = _dt(f["scheduled_departure"])
    out: dict[str, Any] = {
        "carrier_type": f.get("carrier_type", "indian_scheduled"),
        "sector": f.get("sector"),
        "sector_distance_km": f.get("sector_distance_km"),
        "block_time_minutes": f.get("block_time_minutes"),
        "basic_fare_inr": b.get("basic_fare_inr"),
        "fuel_charge_inr": b.get("fuel_charge_inr"),
        "total_ticket_cost_inr": b.get("total_ticket_cost_inr"),
        "booked_class": b.get("booked_class"),
        "payment_mode": b.get("payment_mode"),
        "contact_details_provided": p.get("contact_details_provided_at_booking"),
        "category": d.get("type"),
        "informed_at_airport": d.get("informed_at_airport"),
        "notice_minutes": d.get("notice_minutes_before_departure"),
        "delay_minutes": d.get("delay_minutes"),
        "refund_requested": d.get("refund_requested"),
        "airline_stated_cause": d.get("airline_stated_cause"),
        "cause_disputed": d.get("passenger_disputes_cause"),
        "presented_on_time": d.get("presented_for_checkin_on_time"),
        "volunteered": d.get("volunteered"),
        "flown_class": d.get("flown_class"),
        "downgrade_requested": d.get("downgrade_requested_by_passenger"),
        "baggage_claim": d.get("baggage_claim"),
        "requested_remedy": claim.get("requested_remedy"),
        "scheduled_departure_local_hour": sched.hour if sched else None,
    }
    if out["notice_minutes"] is None and d.get("notified_at") and sched:
        out["notice_minutes"] = max(0, int((sched - _dt(d["notified_at"])).total_seconds() // 60))
    if out["delay_minutes"] is None and d.get("actual_departure") and sched:
        out["delay_minutes"] = max(0, int((_dt(d["actual_departure"]) - sched).total_seconds() // 60))
    alt = d.get("alternate_flight")
    if alt:
        out["alternate_offered"] = alt.get("offered")
        out["alternate_accepted"] = alt.get("accepted")
        if alt.get("scheduled_departure") and sched:
            out["alternate_delay_minutes"] = int((_dt(alt["scheduled_departure"]) - sched).total_seconds() // 60)
    else:
        out["alternate_offered"] = None
        out["alternate_accepted"] = None
    return out


def _dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


LLM_FIELDS = {
    "category": str, "notice_minutes": int, "informed_at_airport": bool, "alternate_offered": bool,
    "alternate_accepted": bool, "alternate_delay_minutes": int, "block_time_minutes": int, "delay_minutes": int,
    "contact_details_provided": bool, "airline_stated_cause": str, "cause_disputed": bool, "refund_requested": bool,
    "presented_on_time": bool, "volunteered": bool, "flown_class": str, "baggage_claim": bool,
}
VALID_CATEGORIES = {"denied_boarding", "cancellation", "delay", "downgrading", "baggage"}


def clean_llm_extraction(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Keep only well-typed fields from the model output; anything else is dropped, never trusted."""
    out: dict[str, Any] = {}
    if not isinstance(raw, dict):
        return out
    for k, typ in LLM_FIELDS.items():
        v = raw.get(k)
        if v is None:
            continue
        try:
            if typ is bool and isinstance(v, bool):
                out[k] = v
            elif typ is int and isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 20000:
                out[k] = int(v)
            elif typ is str and isinstance(v, str) and 0 < len(v) <= 200:
                out[k] = v.strip().lower()
        except (TypeError, ValueError):
            continue
    if out.get("category") not in VALID_CATEGORIES:
        out.pop("category", None)
    return out


def merge(structured: dict[str, Any], llm: dict[str, Any], rules: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    keys = set(structured) | set(llm) | {k for k in rules if not k.startswith("_")}
    facts: dict[str, Any] = {}
    sources: dict[str, str] = {}
    conflicts: list[str] = []
    for k in sorted(keys):
        for name, src in (("structured", structured), ("llm", llm), ("rules", rules)):
            if src.get(k) is not None:
                facts[k], sources[k] = src[k], name
                break
        else:
            facts[k] = None
        if k in DECISIVE:
            vals = {n: s.get(k) for n, s in (("structured", structured), ("llm", llm), ("rules", rules)) if s.get(k) is not None}
            distinct = set(map(str, vals.values()))
            if len(distinct) > 1 and not _close(list(vals.values())):
                conflicts.append(f"{k}: " + ", ".join(f"{n}={v}" for n, v in vals.items()))
    return facts, sources, conflicts


def _close(values: list[Any]) -> bool:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) == len(values) and nums:
        return max(nums) - min(nums) <= max(15, 0.1 * max(nums))
    return False
