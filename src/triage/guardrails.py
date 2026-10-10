"""Guardrails: prompt-injection defence, toxicity detection, and output filtering.

Input side (intake agent, before any model sees the text):
  * normalise (NFKC, strip zero-width/control characters) so look-alike tricks do not evade the patterns;
  * score every free-text field against injection patterns. A detected injection never reaches a model holding
    tools: the narrative is withheld, the run continues on validated structured fields only, and is escalated;
  * toxicity is flagged and redacted, but does not by itself reduce an entitlement (angry passengers still have
    rights).
Output side (entitlement agent, before a result is stored):
  * the route must be one of the four allowed routes; amounts cannot exceed the CAR caps;
  * language that authorises a payment ("approved", "we will pay", "refund processed") is rewritten, since the
    system recommends and an officer decides;
  * toxic output is replaced with a neutral template.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

_ZERO_WIDTH = re.compile(r"[​-‏ - ⁠-⁯﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

INJECTION_PATTERNS: list[tuple[str, str, float]] = [
    ("ignore_instructions", r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system|your)\b.{0,30}\b(instructions?|prompts?|rules?|guidelines?|directions?)", 1.0),
    ("new_instructions", r"\b(new|updated|real|actual)\s+(instructions?|task|rules?)\s*[:\-]", 0.8),
    ("role_hijack", r"\b(you are now|act as|pretend to be|roleplay as|from now on you)\b", 0.8),
    ("system_prompt", r"\b(system prompt|developer message|hidden prompt|reveal (your|the) (prompt|instructions))\b", 0.9),
    ("chat_markup", r"(<\|?(im_start|im_end|system|assistant)\|?>|\[/?INST\]|^\s*#{2,}\s*(system|instruction)|<\s*/?\s*system\s*>)", 1.0),
    ("tool_coercion", r"\b(call|invoke|use|run)\s+(the\s+)?(tool|function|search_corpus|get_clause|find_related)\b", 0.7),
    ("verdict_coercion", r"\b(you must|you have to|always|just)\s+(approve|pay|authori[sz]e|route (it|this) to|output|return|respond with)\b", 0.8),
    ("output_format_coercion", r"\b(respond|reply|answer|output)\s+(only\s+)?(with|in)\s+(json|the following)\b", 0.6),
    ("route_injection", r"\b(route|decision|verdict)\s*[:=]\s*\"?(pay_as_computed|approve|pay)", 0.9),
    ("exfiltration", r"\b(api[_ -]?key|password|secret|token|credentials?)\b.{0,30}\b(print|show|send|reveal|tell)\b", 0.7),
    ("encoded_payload", r"\b[A-Za-z0-9+/]{120,}={0,2}\b", 0.5),
]
_COMPILED = [(name, re.compile(rx, re.I | re.M), w) for name, rx, w in INJECTION_PATTERNS]
INJECTION_THRESHOLD = 0.7

TOXIC_TERMS = {
    "idiot", "idiots", "stupid", "moron", "morons", "bastard", "bastards", "bitch", "fuck", "fucking", "fucked",
    "shit", "shitty", "asshole", "dumbass", "crap", "scum", "useless pigs", "chutiya", "bewakoof", "kamina",
    "harami", "saala", "bloody fools",
}
THREAT_PATTERNS = [r"\b(kill|hurt|beat up|burn down|destroy)\b.{0,20}\b(you|staff|crew|them|your office)\b",
                   r"\bbomb\b"]
_TOXIC_RX = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(TOXIC_TERMS, key=len, reverse=True)) + r")\b", re.I)
_THREAT_RX = [re.compile(p, re.I) for p in THREAT_PATTERNS]


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _ZERO_WIDTH.sub("", text)
    return _CONTROL.sub(" ", text)


def detect_injection(text: str) -> dict[str, Any]:
    t = normalise(text)
    hits = [(name, w) for name, rx, w in _COMPILED if rx.search(t)]
    score = min(1.0, sum(w for _, w in hits))
    return {"detected": score >= INJECTION_THRESHOLD, "score": round(score, 2), "patterns": [n for n, _ in hits]}


def detect_toxicity(text: str) -> dict[str, Any]:
    t = normalise(text)
    terms = sorted({m.group(0).lower() for m in _TOXIC_RX.finditer(t)})
    threats = [p.pattern for p in _THREAT_RX if p.search(t)]
    words = max(1, len(t.split()))
    score = min(1.0, 3 * len(terms) / words * 10 + (0.8 if threats else 0.0))
    return {"detected": bool(terms or threats), "score": round(score, 2), "terms": terms, "threat": bool(threats)}


def redact_toxic(text: str) -> str:
    return _TOXIC_RX.sub("[redacted]", text)


def spotlight(text: str) -> str:
    """Delimit untrusted text so prompts can tell the model it is data, not instructions."""
    clean = normalise(text).replace("<<<", "«").replace(">>>", "»")
    return f"<<<CLAIM_TEXT\n{clean[:4000]}\nCLAIM_TEXT>>>"


ALLOWED_ROUTES = {"pay_as_computed", "request_further_evidence", "decline", "escalate_to_human"}
CAPS_INR = {"denied_boarding": 20000.0, "cancellation": 10000.0, "delay": 0.0, "downgrading": None}
_AUTHORISING = [
    (re.compile(r"\b(payment|compensation|refund|claim)\s+(has been|is|was)\s+(approved|authori[sz]ed|sanctioned|processed|released)\b", re.I),
     r"\1 is recommended for officer review"),
    (re.compile(r"\bwe (will|shall|have) (pay|paid|refund|refunded|credit|credited|transfer|transferred)\b", re.I),
     "the officer may decide to pay"),
    (re.compile(r"\b(i|we) (approve|authori[sz]e|sanction)\b", re.I), "the system recommends"),
    (re.compile(r"\byou will (receive|get|be paid)\b", re.I), "you may be entitled to"),
    (re.compile(r"\bguarantee[sd]?\b", re.I), "recommend"),
]


def filter_output(result: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    flags: list[str] = []
    if result.get("route") not in ALLOWED_ROUTES:
        flags.append("output_route_not_allowed")
        result["route"] = "escalate_to_human"
        result.setdefault("escalation", {}).setdefault("reasons", []).append("Output filter: route not permitted")
    ent = result.setdefault("entitlement", {})
    cat = result.get("disruption_category")
    cap = CAPS_INR.get(cat or "", None)
    comp = float(ent.get("compensation_inr") or 0)
    if comp < 0 or (cap is not None and comp > cap):
        flags.append("output_amount_exceeds_cap")
        result["route"] = "escalate_to_human"
        result.setdefault("escalation", {}).setdefault("reasons", []).append(
            f"Output filter: computed compensation {comp} outside the permitted range for {cat}")
    if comp > 0 and result.get("route") == "decline":
        flags.append("output_inconsistent_route")
        result["route"] = "escalate_to_human"
    text = result.get("justification", "")
    for rx, repl in _AUTHORISING:
        if rx.search(text):
            flags.append("output_authorising_language")
            text = rx.sub(repl, text)
    tox = detect_toxicity(text)
    if tox["detected"]:
        flags.append("output_toxicity")
        text = "The recommendation and its supporting provisions are listed below."
    result["justification"] = text
    return result, sorted(set(flags))
