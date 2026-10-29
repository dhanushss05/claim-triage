"""Answer verification (A3). Runs before any answer is returned.

A candidate answer passes only if every check passes:
  citations_resolve      every cited clause id is in the evidence the agent retrieved during this run
  quotes_in_text         every quote is verbatim (whitespace/case-normalised) text of the cited passage, and every
                         rule component's required phrases appear in its cited passage
  required_retrieved     the conditions and exemptions that could remove the entitlement were retrieved (F4)
  route_has_basis        a decline cites a provision that removes or conditions compensation; a payment cites an amount;
                         an escalation carries reasons
  consistent_with_rules  (model answers) route and amount agree with the deterministic engine evaluated on the same
                         facts; disagreement means the model's answer is not supported by the provisions
"""
from __future__ import annotations

import re
from typing import Any, Optional

NEGATIVE = re.compile(r"\b(no (?:such |financial |monetary )?compensation|not be liable|not liable|not payable|"
                      r"not entitled|does not apply|shall not apply|not covered|not required|not be obliged|"
                      r"not have the obligation)\b", re.I)


# Facility kinds the CAR prescribes (para 3.8.1). A model answer must not drop one the engine finds owed.
FACILITY_KINDS = ("meal", "hotel")


def omitted_facilities(required: list[str], offered: list[str]) -> list[str]:
    req, off = " ".join(required).lower(), " ".join(offered).lower()
    return [k for k in FACILITY_KINDS if k in req and k not in off]


def has_decline_basis(cited_text: str) -> bool:
    """A decline needs a provision that removes compensation, or one that makes it conditional on something the
    passenger did not meet or prescribes a different remedy (corpus-specific markers in car_rules.yaml)."""
    if NEGATIVE.search(cited_text):
        return True
    from ..rules.engine import rules
    t = norm(cited_text)
    return any(norm(m) in t for m in rules().get("decline_markers", []))


def norm(text: str) -> str:
    t = (text or "").lower().replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[\"'`]", "", t)
    return " ".join(t.split())


def quote_in(quote: str, passage_text: str) -> bool:
    q = norm(quote).strip(" .;,")
    if len(q) < 12:
        return False
    return q in norm(passage_text)


def best_sentence(text: str, terms: list[str]) -> str:
    """Verbatim sentence (or sub-para) of `text` that contains the most support terms - used as the citation quote."""
    pieces = [p.strip() for p in re.split(r"(?<=[.;:])\s+|\n(?=\([a-z]\))", " ".join(text.split("\n"))) if p.strip()]
    if not pieces:
        return text[:300]
    scored = sorted(pieces, key=lambda p: (-sum(norm(t) in norm(p) for t in terms), len(p)))
    return scored[0][:500]


def verify(candidate: dict[str, Any], evidence: dict[str, dict[str, Any]], required: list[str],
           rule_position: Optional[dict[str, Any]], is_model_answer: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    cits = candidate.get("citations") or []
    unresolved = [c.get("clause_id") for c in cits if c.get("clause_id") not in evidence]
    checks.append({"name": "citations_resolve", "passed": bool(cits) and not unresolved,
                   "detail": f"unresolved: {unresolved}" if unresolved else f"{len(cits)} citations resolve"})

    unsupported = []
    for c in cits:
        p = evidence.get(c.get("clause_id"))
        if not p:
            continue
        if c.get("quote") and not quote_in(c["quote"], p["text"]):
            unsupported.append({"clause_id": c["clause_id"], "problem": "quote not found in passage"})
        for t in c.get("terms") or []:
            if norm(t) not in norm(p["text"]):
                unsupported.append({"clause_id": c["clause_id"], "problem": f"phrase '{t}' not in passage"})
    checks.append({"name": "quotes_in_text", "passed": not unsupported,
                   "detail": unsupported if unsupported else "all quotes/phrases found verbatim"})

    missing = [r for r in required if r not in evidence]
    checks.append({"name": "required_retrieved", "passed": not missing,
                   "detail": f"missing: {missing}" if missing else f"{len(required)} conditions/exemptions retrieved"})

    route = candidate.get("route")
    basis_ok, why = True, "ok"
    cited_text = " ".join(evidence[c["clause_id"]]["text"] for c in cits if c.get("clause_id") in evidence)
    if route == "decline":
        basis_ok = has_decline_basis(cited_text)
        why = "decline cites a provision removing or conditioning the entitlement" if basis_ok else "decline cites no provision removing the entitlement"
    elif route == "pay_as_computed":
        amt = float(candidate.get("compensation_inr") or 0) + float(candidate.get("downgrade_refund_inr") or 0)
        basis_ok = amt > 0 and bool(re.search(r"INR\s?[\d,]+|\d+%", cited_text))
        why = "payment cites an amount provision" if basis_ok else "payment without a cited amount provision"
    elif route == "escalate_to_human":
        basis_ok = bool(candidate.get("escalate_reasons"))
        why = "escalation carries reasons" if basis_ok else "escalation without reasons"
    elif route == "request_further_evidence":
        basis_ok = bool(candidate.get("missing_evidence"))
        why = "lists the evidence needed" if basis_ok else "no evidence listed"
    else:
        basis_ok, why = False, f"route {route!r} not allowed"
    checks.append({"name": "route_has_basis", "passed": basis_ok, "detail": why})

    if is_model_answer and rule_position is not None:
        decisive = sorted({c["clause_id"] for c in rule_position.get("components", [])
                           if c["kind"] in ("basis", "amount", "exemption")})
        cited_ids = {c.get("clause_id") for c in cits}
        not_cited = [d for d in decisive if d not in cited_ids]
        checks.append({"name": "cites_decisive_provision", "passed": not not_cited,
                       "detail": f"answer does not cite {not_cited}" if not_cited else "cites the provisions the outcome rests on"})
        r_route = rule_position["route"]
        r_amt = float(rule_position.get("compensation_inr") or 0) + float(rule_position.get("downgrade_refund_inr") or 0)
        m_amt = float(candidate.get("compensation_inr") or 0) + float(candidate.get("downgrade_refund_inr") or 0)
        omitted = omitted_facilities(rule_position.get("facilities") or [], candidate.get("facilities") or [])
        agree = route == r_route and abs(r_amt - m_amt) <= 1.0 and not omitted
        if agree:
            detail = "agrees"
        elif omitted:
            detail = f"answer omits facilities the provisions require: {omitted}"
        else:
            detail = f"model route={route} amount={m_amt}; provisions give route={r_route} amount={r_amt}"
        checks.append({"name": "consistent_with_rules", "passed": agree, "detail": detail})

    passed = all(c["passed"] for c in checks)
    return {"passed": passed, "checks": checks, "missing_clauses": missing,
            "unsupported": unsupported + [{"clause_id": u, "problem": "not retrieved"} for u in unresolved]}
