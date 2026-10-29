"""Answer-quality metrics, used by the offline evaluation (eval/) and by live sampling in the entitlement worker.

These are the four RAG-triad style metrics TruLens reports, computed deterministically (no LLM judge, so they are
free, reproducible, and runnable offline):

groundedness       share of justification sentences supported by the cited passages or the claim's own facts.
                   A sentence is supported when >= 50% of its content words (stop-words removed) appear in one cited
                   passage or in the rendered fact sheet. Clause tags like [CAR:3.3.2] are stripped first.
context_relevance  share of retrieved passages that are relevant. Offline: relevant = the scenario's gold clauses
                   plus the conditions/exemptions required for its category. Live (no gold): relevant = passages the
                   answer cites or that the category requires (a proxy, reported as such).
faithfulness       share of the answer's atomic claims that the cited passages entail: the route's basis (a decline
                   cites a provision that removes or conditions compensation; a payment cites an amount provision
                   whose figure bounds the amount; an escalation carries reasons), each facility owed (cited text
                   mentions it), and each quote (verbatim).
hallucination      per run, 1 if any citation does not resolve to retrieved text, any quote is not verbatim, or an
                   amount is not bounded by a cited figure; the rate is the mean over runs.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .agents.verifier import has_decline_basis, norm, quote_in

_STOP = set("""a an the of to and or in on for by is are be as at with from that this which shall any its it under not
no if such than other all has have was were so you your their they he she his her we our may can will would been being
there these those into also only per para paragraph sub""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9,%]*", text.lower().replace(",", "")) if w not in _STOP and len(w) > 2}


def _fact_sheet(facts: dict[str, Any]) -> str:
    parts = []
    for k, v in facts.items():
        if v is None:
            continue
        parts.append(f"{k.replace('_', ' ')} {v}")
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k.endswith("minutes"):
            h, m = divmod(int(v), 60)
            parts.append(f"{v} minutes {h} hours {h} hour {m} minutes {int(v) // 1440} days")
    return " ".join(parts)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\[[A-Z]+:[0-9.]+\]", "", text or "")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip().split()) >= 4]


def groundedness(result: dict[str, Any], evidence: dict[str, dict], facts: dict[str, Any]) -> float:
    sents = split_sentences(result.get("justification", ""))
    if not sents:
        return 0.0
    cited = [evidence[c["clause_id"]]["text"] for c in result.get("citations", []) if c["clause_id"] in evidence]
    sources = [content_words(t) for t in cited] + [content_words(_fact_sheet(facts))]
    ok = 0
    for s in sents:
        w = content_words(s)
        if not w:
            ok += 1
            continue
        if any(len(w & src) / len(w) >= 0.5 for src in sources):
            ok += 1
    return ok / len(sents)


def context_relevance(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    return sum(1 for r in retrieved if r in relevant) / len(retrieved)


def faithfulness_claims(result: dict[str, Any], evidence: dict[str, dict]) -> list[tuple[str, bool]]:
    cits = [c for c in result.get("citations", [])]
    cited_text = " ".join(evidence[c["clause_id"]]["text"] for c in cits if c["clause_id"] in evidence)
    claims: list[tuple[str, bool]] = []
    route = result.get("route")
    ent = result.get("entitlement", {})
    if route == "decline":
        claims.append(("decline basis", has_decline_basis(cited_text)))
    elif route == "pay_as_computed":
        amt = float(ent.get("compensation_inr") or 0) + float(ent.get("downgrade_refund_inr") or 0)
        figures = [float(x.replace(",", "")) for x in re.findall(r"INR\s?([\d,]+)", cited_text)]
        pct = bool(re.search(r"\d+%", cited_text))
        claims.append(("amount bounded by cited figure", amt > 0 and (any(amt <= f + 1 for f in figures) or pct)))
    elif route == "escalate_to_human":
        claims.append(("escalation reasons", bool(result.get("escalation", {}).get("reasons"))))
    elif route == "request_further_evidence":
        claims.append(("evidence requested", bool(result.get("missing_evidence"))))
    for fac in ent.get("facilities", []) or []:
        key = "meals" if "meal" in fac else "hotel" if "hotel" in fac else fac.split()[0].lower()
        claims.append((f"facility {key}", key in cited_text.lower()))
    for c in cits:
        p = evidence.get(c["clause_id"])
        claims.append((f"quote {c['clause_id']}", bool(p) and quote_in(c.get("quote", ""), p["text"])))
    return claims


def faithfulness(result: dict[str, Any], evidence: dict[str, dict]) -> float:
    claims = faithfulness_claims(result, evidence)
    return sum(ok for _, ok in claims) / len(claims) if claims else 0.0


def hallucinated(result: dict[str, Any], evidence: dict[str, dict], corpus_ids: Optional[set[str]] = None) -> bool:
    for c in result.get("citations", []):
        if c["clause_id"] not in evidence or (corpus_ids is not None and c["clause_id"] not in corpus_ids):
            return True
        if not quote_in(c.get("quote", ""), evidence[c["clause_id"]]["text"]):
            return True
    claims = dict(faithfulness_claims(result, evidence))
    return claims.get("amount bounded by cited figure") is False


def score_run(result: dict[str, Any], evidence: dict[str, dict], facts: dict[str, Any],
              relevant: Optional[set[str]] = None) -> dict[str, float]:
    retrieved = list(evidence)
    if relevant is None:
        from .rules.engine import rules

        relevant = {c["clause_id"] for c in result.get("citations", [])}
        relevant |= set(rules()["required_by_category"].get(result.get("disruption_category") or "", []))
    return {"groundedness": round(groundedness(result, evidence, facts), 4),
            "context_relevance": round(context_relevance(retrieved, relevant), 4),
            "faithfulness": round(faithfulness(result, evidence), 4),
            "hallucination": 1.0 if hallucinated(result, evidence) else 0.0}


__all__ = ["score_run", "groundedness", "context_relevance", "faithfulness", "hallucinated", "norm"]
