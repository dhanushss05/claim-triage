"""Intake agent: guardrails, disruption classification and entity extraction.

Input:  the validated ClaimRequest (validation already happened at the API boundary).
Output: an IntakeHandoff (contracts.py) placed on the entitlement queue.

The intake agent holds no tools. It is the only component that reads the raw narrative; the entitlement agent,
which holds tools, receives the narrative only if no injection was detected (and then spotlighted as data).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from ..extraction import clean_llm_extraction, extract_rules, merge, structured_facts
from ..guardrails import detect_injection, detect_toxicity, redact_toxic, spotlight
from ..llm_client import ProviderUnavailable
from ..metrics import DEGRADED, GUARDRAIL
from ..prompts import load_prompt
from .runtime import AgentDeps, Recorder

AGENT = "intake"


async def run_intake(run_id: str, claim: dict[str, Any], deps: AgentDeps, rec: Recorder) -> dict[str, Any]:
    flags: list[str] = []
    free_text = {"narrative": claim.get("narrative", ""),
                 "disruption.airline_stated_cause": (claim.get("disruption") or {}).get("airline_stated_cause") or ""}
    with rec.step_timer("guardrails"):
        inj = {k: detect_injection(v) for k, v in free_text.items() if v}
        tox = {k: detect_toxicity(v) for k, v in free_text.items() if v}
    injected = [k for k, r in inj.items() if r["detected"]]
    toxic = [k for k, r in tox.items() if r["detected"]]
    if injected:
        flags.append("prompt_injection")
        GUARDRAIL.labels("prompt_injection", "input").inc()
    if toxic:
        flags.append("toxicity_input")
        GUARDRAIL.labels("toxicity", "input").inc()
    await rec.event("guardrails", injection=inj, toxicity=tox, flags=flags,
                    latency_ms=round(rec.step_ms.get("guardrails", 0), 1))

    narrative = claim.get("narrative", "")
    if toxic:
        narrative = redact_toxic(narrative)
    narrative_for_model: Optional[str] = None if injected else narrative

    with rec.step_timer("extract_rules"):
        rules = extract_rules(narrative)
        structured = structured_facts(claim)
    llm_facts: dict[str, Any] = {}
    degraded = False
    prompt_tag = ""
    if deps.llm is not None and narrative_for_model is not None:
        prompt = load_prompt(deps.prompts_dir, "extraction", deps.prompt_versions.get("extraction", "v1"))
        prompt_tag = prompt.tag
        with rec.step_timer("extract_model"):
            try:
                resp = await deps.llm.chat(messages=prompt.render(claim_text=spotlight(narrative_for_model)),
                                           prompt_id="extraction", prompt_version=prompt.tag, run_id=run_id,
                                           agent=AGENT, step="extract", max_tokens=300)
                await rec.llm("extract", prompt.tag, resp)
                llm_facts = clean_llm_extraction(resp.parsed)
            except ProviderUnavailable as exc:
                await rec.llm("extract", prompt.tag, None, error=str(exc))
                degraded = True
                DEGRADED.labels(AGENT, "extraction_unavailable").inc()
    elif narrative_for_model is None:
        rules = {}  # an injected narrative is not trusted for extraction either: structured fields only

    facts, sources, conflicts = merge(structured, llm_facts, rules)
    category = facts.get("category")
    if sources.get("category") == "structured":
        conf = 1.0
    elif sources.get("category") == "llm":
        conf = 0.9 if rules.get("category") in (None, category) else 0.6
    else:
        conf = float(rules.get("_category_confidence", 0.0))
    handoff = {
        "run_id": run_id, "category": category, "category_confidence": conf, "facts": facts,
        "fact_sources": sources, "conflicts": conflicts, "guardrail_flags": flags,
        "narrative_for_model": spotlight(narrative_for_model) if narrative_for_model else None,
        "prompt_versions": {"extraction": prompt_tag} if prompt_tag else {}, "degraded": degraded,
    }
    await rec.event("extraction", category=category, category_confidence=conf, facts=facts, fact_sources=sources,
                    conflicts=conflicts, model_fields=sorted(llm_facts), rule_fields=sorted(k for k in rules if not k.startswith("_")),
                    degraded=degraded)
    return handoff


def handoff_summary(h: dict[str, Any]) -> str:
    return json.dumps({k: h[k] for k in ("category", "category_confidence", "conflicts", "guardrail_flags")})
