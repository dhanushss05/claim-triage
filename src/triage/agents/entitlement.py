"""Entitlement agent (LangGraph).

Topology of one run:

    plan --(tool)--> act --> plan --(finish / budget)--> determine --> verify --(pass)--> finalize
                                                                       |--(fail, attempts left)--> recover --> plan
                                                                       |--(fail, exhausted)------> fallback --> finalize | escalate

* plan     chooses the next tool. In hybrid mode the model proposes an action; a checklist guard overrides
           "finish" until the provision that appears to apply has been expanded (find_related) and searches derived
           from the text already read have run. Without a model, the checklist drives (degraded mode).
* act      executes one LangChain tool: search_corpus, get_clause, find_related, get_claim_fields.
* determine  the deterministic engine evaluates the facts; in hybrid mode the model also drafts a position with
           verbatim quotes. The model's draft is the candidate.
* verify   checks the candidate against the retrieved passages (see verifier.py).
* recover  changes strategy: exact clause lookups for missing conditions/exemptions and lexical (BM25) searches
           on the phrases that failed, then re-determines with the verifier's feedback.
* fallback after the last attempt, the engine's own position is verified on its own; if that also fails the run
           escalates. An unverified answer is never returned.
Loop bounds: AGENT_MAX_STEPS graph steps, AGENT_MAX_TOOL_CALLS tool calls, AGENT_MAX_WALL_S seconds.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal, Optional, TypedDict

from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from ..config import Settings
from ..guardrails import ALLOWED_ROUTES, filter_output
from ..llm_client import ProviderUnavailable
from ..metrics import DEGRADED, ESCALATIONS, GUARDRAIL, VERIFICATION
from ..prompts import load_prompt
from ..rules import engine
from .runtime import AgentDeps, Recorder, tool_metric
from .verifier import best_sentence, verify

logger = logging.getLogger(__name__)
AGENT = "entitlement"
ESCALATION_PATH = ["Customer relations officer (reviews this recommendation)",
                   "Airline nodal officer for grievances (CAR para 3.8.1)",
                   "AirSewa portal, Ministry of Civil Aviation (passenger escalation)"]


class State(TypedDict, total=False):
    run_id: str
    facts: dict
    category: Optional[str]
    guardrail_flags: list
    narrative: Optional[str]
    conflicts: list
    evidence: dict
    primary_hits: list
    tool_log: list
    steps: int
    tool_calls: int
    tool_errors: int
    t0: float
    done_items: list
    expanded: list
    derived_done: list
    forced: list
    next_action: Optional[dict]
    position: dict
    candidate: dict
    candidate_source: str
    verification: dict
    attempts: int
    feedback: str
    degraded: list
    stop_reason: str
    planner_sources: list
    result: dict


# --- tool schemas -------------------------------------------------------------------------------------------------
class SearchArgs(BaseModel):
    query: str = Field(description="Search text. Use provision wording, not the passenger's words, when refining.")
    mode: Literal["hybrid", "bm25", "dense"] = Field(default="hybrid", description="bm25 = exact phrase/number match")
    k: int = Field(default=5, ge=1, le=8)
    doc_id: Optional[Literal["CAR", "CHARTER"]] = None


class ClauseArgs(BaseModel):
    clause_id: str = Field(description="Clause id such as CAR:3.3.2 or CHARTER:6.2")


class FieldsArgs(BaseModel):
    fields: list[str] = Field(description="Names of claim fields to read")


TOOL_HELP = """search_corpus(query, mode='hybrid'|'bm25'|'dense', k=5, doc_id=None|'CAR'|'CHARTER'): search provisions
get_clause(clause_id): fetch one provision by id, e.g. CAR:3.4.1
find_related(clause_id): provisions that this one cites, provisions that cite it (exemptions, modifiers), siblings, and definitions of terms it uses
get_claim_fields(fields): read validated claim fields (e.g. notice_minutes, alternate_accepted, block_time_minutes)
finish(): stop researching and determine the entitlement"""


def _passage_brief(p: dict[str, Any], full: bool) -> dict[str, Any]:
    out = {"clause_id": p["clause_id"], "relation": p.get("relation"), "score": p.get("score")}
    out["text" if full else "excerpt"] = p["text"] if full else p["text"][:160]
    return out


def primary_query(state: State) -> str:
    cat = (state.get("category") or "flight disruption").replace("_", " ")
    narrative = state.get("narrative")
    if narrative:
        return f"{cat} compensation: {narrative[:300]}"
    return f"{cat} compensation entitlement amount"


def derived_queries(state: State) -> list[tuple[str, dict[str, Any]]]:
    """Queries built from what the agent has read so far (step 3 of the design approach)."""
    f, cat = state["facts"], state.get("category")
    ev = " ".join(p["text"].lower() for p in state.get("evidence", {}).values())
    out: list[tuple[str, dict[str, Any]]] = []
    if "alternate flight" in ev and f.get("alternate_offered"):
        out.append(("alternate", {"query": "passenger accepts the alternate flight no compensation facilities awaiting "
                                           "the alternate flight meals and refreshments waiting time"}))
    if "extraordinary circumstances" in ev and f.get("airline_stated_cause"):
        out.append(("cause", {"query": f"extraordinary circumstances {f['airline_stated_cause']} beyond the control "
                                       "of the airline general test burden"}))
    if f.get("contact_details_provided") is False or ("contact details" in ev and cat == "cancellation"):
        out.append(("contact", {"query": "passenger did not provide contact details at booking airline not liable to "
                                         "pay compensation", "mode": "bm25"}))
    wait = f.get("alternate_delay_minutes") or f.get("delay_minutes")
    if wait and ("para 3.4" in ev or "facilities" in ev or cat == "delay"):
        out.append(("facilities", {"query": "meals and refreshments in relation to the waiting time block time hotel "
                                            "accommodation"}))
    if cat in ("cancellation", "denied_boarding"):
        out.append(("exemptions", {"query": f"{cat.replace('_', ' ')} airline shall not be liable to pay compensation "
                                            "extraordinary circumstances", "mode": "bm25"}))
    if cat == "denied_boarding":
        out.append(("checkin", {"query": "presented for check-in not later than one hour volunteers denied boarding"}))
    if cat == "downgrading":
        out.append(("charter", {"query": "downgraded lower class refund international sectors", "doc_id": "CHARTER"}))
        out.append(("dg_exempt", {"query": "downgrade requested by the passenger refund does not apply", "mode": "bm25"}))
    if cat == "delay":
        out.append(("delay_comp", {"query": "no monetary compensation for delay facilities irrespective of the cause",
                                   "mode": "bm25"}))
    return out


def heuristic_next(state: State, settings: Settings) -> dict[str, Any]:
    done = set(state.get("done_items", []))
    if "fields" not in done:
        return {"tool": "get_claim_fields", "item": "fields",
                "args": {"fields": ["category", "notice_minutes", "informed_at_airport", "alternate_accepted",
                                    "alternate_delay_minutes", "block_time_minutes", "delay_minutes",
                                    "contact_details_provided", "airline_stated_cause", "cause_disputed", "sector"]}}
    if "primary" not in done:
        return {"tool": "search_corpus", "item": "primary", "args": {"query": primary_query(state), "k": 5}}
    if settings.naive_baseline:
        return {"tool": "finish", "item": "finish", "args": {}}
    if settings.enable_modifier_search:
        for cid in state.get("primary_hits", [])[:2]:
            if cid not in state.get("expanded", []):
                return {"tool": "find_related", "item": f"expand:{cid}", "args": {"clause_id": cid}}
    derived_done = state.get("derived_done", [])
    for key, args in derived_queries(state):
        if key not in derived_done and len(derived_done) < 4:
            return {"tool": "search_corpus", "item": f"derived:{key}", "args": {"k": settings.derived_k, **args}}
    if "applicability" not in done:
        return {"tool": "search_corpus", "item": "applicability",
                "args": {"query": "applicability of this CAR scheduled Indian air carriers foreign air carriers "
                                  "baggage claims", "k": 3, "mode": "bm25"}}
    return {"tool": "finish", "item": "finish", "args": {}}


def _rules_candidate(position: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    by_clause: dict[str, list[str]] = {}
    for comp in position["components"]:
        by_clause.setdefault(comp["clause_id"], [])
        for t in comp["terms"]:
            if t not in by_clause[comp["clause_id"]]:
                by_clause[comp["clause_id"]].append(t)
    cits = []
    for cid, terms in by_clause.items():
        p = evidence.get(cid)
        cits.append({"clause_id": cid, "terms": terms, "quote": best_sentence(p["text"], terms) if p else ""})
    pos = engine.Position(category=position["category"])
    pos.components = [engine.Component(**c) for c in position["components"]]
    pos.missing_facts, pos.escalate_reasons = position["missing_facts"], position["escalate_reasons"]
    return {"route": position["route"], "compensation_inr": position["compensation_inr"],
            "downgrade_refund_inr": position["downgrade_refund_inr"], "refund_due": position["refund_due"],
            "facilities": position["facilities"], "computation": position["computation"], "citations": cits,
            "justification": engine.justification(pos), "missing_evidence": position["missing_facts"],
            "escalate_reasons": position["escalate_reasons"], "notes": position["notes"]}


def _clean_model_candidate(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict) or raw.get("route") not in ALLOWED_ROUTES:
        return None
    cits = []
    for c in raw.get("citations") or []:
        if isinstance(c, dict) and isinstance(c.get("clause_id"), str):
            cid = c["clause_id"].strip()
            if ":" not in cid:
                cid = "CAR:" + cid.replace("para", "").strip()
            cits.append({"clause_id": cid, "quote": str(c.get("quote") or "")[:600]})

    def num(v: Any) -> float:
        try:
            return float(v or 0)
        except (TypeError, ValueError):
            return -1.0

    return {"route": raw["route"], "compensation_inr": num(raw.get("compensation_inr")),
            "downgrade_refund_inr": num(raw.get("downgrade_refund_inr")), "refund_due": bool(raw.get("refund_due")),
            "facilities": [str(x)[:120] for x in (raw.get("facilities") or [])][:6], "computation": str(raw.get("computation") or "")[:300],
            "citations": cits, "justification": str(raw.get("justification") or "")[:3000],
            "missing_evidence": [str(x)[:200] for x in (raw.get("missing_evidence") or [])][:6],
            "escalate_reasons": [str(x)[:300] for x in (raw.get("escalate_reasons") or [])][:6], "notes": []}


def build_graph(deps: AgentDeps, settings: Settings):
    async def t_search(query: str, mode: str = "hybrid", k: int = 5, doc_id: Optional[str] = None):
        return await deps.retriever.search(query, k=k, mode=mode, doc_id=doc_id)

    async def t_clause(clause_id: str):
        p = await deps.retriever.clause(clause_id)
        return [p] if p else []

    async def t_related(clause_id: str):
        return await deps.retriever.related(clause_id, settings.related_relations)

    tools = {
        "search_corpus": StructuredTool.from_function(coroutine=t_search, name="search_corpus",
                                                      description="Search CAR and charter provisions", args_schema=SearchArgs),
        "get_clause": StructuredTool.from_function(coroutine=t_clause, name="get_clause",
                                                   description="Fetch one provision by clause id", args_schema=ClauseArgs),
        "find_related": StructuredTool.from_function(coroutine=t_related, name="find_related",
                                                     description="Provisions that cite, are cited by, or define terms of a clause",
                                                     args_schema=ClauseArgs),
    }

    def rec_of(config) -> Recorder:
        return config["configurable"]["rec"]

    def budget_left(state: State) -> Optional[str]:
        if state.get("tool_calls", 0) >= settings.agent_max_tool_calls:
            return "max_tool_calls"
        if state.get("steps", 0) >= settings.agent_max_steps:
            return "max_steps"
        if time.monotonic() - state["t0"] >= settings.agent_max_wall_s:
            return "max_wall_clock"
        if state.get("tool_errors", 0) >= 3:
            return "tool_errors"
        return None

    async def llm_plan(state: State, rec: Recorder) -> Optional[dict[str, Any]]:
        prompt = load_prompt(deps.prompts_dir, "planner", deps.prompt_versions.get("planner", "v1"))
        ev = "\n".join(f"- {cid}: {p['text'][:140]}" for cid, p in state.get("evidence", {}).items()) or "(none yet)"
        hist = "\n".join(f"- {t['tool']} {json.dumps(t['args'])}" for t in state.get("tool_log", [])) or "(none)"
        facts = {k: v for k, v in state["facts"].items() if v is not None}
        msgs = prompt.render(tools=TOOL_HELP, facts=json.dumps(facts), evidence=ev, history=hist,
                             category=state.get("category") or "unknown")
        try:
            resp = await deps.llm.chat(messages=msgs, prompt_id="planner", prompt_version=prompt.tag,
                                       run_id=state["run_id"], agent=AGENT, step="plan", max_tokens=200)
        except ProviderUnavailable as exc:
            await rec.llm("plan", prompt.tag, None, error=str(exc))
            state.setdefault("degraded", []).append(f"planner model unavailable: {exc}")
            DEGRADED.labels(AGENT, "planner_unavailable").inc()
            return None
        await rec.llm("plan", prompt.tag, resp)
        a = resp.parsed or {}
        action = a.get("action")
        if action == "finish":
            return {"tool": "finish", "args": {}, "item": "finish"}
        if action in tools and isinstance(a.get("args"), dict):
            return {"tool": action, "args": a["args"], "item": f"llm:{action}"}
        return None

    async def plan(state: State, config) -> State:
        rec = rec_of(config)
        state["steps"] = state.get("steps", 0) + 1
        with rec.step_timer("plan"):
            stop = budget_left(state)
            source = "heuristic"
            if stop:
                action = {"tool": "finish", "args": {}, "item": "finish"}
                state["stop_reason"] = stop
                source = "budget"
            elif settings.naive_baseline:
                action = heuristic_next(state, settings)
            elif state.get("forced"):
                action = state["forced"].pop(0)
                source = "recovery"
            else:
                guard = heuristic_next(state, settings)
                action = guard
                use_llm = (settings.planner_mode in ("hybrid", "llm") and deps.llm is not None
                           and not any("unavailable" in d for d in state.get("degraded", [])))
                if use_llm:
                    proposed = await llm_plan(state, rec)
                    if proposed is not None:
                        dup = any(t["tool"] == proposed["tool"] and t["args"] == proposed["args"]
                                  for t in state.get("tool_log", []))
                        if proposed["tool"] == "finish" and guard["tool"] != "finish" and settings.planner_mode == "hybrid":
                            source = "guard_override"  # model wanted to stop before the modifiers were checked
                        elif dup:
                            source = "guard_override_duplicate"
                        else:
                            action, source = proposed, "model"
        state["next_action"] = action
        state.setdefault("planner_sources", []).append(source)
        await rec.event("plan", step_no=state["steps"], action=action["tool"], args=action.get("args"),
                        source=source, item=action.get("item"), latency_ms=round(rec.step_ms.get("plan", 0), 1))
        return state

    async def act(state: State, config) -> State:
        rec = rec_of(config)
        action = state["next_action"]
        name, args = action["tool"], dict(action.get("args") or {})
        state["tool_calls"] = state.get("tool_calls", 0) + 1
        evidence = state.setdefault("evidence", {})
        t0 = time.perf_counter()
        result: Any
        ok = True
        try:
            with rec.step_timer(f"tool:{name}"):
                if name == "get_claim_fields":
                    fields = args.get("fields") or list(state["facts"])
                    result = {k: state["facts"].get(k) for k in fields if k in state["facts"]}
                else:
                    result = await tools[name].ainvoke(args)
        except Exception as exc:  # retries exhausted inside the client
            ok = False
            result = []
            state["tool_errors"] = state.get("tool_errors", 0) + 1
            state.setdefault("degraded", []).append(f"tool {name} failed: {type(exc).__name__}")
            DEGRADED.labels(AGENT, "tool_failure").inc()
        latency = (time.perf_counter() - t0) * 1000
        tool_metric(AGENT, name, ok)
        new_ids: list[str] = []
        briefs = []
        if isinstance(result, list):
            for p in result:
                full = p["clause_id"] not in evidence
                if full:
                    new_ids.append(p["clause_id"])
                    evidence[p["clause_id"]] = p
                briefs.append(_passage_brief(p, full))
        item = action.get("item", "")
        done = state.setdefault("done_items", [])
        if item == "fields" or name == "get_claim_fields":
            done.append("fields")
        elif item == "primary":
            done.append("primary")
            section = {"denied_boarding": "CAR:3.2.", "cancellation": "CAR:3.3.", "delay": "CAR:3.4.",
                       "downgrading": "CAR:3.5."}.get(state.get("category") or "", "CAR:3.")
            hits = [p["clause_id"] for p in result if p["clause_id"].startswith("CAR:3.")] if isinstance(result, list) else []
            # expand the provisions of the category's own section first, then whatever else ranked highest
            state["primary_hits"] = (sorted(hits, key=lambda c: not c.startswith(section)))[:2]
        elif item.startswith("expand:") or name == "find_related":
            state.setdefault("expanded", []).append(args.get("clause_id"))
        elif item.startswith("derived:"):
            state.setdefault("derived_done", []).append(item.split(":", 1)[1])
        elif item == "applicability":
            done.append("applicability")
        state.setdefault("tool_log", []).append({"tool": name, "args": args, "ok": ok, "new": new_ids})
        await rec.event("tool_call", tool=name, args=args, ok=ok, latency_ms=round(latency, 1),
                        passages=briefs if isinstance(result, list) else None,
                        fields=result if isinstance(result, dict) else None, new_clauses=new_ids)
        return state

    async def determine(state: State, config) -> State:
        rec = rec_of(config)
        with rec.step_timer("determine"):
            position = engine.evaluate(state["facts"], state.get("guardrail_flags")).to_dict()
            if state.get("conflicts"):
                position.setdefault("notes", []).append("Extraction conflicts: " + "; ".join(state["conflicts"]))
            state["position"] = position
            candidate, source = None, "rules"
            use_model = (settings.determination_mode == "hybrid" and deps.llm is not None
                         and "prompt_injection" not in state.get("guardrail_flags", [])
                         and not any("unavailable" in d for d in state.get("degraded", [])))
            if use_model:
                prompt = load_prompt(deps.prompts_dir, "determination", deps.prompt_versions.get("determination", "v1"))
                ev = "\n\n".join(f"[{cid}] ({p.get('title_path', '')})\n{p['text']}" for cid, p in state["evidence"].items())
                facts = {k: v for k, v in state["facts"].items() if v is not None}
                msgs = prompt.render(facts=json.dumps(facts, indent=1), evidence=ev,
                                     narrative=state.get("narrative") or "(withheld)",
                                     feedback=state.get("feedback") or "none", category=state.get("category") or "unknown")
                try:
                    resp = await deps.llm.chat(messages=msgs, prompt_id="determination", prompt_version=prompt.tag,
                                               run_id=state["run_id"], agent=AGENT, step="determine", max_tokens=900)
                    await rec.llm("determine", prompt.tag, resp)
                    candidate = _clean_model_candidate(resp.parsed or {})
                    source = "model" if candidate else "rules"
                    if candidate is None:
                        state.setdefault("degraded", []).append("model determination unparseable")
                except ProviderUnavailable as exc:
                    await rec.llm("determine", prompt.tag, None, error=str(exc))
                    state.setdefault("degraded", []).append(f"determination model unavailable: {exc}")
                    DEGRADED.labels(AGENT, "determination_unavailable").inc()
            if candidate is None:
                candidate = _rules_candidate(position, state["evidence"])
        state["candidate"], state["candidate_source"] = candidate, source
        await rec.event("determination", source=source, route=candidate["route"],
                        compensation_inr=candidate.get("compensation_inr"), rule_route=position["route"],
                        rule_compensation_inr=position["compensation_inr"], citations=candidate["citations"],
                        missing_facts=position["missing_facts"], escalate_reasons=position["escalate_reasons"],
                        latency_ms=round(rec.step_ms.get("determine", 0), 1))
        return state

    async def verify_node(state: State, config) -> State:
        rec = rec_of(config)
        with rec.step_timer("verify"):
            v = verify(state["candidate"], state["evidence"], state["position"]["required_clauses"],
                       state["position"], is_model_answer=state["candidate_source"] == "model")
        state["attempts"] = state.get("attempts", 0) + 1
        v["attempt"], v["candidate_source"] = state["attempts"], state["candidate_source"]
        state["verification"] = v
        VERIFICATION.labels(AGENT, "pass" if v["passed"] else "fail").inc()
        await rec.event("verification", **v)
        return state

    async def recover(state: State, config) -> State:
        rec = rec_of(config)
        v, pos, ev = state["verification"], state["position"], state["evidence"]
        forced: list[dict[str, Any]] = []
        wanted = list(v["missing_clauses"]) + [u["clause_id"] for u in v["unsupported"] if u["clause_id"] not in ev]
        wanted += [c["clause_id"] for c in pos["components"] if c["clause_id"] not in ev]
        for cid in dict.fromkeys(wanted):
            forced.append({"tool": "get_clause", "args": {"clause_id": cid}, "item": f"recover:{cid}"})
        terms = [t for c in pos["components"] for t in c["terms"]][:6]
        if terms:
            forced.append({"tool": "search_corpus", "args": {"query": " ".join(terms), "mode": "bm25", "k": 5},
                           "item": "recover:lexical"})
        state["forced"] = forced
        failed = [c for c in v["checks"] if not c["passed"]]
        state["feedback"] = ("Your previous answer failed verification: " +
                             "; ".join(f"{c['name']}: {c['detail']}" for c in failed) +
                             ". Quote provisions verbatim, consider every condition and exemption, and only cite "
                             "passages listed in the evidence.")
        await rec.event("recovery", strategy="exact clause lookup + lexical search", forced=forced,
                        failed_checks=[c["name"] for c in failed])
        return state

    async def fallback(state: State, config) -> State:
        rec = rec_of(config)
        if state["candidate_source"] == "model":
            cand = _rules_candidate(state["position"], state["evidence"])
            v = verify(cand, state["evidence"], state["position"]["required_clauses"], None, is_model_answer=False)
            v["attempt"], v["candidate_source"] = state["attempts"] + 1, "rules_fallback"
            VERIFICATION.labels(AGENT, "pass" if v["passed"] else "fail").inc()
            await rec.event("verification", **v)
            if v["passed"]:
                state["candidate"], state["candidate_source"], state["verification"] = cand, "rules_fallback", v
                return state
        state["candidate_source"] = "unverified"
        return state

    def finalize_result(state: State, escalate_reason: str = "") -> dict[str, Any]:
        cand, pos, v = state["candidate"], state["position"], state.get("verification", {})
        degraded = bool(state.get("degraded"))
        route = cand["route"]
        reasons = list(cand.get("escalate_reasons") or [])
        citations = []
        for c in cand["citations"]:
            p = state["evidence"].get(c["clause_id"])
            if not p:
                continue
            label = (f"DGCA CAR S3 Series M Part IV, para {p['number']}" if p["doc_id"] == "CAR"
                     else f"Airline passenger charter, para {p['number']}")
            citations.append({"clause_id": c["clause_id"], "label": label, "doc_id": p["doc_id"],
                              "quote": c.get("quote") or best_sentence(p["text"], c.get("terms", []))})
        if escalate_reason:
            route = "escalate_to_human"
            reasons.append(escalate_reason)
        source = state.get("candidate_source")
        confidence = "high" if (source == "model" and v.get("attempt") == 1) else "medium"
        if degraded or route == "escalate_to_human":
            confidence = "low" if degraded else "medium"
        result = {
            "run_id": state["run_id"],
            "status": "escalated" if route == "escalate_to_human" else "completed",
            "route": route,
            "disruption_category": state.get("category"),
            "entitlement": {"compensation_inr": max(0.0, float(cand.get("compensation_inr") or 0)),
                            "computation": cand.get("computation") or pos.get("computation", ""),
                            "refund_due": bool(cand.get("refund_due")), "facilities": cand.get("facilities") or [],
                            "downgrade_refund_inr": max(0.0, float(cand.get("downgrade_refund_inr") or 0))},
            "citations": citations,
            "justification": cand.get("justification", ""),
            "missing_evidence": cand.get("missing_evidence") or [],
            "escalation": {"required": route == "escalate_to_human", "reasons": reasons, "path": ESCALATION_PATH},
            "verification": {"passed": bool(v.get("passed")) and not escalate_reason, "attempts": state.get("attempts", 0),
                             "checks": v.get("checks", [])},
            "guardrail_flags": list(state.get("guardrail_flags", [])),
            "degraded": degraded,
            "confidence": confidence,
            "basis": {"model": "model answer verified against retrieved provisions",
                      "rules": "deterministic engine verified against retrieved provisions",
                      "rules_fallback": "model answer failed verification; engine position verified instead",
                      "unverified": "verification failed; escalated"}.get(source or "", source or ""),
            "notes": cand.get("notes") or pos.get("notes", []),
            "agent": {"steps": state.get("steps", 0), "tool_calls": state.get("tool_calls", 0),
                      "planner_sources": state.get("planner_sources", []), "stop_reason": state.get("stop_reason", ""),
                      "degraded_reasons": state.get("degraded", []), "evidence": list(state.get("evidence", {}))},
        }
        if settings.naive_baseline:
            result["basis"] = "naive baseline: single retrieval pass, not verified (evaluation only)"
        result, flags = filter_output(result)
        for fl in flags:
            GUARDRAIL.labels(fl, "output").inc()
        result["guardrail_flags"] = sorted(set(result["guardrail_flags"]) | set(flags))
        if result["route"] == "escalate_to_human":
            result["status"] = "escalated"
            result["escalation"]["required"] = True
            for r in result["escalation"]["reasons"] or ["unspecified"]:
                ESCALATIONS.labels(_reason_label(r)).inc()
        return result

    async def finalize(state: State, config) -> State:
        rec = rec_of(config)
        state["result"] = finalize_result(state)
        await rec.event("final", route=state["result"]["route"], status=state["result"]["status"],
                        verification_passed=state["result"]["verification"]["passed"],
                        guardrail_flags=state["result"]["guardrail_flags"])
        return state

    async def escalate(state: State, config) -> State:
        rec = rec_of(config)
        v = state.get("verification", {})
        failed = [c["name"] for c in v.get("checks", []) if not c["passed"]]
        state["result"] = finalize_result(
            state, f"Self-verification failed after {state.get('attempts', 0)} attempts ({', '.join(failed)}); "
                   "no verified answer could be produced.")
        await rec.event("final", route="escalate_to_human", status="escalated", verification_passed=False,
                        guardrail_flags=state["result"]["guardrail_flags"])
        return state

    def after_plan(state: State) -> str:
        return "determine" if state["next_action"]["tool"] == "finish" else "act"

    def after_verify(state: State) -> str:
        if state["verification"]["passed"] or settings.naive_baseline:
            return "finalize"
        if state.get("attempts", 0) < settings.agent_max_verify_attempts and not budget_left(state):
            return "recover"
        return "fallback"

    def after_fallback(state: State) -> str:
        return "escalate" if state["candidate_source"] == "unverified" else "finalize"

    g = StateGraph(State)
    for name, fn in (("plan", plan), ("act", act), ("determine", determine), ("verify", verify_node),
                     ("recover", recover), ("fallback", fallback), ("finalize", finalize), ("escalate", escalate)):
        g.add_node(name, fn)
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", after_plan, {"act": "act", "determine": "determine"})
    g.add_edge("act", "plan")
    g.add_edge("determine", "verify")
    g.add_conditional_edges("verify", after_verify, {"finalize": "finalize", "recover": "recover", "fallback": "fallback"})
    g.add_edge("recover", "plan")
    g.add_conditional_edges("fallback", after_fallback, {"finalize": "finalize", "escalate": "escalate"})
    g.add_edge("finalize", END)
    g.add_edge("escalate", END)
    return g.compile()


def _reason_label(reason: str) -> str:
    r = reason.lower()
    for key, label in (("disputed", "cause_disputed"), ("general test", "general_test"), ("conflict", "provision_conflict"),
                       ("outside the corpus", "outside_corpus"), ("injection", "prompt_injection"),
                       ("verification", "verification_failed"), ("output filter", "output_filter"),
                       ("worker failed", "worker_failure"), ("deadline", "deadline")):
        if key in r:
            return label
    return "other"


def initial_state(run_id: str, handoff: dict[str, Any]) -> State:
    return {"run_id": run_id, "facts": handoff["facts"], "category": handoff.get("category"),
            "guardrail_flags": handoff.get("guardrail_flags", []), "narrative": handoff.get("narrative_for_model"),
            "conflicts": handoff.get("conflicts", []), "evidence": {}, "t0": time.monotonic(), "steps": 0,
            "tool_calls": 0, "attempts": 0, "degraded": ["intake ran without the model"] if handoff.get("degraded") else []}
