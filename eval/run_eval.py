"""Batch evaluation from the command line.

    python -m eval.run_eval --mode inprocess --label baseline
    python -m eval.run_eval --mode inprocess --label no-modifier-search --set ENABLE_MODIFIER_SEARCH=false
    python -m eval.run_eval --mode inprocess --label det-v2 --set PROMPT_VERSION_DETERMINATION=v2 --llm-url http://llm-gateway:8000
    python -m eval.run_eval --mode api --base-url http://api:8000 --label live-stack

inprocess: runs both agents in this process against a locally built index (fast, deterministic; the LLM is the
           in-process stub unless --llm-url points at a gateway).
api:       submits every scenario to a running stack, polls, and rebuilds evidence from each run's trace.
Writes eval/results/<label>.json and prints a metrics table. See triage/quality.py for metric definitions.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import statistics
import sys
import tempfile
import time
from typing import Any

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_scenarios(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    out = []
    for sc in data["scenarios"]:
        claim = deep_merge(data["base"], sc["claim"])
        claim["client_reference"] = f"EVAL-{sc['id']}"
        out.append({**sc, "claim": claim})
    return out


def check_supporting_passages(scenarios, index) -> list[str]:
    """The eval set itself must be grounded: every supporting quote must exist in the corpus."""
    from triage.agents.verifier import quote_in

    problems = []
    for sc in scenarios:
        sp = sc["expected"]["supporting_passage"]
        p = index.passage(sp["clause_id"])
        if not p or not quote_in(sp["quote"], p["text"]):
            problems.append(f"{sc['id']}: supporting passage not found in {sp['clause_id']}")
    return problems


def score(sc: dict[str, Any], result: dict[str, Any], evidence: dict[str, dict], facts: dict[str, Any]) -> dict[str, Any]:
    from triage.quality import context_relevance, faithfulness, groundedness, hallucinated
    from triage.rules.engine import rules

    exp = sc["expected"]
    ent = result.get("entitlement", {})
    route_ok = result.get("route") == exp["route"]
    amount_ok = True
    if "compensation_inr" in exp:
        amount_ok = abs(float(ent.get("compensation_inr") or 0) - exp["compensation_inr"]) <= 1
    if "downgrade_refund_inr" in exp:
        amount_ok = amount_ok and abs(float(ent.get("downgrade_refund_inr") or 0) - exp["downgrade_refund_inr"]) <= 1
    facs = " ".join(ent.get("facilities") or []).lower()
    fac_ok = all(f in facs for f in exp.get("facilities_include", [])) and not any(f in facs for f in exp.get("facilities_exclude", []))
    cited = {c["clause_id"] for c in result.get("citations", [])}
    must = exp.get("must_cite", [])
    cite_recall = (sum(1 for m in must if m in cited) / len(must)) if must else 1.0
    retrieved_recall = (sum(1 for m in must if m in evidence) / len(must)) if must else 1.0
    cat = result.get("disruption_category") or ""
    relevant = set(must) | set(rules()["required_by_category"].get(cat, [])) | {exp["supporting_passage"]["clause_id"]}
    support_retrieved = exp["supporting_passage"]["clause_id"] in evidence
    reason_ok = True
    if exp.get("escalation_reason"):
        from triage.agents.entitlement import _reason_label

        reason_ok = exp["escalation_reason"] in {_reason_label(x) for x in result.get("escalation", {}).get("reasons", [])}
    return {
        "id": sc["id"], "title": sc["title"], "expected_route": exp["route"], "route": result.get("route"),
        "route_ok": route_ok, "amount_ok": amount_ok, "facilities_ok": fac_ok, "escalation_reason_ok": reason_ok,
        "correct": route_ok and amount_ok and fac_ok and reason_ok,
        "compensation_inr": ent.get("compensation_inr"), "downgrade_refund_inr": ent.get("downgrade_refund_inr"),
        "must_cite_recall": cite_recall, "context_recall": retrieved_recall, "supporting_passage_retrieved": support_retrieved,
        "groundedness": groundedness(result, evidence, facts),
        "context_relevance": context_relevance(list(evidence), relevant),
        "faithfulness": faithfulness(result, evidence),
        "hallucination": 1.0 if hallucinated(result, evidence) else 0.0,
        "verification_passed": result.get("verification", {}).get("passed"),
        "verification_attempts": result.get("verification", {}).get("attempts"),
        "basis": result.get("basis"), "degraded": result.get("degraded"),
        "steps": result.get("agent", {}).get("steps"), "tool_calls": result.get("agent", {}).get("tool_calls"),
        "cost": result.get("cost", {}).get("total", {}), "latency_ms": result.get("timings_ms", {}).get("end_to_end"),
        "retrieved": len(evidence),
    }


async def run_inprocess(scenarios, args) -> list[dict[str, Any]]:
    os.environ.setdefault("EMBEDDING_BACKEND", "hashing")
    from triage.agents.entitlement import build_graph, initial_state
    from triage.agents.intake import run_intake
    from triage.agents.runtime import AgentDeps, LocalRetriever, Recorder
    from triage.config import get_settings
    from triage.contracts import ClaimRequest, cross_field_errors
    from triage.llm_client import GatewayLLM
    from triage.prompts import load_prompt
    from triage.retrieval.embeddings import build_embeddings
    from triage.retrieval.index import ClauseIndex, current_version_dir
    from triage.retrieval.pipeline import ingest
    from triage.stub_responses import InProcessStubLLM

    s = get_settings()
    index_dir = args.index_dir or tempfile.mkdtemp(prefix="eval-index-")
    emb = build_embeddings(s.embedding_backend, s.embedding_model, True)
    ingest(os.path.join(ROOT, "corpus", "manifest.yaml"), index_dir, emb, chunking=os.environ.get("CHUNKING", "clause"))
    index = ClauseIndex(current_version_dir(index_dir), emb)
    problems = check_supporting_passages(scenarios, index)
    if problems and os.environ.get("CHUNKING", "clause") == "clause":
        print("EVAL SET PROBLEMS:\n" + "\n".join(problems))
        sys.exit(1)
    if args.llm_url:
        llm = GatewayLLM(args.llm_url, s.llm_call_timeout_s, s.llm_max_retries, s.retry_base_delay_s)
    else:
        llm = InProcessStubLLM(float(os.environ.get("LLM_PRICE_PROMPT_PER_1M_USD", "0.10")),
                               float(os.environ.get("LLM_PRICE_COMPLETION_PER_1M_USD", "0.40")))
    prompts_dir = os.path.join(ROOT, "prompts")
    versions = {p: s.prompt_version(p) for p in ("extraction", "planner", "determination")}
    deps = AgentDeps(llm=llm, retriever=LocalRetriever(index), prompts_dir=prompts_dir, prompt_versions=versions)
    graph = build_graph(deps, s)
    rows = []
    for sc in scenarios:
        claim = ClaimRequest.model_validate(sc["claim"])
        errs = cross_field_errors(claim)
        if errs:
            print(f"{sc['id']} invalid claim: {errs}")
            sys.exit(1)
        run_id = f"eval-{sc['id']}"
        t0 = time.perf_counter()
        rec_i = Recorder(run_id, "intake")
        handoff = await run_intake(run_id, claim.model_dump(mode="json"), deps, rec_i)
        rec = Recorder(run_id, "entitlement")
        final = await graph.ainvoke(initial_state(run_id, handoff), config={"configurable": {"rec": rec}, "recursion_limit": 100})
        result = final["result"]
        ci, ce = rec_i.cost_summary()["total"], rec.cost_summary()["total"]
        result["cost"] = {"total": {k: ci[k] + ce[k] for k in ci}}
        result["timings_ms"] = {"end_to_end": round((time.perf_counter() - t0) * 1000, 1)}
        result["prompt_versions"] = {p: load_prompt(prompts_dir, p, v).tag for p, v in versions.items()}
        row = score(sc, result, final["evidence"], final["facts"])
        row["cost_by_step"] = {**rec_i.cost_summary()["by_step"], **rec.cost_summary()["by_step"]}
        row["result"] = result
        rows.append(row)
        print(f"{sc['id']:4} {'OK ' if row['correct'] else 'BAD'} route={result['route']:<25} expected={sc['expected']['route']:<25} "
              f"comp={row['compensation_inr']} G={row['groundedness']:.2f} F={row['faithfulness']:.2f} H={row['hallucination']:.0f} "
              f"steps={row['steps']} tools={row['tool_calls']} basis={row['basis']}")
    return rows


async def run_api(scenarios, args) -> list[dict[str, Any]]:
    import httpx

    rows = []
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as c:
        for sc in scenarios:
            claim = dict(sc["claim"], client_reference=f"EVAL-{sc['id']}-{int(time.time())}")
            t0 = time.perf_counter()
            resp = await c.post("/v1/claims", json=claim, headers={"X-Client-Id": "eval-cli"})
            if resp.status_code != 202:
                print(sc["id"], "submit failed", resp.status_code, resp.text[:300])
                continue
            run_id = resp.json()["run_id"]
            while True:
                data = (await c.get(f"/v1/runs/{run_id}")).json()
                if data.get("result"):
                    break
                if time.perf_counter() - t0 > 180:
                    data = {"result": {"route": "timeout"}}
                    break
                await asyncio.sleep(0.5)
            trace = (await c.get(f"/v1/runs/{run_id}/trace")).json()
            evidence: dict[str, dict] = {}
            for e in trace["events"]:
                if e["type"] == "tool_call" and e.get("passages"):
                    for p in e["passages"]:
                        if "text" in p:
                            evidence[p["clause_id"]] = {"text": p["text"]}
            facts = (trace.get("handoff") or {}).get("facts", {})
            result = data["result"]
            row = score(sc, result, evidence, facts)
            row["run_id"] = run_id
            row["result"] = result
            rows.append(row)
            print(f"{sc['id']:4} {'OK ' if row['correct'] else 'BAD'} route={result.get('route')} expected={sc['expected']['route']} "
                  f"latency={row['latency_ms']}ms")
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    def mean(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 4) if vals else None

    esc_exp = [r for r in rows if r["expected_route"] == "escalate_to_human"]
    esc_got = [r for r in rows if r["route"] == "escalate_to_human"]
    tp = sum(1 for r in rows if r["expected_route"] == r["route"] == "escalate_to_human")
    costs = [r["cost"].get("cost_usd", 0) for r in rows if r.get("cost")]
    tokens_p = [r["cost"].get("prompt_tokens", 0) for r in rows if r.get("cost")]
    tokens_c = [r["cost"].get("completion_tokens", 0) for r in rows if r.get("cost")]
    step_cost: dict[str, float] = {}
    for r in rows:
        for step, v in (r.get("cost_by_step") or {}).items():
            step_cost[step] = step_cost.get(step, 0) + v.get("cost_usd", 0)
    return {
        "scenarios": n, "accuracy": mean("correct"), "route_accuracy": mean("route_ok"), "amount_accuracy": mean("amount_ok"),
        "must_cite_recall": mean("must_cite_recall"), "context_recall": mean("context_recall"),
        "groundedness": mean("groundedness"), "context_relevance": mean("context_relevance"),
        "faithfulness": mean("faithfulness"), "hallucination_rate": mean("hallucination"),
        "escalation_precision": round(tp / len(esc_got), 4) if esc_got else None,
        "escalation_recall": round(tp / len(esc_exp), 4) if esc_exp else None,
        "verification_pass_rate": mean("verification_passed"), "mean_steps": mean("steps"),
        "mean_tool_calls": mean("tool_calls"), "mean_retrieved": mean("retrieved"),
        "mean_prompt_tokens": round(statistics.mean(tokens_p), 1) if tokens_p else None,
        "mean_completion_tokens": round(statistics.mean(tokens_c), 1) if tokens_c else None,
        "cost_per_100_runs_usd": round(100 * statistics.mean(costs), 6) if costs else None,
        "cost_share_by_step": {k: round(v / sum(step_cost.values()), 3) for k, v in sorted(step_cost.items(), key=lambda kv: -kv[1])} if step_cost and sum(step_cost.values()) else {},
        "mean_latency_ms": mean("latency_ms"),
        "s01_difficult_case_correct": next((r["correct"] for r in rows if r["id"] == "S01"), None),
        # Runs that completed without the model (breaker open, provider quota). A model comparison is only valid
        # when this is 0 or explained (INC-16).
        "degraded_runs": [r["id"] for r in rows if r.get("degraded")],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inprocess", "api"], default="inprocess")
    ap.add_argument("--label", default="run")
    ap.add_argument("--scenarios", default=os.path.join(HERE, "scenarios.yaml"))
    ap.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", ""))
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--index-dir", default="")
    ap.add_argument("--only", default="", help="comma-separated scenario ids")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE environment override for this run")
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    sys.path.insert(0, os.path.join(ROOT, "src"))
    os.environ.setdefault("PROMPTS_DIR", os.path.join(ROOT, "prompts"))
    scenarios = load_scenarios(args.scenarios)
    if args.only:
        keep = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep]
    rows = asyncio.run(run_inprocess(scenarios, args) if args.mode == "inprocess" else run_api(scenarios, args))
    summary = summarise(rows)
    summary.update({"label": args.label, "mode": args.mode, "overrides": args.set,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results", f"{args.label}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, default=str)
    print(json.dumps(summary, indent=2))
    print(f"written {out}")
    if summary.get("degraded_runs"):
        print(f"WARNING: {len(summary['degraded_runs'])} runs completed without the model: "
              f"{','.join(summary['degraded_runs'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
