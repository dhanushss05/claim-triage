"""Generates observability/grafana/dashboards/claim-triage.json (run: python scripts/make_dashboard.py)."""
import json
import os

panels = []
y = 0


def row(title):
    global y
    panels.append({"type": "row", "title": title, "collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "id": len(panels) + 1, "panels": []})
    y += 1


def ts(title, targets, x, w=8, h=8, unit="short", kind="timeseries", thresholds=None, desc=""):
    p = {"type": kind, "title": title, "description": desc, "datasource": {"type": "prometheus", "uid": "prom"},
         "gridPos": {"h": h, "w": w, "x": x, "y": y}, "id": len(panels) + 1,
         "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": l, "datasource": {"type": "prometheus", "uid": "prom"}} for i, (e, l) in enumerate(targets)],
         "fieldConfig": {"defaults": {"unit": unit}, "overrides": []}, "options": {"legend": {"displayMode": "list", "placement": "bottom"}}}
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": thresholds}
        p["fieldConfig"]["defaults"]["custom"] = {"thresholdsStyle": {"mode": "line"}}
    panels.append(p)


def next_row(h=8):
    global y
    y += h


row("Service level (SLO: p95 complete run <= 30 s)")
ts("Run latency, submission to result (p50 / p95)", [
    ("histogram_quantile(0.5, sum by (le) (rate(run_e2e_seconds_bucket[1m])))", "p50"),
    ("histogram_quantile(0.95, sum by (le) (rate(run_e2e_seconds_bucket[1m])))", "p95")], 0, unit="s",
   thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 30}])
ts("Throughput", [("sum(rate(runs_finished_total[1m]))", "runs finished /s"), ("sum(rate(claims_accepted_total[1m]))", "claims accepted /s"),
                  ("sum(rate(claims_rejected_total[1m]))", "claims rejected /s")], 8, unit="reqps")
ts("Error rate", [("sum(rate(http_requests_total{status=~\"5..\"}[1m])) / clamp_min(sum(rate(http_requests_total[1m])), 1e-9)", "HTTP 5xx ratio"),
                  ("sum(rate(runs_finished_total{status=\"failed\"}[5m])) / clamp_min(sum(rate(runs_finished_total[5m])), 1e-9)", "failed runs ratio")], 16, unit="percentunit")
next_row()
ts("Availability (targets up)", [("avg by (job) (up)", "{{job}}")], 0, unit="percentunit")
ts("API latency by route (p50 / p95)", [
    ("histogram_quantile(0.5, sum by (le, route) (rate(http_request_duration_seconds_bucket{service=\"api\"}[1m])))", "p50 {{route}}"),
    ("histogram_quantile(0.95, sum by (le, route) (rate(http_request_duration_seconds_bucket{service=\"api\"}[1m])))", "p95 {{route}}")], 8, unit="s")
ts("Queue depth (lag + pending)", [("max by (stream) (queue_depth)", "{{stream}}"), ("sum by (agent) (worker_inflight_runs)", "in flight {{agent}}")], 16)
next_row()
row("Agents")
ts("Task success rate per agent", [("sum by (agent) (rate(agent_tasks_total{outcome=\"success\"}[5m])) / clamp_min(sum by (agent) (rate(agent_tasks_total[5m])), 1e-9)", "{{agent}}")], 0, unit="percentunit")
ts("Steps and tool calls per run", [("sum(rate(agent_steps_per_run_sum[5m])) / clamp_min(sum(rate(agent_steps_per_run_count[5m])), 1e-9)", "steps / run"),
                                    ("sum(rate(agent_tool_calls_per_run_sum[5m])) / clamp_min(sum(rate(agent_tool_calls_per_run_count[5m])), 1e-9)", "tool calls / run")], 8)
ts("Per-step latency p95", [("histogram_quantile(0.95, sum by (le, agent, step) (rate(agent_step_seconds_bucket[5m])))", "{{agent}} {{step}}")], 16, unit="s")
next_row()
ts("Agent task latency p50/p95", [("histogram_quantile(0.5, sum by (le, agent) (rate(agent_task_seconds_bucket[5m])))", "p50 {{agent}}"),
                                  ("histogram_quantile(0.95, sum by (le, agent) (rate(agent_task_seconds_bucket[5m])))", "p95 {{agent}}")], 0, unit="s")
ts("Retry rate", [("sum by (agent, target) (rate(retries_total[5m])) / clamp_min(sum by (agent, target) (rate(calls_total[5m])), 1e-9)", "{{agent}} -> {{target}}")], 8, unit="percentunit")
ts("Container uptime", [("time() - process_start_time_seconds", "{{job}} {{instance}}")], 16, unit="s")
next_row()
ts("Verification and escalation", [("sum by (outcome) (rate(verification_total[5m]))", "verification {{outcome}}"),
                                   ("sum by (reason) (rate(escalations_total[5m]))", "escalation {{reason}}"),
                                   ("sum by (reason) (rate(degraded_runs_total[5m]))", "degraded {{reason}}")], 0)
ts("Routes", [("sum by (route) (rate(runs_finished_total[5m]))", "{{route}}")], 8)
ts("Guardrail detections", [("sum by (kind, stage) (rate(guardrail_events_total[5m]))", "{{stage}} {{kind}}")], 16)
next_row()
row("Resources per container")
ts("CPU (cores)", [("rate(process_cpu_seconds_total[1m])", "{{job}} {{instance}}")], 0, w=12)
ts("Memory (RSS)", [("process_resident_memory_bytes", "{{job}} {{instance}}")], 12, w=12, unit="bytes")
next_row()
row("Model, cost and caching")
ts("Cost per run (USD)", [("sum(increase(llm_cost_usd_total[10m])) / clamp_min(sum(increase(runs_finished_total[10m])), 1)", "USD / run"),
                          ("sum by (step) (increase(llm_cost_usd_total[10m])) / clamp_min(sum(increase(runs_finished_total[10m])), 1)", "{{step}}")], 0, unit="currencyUSD")
ts("Tokens by step", [("sum by (step, kind) (rate(llm_tokens_total[5m]))", "{{step}} {{kind}}")], 8)
ts("Provider circuit breaker and budget", [("max(llm_circuit_state)", "breaker (0 closed, 1 half-open, 2 open)"), ("max(llm_budget_spent_usd)", "spent today USD"),
                                           ("sum by (outcome) (rate(llm_requests_total[1m]))", "provider {{outcome}}")], 16)
next_row()
ts("Cache hit ratio", [("sum(rate(run_cache_total{result=\"hit\"}[5m])) / clamp_min(sum(rate(run_cache_total[5m])), 1e-9)", "whole-run cache"),
                       ("sum(rate(retrieval_cache_total{result=\"hit\"}[5m])) / clamp_min(sum(rate(retrieval_cache_total[5m])), 1e-9)", "retrieval cache"),
                       ("sum(rate(embedding_cache_total{result=\"hit\"}[5m])) / clamp_min(sum(rate(embedding_cache_total[5m])), 1e-9)", "query embedding cache")], 0, unit="percentunit")
ts("Rate limiting", [("sum by (kind) (rate(rate_limited_total[1m]))", "{{kind}}")], 8, unit="reqps")
ts("Index freshness", [("max(index_chunks)", "chunks"), ("max(index_age_seconds) / 3600", "age (h)"), ("max(ingest_throughput_chunks_per_second)", "ingest chunks/s")], 16)
next_row()
row("Quality")
ts("Live-run quality (sampled)", [("avg by (metric) (live_eval_score)", "{{metric}}"),
                                  ("sum by (metric) (rate(live_eval_observation_sum[15m])) / clamp_min(sum by (metric) (rate(live_eval_observation_count[15m])), 1e-9)", "15m mean {{metric}}")], 0, w=12, unit="percentunit")
ts("Feedback", [("sum by (verdict) (increase(feedback_total[1h]))", "{{verdict}}")], 12, w=12)
next_row()

dash = {"title": "Claim Triage", "uid": "claim-triage", "schemaVersion": 39, "version": 1, "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"}, "tags": ["mlops"], "timezone": "browser", "panels": panels,
        "templating": {"list": []}, "annotations": {"list": []}}
out = os.path.join(os.path.dirname(__file__), "..", "observability", "grafana", "dashboards", "claim-triage.json")
with open(out, "w") as fh:
    json.dump(dash, fh, indent=1)
print("wrote", out, len(panels), "panels")
