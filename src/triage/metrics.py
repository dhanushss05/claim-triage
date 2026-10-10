"""Prometheus metrics shared by every service.

Each container runs one process, so the default process collector (CPU seconds, resident memory, open fds,
start time) is the per-container resource and uptime signal on the dashboard.
"""
from __future__ import annotations

import threading

from prometheus_client import Counter, Gauge, Histogram, start_http_server

LAT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180)
FAST_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)

# --- HTTP (every FastAPI service) ---------------------------------------------------------------------------
HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["service", "route", "method", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["service", "route"],
                         buckets=FAST_BUCKETS + (10, 30))

# --- ingestion boundary --------------------------------------------------------------------------------------
CLAIMS_ACCEPTED = Counter("claims_accepted_total", "Claims accepted and queued")
CLAIMS_REJECTED = Counter("claims_rejected_total", "Claims rejected at the ingestion boundary", ["reason"])
RATE_LIMITED = Counter("rate_limited_total", "Requests refused by the per-client rate limiter", ["kind"])
RUN_CACHE = Counter("run_cache_total", "Whole-run cache lookups", ["result"])
FEEDBACK = Counter("feedback_total", "Feedback received", ["verdict"])
QUEUE_DEPTH = Gauge("queue_depth", "Messages waiting in a stream", ["stream"])

# --- runs / agents ------------------------------------------------------------------------------------------
RUN_E2E = Histogram("run_e2e_seconds", "Run latency from submission to final result", ["route"], buckets=LAT_BUCKETS)
RUNS_FINISHED = Counter("runs_finished_total", "Runs finished", ["status", "route", "degraded"])
AGENT_TASKS = Counter("agent_tasks_total", "Agent tasks by outcome", ["agent", "outcome"])
AGENT_TASK_LATENCY = Histogram("agent_task_seconds", "Agent task latency", ["agent"], buckets=LAT_BUCKETS)
AGENT_STEP_LATENCY = Histogram("agent_step_seconds", "Agent step latency", ["agent", "step"], buckets=FAST_BUCKETS + (10, 30))
AGENT_STEPS = Histogram("agent_steps_per_run", "Graph steps per run", ["agent"], buckets=(1, 2, 4, 6, 8, 10, 12, 16, 20, 30))
AGENT_TOOL_CALLS = Histogram("agent_tool_calls_per_run", "Tool calls per run", ["agent"], buckets=(0, 1, 2, 4, 6, 8, 10, 12, 16, 20))
TOOL_CALLS = Counter("tool_calls_total", "Tool calls", ["agent", "tool", "outcome"])
RETRIES = Counter("retries_total", "Retried calls", ["agent", "target"])
CALLS = Counter("calls_total", "Calls subject to retry (denominator for retry rate)", ["agent", "target"])
VERIFICATION = Counter("verification_total", "Verification outcomes", ["agent", "outcome"])
ESCALATIONS = Counter("escalations_total", "Escalations to a human", ["reason"])
GUARDRAIL = Counter("guardrail_events_total", "Guardrail detections", ["kind", "stage"])
DEGRADED = Counter("degraded_runs_total", "Runs that used the fallback path", ["agent", "reason"])
WORKER_INFLIGHT = Gauge("worker_inflight_runs", "Runs currently executing in this worker", ["agent"])

# --- LLM ----------------------------------------------------------------------------------------------------
LLM_REQUESTS = Counter("llm_requests_total", "LLM provider requests", ["provider", "outcome"])
LLM_LATENCY = Histogram("llm_request_seconds", "LLM provider latency", ["provider", "step"], buckets=LAT_BUCKETS)
LLM_TOKENS = Counter("llm_tokens_total", "Tokens", ["agent", "step", "kind"])
LLM_COST = Counter("llm_cost_usd_total", "Cost in USD at call time", ["agent", "step"])
BREAKER_STATE = Gauge("llm_circuit_state", "Circuit breaker state (0 closed, 1 half-open, 2 open)")
BUDGET_SPENT = Gauge("llm_budget_spent_usd", "LLM spend today in USD")

# --- retrieval / index -----------------------------------------------------------------------------------------
SEARCH_LATENCY = Histogram("retrieval_seconds", "Retriever search latency", ["mode"], buckets=FAST_BUCKETS)
RETRIEVAL_CACHE = Counter("retrieval_cache_total", "Retrieval cache lookups", ["result"])
EMBED_CACHE = Counter("embedding_cache_total", "Query-embedding cache lookups", ["result"])
INDEX_CHUNKS = Gauge("index_chunks", "Chunks in the served index")
INDEX_DOCS = Gauge("index_documents", "Documents in the served index")
INDEX_BUILT = Gauge("index_built_timestamp_seconds", "When the served index was built (unix time)")
INDEX_AGE = Gauge("index_age_seconds", "Age of the served index")
INGEST_THROUGHPUT = Gauge("ingest_throughput_chunks_per_second", "Chunks processed per second in the last ingest")
INGEST_DURATION = Gauge("ingest_duration_seconds", "Duration of the last ingest")
INGEST_EMBEDDED = Gauge("ingest_chunks_embedded", "Chunks (re-)embedded in the last ingest")
EMBEDDING_INFO = Gauge("embedding_backend_info", "Active embedding backend", ["backend", "model"])

# --- quality ---------------------------------------------------------------------------------------------------
LIVE_EVAL = Gauge("live_eval_score", "Rolling live-run quality score", ["metric"])
LIVE_EVAL_HIST = Histogram("live_eval_observation", "Live-run quality observations", ["metric"],
                           buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))

_started = threading.Event()


def serve_metrics(port: int) -> None:
    """Expose /metrics on a side port (used by queue workers that have no HTTP app)."""
    if not _started.is_set():
        start_http_server(port)
        _started.set()
