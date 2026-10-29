# Runbook

This is for whoever is on call for the claim triage stack. All commands run from the repository root on the Docker
host.

**Where to look first**

* Grafana: <http://localhost:3000>, dashboard **Claim triage**.
* Jaeger: <http://localhost:16686>, service `api`.
* API readiness: `curl localhost:8080/ready`.
* One run's full history: `curl localhost:8080/v1/runs/<run_id>/trace`.

Every log line carries `run_id` (which is also the correlation id) and `agent`, so you can follow one run with
`docker compose logs | grep <run_id>`.

**What the system promises when it is degraded.** It never returns an unverified answer. A run either passes
self-verification, possibly in degraded mode, or it is closed as `escalated` or `failed` with
`escalation.required = true`. Both outcomes go to a customer relations officer. So in a degraded state, look for
claims that are slow or escalated. You will not find wrong answers there.

---

## Failure mode 1: model provider outage (circuit breaker open)

**Symptoms**

* Grafana's *Provider circuit breaker and budget* panel shows `llm_circuit_state` at 2 (open), and *provider error*
  is rising.
* *Verification and escalation* shows `degraded_runs_total` climbing.
* Runs complete with `"degraded": true`, and their `basis` mentions the rule extractor or the checklist planner.
* Latency usually *drops*, because degraded runs make no model calls.
* If the provider is slow rather than down, latency climbs instead, and `retries_total{target="llm"}` rises before
  the breaker opens.

**Confirm**

```sh
curl -s localhost:8080/ready                                   # api + workers up
docker compose exec llm-gateway python -c "import httpx;print(httpx.get('http://localhost:8000/admin/breaker').json())"
docker compose logs llm-gateway --since 5m | grep -E 'breaker|provider_error' | tail
```

The breaker reports `state: open`, the number of consecutive failures, and when the next trial is allowed. The
gateway log shows the provider's real error, for example HTTP 429 (rate limit or quota), 401 (bad key) or a timeout.

**Recover**

1. **Provider down or slow.** Do nothing. The breaker sends a single trial call every `BREAKER_OPEN_SECONDS`
   (default 30 s) and closes by itself when the trial succeeds. Runs keep completing in degraded mode in the
   meantime.
2. **HTTP 401/403.** The key is wrong or was revoked. Fix `LLM_API_KEY` in `.env`, then run
   `docker compose up -d llm-gateway`. Only the gateway holds the key.
3. **HTTP 429 "requests per minute limit exceeded"** (the gateway log's `provider_body` names the quota). Set
   `LLM_MAX_RPM` just below the provider's limit (14 for Gemini free tier) and restart `llm-gateway`. The gateway
   then paces itself, and calls that would queue too long fall back instead of opening the breaker. A 429 for the
   daily quota ("requests per day", 500 for `gemini-3.5-flash-lite` on the free tier) clears at the provider's reset
   time (midnight Pacific time); the error body gives the exact wait. Until then runs complete in degraded mode.
4. **Daily budget exhausted** (`llm_budget_spent_usd` ≥ `LLM_DAILY_BUDGET_USD`). Either raise the budget or reduce
   load by lowering `WORKER_CONCURRENCY`.
5. **Switch provider.** Change `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` in `.env` and restart
   `llm-gateway`. Nothing else changes.
6. **Close the breaker by hand** after a fix:
   `docker compose exec llm-gateway python -c "import httpx;print(httpx.post('http://localhost:8000/admin/breaker/reset').json())"`.
7. **Afterwards**, find the degraded runs in the run store (`degraded: true`). You can resubmit them with the header
   `X-Cache-Bypass: 1` if you want a model-drafted position, but they are already verified.

**Drill (verified 2026-10-10).** `make demo` item `breaker` runs this drill. It sets the stub to `error`: the breaker
opens after 5 failures, runs complete degraded and verified, and then the stub is restored. The breaker goes
half-open, the trial succeeds, and the breaker closes.

---

## Failure mode 2: retriever down or index unavailable

**Symptoms**

* The *Availability (targets up)* panel shows `retriever` at 0, or `/ready` reports `"ready": false`.
* Runs end `escalated`. `escalation.reasons` says self-verification failed on `citations_resolve` and
  `required_retrieved`, because the agent could not read any provisions.
* `tool_calls_total{outcome="error"}` rises.
* If the index is stale rather than missing, *Index freshness* shows `index_age_seconds` growing and an old
  `index_built_timestamp_seconds`.

**Confirm**

```sh
docker compose ps retriever
docker compose logs retriever --since 10m | tail -30
docker compose exec retriever python -c "import httpx;print(httpx.get('http://localhost:8000/index/info').json())"
docker compose run --rm ingest python -m triage.retrieval.pipeline info
```

**Recover**

1. **Container exited.** Run `docker compose up -d retriever`. It loads `index/CURRENT` on start.
2. **Index missing or corrupt** (the log says it could not load `CURRENT`). Rebuild it with `make ingest`, which
   is idempotent. To roll back to a previous version instead, list `/data/index/v_*` in the `index-data` volume (`docker compose run --rm ingest ls /data/index`), repoint
   `CURRENT`, then call `POST /admin/reload` on the retriever.
3. **Corpus updated but answers still old.** Run `make ingest`; the retriever hot-reloads when `CURRENT` changes.
   If you changed the embedding model, use `make reembed`.
4. **Afterwards**, resubmit the runs that escalated during the outage.

**Drill (verified 2026-10-10).** We stopped the retriever and submitted the section 4.1 claim. After 7.6 s it ended
`escalated`, `degraded: true`, with failed checks `citations_resolve, required_retrieved, route_has_basis,
consistent_with_rules`. No answer was returned. Running `docker compose start retriever` restored normal runs.

---

## Other situations

| Symptom | Likely cause | Action |
|---|---|---|
| `queue_depth{stream="runs:entitlement"}` keeps growing, p95 over 30 s | Agent slots saturated (see report §6) | `docker compose up -d --scale agent-worker=N`, or raise `WORKER_CONCURRENCY` |
| A worker crashed mid-run | Its message stays pending | Nothing to do: another replica reclaims it after `CLAIM_IDLE_MS` (150 s). After `MAX_DELIVERIES` (3) the run is closed `failed` and escalated, and the message goes to `runs:dead` |
| Many HTTP 429 from the API | A client exceeded the token bucket | Expected. Raise `RATE_LIMIT_PER_MINUTE` / `RATE_LIMIT_BURST` only if the client is legitimate |
| Many 422s, quarantine growing | A client sends malformed claims | `GET /v1/quarantine` shows the errors per field. Contact the client with `contracts/CONTRACTS.md` |
| Live quality score drops (*Live-run quality* panel) | Prompt or model change | Compare `prompt_versions` on recent results, roll back the prompt version in `.env`, and run `make eval` |
