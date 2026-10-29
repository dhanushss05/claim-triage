# Air Passenger Compensation Claim Triage Service

Mini Project 04, UE24AM342AA5 ML System Design and AgentOps, Dept of CSE (AI & ML), PES University.

| Team member | SRN |
|---|---|
| Dhanush S S | PES1UG24AM082 |
| Mahipal S | PES1UG24AM153 |
| Vivaan A | PES1UG24AM071 |

The service is multi-agent. It accepts a passenger's claim for denied boarding, cancellation, delay or downgrading,
validates it, and works out what DGCA CAR Section 3 Series M Part IV says the passenger is owed. It then routes the
claim with clause-level citations to one of four outcomes: **pay as computed**, **request further evidence**,
**decline (with the provision)** or **escalate to a human**. It never authorises a payment. A customer relations
officer decides.

* **Report** (design, evaluation, load test, cost, what did not work): [`REPORT.md`](REPORT.md), also as
  `REPORT.pdf`.
* **Architecture**, including the chunking and embedding strategy: [`docs/architecture.md`](docs/architecture.md).
* **Operations**:
  * [`docs/runbook.md`](docs/runbook.md)
  * [`docs/incident_log.md`](docs/incident_log.md)
  * [`docs/slo.md`](docs/slo.md)
* **Interfaces**: [`contracts/CONTRACTS.md`](contracts/CONTRACTS.md), with the JSON Schemas next to it.
* **Evaluation set**: [`eval/scenarios.yaml`](eval/scenarios.yaml), 27 scenarios. Results are in `eval/results/`.

---

## 1. What you need

* **Docker** with **Docker Compose v2** (`docker compose version` should print v2.x). Docker Desktop on
  Windows/macOS works.
* About **6 GB of free disk** and **4 GB of RAM** for Docker. The retriever image includes CPU PyTorch for local
  embeddings.
* An internet connection for the first build, to pull images and Python packages and to download the embedding
  model once.
* **No API key is needed.** By default the model is a bundled stub. Section 5 shows how to switch to a free Gemini
  key.

Nothing has to be installed on the host apart from Docker. Every component, including tests, evaluation and load
tests, runs in containers.

## 2. Start it

```sh
git clone <this repository> claim-triage && cd claim-triage
cp .env.example .env          # all configuration lives here; defaults work as-is
docker compose up -d --build  # first build takes several minutes (PyTorch download)
docker compose ps             # wait until services show "healthy" (about 1-2 minutes)
```

`make up` does the same, if you have `make`.

What starts:

1. `ingest` builds the search index from `corpus/` and exits. This is expected.
2. The long-running services come up: `redis`, `llm-stub`, `llm-gateway`, `retriever`, `api`, `intake-agent`,
   `agent-worker`, `jaeger`, `prometheus` and `grafana`.

| Open | URL |
|---|---|
| **Web client**: submit a claim and see the recommendation, citations and escalation path | <http://localhost:8080> |
| API docs (OpenAPI) | <http://localhost:8080/docs> |
| Grafana dashboard **Claim triage**, login from `.env` (`admin` / `change-me`) | <http://localhost:3000> |
| Jaeger traces (service `api`) | <http://localhost:16686> |
| Prometheus | <http://localhost:9090> |

**Check it works.**

1. Run `curl localhost:8080/ready`. It should print `{"ready":true,...}`.
2. In the web client, click **Section 4.1 case**, then **Submit claim**.
3. After about 10 seconds you should see **decline, INR 0, meals and refreshments owed**, citing CAR 3.3.2 and
   3.4.1, with *verification passed*.

### Scale the entitlement agent

```sh
docker compose up -d --scale agent-worker=3
```

Prometheus discovers the new replicas automatically.

### Stop

`docker compose down` stops everything and keeps the data volumes. Add `-v` to delete them too.

## 3. Use the API

```sh
# Submit: returns 202 with a run_id at once; the run happens in the background
curl -s -X POST localhost:8080/v1/claims -H 'Content-Type: application/json' -H 'X-Client-Id: me' -d '{
  "client_reference": "CR-1001",
  "passenger": {"full_name": "Asha Rao", "email": "asha@example.com"},
  "flight": {"airline_code": "DA", "flight_number": "611", "origin": "BLR", "destination": "DEL",
             "scheduled_departure": "2026-09-14T09:30:00+05:30", "block_time_minutes": 150},
  "booking": {"pnr": "QX7TZ2", "basic_fare_inr": 6200, "fuel_charge_inr": 1100},
  "narrative": "My flight was cancelled when I reached the airport. I declined the alternate flight and asked for a refund."
}'
# -> {"run_id":"run_...","status":"queued","status_url":"/v1/runs/run_...", ...}

curl -s localhost:8080/v1/runs/<run_id>          # poll until "result" is present (or pass "callback_url" when submitting)
curl -s localhost:8080/v1/runs/<run_id>/trace    # every step: tool calls, passages, prompt versions, tokens, cost, latency
curl -s -X POST localhost:8080/v1/runs/<run_id>/feedback -H 'Content-Type: application/json' \
     -d '{"verdict": "unsatisfactory", "reason": "missed the hotel entitlement"}'
curl -s localhost:8080/v1/feedback/summary       # aggregated feedback
curl -s localhost:8080/v1/quality                # live quality scores (sampled runs)
curl -s localhost:8080/v1/quarantine             # rejected requests with field-level errors
```

A malformed request gets **HTTP 422** with one error per field. It is written to quarantine and **no run starts**.
Exceeding the per-client rate limit returns **HTTP 429** with `Retry-After`.

## 4. Demo, tests, evaluation, load tests

The stack must be running for all of these except `eval-inprocess` and `test`.

| Command | What it does |
|---|---|
| `make demo` | Walks through the demo items against the live stack: valid request, end-to-end trace, malformed request, outside the corpus, self-verification, prompt injection, circuit breaker, the 4.1 case, and the cache. Run a single item with `docker compose --profile tools run --rm --entrypoint python eval scripts/demo.py breaker` (items: `valid trace malformed outside verify injection breaker difficult cache`) |
| `make test` | pytest (28 tests) inside the tools container |
| `make eval` | Runs the 27-scenario evaluation set through the live API and writes `eval/results/stack.json` |
| `make eval-inprocess` | The same scenarios in one process with the stub, with no stack needed |
| `make ingest` / `make reembed` | Re-runs the idempotent corpus pipeline, or re-embeds every chunk (for example after changing `EMBEDDING_MODEL`). The retriever hot-reloads |
| `WORKERS=1 make loadtest` | Locust at 5, 10, 20 and 40 concurrent users for 90 s each, printing RPS, p50 and p95 against the SLO. Results go to `loadtest/results/` |

If you don't have `make`, open the `Makefile`; each target is a single `docker compose ...` command. The Locust web
UI is available with `docker compose --profile loadtest up locust`, at <http://localhost:8089>.

**Reproducing the measurements in the report**

* **L3 before/after:**

  ```sh
  docker compose --profile tools run --rm eval --mode inprocess --label L3-before \
    --set FIND_RELATED_RELATIONS=referenced_by_this,cites_this,cites_parent,sibling,definition --set DERIVED_K=5
  docker compose --profile tools run --rm eval --mode inprocess --label L3-after
  ```

* **L2 prompt variants:** add `--set PROMPT_VERSION_DETERMINATION=v1` (or `v2`) and a matching `--label`.
* **Scaling:**

  ```sh
  docker compose up -d --scale agent-worker=3
  WORKERS=3 make loadtest
  ```

  To test more slots per replica, set `WORKER_CONCURRENCY` in `.env` first.

## 5. Use a real model (optional, free)

The model sits behind `llm-gateway`, so switching provider is a configuration change only. We ran the evaluation
with Google Gemini's free tier (`gemini-3.5-flash-lite`); the results are in the report, §5.

1. Get a free API key at <https://aistudio.google.com>.
2. In `.env`, replace the four `LLM_*` lines with the Gemini block that is commented out in `.env.example`, and set
   the request pacing:

   ```sh
   LLM_PROVIDER=gemini
   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
   LLM_MODEL=gemini-3.5-flash-lite
   LLM_API_KEY=<your key>
   LLM_MAX_RPM=14        # the free tier allows 15 requests per minute for this model
   ```

3. Run `docker compose up -d`. Only `llm-gateway` receives the key.

Any OpenAI-compatible endpoint works the same way, including OpenAI, Groq and a local Ollama. Google retires model
ids from time to time; if the gateway log shows `404 ... no longer available`, pick a current id from the message.

**Free tiers are rate limited.** A run makes about 14 model calls. Gemini's free tier for this model allows 15
requests per minute and **500 requests per day**, so one run takes about a minute, the full 27-scenario evaluation
takes about 30 minutes and about 390 requests, and only one full evaluation fits in a day. Try a few scenarios
first, through the gateway:

```sh
docker compose --profile tools run --rm eval --mode inprocess --label gemini --only S01,S02,S12 \
  --llm-url http://llm-gateway:8000
```

To compare prompt versions on the real model, run the same command twice, adding
`--set PROMPT_VERSION_DETERMINATION=v1` and then `v2`, each with its own `--label`. On the free tier, use the same
`--only` subset for both so the pair fits in one day's quota. Check `degraded_runs` in the summary: a run that
completed without the model (quota, breaker) is listed there, and the eval prints a warning.

If the provider fails, the circuit breaker opens and runs complete in a degraded but verified mode. See the runbook.
Load tests should always use the stub (`LLM_PROVIDER=stub`, `LLM_MAX_RPM=0`).

## 6. The corpus

`corpus/manifest.yaml` lists the documents:

* `corpus/raw/dgca_car_s3_m_p4_rev4.pdf`: the **official** DGCA CAR Section 3, Series M, Part IV, Issue I,
  Revision 4 dated 25 Jan 2023 (effective 15 Feb 2023), as published on <https://www.dgca.gov.in>. It is ingested
  from its text layer; running page headers are stripped.
* `corpus/raw/airline_passenger_charter.md`: a **synthetic** airline charter ("Demo Air"). It deliberately conflicts
  with the CAR on international downgrades, to exercise conflict handling. Replace it with a real airline's charter
  by adding a manifest entry with `authority: airline_policy`.

To ingest a new CAR revision:

1. Put the PDF in `corpus/raw/`, and update `path` and `version` in the manifest.
2. Run `make ingest` (only changed documents are re-processed).
3. Run `make test`. `test_rule_terms_match_the_corpus` lists any phrase in `src/triage/rules/car_rules.yaml` that the
   new text no longer contains. Update those entries, then run `make eval` to confirm the outcomes.

A PDF without a text layer is rejected with a message to OCR it first (for example with `ocrmypdf`).

## 7. Configuration

Every setting is an environment variable documented in [`.env.example`](.env.example): endpoints, the model
provider, prices, the budget, the breaker, agent loop limits, prompt versions, rate limits, caching and
observability.

The repository contains **no credentials, endpoints or model identifiers** in code. `.env` is git-ignored; never
commit it.

## 8. Repository layout

```
src/triage/
  contracts.py            data contracts (Pydantic) + cross-field validation
  agents/intake.py        agent 1: guardrails, classification, extraction
  agents/entitlement.py   agent 2: LangGraph loop (plan/act/determine/verify/recover/fallback)
  agents/verifier.py      answer verification (A3)
  rules/                  deterministic CAR rule engine (cross-check and fallback)
  retrieval/              chunking, embeddings, FAISS+BM25 index, ingest CLI
  services/               api, workers, retriever, llm-gateway, llm-stub, web client (static/)
  guardrails.py quality.py prompts.py llm_client.py store.py queue.py metrics.py tracing.py
prompts/                  versioned prompts (extraction v1/v2, planner v1, determination v1/v2)
corpus/                   source documents + manifest
contracts/                JSON Schemas + CONTRACTS.md (generated by scripts/export_contracts.py)
eval/                     scenarios.yaml (27), run_eval.py, results/
loadtest/                 locustfile.py, run_matrix.sh, summarise.py, results/
observability/            prometheus.yml, Grafana provisioning + dashboard
docs/                     architecture, runbook, incident log, SLO, images
scripts/                  demo.py, export_contracts.py, make_dashboard.py
tests/                    pytest
```

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| Build fails downloading PyTorch, or the network is slow or restricted | Build without local embeddings: `INSTALL_EMBEDDINGS=false docker compose build`, and set `EMBEDDING_BACKEND=hashing` in `.env`. Retrieval still works (BM25 plus hashing vectors) |
| `retriever` unhealthy on first start | It downloads the embedding model (about 90 MB) on first start. Wait about 2 minutes, then check `docker compose logs retriever`. If the download is blocked it falls back to hashing embeddings (`EMBEDDING_ALLOW_FALLBACK=true`) |
| `ingest` exited | That is normal: it is a one-shot job. `docker compose logs ingest` should end with `"status": "built"` or `"up to date"` |
| Port already in use (8080, 3000, 9090, 16686) | Change the left-hand port in `docker-compose.yml`, for example `"8081:8000"` |
| `toomanyrequests` from Docker Hub | Wait, or `docker login`. The images are also on `mirror.gcr.io` |
| Behind a corporate proxy with its own root CA | Put the certificate as `docker/certs/<name>.crt` and rebuild. It is trusted at build and run time, and `*.crt` files are git-ignored |
| A run shows `degraded: true` | The model provider was unavailable and the fallback path answered. See runbook failure mode 1 |
| Gateway log shows `provider_error ... 429` | The provider's per-minute quota. Set `LLM_MAX_RPM` below it (14 for Gemini free tier) and restart `llm-gateway` |
| Everything is escalated | Probably the retriever is down. See runbook failure mode 2 |
