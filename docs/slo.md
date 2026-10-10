# Service level objective (declared before any load test)

Declared: 2026-10-10, before the first Locust run. The git history shows this file committed before
`loadtest/results/` existed.

## Objective

**p95 latency of a complete run, from `POST /v1/claims` to the final result being available on
`GET /v1/runs/{id}`, is at most 30 seconds, while sustaining 2 runs per second, with fewer than 1% of runs
failing.** Measured with Locust against the stack running the LLM stub (fixed responses after ~0.8 s ± 35%).

Supporting targets: availability of the API 99.5% (Prometheus `up`); HTTP 5xx ratio below 1%.

## Why 30 seconds (section 1 of the brief)

* The decision is made by a customer relations officer, not by the system. The recommendation has to be ready
  before the officer opens the claim, which in an airline's claims queue is minutes to hours later. Anything far
  below that is not worth paying for.
* The web client (A8) shows the result to whoever submitted it. Past roughly half a minute a person stops
  waiting on a page, so 30 s is the upper bound for an interactive submission.
* A run makes several model calls (extraction, 2-6 planner calls, 1-3 determination calls) and 8-12 tool calls.
  With a provider at ~1 s per call a healthy run takes 8-15 s; 30 s leaves room for one verification retry and a
  retried provider call, so breaching it means queueing or a struggling component, which is what an SLO should
  detect.

## Why 2 runs per second

A mass disruption (for example winter fog at Delhi cancelling a morning bank of flights) can affect ~20,000
passengers of one airline in a day. If 10% claim and half of those arrive within the following three hours,
that is ~1,000 claims in 3 hours, about 0.1 per second. 2 runs/s is a 20x headroom over that peak, so the target
is about capacity for bursts and retries, not average load.

## How it is measured

`loadtest/locustfile.py` reports a custom `full_run` request (submit, then poll every 0.5 s until a result exists).
Claims are randomised so the whole-run cache does not serve them. Concurrency is stepped (`loadtest/run_matrix.sh`),
once with one `agent-worker` replica and once with `--scale agent-worker=3`.
