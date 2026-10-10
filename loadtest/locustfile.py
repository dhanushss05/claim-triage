"""Locust load test: measures a COMPLETE run (submission -> final result), not just the submission call.

Each simulated client submits a claim, polls GET /v1/runs/{id} until a result exists, and reports the whole
duration as a custom request named "full_run" (type RUN). The LLM is the stub service, so the test measures this
system rather than a provider's rate limits. Claims carry a random fare so the whole-run cache does not serve them
(set LOADTEST_CACHEABLE=1 to measure the cache instead).

Headless example (inside the tools container):
  locust -f loadtest/locustfile.py --host http://api:8000 --headless -u 20 -r 5 -t 3m --csv loadtest/results/u20
"""
import os
import random
import time
import uuid

from locust import HttpUser, between, events, task

NARRATIVES = [
    ("My flight was cancelled. I was already at the airport and was told 90 minutes before departure. The airline put me on a flight four hours later, which I took. I want the compensation for cancellation.", {}),
    ("My flight was cancelled when I reached the airport. I declined the alternate flight and asked for a full refund.", {"block_time_minutes": 150}),
    ("I was denied boarding because the flight was overbooked. I did not volunteer. They put me on a flight three hours later, which I took.", {}),
    ("My flight was delayed by 5 hours. I want compensation for the delay.", {"block_time_minutes": 180}),
]
CACHEABLE = os.environ.get("LOADTEST_CACHEABLE") == "1"
POLL_S = float(os.environ.get("LOADTEST_POLL_S", "0.5"))
RUN_TIMEOUT_S = float(os.environ.get("LOADTEST_RUN_TIMEOUT_S", "180"))


def claim():
    text, flight_extra = random.choice(NARRATIVES)
    fare = 5000 if CACHEABLE else random.randint(2000, 9000)
    return {
        "client_reference": "LT-" + uuid.uuid4().hex[:12],
        "passenger": {"full_name": "Load Test", "email": "lt@example.com"},
        "flight": {"airline_code": "DA", "flight_number": "611", "origin": "BLR", "destination": "DEL",
                   "scheduled_departure": "2026-09-14T09:30:00+05:30", **flight_extra},
        "booking": {"pnr": "LT0001", "basic_fare_inr": fare, "fuel_charge_inr": 500},
        "narrative": text,
    }


class Claimant(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.client_id = "locust-" + uuid.uuid4().hex[:8]

    @task
    def submit_and_wait(self):
        t0 = time.perf_counter()
        with self.client.post("/v1/claims", json=claim(), headers={"X-Client-Id": self.client_id},
                              name="POST /v1/claims", catch_response=True) as resp:
            if resp.status_code != 202:
                resp.failure(f"status {resp.status_code}")
                return
            run_id = resp.json()["run_id"]
        status = "timeout"
        while time.perf_counter() - t0 < RUN_TIMEOUT_S:
            r = self.client.get(f"/v1/runs/{run_id}", name="GET /v1/runs/{id}")
            if r.status_code == 200 and r.json().get("result"):
                status = r.json()["result"]["status"]
                break
            time.sleep(POLL_S)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        events.request.fire(request_type="RUN", name="full_run", response_time=elapsed_ms, response_length=0,
                            exception=None if status in ("completed", "escalated") else Exception(f"run {status}"),
                            context={})
