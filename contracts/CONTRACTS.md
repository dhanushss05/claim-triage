# Data contracts

Every external interface has a published contract. JSON Schemas in this folder are generated from the Pydantic
models in `src/triage/contracts.py` (`make contracts`) and served live at `GET /v1/contracts/{claim|result|feedback}`.

| Interface | Direction | Contract |
|---|---|---|
| `POST /v1/claims` | client to API | `claim_request.schema.json` + cross-field rules below |
| `202` response | API to client | `run_accepted.schema.json` |
| `GET /v1/runs/{id}` result, and `callback_url` POST | API/worker to client | `run_result.schema.json` |
| `POST /v1/runs/{id}/feedback` | client to API | `feedback.schema.json` |
| Queue `runs:entitlement` | intake agent to entitlement agent | `intake_handoff.schema.json` |
| LLM gateway `POST /v1/chat` | agents to gateway | `ChatRequest` in `services/llm_gateway.py` |
| Retriever `POST /search`, `GET /clause/{id}`, `GET /related/{id}` | agent to retriever | `SearchReq` in `services/retriever_app.py` |

## Enforcement at the ingestion boundary (D1)

Validation runs in the API before anything is queued. A request that fails any rule gets `422` with field-level
errors, is written to the quarantine (Redis list `quarantine` and `/data/quarantine/quarantine.jsonl` in the API
volume) with those errors, and **no run id is issued, so no agent starts**. `GET /v1/quarantine` lists rejects.

1. **Schema and type**: strict; unknown fields are rejected (`extra=forbid`). Datetimes must carry a UTC offset.
2. **Range**: `block_time_minutes` 15-1200; fares 0-1,000,000 INR; `notice_minutes_before_departure` 0-525,600;
   `delay_minutes` 0-10,080; `sector_distance_km` 50-20,000; narrative 10-4,000 characters.
3. **Category**: `carrier_type`, `sector`, cabin classes, disruption `type`, `requested_remedy`, `payment_mode` are enums.
4. **Format**: IATA airport codes `^[A-Z]{3}$`, airline code `^[A-Z0-9]{2}$`, PNR `^[A-Z0-9]{6}$`, e-mail, phone.
5. **Cross-field** (`cross_field_errors`):
   * destination differs from origin;
   * the flight was due to depart (not more than 1 day in the future) and is not older than `CLAIM_MAX_AGE_DAYS`;
   * `notified_at` not more than 24 h after scheduled departure, and consistent (±5 min) with `notice_minutes_before_departure`;
   * `actual_departure` not before scheduled departure, and consistent (±5 min) with `delay_minutes`;
   * an alternate flight cannot be `accepted` unless `offered`; its departure is within 7 days of the original;
   * `type=downgrading` requires `flown_class`, which must be lower than `booked_class`; a lower flown class with
     another type is inconsistent;
   * `passenger_disputes_cause=true` requires `airline_stated_cause`;
   * `volunteered` only applies to denied boarding;
   * `total_ticket_cost_inr` is at least basic fare plus fuel charge.

Free-text fields (`narrative`, `airline_stated_cause`) pass validation but are screened by the intake agent's
guardrails before any model sees them.
