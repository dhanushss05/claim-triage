"""Published data contracts for every external interface.

* ClaimRequest   - POST /v1/claims (ingestion boundary). Strict: unknown fields are rejected.
* RunAccepted    - 202 response carrying the run identifier.
* RunResult      - GET /v1/runs/{run_id} result payload (also POSTed to callback_url).
* Feedback       - POST /v1/runs/{run_id}/feedback.
* IntakeHandoff  - internal hand-off from the intake agent to the entitlement agent (over the queue).

The JSON Schemas in contracts/ are generated from these models (`make contracts`). Cross-field rules that JSON
Schema cannot express are implemented in `cross_field_errors` and documented in contracts/CONTRACTS.md.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

SCHEMA_VERSION = "1.0"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CabinClass(str, Enum):
    economy = "economy"
    premium_economy = "premium_economy"
    business = "business"
    first = "first"


CLASS_RANK = {"economy": 0, "premium_economy": 1, "business": 2, "first": 3}


class DisruptionType(str, Enum):
    denied_boarding = "denied_boarding"
    cancellation = "cancellation"
    delay = "delay"
    downgrading = "downgrading"


class Route(str, Enum):
    pay_as_computed = "pay_as_computed"
    request_further_evidence = "request_further_evidence"
    decline = "decline"
    escalate_to_human = "escalate_to_human"


class Passenger(Strict):
    full_name: str = Field(min_length=2, max_length=100)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$", max_length=254)
    phone: Optional[str] = Field(default=None, pattern=r"^\+?[0-9 ]{8,16}$")
    contact_details_provided_at_booking: Optional[bool] = Field(
        default=None, description="Did the passenger give a mobile number or e-mail at booking?")


class Flight(Strict):
    airline_code: str = Field(pattern=r"^[A-Z0-9]{2}$")
    flight_number: str = Field(pattern=r"^[0-9]{1,4}[A-Z]?$")
    carrier_type: Literal["indian_scheduled", "foreign"] = "indian_scheduled"
    origin: str = Field(pattern=r"^[A-Z]{3}$", description="IATA airport code")
    destination: str = Field(pattern=r"^[A-Z]{3}$")
    sector: Optional[Literal["domestic", "international"]] = None
    scheduled_departure: datetime = Field(description="Booked scheduled departure, ISO-8601 with UTC offset")
    block_time_minutes: Optional[int] = Field(default=None, ge=15, le=1200)
    sector_distance_km: Optional[int] = Field(default=None, ge=50, le=20000)

    @field_validator("scheduled_departure")
    @classmethod
    def tz_required(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("must include a UTC offset, e.g. 2026-09-14T09:30:00+05:30")
        return v


class Booking(Strict):
    pnr: str = Field(pattern=r"^[A-Z0-9]{6}$")
    booked_class: CabinClass = CabinClass.economy
    basic_fare_inr: Optional[float] = Field(default=None, ge=0, le=1_000_000)
    fuel_charge_inr: Optional[float] = Field(default=None, ge=0, le=200_000)
    total_ticket_cost_inr: Optional[float] = Field(default=None, ge=0, le=2_000_000,
                                                   description="Sector ticket cost including taxes")
    payment_mode: Optional[Literal["card", "cash", "upi", "agent", "other"]] = None


class AlternateFlight(Strict):
    offered: bool
    accepted: Optional[bool] = None
    scheduled_departure: Optional[datetime] = None


class Disruption(Strict):
    type: Optional[DisruptionType] = Field(default=None, description="Optional; the intake agent classifies it")
    notified_at: Optional[datetime] = None
    notice_minutes_before_departure: Optional[int] = Field(default=None, ge=0, le=525_600)
    informed_at_airport: Optional[bool] = None
    actual_departure: Optional[datetime] = None
    delay_minutes: Optional[int] = Field(default=None, ge=0, le=10_080)
    alternate_flight: Optional[AlternateFlight] = None
    refund_requested: Optional[bool] = None
    airline_stated_cause: Optional[str] = Field(default=None, max_length=200)
    passenger_disputes_cause: Optional[bool] = None
    presented_for_checkin_on_time: Optional[bool] = None
    volunteered: Optional[bool] = None
    flown_class: Optional[CabinClass] = None
    downgrade_requested_by_passenger: Optional[bool] = None
    baggage_claim: Optional[bool] = Field(default=None, description="True if the claim is about baggage")


class ClaimRequest(Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    client_reference: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    passenger: Passenger
    flight: Flight
    booking: Booking
    disruption: Disruption = Field(default_factory=Disruption)
    narrative: str = Field(min_length=10, max_length=4000, description="Passenger's own account (free text)")
    requested_remedy: Optional[Literal["compensation", "refund", "facilities_reimbursement", "other"]] = None
    callback_url: Optional[str] = Field(default=None, pattern=r"^https?://[^\s]{3,500}$")


class FieldError(BaseModel):
    field: str
    code: str
    message: str


def _loc(loc: tuple[Any, ...]) -> str:
    return ".".join(str(p) for p in loc if p not in ("body",))


def schema_errors(exc: ValidationError) -> list[FieldError]:
    return [FieldError(field=_loc(e["loc"]) or "$", code=e["type"], message=e["msg"]) for e in exc.errors()]


def cross_field_errors(c: ClaimRequest, now: Optional[datetime] = None) -> list[FieldError]:
    """Rules spanning several fields. Every rule is listed in contracts/CONTRACTS.md."""
    now = now or datetime.now(timezone.utc)
    errs: list[FieldError] = []
    d, f, b = c.disruption, c.flight, c.booking
    sched = f.scheduled_departure

    def add(field: str, code: str, msg: str) -> None:
        errs.append(FieldError(field=field, code=code, message=msg))

    if f.origin == f.destination:
        add("flight.destination", "same_as_origin", "destination must differ from origin")
    max_age = int(os.environ.get("CLAIM_MAX_AGE_DAYS", "730"))
    if sched > now + timedelta(days=1):
        add("flight.scheduled_departure", "in_future", "a claim can only be made for a flight that was due to depart")
    if sched < now - timedelta(days=max_age):
        add("flight.scheduled_departure", "too_old", f"flight is older than {max_age} days")
    for name in ("notified_at", "actual_departure"):
        v = getattr(d, name)
        if v is not None and v.tzinfo is None:
            add(f"disruption.{name}", "timezone_missing", "must include a UTC offset")
    if d.notified_at is not None and d.notified_at.tzinfo is not None:
        if d.notified_at > sched + timedelta(hours=24):
            add("disruption.notified_at", "after_departure", "notification cannot be more than 24h after scheduled departure")
        if d.notice_minutes_before_departure is not None:
            implied = (sched - d.notified_at).total_seconds() / 60
            if abs(implied - d.notice_minutes_before_departure) > 5:
                add("disruption.notice_minutes_before_departure", "inconsistent",
                    f"conflicts with notified_at (implies {int(implied)} minutes)")
    if d.actual_departure is not None and d.actual_departure.tzinfo is not None:
        if d.actual_departure < sched - timedelta(minutes=1):
            add("disruption.actual_departure", "before_scheduled", "actual departure precedes the scheduled departure")
        if d.delay_minutes is not None:
            implied = (d.actual_departure - sched).total_seconds() / 60
            if abs(implied - d.delay_minutes) > 5:
                add("disruption.delay_minutes", "inconsistent", f"conflicts with actual_departure (implies {int(implied)} minutes)")
    alt = d.alternate_flight
    if alt is not None:
        if alt.accepted and not alt.offered:
            add("disruption.alternate_flight.accepted", "not_offered", "an alternate flight cannot be accepted if none was offered")
        if alt.scheduled_departure is not None:
            if alt.scheduled_departure.tzinfo is None:
                add("disruption.alternate_flight.scheduled_departure", "timezone_missing", "must include a UTC offset")
            elif abs((alt.scheduled_departure - sched).total_seconds()) > 7 * 86400:
                add("disruption.alternate_flight.scheduled_departure", "out_of_range", "must be within 7 days of the original departure")
    if d.type == DisruptionType.downgrading:
        if d.flown_class is None:
            add("disruption.flown_class", "required", "required when type is downgrading")
        elif CLASS_RANK[d.flown_class.value] >= CLASS_RANK[b.booked_class.value]:
            add("disruption.flown_class", "not_lower", "flown class must be lower than the booked class for a downgrade")
    if d.flown_class is not None and d.type not in (None, DisruptionType.downgrading):
        if CLASS_RANK[d.flown_class.value] < CLASS_RANK[b.booked_class.value]:
            add("disruption.type", "inconsistent", "flown class is lower than booked class but type is not downgrading")
    if d.passenger_disputes_cause and not d.airline_stated_cause:
        add("disruption.airline_stated_cause", "required", "required when passenger_disputes_cause is true")
    if d.volunteered and d.type not in (None, DisruptionType.denied_boarding):
        add("disruption.volunteered", "not_applicable", "only meaningful for denied boarding")
    if None not in (b.total_ticket_cost_inr, b.basic_fare_inr):
        if b.total_ticket_cost_inr + 1 < (b.basic_fare_inr or 0) + (b.fuel_charge_inr or 0):
            add("booking.total_ticket_cost_inr", "less_than_components", "must be at least basic fare plus fuel charge")
    return errs


class RunAccepted(BaseModel):
    run_id: str
    status: str
    status_url: str
    trace_url: str
    cached: bool = False


class Citation(BaseModel):
    clause_id: str
    label: str
    quote: str
    doc_id: str = ""


class Entitlement(BaseModel):
    compensation_inr: float = 0.0
    computation: str = ""
    refund_due: bool = False
    facilities: list[str] = []
    downgrade_refund_inr: float = 0.0


class Escalation(BaseModel):
    required: bool = False
    reasons: list[str] = []
    path: list[str] = []


class Verification(BaseModel):
    passed: bool = False
    attempts: int = 0
    checks: list[dict[str, Any]] = []


class RunResult(BaseModel):
    run_id: str
    status: Literal["completed", "escalated", "failed"]
    route: Route
    disruption_category: Optional[str] = None
    entitlement: Entitlement = Entitlement()
    citations: list[Citation] = []
    justification: str = ""
    missing_evidence: list[str] = []
    escalation: Escalation = Escalation()
    verification: Verification = Verification()
    guardrail_flags: list[str] = []
    degraded: bool = False
    confidence: Literal["high", "medium", "low"] = "low"
    basis: str = ""
    prompt_versions: dict[str, str] = {}
    cost: dict[str, Any] = {}
    timings_ms: dict[str, float] = {}
    disclaimer: str = ("This is an entitlement recommendation, not a payment authorisation. "
                       "A customer relations officer decides.")


class Feedback(Strict):
    verdict: Literal["satisfactory", "unsatisfactory"]
    reason: str = Field(min_length=3, max_length=1000)
    reviewer: Optional[str] = Field(default=None, max_length=100)


class IntakeHandoff(BaseModel):
    """Contract between intake agent (producer) and entitlement agent (consumer)."""
    run_id: str
    category: Optional[str]
    category_confidence: float
    facts: dict[str, Any]
    fact_sources: dict[str, str]
    conflicts: list[str]
    guardrail_flags: list[str]
    narrative_for_model: Optional[str]
    prompt_versions: dict[str, str]
    degraded: bool = False
