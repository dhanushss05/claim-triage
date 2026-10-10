import copy

import pytest
from pydantic import ValidationError

from triage.contracts import ClaimRequest, cross_field_errors

GOOD = {"client_reference": "T-1", "passenger": {"full_name": "Asha Rao", "email": "a@example.com"},
        "flight": {"airline_code": "DA", "flight_number": "611", "origin": "BLR", "destination": "DEL",
                   "scheduled_departure": "2026-09-14T09:30:00+05:30"},
        "booking": {"pnr": "QX7TZ2"}, "narrative": "My flight was cancelled at the airport."}


def test_valid_claim_passes():
    c = ClaimRequest.model_validate(GOOD)
    assert cross_field_errors(c) == []


@pytest.mark.parametrize("path,value", [
    (("flight", "airline_code"), "TOOLONG"), (("booking", "pnr"), "1"), (("passenger", "email"), "nope"),
    (("flight", "block_time_minutes"), 5), (("narrative",), "short"), (("flight", "scheduled_departure"), "2026-09-14T09:30:00"),
])
def test_schema_and_range_rejects(path, value):
    bad = copy.deepcopy(GOOD)
    tgt = bad
    for p in path[:-1]:
        tgt = tgt[p]
    tgt[path[-1]] = value
    with pytest.raises(ValidationError):
        ClaimRequest.model_validate(bad)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        ClaimRequest.model_validate({**GOOD, "approve": True})


def test_cross_field_rules():
    bad = copy.deepcopy(GOOD)
    bad["flight"]["destination"] = "BLR"
    bad["disruption"] = {"type": "downgrading", "flown_class": "economy", "alternate_flight": {"offered": False, "accepted": True}}
    errs = {e.field for e in cross_field_errors(ClaimRequest.model_validate(bad))}
    assert {"flight.destination", "disruption.flown_class", "disruption.alternate_flight.accepted"} <= errs
