from triage.extraction import extract_rules, merge
from triage.rules.engine import evaluate

DIFFICULT = ("My flight was cancelled. I was already at the airport and was told 90 minutes before departure. "
             "The airline put me on a flight four hours later, which I took. I want the compensation for cancellation.")


def test_extracts_difficult_case():
    f = extract_rules(DIFFICULT)
    assert f["category"] == "cancellation"
    assert f["notice_minutes"] == 90 and f["informed_at_airport"]
    assert f["alternate_accepted"] is True and f["alternate_delay_minutes"] == 240


def test_difficult_case_declines_compensation_but_owes_meals():
    facts, _, _ = merge({}, {}, extract_rules(DIFFICULT))
    p = evaluate(facts)
    assert p.route == "decline" and p.compensation_inr == 0
    assert any("meals" in x for x in p.facilities)
    cited = p.cited_clauses()
    assert "CAR:3.3.2" in cited and "CAR:3.4.1" in cited


def test_cancellation_amount_whichever_is_less():
    p = evaluate({"category": "cancellation", "notice_minutes": 60, "alternate_accepted": False, "refund_requested": True,
                  "block_time_minutes": 150, "basic_fare_inr": 6200, "fuel_charge_inr": 1100})
    assert p.route == "pay_as_computed" and p.compensation_inr == 7300


def test_disputed_cause_escalates():
    p = evaluate({"category": "cancellation", "notice_minutes": 0, "airline_stated_cause": "weather", "cause_disputed": True})
    assert p.route == "escalate_to_human"


def test_delay_cause_dispute_does_not_escalate():
    p = evaluate({"category": "delay", "delay_minutes": 240, "block_time_minutes": 100, "cause_disputed": True,
                  "airline_stated_cause": "air traffic"})
    assert p.route == "decline"


def test_foreign_carrier_outside_corpus():
    assert evaluate({"category": "cancellation", "carrier_type": "foreign"}).route == "escalate_to_human"
