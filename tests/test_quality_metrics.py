"""Deterministic answer-quality metrics (triage.quality)."""
import pytest

from triage.quality import content_words, context_relevance, groundedness, split_sentences


def test_content_words_drops_stop_words_short_words_and_commas():
    assert content_words("The airline shall pay INR 10,000 to the passenger") == {"airline", "pay", "inr", "10000",
                                                                                "passenger"}


def test_split_sentences_strips_clause_tags_and_short_fragments():
    text = "Compensation is owed under the CAR [CAR:3.3.2]. Too short. Meals are owed for the long delay!"
    assert split_sentences(text) == ["Compensation is owed under the CAR .", "Meals are owed for the long delay!"]
    assert split_sentences("") == [] and split_sentences(None) == []


@pytest.mark.parametrize("retrieved, relevant, want", [
    ([], {"a"}, 0.0),
    (["a", "b"], {"a"}, 0.5),
    (["a", "b"], {"a", "b", "c"}, 1.0),
    (["x"], set(), 0.0),
])
def test_context_relevance(retrieved, relevant, want):
    assert context_relevance(retrieved, relevant) == want


EVIDENCE = {
    "3.3.2": {"text": "In case of cancellation the airline shall pay compensation of INR 10,000 or the booked "
                      "one way basic fare plus airline fuel charge, whichever is less."},
}


def test_groundedness_supported_and_unsupported_sentences():
    result = {"justification": "The airline shall pay compensation for the cancellation [CAR:3.3.2]. "
                               "Passengers also receive free spa vouchers and lounge massages.",
              "citations": [{"clause_id": "3.3.2"}]}
    assert groundedness(result, EVIDENCE, {}) == 0.5


def test_groundedness_uses_the_fact_sheet():
    result = {"justification": "The flight was delayed by 180 minutes on that route.", "citations": []}
    assert groundedness(result, {}, {"delay_minutes": 180, "route": "BLR-DEL"}) == 1.0


def test_groundedness_ignores_uncited_evidence_and_empty_answers():
    result = {"justification": "The airline shall pay compensation for the cancellation.", "citations": []}
    assert groundedness(result, EVIDENCE, {}) == 0.0
    assert groundedness({"justification": ""}, EVIDENCE, {}) == 0.0
