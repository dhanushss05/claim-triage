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
