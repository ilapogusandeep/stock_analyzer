"""Tests for the keyword-based _analyze_text_sentiment classifier in
stockiq.data.collector.

The classifier is a method on EnhancedDataCollector but doesn't use any
instance state, so we call it bound to an empty shell object rather
than constructing a full collector (which fetches yfinance data).
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from stockiq.data.collector import EnhancedDataCollector


@pytest.fixture(scope="module")
def score():
    """Return the unbound method so each test calls it cheaply.

    EnhancedDataCollector.__init__ fetches yfinance data, which is slow
    and flaky in tests. Bypass it by calling the method on a dummy.
    """
    fn = EnhancedDataCollector._analyze_text_sentiment

    def run(text: str) -> float:
        return fn(SimpleNamespace(), text)

    return run


# ---------------------------------------------------------------------------
# Polarity detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "AMD shares soar 12% on strong earnings",
    "Stock jumps after analyst upgrade",
    "Company beats expectations, profit surges",
    "Breakout confirmed as momentum builds",
])
def test_clearly_positive_headlines(score, text):
    assert score(text) > 0


@pytest.mark.parametrize("text", [
    "Stock plunges after weak guidance",
    "Analyst downgrade sends shares tumbling",
    "Company misses earnings, warning issued",
    "Lawsuit fears drag shares lower",
])
def test_clearly_negative_headlines(score, text):
    assert score(text) < 0


def test_mixed_sentiment_lands_near_zero(score):
    """Equal pos + neg words -> score of 0 from the (p-n)/(p+n) formula."""
    # "soars" (+1 pos) and "plunges" (+1 neg) cancel.
    s = score("Company soars after product launch then plunges on guidance")
    assert abs(s) < 0.5


def test_no_matching_words_returns_zero(score):
    s = score("Random text with none of the finance keywords at all")
    assert s == 0.0


def test_case_insensitive(score):
    """SOARED, Soared, soared should all score identically."""
    assert score("SOARED") == score("Soared") == score("soared") > 0


def test_word_boundary_not_substring(score):
    """'win' is in positive_words; 'winter' should NOT match."""
    # "winter" contains "win" but isn't a word boundary match.
    s = score("Winter approaches the northern hemisphere")
    assert s == 0.0


def test_score_is_bounded_minus_one_to_one(score):
    """The (p - n) / (p + n) formula clamps by construction."""
    for text in [
        "soar surge rally jump skyrocket spike",
        "plunge crash tumble dive slump fall",
        "soar plunge",
        "Apple and Microsoft are companies",
    ]:
        s = score(text)
        assert -1.0 <= s <= 1.0
