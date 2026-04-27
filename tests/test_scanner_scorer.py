"""Tests for stockiq.scanner.scorer — composite signal scoring."""

from __future__ import annotations

import pytest

from stockiq.scanner.scorer import rank_signals, score_signal


def _snap(**overrides) -> dict:
    base = {
        "ticker": "TEST",
        "price": 100.0,
        "change_1d": 0.0,
        "unusual_count": 0,
        "aggressor_net": 0,
        "news_velocity": 0.0,
    }
    base.update(overrides)
    return base


def test_zero_signal_yields_zero_score():
    out = score_signal(_snap())
    assert out["score"] == 0
    assert out["bias"] == "NEUTRAL"


def test_full_signal_yields_max_100():
    """5+ unusual strikes, |B-S| of 5+, 3x news velocity, 5%+ move = 100."""
    out = score_signal(_snap(
        unusual_count=5,
        aggressor_net=5,
        news_velocity=3.0,
        change_1d=0.05,
    ))
    assert out["score"] == 100.0
    assert out["bias"] == "BULLISH"


def test_components_clamped_so_one_dim_cant_dominate():
    """A signal that's wildly extreme on one axis maxes that axis at 25
    but doesn't bleed into other axes."""
    out = score_signal(_snap(unusual_count=100))  # absurdly high
    assert out["score_options"] == 25.0
    # Other components stay at zero
    assert out["score_aggressor"] == 0
    assert out["score_news"] == 0
    assert out["score_move"] == 0
    assert out["score"] == 25.0


def test_negative_aggressor_with_negative_change_is_bearish():
    out = score_signal(_snap(aggressor_net=-3, change_1d=-0.02))
    assert out["bias"] == "BEARISH"


def test_strong_buyer_with_flat_price_is_bullish():
    """Aggressor net +2 and price ~flat shouldn't fall into NEUTRAL —
    strong-direction aggressor gets the benefit of the doubt."""
    out = score_signal(_snap(aggressor_net=3, change_1d=0.001))
    assert out["bias"] == "BULLISH"


def test_conflicting_signal_is_neutral():
    out = score_signal(_snap(aggressor_net=4, change_1d=-0.04))
    assert out["bias"] == "NEUTRAL"


def test_rank_signals_sorts_descending():
    snaps = [
        _snap(ticker="LOW", unusual_count=1),
        _snap(ticker="HIGH", unusual_count=5, aggressor_net=5),
        _snap(ticker="MID", news_velocity=3.0),
    ]
    ranked = rank_signals(snaps)
    assert [r["ticker"] for r in ranked] == ["HIGH", "MID", "LOW"]


def test_rank_signals_handles_empty():
    assert rank_signals([]) == []
