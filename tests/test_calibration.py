"""Unit tests for calibrate_probs + calibrate_confidence.

These are pure-math functions with no I/O, so tests run fast and deterministic.
"""

from __future__ import annotations

from stockiq.core.prediction_log import calibrate_confidence, calibrate_probs


def test_probs_not_enough_data_no_shrinkage():
    """Below 5 resolved predictions -> leave raw probs untouched."""
    b, n, d, meta = calibrate_probs(
        0.7, 0.2, 0.1, {"resolved": 2, "hit_rate": 0.5}
    )
    assert (b, n, d) == (0.7, 0.2, 0.1)
    assert meta["applied"] is False
    assert meta["reason"] == "untracked"


def test_probs_random_hit_rate_full_shrinkage():
    """50% hit rate = zero edge, pull to uniform (1/3, 1/3, 1/3)."""
    b, n, d, meta = calibrate_probs(
        0.7, 0.2, 0.1, {"resolved": 50, "hit_rate": 0.5}
    )
    assert abs(b - 1 / 3) < 1e-9
    assert abs(n - 1 / 3) < 1e-9
    assert abs(d - 1 / 3) < 1e-9
    assert meta["applied"] is True
    assert meta["trust"] == 0.0


def test_probs_perfect_hit_rate_no_shrinkage():
    """100% hit rate = full trust, return raw probs."""
    b, n, d, meta = calibrate_probs(
        0.7, 0.2, 0.1, {"resolved": 50, "hit_rate": 1.0}
    )
    assert abs(b - 0.7) < 1e-9
    assert abs(n - 0.2) < 1e-9
    assert abs(d - 0.1) < 1e-9
    assert meta["trust"] == 1.0


def test_probs_moderate_hit_rate_halfway_shrinkage():
    """75% hit rate -> trust 0.5, halfway between raw and uniform."""
    b, n, d, meta = calibrate_probs(
        0.7, 0.2, 0.1, {"resolved": 50, "hit_rate": 0.75}
    )
    assert meta["trust"] == 0.5
    # Halfway between raw and 1/3
    assert abs(b - (0.7 + (1 / 3 - 0.7) * 0.5)) < 1e-9


def test_probs_anti_signal_shrinks_not_inverts():
    """Below-50% hit rate still shrinks rather than inverts — a small
    bad sample is more likely noise than a true anti-signal."""
    b, n, d, meta = calibrate_probs(
        0.7, 0.2, 0.1, {"resolved": 50, "hit_rate": 0.35}
    )
    # Bullish stays the largest; just squeezed toward uniform.
    assert b > n > d
    assert b < 0.7  # shrunk
    assert abs(meta["trust"] - 0.3) < 1e-9  # 2 * |0.35 - 0.5|, float-safe


def test_probs_sum_to_one_after_calibration():
    """Regardless of input, calibrated probs must renormalize to 1.0."""
    for hr in (0.5, 0.55, 0.6, 0.65, 0.7, 0.8):
        b, n, d, _ = calibrate_probs(
            0.6, 0.25, 0.15, {"resolved": 20, "hit_rate": hr}
        )
        assert abs(b + n + d - 1.0) < 1e-9, f"hit_rate={hr}: {b + n + d}"


def test_confidence_not_enough_data():
    cal, meta = calibrate_confidence(0.82, {"resolved": 0, "hit_rate": None})
    assert cal == 0.82
    assert meta["applied"] is False


def test_confidence_random_hit_rate_pulls_to_50():
    """50% hit rate should pull raw confidence all the way to 50%."""
    cal, meta = calibrate_confidence(0.82, {"resolved": 50, "hit_rate": 0.5})
    assert abs(cal - 0.5) < 1e-9
    assert meta["applied"] is True


def test_confidence_halfway_shrinkage_at_75pct():
    """75% hit rate -> trust 0.5 -> halfway between raw and 50%."""
    cal, meta = calibrate_confidence(0.80, {"resolved": 50, "hit_rate": 0.75})
    assert abs(cal - 0.65) < 1e-9  # 0.5 + (0.8 - 0.5) * 0.5
    assert meta["trust"] == 0.5
