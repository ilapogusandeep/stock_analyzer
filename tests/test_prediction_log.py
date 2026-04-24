"""Unit tests for PredictionLog — no yfinance network calls in the basic
path; resolution is covered with a stub."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from stockiq.core.prediction_log import PredictionLog


def test_log_and_read(tmp_path):
    path = tmp_path / "p.parquet"
    plog = PredictionLog(path=path, horizon_days=1)

    plog.log(
        ticker="AAPL",
        ml={
            "direction": "BULLISH",
            "confidence": 0.7,
            "scenario_probabilities": {"bullish": 0.7, "neutral": 0.2, "bearish": 0.1},
            "price_target": 110.0,
        },
        price=100.0,
    )
    plog.log(
        ticker="MSFT",
        ml={
            "direction": "BEARISH",
            "confidence": 0.6,
            "scenario_probabilities": {"bullish": 0.1, "neutral": 0.3, "bearish": 0.6},
            "price_target": 95.0,
        },
        price=100.0,
    )

    df = plog._read()
    assert len(df) == 2
    assert set(df["ticker"]) == {"AAPL", "MSFT"}
    assert df["hit"].isna().all()  # unresolved


def test_summary_empty(tmp_path):
    path = tmp_path / "p.parquet"
    plog = PredictionLog(path=path)
    s = plog.summary()
    assert s["total"] == 0
    assert s["resolved"] == 0
    assert s["recent"] == []
    assert s["calibration"] == []


def test_summary_with_resolved(tmp_path):
    """Write rows that are already resolved and verify aggregation."""
    path = tmp_path / "p.parquet"
    plog = PredictionLog(path=path)

    now = datetime.now(timezone.utc)
    rows = []
    # 5 correct high-confidence bullish, 3 wrong low-confidence bullish
    for i in range(5):
        rows.append({
            "id": f"h{i}", "timestamp": now - timedelta(days=10),
            "ticker": "AAPL", "direction": "BULLISH",
            "confidence": 0.8, "bullish_prob": 0.8, "neutral_prob": 0.15, "bearish_prob": 0.05,
            "price_at_prediction": 100, "price_target": 110,
            "resolution_horizon_days": 5,
            "resolved_at": now, "price_at_resolution": 105,
            "actual_return": 0.05, "hit": True,
        })
    for i in range(3):
        rows.append({
            "id": f"l{i}", "timestamp": now - timedelta(days=10),
            "ticker": "MSFT", "direction": "BULLISH",
            "confidence": 0.45, "bullish_prob": 0.45, "neutral_prob": 0.35, "bearish_prob": 0.20,
            "price_at_prediction": 100, "price_target": 105,
            "resolution_horizon_days": 5,
            "resolved_at": now, "price_at_resolution": 95,
            "actual_return": -0.05, "hit": False,
        })
    plog._write(pd.DataFrame(rows))

    s = plog.summary()
    assert s["total"] == 8
    assert s["resolved"] == 8
    assert s["hits"] == 5
    assert abs(s["hit_rate"] - 5 / 8) < 1e-9
    # calibration has >=5 resolved so it should produce buckets
    assert len(s["calibration"]) >= 1
