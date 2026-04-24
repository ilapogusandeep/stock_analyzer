"""Append-only log of model predictions so we can audit accuracy over time.

Storage is a parquet file under ``data/predictions.parquet`` at the repo
root. On Streamlit Cloud this persists across reruns within a single
container lifetime but resets on redeploy — good enough for MVP; a
database would be the follow-up.

Every analyze() call appends one row. When the UI renders, we lazily
resolve any old-enough predictions: fetch current price via yfinance
and compare to the stored price-at-prediction to mark directional hit
or miss.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "predictions.parquet"

# Columns the parquet file carries (fixed schema — order matters for
# first-write / append consistency).
COLUMNS = [
    "id",               # uuid4 hex
    "timestamp",        # UTC datetime
    "ticker",
    "direction",        # BULLISH / NEUTRAL / BEARISH
    "confidence",       # 0..1
    "bullish_prob",
    "neutral_prob",
    "bearish_prob",
    "price_at_prediction",
    "price_target",
    "resolution_horizon_days",
    "resolved_at",      # UTC datetime or NaT
    "price_at_resolution",
    "actual_return",    # (final / initial) - 1
    "hit",              # True/False/None — None while pending
]


class PredictionLog:
    def __init__(self, path: Optional[Path | str] = None,
                 horizon_days: int = 5) -> None:
        self.path = Path(path) if path else _DEFAULT_PATH
        self.horizon_days = horizon_days
        # Allow an env override so tests can point elsewhere.
        override = os.environ.get("STOCKIQ_PRED_LOG_PATH")
        if override:
            self.path = Path(override)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def _read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=COLUMNS)
        try:
            return pd.read_parquet(self.path)
        except Exception:
            return pd.DataFrame(columns=COLUMNS)

    def _write(self, df: pd.DataFrame) -> None:
        # Ensure the schema columns are present, in order.
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[COLUMNS]
        df.to_parquet(self.path, index=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, ticker: str, ml: dict, price: float) -> None:
        """Record one prediction. Silently no-ops if ``ml`` is empty."""
        if not ml or price is None:
            return
        probs = ml.get("scenario_probabilities", {}) or {}
        row = {
            "id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc),
            "ticker": ticker,
            "direction": (ml.get("direction") or "NEUTRAL").upper(),
            "confidence": float(ml.get("confidence") or 0),
            "bullish_prob": float(probs.get("bullish") or 0),
            "neutral_prob": float(probs.get("neutral") or 0),
            "bearish_prob": float(probs.get("bearish") or 0),
            "price_at_prediction": float(price),
            "price_target": float(ml.get("price_target") or 0) or None,
            "resolution_horizon_days": int(self.horizon_days),
            "resolved_at": pd.NaT,
            "price_at_resolution": None,
            "actual_return": None,
            "hit": None,
        }
        df = self._read()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._write(df)

    def resolve_pending(self, max_to_resolve: int = 50) -> int:
        """Score any prediction older than horizon_days. Returns #resolved."""
        df = self._read()
        if df.empty:
            return 0

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.horizon_days)

        # Ensure timestamp is tz-aware UTC
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        pending_mask = df["hit"].isna() & (ts <= cutoff)
        pending_idx = df.index[pending_mask].tolist()[:max_to_resolve]
        if not pending_idx:
            return 0

        try:
            import yfinance as yf
        except ImportError:
            return 0

        # One yfinance call per unique ticker among pending rows.
        for tkr in df.loc[pending_idx, "ticker"].unique():
            try:
                hist = yf.Ticker(tkr).history(period="1mo")
                if hist.empty:
                    continue
                now_price = float(hist["Close"].iloc[-1])
            except Exception:
                continue

            mask = (df.index.isin(pending_idx)) & (df["ticker"] == tkr)
            for i in df.index[mask]:
                p0 = df.at[i, "price_at_prediction"]
                if not p0:
                    continue
                ret = (now_price / p0) - 1
                direction = (df.at[i, "direction"] or "").upper()
                if direction == "BULLISH":
                    hit = ret > 0
                elif direction == "BEARISH":
                    hit = ret < 0
                else:
                    # NEUTRAL: within ±2% = hit
                    hit = abs(ret) <= 0.02
                df.at[i, "resolved_at"] = now
                df.at[i, "price_at_resolution"] = now_price
                df.at[i, "actual_return"] = float(ret)
                df.at[i, "hit"] = bool(hit)

        self._write(df)
        return len(pending_idx)

    def summary(self) -> dict[str, Any]:
        """Overall counts + hit-rate + last N rows."""
        df = self._read()
        total = len(df)
        resolved_df = df[df["hit"].notna()]
        resolved = len(resolved_df)
        hits = int(resolved_df["hit"].sum()) if resolved else 0
        hit_rate = (hits / resolved) if resolved else None
        pending = total - resolved

        # Recent predictions for the UI (newest first)
        if total:
            recent = df.sort_values("timestamp", ascending=False).head(8).to_dict("records")
        else:
            recent = []

        # Calibration by confidence bucket (only when we have enough signal)
        calibration = []
        if resolved >= 5:
            bins = [(0.33, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.01)]
            for lo, hi in bins:
                bucket = resolved_df[
                    (resolved_df["confidence"] >= lo) & (resolved_df["confidence"] < hi)
                ]
                if len(bucket) == 0:
                    continue
                calibration.append({
                    "range": f"{int(lo*100)}–{int(hi*100)}%",
                    "n": int(len(bucket)),
                    "hits": int(bucket["hit"].sum()),
                    "hit_rate": float(bucket["hit"].mean()),
                    "avg_confidence": float(bucket["confidence"].mean()),
                })

        return {
            "total": total,
            "resolved": resolved,
            "pending": pending,
            "hits": hits,
            "hit_rate": hit_rate,
            "recent": recent,
            "calibration": calibration,
        }
