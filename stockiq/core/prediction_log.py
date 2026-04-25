"""Append-only log of model predictions so we can audit accuracy over time.

Two storage backends:

1. **Supabase** (preferred). Activated when ``SUPABASE_URL`` and
   ``SUPABASE_KEY`` are present in ``st.secrets``. Survives Streamlit
   Cloud redeploys, so the Track Record panel shows real long-running
   calibration. Run ``migrations/001_predictions.sql`` once in your
   Supabase SQL editor to create the schema.

2. **Parquet fallback** at ``data/predictions.parquet`` (repo root). On
   Streamlit Cloud this persists across reruns within a single container
   lifetime but resets on redeploy. Used when Supabase secrets are not
   configured, so the dashboard keeps working out of the box.

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


def _read_supabase_secrets() -> tuple[Optional[str], Optional[str]]:
    """Pull SUPABASE_URL / SUPABASE_KEY from st.secrets if available.

    Falls back to environment variables so tests and CLI runs can set
    them outside Streamlit. Returns (None, None) if not configured —
    the caller silently uses the parquet backend in that case.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not (url and key):
        try:
            import streamlit as st
            url = url or st.secrets.get("SUPABASE_URL")
            key = key or st.secrets.get("SUPABASE_KEY")
        except Exception:
            pass
    return (url or None), (key or None)


class PredictionLog:
    def __init__(self, path: Optional[Path | str] = None,
                 horizon_days: int = 5) -> None:
        self.horizon_days = horizon_days
        self._sb_url, self._sb_key = _read_supabase_secrets()

        # Parquet fallback path — only used when Supabase isn't configured.
        self.path = Path(path) if path else _DEFAULT_PATH
        override = os.environ.get("STOCKIQ_PRED_LOG_PATH")
        if override:
            self.path = Path(override)
        if not self._using_supabase():
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _using_supabase(self) -> bool:
        return bool(self._sb_url and self._sb_key)

    # ------------------------------------------------------------------
    # Supabase HTTP helpers
    # ------------------------------------------------------------------

    def _sb_headers(self, prefer: str = "return=representation") -> dict:
        return {
            "apikey": self._sb_key or "",
            "Authorization": f"Bearer {self._sb_key or ''}",
            "Content-Type": "application/json",
            "Prefer": prefer,
        }

    def _sb_url_for(self, query: str = "") -> str:
        return f"{self._sb_url}/rest/v1/predictions{query}"

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

    def _build_row(self, ticker: str, ml: dict, price: float, horizon_days: int) -> dict:
        probs = ml.get("scenario_probabilities", {}) or {}
        return {
            "ticker": ticker,
            "direction": (ml.get("direction") or "NEUTRAL").upper(),
            "confidence": float(ml.get("confidence") or 0),
            "bullish_prob": float(probs.get("bullish") or 0),
            "neutral_prob": float(probs.get("neutral") or 0),
            "bearish_prob": float(probs.get("bearish") or 0),
            "price_at_prediction": float(price),
            "price_target": float(ml.get("price_target") or 0) or None,
            "resolution_horizon_days": int(horizon_days),
        }

    def log(self, ticker: str, ml: dict, price: float,
            horizon_days: Optional[int] = None) -> None:
        """Record one prediction. Silently no-ops if ``ml`` is empty.

        ``horizon_days`` overrides the instance default so a single
        PredictionLog can track 1w / 1m / 3m predictions side by side.
        """
        if not ml or price is None:
            return
        h = int(horizon_days) if horizon_days is not None else self.horizon_days
        if self._using_supabase():
            self._log_supabase(ticker, ml, price, h)
        else:
            self._log_parquet(ticker, ml, price, h)

    def resolve_pending(self, max_to_resolve: int = 50) -> int:
        """Score any prediction older than horizon_days. Returns #resolved."""
        if self._using_supabase():
            return self._resolve_supabase(max_to_resolve)
        return self._resolve_parquet(max_to_resolve)

    def summary(self, horizon_days: Optional[int] = None) -> dict[str, Any]:
        """Overall counts + hit-rate + last N rows + calibration buckets.

        ``horizon_days`` filters to predictions made at that horizon
        (5 for 1w, 21 for 1m, 63 for 3m). Returns the unfiltered totals
        when ``None``.
        """
        if self._using_supabase():
            df = self._read_all_supabase()
        else:
            df = self._read()
        if horizon_days is not None and "resolution_horizon_days" in df.columns:
            try:
                df = df[df["resolution_horizon_days"].astype(int) == int(horizon_days)]
            except Exception:
                df = df.iloc[0:0]
        return self._summarize_df(df)

    # ------------------------------------------------------------------
    # Supabase-backed implementations
    # ------------------------------------------------------------------

    def _log_supabase(self, ticker: str, ml: dict, price: float, horizon_days: int) -> None:
        try:
            import requests
        except ImportError:
            return
        payload = self._build_row(ticker, ml, price, horizon_days)
        # Supabase fills timestamp default + id; client doesn't send them.
        try:
            requests.post(
                self._sb_url_for(),
                headers=self._sb_headers(prefer="return=minimal"),
                json=payload,
                timeout=8,
            )
        except Exception:
            pass

    def _read_all_supabase(self) -> pd.DataFrame:
        try:
            import requests
        except ImportError:
            return pd.DataFrame(columns=COLUMNS)
        try:
            r = requests.get(
                self._sb_url_for("?select=*&order=timestamp.desc&limit=2000"),
                headers=self._sb_headers(),
                timeout=8,
            )
            if r.status_code != 200:
                return pd.DataFrame(columns=COLUMNS)
            rows = r.json() or []
            if not rows:
                return pd.DataFrame(columns=COLUMNS)
            df = pd.DataFrame(rows)
            # Backfill any missing schema columns the migration didn't add
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
            return df
        except Exception:
            return pd.DataFrame(columns=COLUMNS)

    def _resolve_supabase(self, max_to_resolve: int) -> int:
        try:
            import requests
        except ImportError:
            return 0
        try:
            # Pull all unresolved rows; filter by per-row horizon in
            # Python since Postgres expressions like
            # `timestamp <= now() - resolution_horizon_days * interval '1 day'`
            # aren't expressible via PostgREST query params.
            r = requests.get(
                self._sb_url_for(
                    "?select=*&hit=is.null&order=timestamp.asc&limit=500"
                ),
                headers=self._sb_headers(),
                timeout=8,
            )
            if r.status_code != 200:
                return 0
            all_pending = r.json() or []
            now = datetime.now(timezone.utc)
            pending = []
            for row in all_pending:
                ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                hd = int(row.get("resolution_horizon_days") or self.horizon_days)
                if (now - ts.to_pydatetime()) >= timedelta(days=hd):
                    pending.append(row)
            pending = pending[:max_to_resolve]
            if not pending:
                return 0
        except Exception:
            return 0

        try:
            import yfinance as yf
        except ImportError:
            return 0

        # Group by ticker so we hit yfinance once per ticker.
        by_ticker: dict[str, list[dict]] = {}
        for row in pending:
            by_ticker.setdefault(row["ticker"], []).append(row)

        resolved = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        for tkr, rows in by_ticker.items():
            try:
                hist = yf.Ticker(tkr).history(period="1mo")
                if hist.empty:
                    continue
                now_price = float(hist["Close"].iloc[-1])
            except Exception:
                continue

            for row in rows:
                p0 = row.get("price_at_prediction")
                if not p0:
                    continue
                ret = (now_price / float(p0)) - 1
                direction = (row.get("direction") or "").upper()
                if direction == "BULLISH":
                    hit = ret > 0
                elif direction == "BEARISH":
                    hit = ret < 0
                else:
                    hit = abs(ret) <= 0.02
                update_payload = {
                    "resolved_at": now_iso,
                    "price_at_resolution": now_price,
                    "actual_return": float(ret),
                    "hit": bool(hit),
                }
                try:
                    requests.patch(
                        self._sb_url_for(f"?id=eq.{row['id']}"),
                        headers=self._sb_headers(prefer="return=minimal"),
                        json=update_payload,
                        timeout=8,
                    )
                    resolved += 1
                except Exception:
                    continue
        return resolved

    # ------------------------------------------------------------------
    # Parquet-backed implementations (fallback when Supabase isn't set up)
    # ------------------------------------------------------------------

    def _log_parquet(self, ticker: str, ml: dict, price: float, horizon_days: int) -> None:
        row = self._build_row(ticker, ml, price, horizon_days)
        row.update({
            "id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc),
            "resolved_at": pd.NaT,
            "price_at_resolution": None,
            "actual_return": None,
            "hit": None,
        })
        df = self._read()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._write(df)

    def _resolve_parquet(self, max_to_resolve: int) -> int:
        df = self._read()
        if df.empty:
            return 0

        now = datetime.now(timezone.utc)
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        # Per-row horizon — supports rows logged at different horizons.
        hd_days = pd.to_numeric(
            df.get("resolution_horizon_days", self.horizon_days),
            errors="coerce",
        ).fillna(self.horizon_days)
        age_days = (now - ts).dt.total_seconds() / 86400.0
        pending_mask = df["hit"].isna() & (age_days >= hd_days)
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

    # ------------------------------------------------------------------
    # Shared aggregator (works on either backend's DataFrame)
    # ------------------------------------------------------------------

    def _summarize_df(self, df: pd.DataFrame) -> dict[str, Any]:
        """Overall counts + hit-rate + last N rows + calibration."""
        if df.empty:
            return {
                "total": 0, "resolved": 0, "pending": 0, "hits": 0,
                "hit_rate": None, "recent": [], "calibration": [],
            }
        # Coerce hit -> nullable bool: Supabase returns true/false/None,
        # parquet returns Python bool/None. Both round-trip via pandas
        # but the Supabase JSON parse leaves "hit" as bool/None.
        total = len(df)
        resolved_df = df[df["hit"].notna()] if "hit" in df.columns else df.iloc[0:0]
        resolved = len(resolved_df)
        hits = int(resolved_df["hit"].sum()) if resolved else 0
        hit_rate = (hits / resolved) if resolved else None
        pending = total - resolved

        if total:
            recent = df.sort_values("timestamp", ascending=False).head(8).to_dict("records")
        else:
            recent = []

        # Calibration by confidence bucket (only with enough samples)
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


# ---------------------------------------------------------------------------
# Confidence calibration — backend feedback loop
# ---------------------------------------------------------------------------

# How many resolved rows we need before applying calibration. Below this
# the historical hit rate is too noisy to trust as a shrinkage signal.
_CALIBRATION_MIN_N = 5


def calibrate_probs(
    bullish: float, neutral: float, bearish: float, summary: dict
) -> tuple[float, float, float, dict]:
    """Shrink raw model probabilities toward the no-edge baseline (1/3
    each) based on the model's historical hit rate at this horizon.

    The amount of shrinkage scales with how much edge we've actually
    demonstrated. If the model has been right 50% of the time on
    binary-direction predictions, the bars get pulled all the way to
    uniform (1/3, 1/3, 1/3). At 75% the bars stay roughly where the
    model said. Below 50% (anti-signal), we still shrink rather than
    invert -- a small bad sample is more likely noise than a true
    inverted signal.

    Returns ``(bull', neut', bear', meta)`` where meta describes whether
    calibration was applied so the UI can label it.
    """
    resolved = int(summary.get("resolved", 0) or 0)
    hit_rate = summary.get("hit_rate")

    if resolved < _CALIBRATION_MIN_N or hit_rate is None:
        return bullish, neutral, bearish, {
            "applied": False,
            "reason": "untracked",
            "n": resolved,
            "hit_rate": hit_rate,
        }

    # Trust scales linearly with edge: 50% hit rate -> 0 trust, 100% -> 1.
    edge = max(0.0, abs(float(hit_rate) - 0.5))
    trust = min(1.0, edge * 2.0)

    target = 1.0 / 3.0
    cal_bull = bullish + (target - bullish) * (1 - trust)
    cal_neut = neutral + (target - neutral) * (1 - trust)
    cal_bear = bearish + (target - bearish) * (1 - trust)

    # Renormalize to defend against any tiny floating-point drift.
    s = cal_bull + cal_neut + cal_bear
    if s > 0:
        cal_bull /= s; cal_neut /= s; cal_bear /= s

    return cal_bull, cal_neut, cal_bear, {
        "applied": True,
        "trust": trust,
        "n": resolved,
        "hit_rate": float(hit_rate),
    }


def calibrate_confidence(raw_conf: float, summary: dict) -> tuple[float, dict]:
    """Shrink a single direction's confidence toward 50% by the same
    edge-scaled trust factor used in ``calibrate_probs``."""
    resolved = int(summary.get("resolved", 0) or 0)
    hit_rate = summary.get("hit_rate")

    if resolved < _CALIBRATION_MIN_N or hit_rate is None:
        return raw_conf, {"applied": False, "n": resolved, "hit_rate": hit_rate}

    edge = max(0.0, abs(float(hit_rate) - 0.5))
    trust = min(1.0, edge * 2.0)
    cal = 0.5 + (raw_conf - 0.5) * trust
    return cal, {
        "applied": True,
        "trust": trust,
        "n": resolved,
        "hit_rate": float(hit_rate),
    }
