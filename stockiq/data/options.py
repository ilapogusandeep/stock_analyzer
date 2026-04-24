"""Fetch and summarize options flow from yfinance.

Exposes one function ``get_options_flow(ticker, spot=None)`` that returns
a dict of:
    put_call_vol_ratio   — today's put vol / call vol across all strikes
    put_call_oi_ratio    — put OI / call OI
    call_volume          — total call volume across nearest expiry
    put_volume           — total put volume across nearest expiry
    atm_iv               — average IV of ATM calls+puts (within ±5% of spot)
    expiry               — ISO date used
    days_to_expiry       — calendar days
    tilt                 — coarse "bullish" / "bearish" / "neutral" label
                          (P/C vol ratio: >1.2 bearish, <0.8 bullish)

Returns ``{}`` on any failure — never raises.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd


def _label_from_pc(pc_vol: Optional[float]) -> str:
    if pc_vol is None:
        return "—"
    if pc_vol >= 1.2:
        return "BEARISH"
    if pc_vol <= 0.80:
        return "BULLISH"
    return "NEUTRAL"


def get_options_flow(ticker: str, spot: Optional[float] = None) -> dict[str, Any]:
    """Return an options-flow snapshot. Safe on any failure (returns {})."""
    try:
        import yfinance as yf
    except ImportError:
        return {}

    try:
        t = yf.Ticker(ticker)
        expiries = t.options or []
        if not expiries:
            return {}

        expiry = expiries[0]  # nearest
        chain = t.option_chain(expiry)
        calls = chain.calls if hasattr(chain, "calls") else pd.DataFrame()
        puts = chain.puts if hasattr(chain, "puts") else pd.DataFrame()
        if calls.empty and puts.empty:
            return {}

        call_vol = float(calls["volume"].fillna(0).sum()) if "volume" in calls else 0.0
        put_vol  = float(puts["volume"].fillna(0).sum())  if "volume" in puts  else 0.0
        call_oi  = float(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls else 0.0
        put_oi   = float(puts["openInterest"].fillna(0).sum())  if "openInterest" in puts  else 0.0

        pc_vol = (put_vol / call_vol) if call_vol > 0 else None
        pc_oi  = (put_oi  / call_oi ) if call_oi  > 0 else None

        # ATM IV: within ±5% of spot. Spot may not be given so fall back to
        # the middle strike on the options chain.
        if spot is None:
            strikes = pd.concat([calls.get("strike", pd.Series(dtype=float)),
                                 puts.get("strike", pd.Series(dtype=float))])
            spot = float(strikes.median()) if not strikes.empty else None

        atm_iv = None
        if spot and "impliedVolatility" in calls and "impliedVolatility" in puts:
            lo, hi = spot * 0.95, spot * 1.05
            atm_calls = calls[(calls["strike"] >= lo) & (calls["strike"] <= hi)]
            atm_puts  = puts [(puts ["strike"] >= lo) & (puts ["strike"] <= hi)]
            ivs = pd.concat([
                atm_calls["impliedVolatility"],
                atm_puts["impliedVolatility"],
            ]).dropna()
            if len(ivs):
                atm_iv = float(ivs.mean())

        try:
            exp_date = datetime.strptime(expiry, "%Y-%m-%d")
            dte = (exp_date.date() - datetime.utcnow().date()).days
        except Exception:
            dte = None

        return {
            "expiry": expiry,
            "days_to_expiry": dte,
            "call_volume": call_vol,
            "put_volume": put_vol,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "put_call_vol_ratio": pc_vol,
            "put_call_oi_ratio": pc_oi,
            "atm_iv": atm_iv,
            "tilt": _label_from_pc(pc_vol),
        }
    except Exception:
        return {}
