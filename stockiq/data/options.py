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
from typing import Any, List, Optional

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


def get_unusual_activity(
    ticker: str,
    top_n: int = 5,
    min_voi_ratio: float = 2.0,
    max_expiries: int = 3,
    min_volume: int = 100,
) -> List[dict]:
    """Detect unusual options activity across the next few expiries.

    An "unusual" strike has today's volume >= min_voi_ratio × open_interest
    (new positions opened today, not just existing contracts). Rows are
    ranked by dollar premium flow = volume × mid_price × 100.

    Returns up to ``top_n`` rows across the next ``max_expiries`` expiries.
    Empty list on any failure — never raises.
    """
    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        t = yf.Ticker(ticker)
        expiries = (t.options or [])[:max_expiries]
        if not expiries:
            return []

        rows: list[dict] = []
        for expiry in expiries:
            try:
                chain = t.option_chain(expiry)
            except Exception:
                continue

            for side, df in (("C", chain.calls), ("P", chain.puts)):
                if df is None or df.empty:
                    continue
                needed = {"strike", "volume", "openInterest", "bid", "ask"}
                if not needed.issubset(df.columns):
                    continue
                d = df.copy()
                d["volume"] = d["volume"].fillna(0)
                d["openInterest"] = d["openInterest"].fillna(0)
                d["bid"] = d["bid"].fillna(0)
                d["ask"] = d["ask"].fillna(0)

                d = d[d["volume"] >= min_volume]
                # V/OI ratio — use 1 as floor so brand-new strikes (OI=0)
                # still rank by absolute volume without exploding to inf.
                d["voi_ratio"] = d["volume"] / d["openInterest"].clip(lower=1)
                d = d[d["voi_ratio"] >= min_voi_ratio]
                if d.empty:
                    continue

                d["mid"] = (d["bid"] + d["ask"]) / 2
                # Fall back to lastPrice if bid/ask are zero (illiquid)
                last_col = "lastPrice" if "lastPrice" in d.columns else None
                if last_col:
                    fallback = d[last_col].fillna(0)
                    d["mid"] = d["mid"].where(d["mid"] > 0, fallback)
                d["premium_flow"] = d["volume"] * d["mid"] * 100

                # Aggressor heuristic: position of last trade within the
                # bid/ask spread. Top third of spread ~ aggressive buy
                # (took the ask); bottom third ~ aggressive sell (hit the
                # bid); middle ~ inconclusive. Requires a real spread and
                # a valid last print; otherwise marked "—".
                def _aggressor(row):
                    try:
                        bid = float(row.get("bid") or 0)
                        ask = float(row.get("ask") or 0)
                        last = float(row.get(last_col) or 0) if last_col else 0
                        if bid <= 0 or ask <= 0 or ask <= bid or last <= 0:
                            return "—"
                        pos = (last - bid) / (ask - bid)
                        if pos >= 0.66:
                            return "BUY"
                        if pos <= 0.34:
                            return "SELL"
                        return "MID"
                    except Exception:
                        return "—"

                for _, r in d.iterrows():
                    rows.append({
                        "side": side,
                        "strike": float(r["strike"]),
                        "expiry": expiry,
                        "volume": int(r["volume"]),
                        "open_interest": int(r["openInterest"]),
                        "voi_ratio": float(r["voi_ratio"]),
                        "mid_price": float(r["mid"]),
                        "premium_flow": float(r["premium_flow"]),
                        "aggressor": _aggressor(r),
                        "cluster": False,  # filled in after sort
                        "iv": (float(r["impliedVolatility"])
                               if "impliedVolatility" in d.columns
                               and pd.notna(r.get("impliedVolatility")) else None),
                    })

        # Cluster detection: within a (side, expiry) group, mark rows as
        # part of a cluster if at least 3 strikes fall within 10% of each
        # other. Three nearby calls all unusual is much stronger signal
        # than one lone strike.
        by_group: dict = {}
        for r in rows:
            by_group.setdefault((r["side"], r["expiry"]), []).append(r)
        for group_rows in by_group.values():
            strikes = sorted([(g["strike"], g) for g in group_rows])
            n = len(strikes)
            for i in range(n):
                base = strikes[i][0]
                neighbors = [s for s, _ in strikes
                             if abs(s - base) / base <= 0.10]
                if len(neighbors) >= 3:
                    strikes[i][1]["cluster"] = True

        rows.sort(key=lambda x: x["premium_flow"], reverse=True)
        return rows[:top_n]
    except Exception:
        return []
