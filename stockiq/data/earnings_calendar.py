"""Upcoming-earnings calendar for the Analyze view's bottom strip.

Reuses the same dict/DataFrame fallback dance as
``stockiq.core.analyzer.StockAnalyzer.get_earnings_calendar`` but
exposes a Streamlit-cached helper that aggregates next earnings dates
across many tickers (watchlist ∪ curated "famous" set) so we can
render a single horizontal calendar strip without paying yfinance
costs on every rerun.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import streamlit as st


# yfinance logs a noisy "No earnings dates found, symbol may be
# delisted" warning every time we ask for a non-equity (ETFs, indexes,
# forex, crypto). Pre-filter so we never make those calls — these
# instruments don't report earnings and we know it ahead of time.
_NON_EQUITY_PREFIXES = ("^",)
_NON_EQUITY_SUFFIXES = ("=X", "-USD", "-USDT")
_KNOWN_NON_EQUITY = frozenset({
    # Broad-market / index ETFs
    "SPY", "QQQ", "DIA", "IWM", "VOO", "VTI", "VEA", "VWO", "EFA", "EEM",
    "FXI", "ACWI", "MCHI",
    # Sector ETFs (the same XL* list the analyzer uses for relative strength)
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLI", "XLE", "XLC", "XLU",
    "XLRE", "XLB",
    # Themed / leveraged ETFs that show up in SCAN_CORE_TICKERS or
    # popular watchlists
    "ARKK", "TQQQ", "SQQQ", "SOXX", "SOXL", "SMH", "XBI", "IBB",
    "KBE", "KRE", "XHB", "ITA",
    # Bond ETFs
    "AGG", "BND", "TLT", "SHY", "SHV", "IEF", "TIP", "HYG", "LQD",
    # Commodity ETFs
    "GLD", "SLV", "USO", "UNG", "DBC", "PDBC",
})


def _is_non_equity(ticker: str) -> bool:
    t = (ticker or "").strip().upper()
    if not t:
        return True
    if t.startswith(_NON_EQUITY_PREFIXES):
        return True
    if any(t.endswith(suf) for suf in _NON_EQUITY_SUFFIXES):
        return True
    return t in _KNOWN_NON_EQUITY


def _to_naive_date(value) -> Optional[date]:
    """Coerce any date-like input to a naive ``datetime.date``. Returns
    None when the input can't be interpreted (yfinance occasionally
    returns ``NaT``, empty lists, or strings)."""
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if getattr(ts, "tz", None) is not None:
        try:
            ts = ts.tz_convert("UTC").tz_localize(None)
        except Exception:
            try:
                ts = ts.tz_localize(None)
            except Exception:
                return None
    return ts.date()


@st.cache_data(ttl=86400, show_spinner=False, max_entries=512)
def _next_earnings_for(ticker: str) -> Optional[dict]:
    """Return ``{ticker, date, est_eps}`` for the next future earnings
    print, or None when yfinance has no upcoming date for the symbol.
    Cached for 24h since earnings dates don't move intraday."""
    ticker = ticker.strip().upper()
    if not ticker or _is_non_equity(ticker):
        return None
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
    except Exception:
        return None

    today = date.today()
    next_d: Optional[date] = None
    est_eps = None

    # Primary: ticker.calendar — recent yfinance returns a dict, older
    # versions returned a DataFrame.
    try:
        cal = t.calendar
    except Exception:
        cal = None
    if isinstance(cal, dict) and cal:
        raw = cal.get("Earnings Date") or []
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        # Pick the earliest future date (calendar can return a [start,
        # end] window for unconfirmed prints).
        for entry in raw:
            d = _to_naive_date(entry)
            if d is not None and d >= today and (next_d is None or d < next_d):
                next_d = d
        eps = cal.get("EPS Estimate")
        if isinstance(eps, (list, tuple)) and eps:
            eps = eps[0]
        if isinstance(eps, (int, float)):
            est_eps = float(eps)
    elif hasattr(cal, "empty") and not cal.empty:
        try:
            next_d = _to_naive_date(cal.iloc[0].name)
        except Exception:
            pass

    # Fallback: ticker.earnings_dates — past + future prints, indexed
    # by date. Use it when calendar is empty.
    if next_d is None:
        try:
            edf = t.earnings_dates
        except Exception:
            edf = None
        if edf is not None and hasattr(edf, "empty") and not edf.empty:
            idx = edf.index
            try:
                now_ts = pd.Timestamp.now(tz=idx.tz) if getattr(idx, "tz", None) else pd.Timestamp.now()
                future = idx[idx > now_ts]
                if len(future):
                    next_d = _to_naive_date(future.min())
            except Exception:
                pass

    if next_d is None:
        return None
    return {"ticker": ticker, "date": next_d, "est_eps": est_eps}


def get_upcoming_earnings(
    tickers: tuple[str, ...] | list[str],
    days_ahead: int = 7,
) -> list[dict]:
    """Return upcoming earnings within ``days_ahead`` days for the given
    tickers, sorted soonest-first. Tickers without a date or with a
    date past the window are silently dropped — the caller decides
    whether to render the strip based on the returned list length."""
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    out: list[dict] = []
    seen: set[str] = set()
    for raw in tickers:
        if not raw:
            continue
        tkr = raw.strip().upper()
        if not tkr or tkr in seen:
            continue
        seen.add(tkr)
        info = _next_earnings_for(tkr)
        if not info:
            continue
        if info["date"] > cutoff:
            continue
        info["days_to"] = (info["date"] - today).days
        out.append(info)
    out.sort(key=lambda x: x["date"])
    return out
