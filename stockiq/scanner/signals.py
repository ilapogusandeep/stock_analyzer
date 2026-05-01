"""Lightweight per-ticker signal pull for the Scanner view.

Returns a small dict per ticker: price, 1-day return, count of unusual
options strikes, net options aggressor (buys − sells), today vs 7-day
news article count. Designed to be cheap enough to run across ~50
tickers within yfinance's rate-limit budget, but rich enough to flag
"something is happening" candidates worth a full Analyze pass.

Each call hits yfinance ~3 times (history, news, options chain), so
the caller MUST cache aggressively.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from stockiq.data.options import get_unusual_activity


def _safe_news(t) -> list[dict]:
    """Pull yfinance native news items. Returns empty list on failure."""
    try:
        raw = getattr(t, "news", None) or []
        return raw[:30] if isinstance(raw, list) else []
    except Exception:
        return []


def _news_publish_ts(item: dict) -> Optional[float]:
    """Extract the Unix timestamp from a yfinance news item, handling
    both old and new schemas."""
    content = item.get("content")
    if isinstance(content, dict):
        pd_str = content.get("pubDate")
        if pd_str:
            try:
                return datetime.fromisoformat(
                    str(pd_str).replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                return None
    raw = item.get("providerPublishTime")
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _news_velocity(items: list[dict]) -> dict:
    """Today's article count vs 7-day-trailing average daily count.

    Returns:
        {
            "count_today": int,
            "count_avg_7d": float,
            "velocity": float,  # today / max(avg, 0.5) — clipped to avoid /0
        }
    """
    now = time.time()
    one_day = 86400.0
    seven_days = 7 * one_day

    count_today = 0
    count_7d = 0
    for it in items:
        ts = _news_publish_ts(it)
        if ts is None:
            continue
        age = now - ts
        if age < 0:
            continue
        if age <= one_day:
            count_today += 1
        if age <= seven_days:
            count_7d += 1
    avg_7d = (count_7d - count_today) / 6 if count_7d > count_today else 0.0
    velocity = count_today / max(avg_7d, 0.5)  # 0.5 floor avoids /0
    return {
        "count_today": count_today,
        "count_avg_7d": round(avg_7d, 2),
        "velocity": round(velocity, 2),
    }


def _aggressor_net(rows: list[dict]) -> int:
    """Direction-aware net options flow.

    Each unusual row's directional sign is set by *both* the aggressor
    side (bid-ask heuristic) AND the option type:

        BUY  CALL → +1  (long upside — bullish)
        SELL PUT  → +1  (short downside — bullish)
        SELL CALL → -1  (short upside — bearish)
        BUY  PUT  → -1  (long downside — bearish)
        MID / —   →  0  (inconclusive)

    Without pairing aggressor with option type, heavy put-buying
    falsely registers as bullish (because aggressor=BUY). e.g. a row
    with five PUT BUYs and one CALL SELL — clearly bearish flow —
    would score +4 with a naive sum.
    """
    score = 0
    for r in rows:
        agg = (r.get("aggressor") or "").upper()
        side = (r.get("side") or "").upper()  # "CALL" or "PUT"
        if agg == "BUY" and side == "CALL":
            score += 1
        elif agg == "SELL" and side == "PUT":
            score += 1
        elif agg == "SELL" and side == "CALL":
            score -= 1
        elif agg == "BUY" and side == "PUT":
            score -= 1
        # MID / unknown side: contributes nothing
    return score


def compute_signals(ticker: str) -> dict[str, Any]:
    """Pull a lightweight signal snapshot for ``ticker``.

    Returns a dict with keys: ticker, price, change_1d, unusual_count,
    aggressor_net, news_count_today, news_avg_7d, news_velocity, error.
    On any failure the dict still has the right keys with zero/None
    defaults so the caller never has to special-case.
    """
    out: dict[str, Any] = {
        "ticker": ticker,
        "price": None,
        "change_1d": None,
        "unusual_count": 0,
        "aggressor_net": 0,
        "news_count_today": 0,
        "news_avg_7d": 0.0,
        "news_velocity": 0.0,
        "error": None,
    }

    try:
        import yfinance as yf
    except ImportError:
        out["error"] = "yfinance unavailable"
        return out

    try:
        t = yf.Ticker(ticker)

        # Price + 1-day return (5d window so weekends don't break us).
        try:
            hist = t.history(period="5d")
            if hist is not None and not hist.empty:
                close = hist["Close"]
                out["price"] = float(close.iloc[-1])
                if len(close) >= 2:
                    prev = float(close.iloc[-2])
                    if prev:
                        out["change_1d"] = (out["price"] / prev) - 1
        except Exception as e:
            out["error"] = f"price: {e}"

        # Unusual options + aggressor net (reuses the existing helper)
        try:
            rows = get_unusual_activity(ticker, top_n=10, min_voi_ratio=2.0)
            out["unusual_count"] = len(rows)
            out["aggressor_net"] = _aggressor_net(rows)
        except Exception:
            pass

        # News velocity
        try:
            items = _safe_news(t)
            v = _news_velocity(items)
            out["news_count_today"] = v["count_today"]
            out["news_avg_7d"] = v["count_avg_7d"]
            out["news_velocity"] = v["velocity"]
        except Exception:
            pass

    except Exception as e:
        out["error"] = str(e)

    return out
