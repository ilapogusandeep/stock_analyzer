"""Curated universe builder for the Scanner's "Top movers" view.

We can't scan all 10K SEC tickers on yfinance free without getting
rate-limited into oblivion. The universe is the union of:

  1. A hand-picked list of ~40 mega/large caps + ETFs that always
     have meaningful options + news activity.
  2. Whatever's currently in the user's Supabase watchlist.
  3. Recently-searched tickers from data/searched_tickers.json (or
     the Supabase searched_tickers table) — capped to the 10 most
     recent so the scan stays under ~50 names.

Order is preserved by source (curated -> watchlist -> recent) and
duplicates are dropped on first occurrence.
"""

from __future__ import annotations

# ~40 names that almost always have unusual options and active news flow.
# Mix of big tech, semis, EVs, popular memes, and broad-market ETFs.
SCAN_CORE_TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    # Semis
    "AMD", "INTC", "QCOM", "MU", "TSM", "ARM",
    # Cloud / SaaS
    "CRM", "ORCL", "ADBE", "SNOW", "PLTR", "NET",
    # Finance
    "JPM", "GS", "BAC", "MS", "V", "MA",
    # Consumer
    "WMT", "HD", "COST", "DIS", "NFLX", "MCD", "SBUX",
    # Energy / industrials
    "XOM", "CVX", "BA",
    # Memes / volatility
    "GME", "AMC",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA",
]


def get_scan_universe(watchlist: list[str] | None = None,
                       searched: list[str] | None = None,
                       max_total: int = 50) -> list[str]:
    """Build the union of curated + watchlist + recent searches.

    De-duplicates while preserving order: curated wins on duplicate
    insertion. Caps total length to ``max_total`` so the scan stays
    within yfinance's per-IP rate-limit budget.
    """
    seen: set[str] = set()
    out: list[str] = []
    sources: list[list[str]] = [
        [t.upper() for t in SCAN_CORE_TICKERS],
        [t.upper() for t in (watchlist or [])],
        [t.upper() for t in (searched or [])][:10],  # cap recent at 10
    ]
    for src in sources:
        for t in src:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= max_total:
                    return out
    return out
