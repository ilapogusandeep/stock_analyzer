"""Lightweight market-scanner: per-ticker signal pull + scoring.

Designed for free yfinance data and a personal app's scale (~50 tickers
per scan, cached aggressively). Not a real UOA / Polygon replacement —
flags candidates for the user to look at, then they hit the Analyze
view for the full breakdown.
"""

from __future__ import annotations
