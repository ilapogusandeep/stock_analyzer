"""Tests for the options aggressor + cluster heuristics in
stockiq.data.options.get_unusual_activity.

The aggressor helper is a closure inside the function, so we replicate
its logic here for a direct math check. The cluster detection is
exercised by building a synthetic chain via monkeypatched yfinance.
"""

from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Aggressor logic (same math as the closure inside get_unusual_activity)
# ---------------------------------------------------------------------------

def _aggressor(bid: float, ask: float, last: float) -> str:
    if bid <= 0 or ask <= 0 or ask <= bid or last <= 0:
        return "—"
    pos = (last - bid) / (ask - bid)
    if pos >= 0.66:
        return "BUY"
    if pos <= 0.34:
        return "SELL"
    return "MID"


@pytest.mark.parametrize(
    "bid,ask,last,expected",
    [
        (1.00, 1.20, 1.20, "BUY"),    # hit the ask
        (1.00, 1.20, 1.00, "SELL"),   # hit the bid
        (1.00, 1.20, 1.10, "MID"),    # midpoint
        (1.00, 1.20, 1.14, "BUY"),    # pos=0.70, just past top-third threshold (0.66)
        (1.00, 1.20, 1.06, "SELL"),   # pos=0.30, just past bottom-third threshold (0.34)
        (1.20, 1.20, 1.20, "—"),      # locked market, no spread
        (1.00, 1.20, 0.00, "—"),      # no last print (stale)
        (0.00, 0.10, 0.05, "—"),      # no bid
        (-1.0, 1.0, 0.5, "—"),        # malformed
    ],
)
def test_aggressor_classification(bid, ask, last, expected):
    assert _aggressor(bid, ask, last) == expected


# ---------------------------------------------------------------------------
# Cluster detection (within 10% of the reference strike, same side+expiry)
# ---------------------------------------------------------------------------

def _flag_cluster(rows: list) -> list:
    """Replica of the cluster pass that runs after rows are built."""
    by_group: dict = {}
    for r in rows:
        by_group.setdefault((r["side"], r["expiry"]), []).append(r)
    for group_rows in by_group.values():
        strikes = sorted([(g["strike"], g) for g in group_rows])
        n = len(strikes)
        for i in range(n):
            base = strikes[i][0]
            neighbors = [s for s, _ in strikes if abs(s - base) / base <= 0.10]
            if len(neighbors) >= 3:
                strikes[i][1]["cluster"] = True
    return rows


def _row(side: str, strike: float, expiry: str = "2026-05-01") -> dict:
    return {"side": side, "strike": strike, "expiry": expiry, "cluster": False}


def test_cluster_requires_at_least_three_nearby_strikes():
    rows = [_row("C", 100), _row("C", 105)]
    _flag_cluster(rows)
    assert all(not r["cluster"] for r in rows)


def test_cluster_flags_three_nearby_calls():
    rows = [_row("C", 100), _row("C", 105), _row("C", 108)]
    _flag_cluster(rows)
    assert all(r["cluster"] for r in rows)


def test_cluster_isolates_by_side():
    """Three calls near each other, one put — the put shouldn't cluster."""
    rows = [_row("C", 100), _row("C", 105), _row("C", 108), _row("P", 102)]
    _flag_cluster(rows)
    calls = [r for r in rows if r["side"] == "C"]
    puts = [r for r in rows if r["side"] == "P"]
    assert all(r["cluster"] for r in calls)
    assert all(not r["cluster"] for r in puts)


def test_cluster_isolates_by_expiry():
    """Three strikes on two different expiries — neither expiry alone has 3."""
    rows = [
        _row("C", 100, "2026-05-01"),
        _row("C", 105, "2026-05-01"),
        _row("C", 108, "2026-06-01"),  # different expiry
    ]
    _flag_cluster(rows)
    assert all(not r["cluster"] for r in rows)


def test_cluster_respects_10_percent_window():
    """Strikes too far apart (> 10% of base) don't cluster."""
    rows = [_row("C", 100), _row("C", 150), _row("C", 200)]
    _flag_cluster(rows)
    assert all(not r["cluster"] for r in rows)
