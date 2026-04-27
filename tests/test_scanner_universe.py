"""Tests for stockiq.scanner.universe — the union-of-sources builder."""

from __future__ import annotations

from stockiq.scanner.universe import SCAN_CORE_TICKERS, get_scan_universe


def test_default_universe_includes_core():
    u = get_scan_universe()
    assert "AAPL" in u
    assert "MSFT" in u
    assert len(u) <= 50


def test_watchlist_appended_after_core():
    u = get_scan_universe(watchlist=["NBIS", "BTC-USD"])
    # Core comes first; watchlist names land later
    assert u.index("AAPL") < u.index("NBIS")
    assert "NBIS" in u
    assert "BTC-USD" in u


def test_dedupe_preserves_first_occurrence():
    """If a watchlist contains a ticker already in core, it shouldn't
    duplicate; core position wins."""
    aapl_idx_default = get_scan_universe().index("AAPL")
    u = get_scan_universe(watchlist=["AAPL", "ZS"])
    assert u.count("AAPL") == 1
    assert u.index("AAPL") == aapl_idx_default
    assert "ZS" in u


def test_searched_capped_to_10():
    """Recent-search list of 50 should add at most 10 to the universe.

    Use ZZ-prefixed sentinel tickers so the filter doesn't accidentally
    match real tickers in SCAN_CORE_TICKERS (e.g. XOM, IWM).
    """
    fake_searched = [f"ZZTEST{i}" for i in range(50)]
    u = get_scan_universe(searched=fake_searched, max_total=200)
    added = [t for t in u if t.startswith("ZZTEST")]
    assert len(added) <= 10


def test_max_total_caps_universe():
    """Even with huge inputs, output respects max_total."""
    big = [f"WL{i}" for i in range(200)]
    u = get_scan_universe(watchlist=big, max_total=20)
    assert len(u) == 20


def test_uppercases_lowercase_input():
    u = get_scan_universe(watchlist=["nbis", "btc-usd"])
    assert "NBIS" in u
    assert "BTC-USD" in u


def test_empty_inputs_return_only_core():
    u = get_scan_universe(watchlist=[], searched=[])
    assert all(t in SCAN_CORE_TICKERS for t in u)
