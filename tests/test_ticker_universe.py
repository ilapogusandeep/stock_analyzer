"""Tests for the merged ticker universe (POPULAR + SEC + searched).

The SEC fetch is patched out so tests stay offline. We verify:
- The merge precedence (POPULAR wins on conflict).
- remember_ticker writes to disk and is read back.
- remember_ticker no-ops on tickers already in POPULAR (no point storing).
- remember_ticker no-ops on falsy inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from stockiq.data import tickers as tickers_module
from stockiq.data.tickers import (
    POPULAR_TICKERS,
    get_all_tickers,
    remember_ticker,
)


@pytest.fixture
def isolated_searched(tmp_path, monkeypatch):
    """Point _SEARCHED_PATH at a tmp file so tests don't pollute the
    repo's data/ directory or stomp on each other."""
    fake = tmp_path / "searched.json"
    monkeypatch.setattr(tickers_module, "_SEARCHED_PATH", fake)
    return fake


def test_remember_writes_and_reads_back(isolated_searched):
    remember_ticker("BTC-USD", "Bitcoin USD")
    assert isolated_searched.exists()
    data = json.loads(isolated_searched.read_text())
    assert data == {"BTC-USD": "Bitcoin USD"}


def test_remember_appends_multiple(isolated_searched):
    remember_ticker("BTC-USD", "Bitcoin USD")
    remember_ticker("ETH-USD", "Ethereum USD")
    data = json.loads(isolated_searched.read_text())
    assert data == {"BTC-USD": "Bitcoin USD", "ETH-USD": "Ethereum USD"}


def test_remember_skips_already_curated(isolated_searched):
    """AAPL is in POPULAR_TICKERS — no point persisting it again."""
    assert "AAPL" in POPULAR_TICKERS
    remember_ticker("AAPL", "Apple Inc.")
    assert not isolated_searched.exists()


def test_remember_noop_on_falsy_inputs(isolated_searched):
    remember_ticker("", "Some Co")
    remember_ticker("FOO", "")
    remember_ticker("FOO", None)
    assert not isolated_searched.exists()


def test_remember_uppercases_ticker(isolated_searched):
    remember_ticker("eurusd=x", "EUR/USD spot")
    data = json.loads(isolated_searched.read_text())
    assert "EURUSD=X" in data


def test_get_all_tickers_curated_wins_on_conflict(isolated_searched):
    """SEC list returns 'NVIDIA CORP'; POPULAR has 'NVIDIA Corporation'.
    The curated name should win in the merged universe."""
    fake_sec = {"NVDA": "NVIDIA CORP"}
    with mock.patch.object(tickers_module, "_fetch_sec_tickers", return_value=fake_sec):
        merged = get_all_tickers()
    assert merged["NVDA"] == "NVIDIA Corporation"


def test_get_all_tickers_includes_searched(isolated_searched):
    remember_ticker("EURUSD=X", "EUR/USD spot")
    with mock.patch.object(tickers_module, "_fetch_sec_tickers", return_value={}):
        merged = get_all_tickers()
    assert merged["EURUSD=X"] == "EUR/USD spot"


def test_get_all_tickers_includes_sec_only_entries(isolated_searched):
    """A ticker present only in SEC should land in the merged dict."""
    fake_sec = {"OBSCUREXYZ": "Obscure Corp"}
    with mock.patch.object(tickers_module, "_fetch_sec_tickers", return_value=fake_sec):
        merged = get_all_tickers()
    assert merged.get("OBSCUREXYZ") == "Obscure Corp"


def test_get_all_tickers_offline_mode(isolated_searched):
    """include_sec=False should skip the network call entirely."""
    with mock.patch.object(tickers_module, "_fetch_sec_tickers") as fetcher:
        merged = get_all_tickers(include_sec=False)
        fetcher.assert_not_called()
    # Still includes POPULAR
    assert merged.get("AAPL") == "Apple Inc."
