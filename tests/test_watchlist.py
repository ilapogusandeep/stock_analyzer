"""Tests for stockiq.data.watchlist — Supabase + file-fallback storage."""

from __future__ import annotations

import json
from unittest import mock

import pytest

from stockiq.data import watchlist as wl


@pytest.fixture
def isolated_file(tmp_path, monkeypatch):
    """Force the file backend (no Supabase env) and point _PATH at tmp."""
    fake = tmp_path / "watchlist.json"
    monkeypatch.setattr(wl, "_PATH", fake)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    monkeypatch.setattr(wl, "_supabase_creds", lambda: (None, None))
    return fake


@pytest.fixture
def supabase_creds(monkeypatch):
    monkeypatch.setattr(
        wl, "_supabase_creds",
        lambda: ("https://fake.supabase.co", "sb_publishable_fake"),
    )


# ---------------------------------------------------------------------------
# File backend
# ---------------------------------------------------------------------------

def test_add_and_list_round_trip(isolated_file):
    wl.add("AAPL")
    wl.add("NBIS")
    assert wl.list_tickers() == ["NBIS", "AAPL"]  # newest-first


def test_add_uppercases(isolated_file):
    wl.add("nbis")
    assert wl.list_tickers() == ["NBIS"]


def test_add_dedupes(isolated_file):
    wl.add("AAPL")
    wl.add("AAPL")
    assert wl.list_tickers() == ["AAPL"]


def test_remove(isolated_file):
    wl.add("AAPL")
    wl.add("NBIS")
    wl.remove("AAPL")
    assert wl.list_tickers() == ["NBIS"]


def test_remove_missing_is_noop(isolated_file):
    wl.add("AAPL")
    wl.remove("MSFT")  # not present
    assert wl.list_tickers() == ["AAPL"]


def test_empty_input_rejected(isolated_file):
    assert wl.add("") is False
    assert wl.add(None) is False
    assert wl.list_tickers() == []


# ---------------------------------------------------------------------------
# Supabase backend dispatch
# ---------------------------------------------------------------------------

def test_add_uses_supabase_when_configured(supabase_creds):
    with mock.patch.object(wl, "_add_supabase", return_value=True) as sb, \
         mock.patch.object(wl, "_add_file") as fs:
        wl.add("NBIS")
    sb.assert_called_once()
    fs.assert_not_called()


def test_add_falls_back_to_file_on_supabase_failure(supabase_creds, tmp_path, monkeypatch):
    fake = tmp_path / "watchlist.json"
    monkeypatch.setattr(wl, "_PATH", fake)
    with mock.patch.object(wl, "_add_supabase", return_value=False):
        wl.add("NBIS")
    assert fake.exists()
    assert json.loads(fake.read_text()) == ["NBIS"]


def test_remove_uses_supabase_when_configured(supabase_creds):
    with mock.patch.object(wl, "_remove_supabase", return_value=True) as sb, \
         mock.patch.object(wl, "_remove_file") as fs:
        wl.remove("NBIS")
    sb.assert_called_once()
    fs.assert_not_called()


def test_list_uses_supabase_when_configured(supabase_creds):
    expected = ["NBIS", "AAPL"]
    with mock.patch.object(wl, "_list_supabase", return_value=expected), \
         mock.patch.object(wl, "_list_file") as fs:
        result = wl.list_tickers()
    fs.assert_not_called()
    assert result == expected
