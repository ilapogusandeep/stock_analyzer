"""Watchlist storage for the Scanner view.

Same dispatch pattern as ``stockiq.data.tickers``: prefer Supabase when
``SUPABASE_URL`` + ``SUPABASE_KEY`` are configured (so the watchlist
survives Streamlit Cloud redeploys), fall back to a local JSON file
otherwise.

Public API:
    add(ticker)      — add to watchlist
    remove(ticker)   — drop from watchlist
    list_tickers()   — return list[str] of tickers, ordered by added_at desc
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "watchlist.json"


# ---------------------------------------------------------------------------
# Secret discovery (mirrors the pattern in tickers.py)
# ---------------------------------------------------------------------------

def _supabase_creds() -> tuple[Optional[str], Optional[str]]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not (url and key):
        try:
            import streamlit as st
            url = url or st.secrets.get("SUPABASE_URL")
            key = key or st.secrets.get("SUPABASE_KEY")
        except Exception:
            pass
    return (url or None), (key or None)


def _sb_headers(key: str, prefer: str = "return=minimal") -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


# ---------------------------------------------------------------------------
# Supabase backend
# ---------------------------------------------------------------------------

def _list_supabase(url: str, key: str) -> list[str]:
    try:
        import requests
    except ImportError:
        return []
    try:
        r = requests.get(
            f"{url}/rest/v1/watchlist?select=ticker&order=added_at.desc&limit=200",
            headers=_sb_headers(key, prefer="count=none"),
            timeout=8,
        )
        if r.status_code != 200:
            return []
        return [str(row["ticker"]).upper() for row in (r.json() or [])
                if row.get("ticker")]
    except Exception:
        return []


def _add_supabase(url: str, key: str, ticker: str) -> bool:
    try:
        import requests
    except ImportError:
        return False
    try:
        r = requests.post(
            f"{url}/rest/v1/watchlist?on_conflict=ticker",
            headers=_sb_headers(
                key, prefer="resolution=ignore-duplicates,return=minimal",
            ),
            json={"ticker": ticker},
            timeout=8,
        )
        return r.status_code in (200, 201, 204, 409)
    except Exception:
        return False


def _remove_supabase(url: str, key: str, ticker: str) -> bool:
    try:
        import requests
    except ImportError:
        return False
    try:
        r = requests.delete(
            f"{url}/rest/v1/watchlist?ticker=eq.{ticker}",
            headers=_sb_headers(key),
            timeout=8,
        )
        return r.status_code in (200, 204)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# File-backend fallback
# ---------------------------------------------------------------------------

def _list_file() -> list[str]:
    try:
        if _PATH.exists():
            with _PATH.open() as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(t).upper() for t in data if t]
    except Exception:
        pass
    return []


def _save_file(tickers: list[str]) -> None:
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PATH.open("w") as f:
            json.dump(list(tickers), f, indent=2)
    except Exception:
        pass


def _add_file(ticker: str) -> None:
    items = _list_file()
    if ticker in items:
        return
    items.insert(0, ticker)  # newest first
    _save_file(items)


def _remove_file(ticker: str) -> None:
    items = [t for t in _list_file() if t != ticker]
    _save_file(items)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_tickers() -> list[str]:
    """Return the user's watchlist, newest-first."""
    url, key = _supabase_creds()
    if url and key:
        return _list_supabase(url, key)
    return _list_file()


def add(ticker: str) -> bool:
    """Add a ticker. Returns True on success (or if already present)."""
    if not ticker:
        return False
    ticker = ticker.strip().upper()
    if not ticker:
        return False
    url, key = _supabase_creds()
    if url and key and _add_supabase(url, key, ticker):
        return True
    _add_file(ticker)
    return True


def remove(ticker: str) -> bool:
    """Remove a ticker. Returns True on success (or if absent)."""
    if not ticker:
        return False
    ticker = ticker.strip().upper()
    if not ticker:
        return False
    url, key = _supabase_creds()
    if url and key and _remove_supabase(url, key, ticker):
        return True
    _remove_file(ticker)
    return True
