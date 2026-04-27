"""Tests for UniversalStockAnalyzer.get_earnings_calendar.

yfinance changed ticker.calendar return type from DataFrame to dict;
both need to produce a usable `next_earnings_date` for the header band.
We stub self.stock.calendar so these tests don't hit the network.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from stockiq.core.analyzer import UniversalStockAnalyzer


def _make_analyzer(fake_stock) -> UniversalStockAnalyzer:
    """Construct an analyzer with a stubbed yfinance Ticker object."""
    # Bypass __init__ (which constructs a real yf.Ticker); patch fields manually.
    a = UniversalStockAnalyzer.__new__(UniversalStockAnalyzer)
    a.ticker = "TEST"
    a.stock = fake_stock
    return a


def _future(days: int) -> dt.date:
    return (dt.datetime.now() + dt.timedelta(days=days)).date()


# ---------------------------------------------------------------------------
# Dict-form calendar (new yfinance)
# ---------------------------------------------------------------------------

def test_dict_calendar_single_future_date():
    target = _future(10)
    fake = SimpleNamespace(
        calendar={"Earnings Date": [target]},
        earnings_dates=pd.DataFrame(),
    )
    result = _make_analyzer(fake).get_earnings_calendar()
    assert result["earnings_expected"] is True
    assert result["next_earnings_date"].date() == target
    assert result["days_to_earnings"] == 10


def test_dict_calendar_picks_earliest_future():
    """A window like [start, end] should pick the earlier one."""
    early = _future(5)
    late = _future(20)
    fake = SimpleNamespace(
        calendar={"Earnings Date": [late, early]},  # order shouldn't matter
        earnings_dates=pd.DataFrame(),
    )
    result = _make_analyzer(fake).get_earnings_calendar()
    assert result["next_earnings_date"].date() == early


def test_dict_calendar_ignores_past_dates():
    past = (dt.datetime.now() - dt.timedelta(days=10)).date()
    fake = SimpleNamespace(
        calendar={"Earnings Date": [past]},
        earnings_dates=pd.DataFrame(),
    )
    result = _make_analyzer(fake).get_earnings_calendar()
    assert result["earnings_expected"] is False
    assert result["next_earnings_date"] is None


def test_dict_calendar_empty_earnings_date_list():
    fake = SimpleNamespace(
        calendar={"Dividend Date": _future(5)},  # no Earnings Date key
        earnings_dates=pd.DataFrame(),
    )
    result = _make_analyzer(fake).get_earnings_calendar()
    assert result["earnings_expected"] is False


# ---------------------------------------------------------------------------
# DataFrame-form calendar (old yfinance)
# ---------------------------------------------------------------------------

def test_dataframe_calendar_parses_first_row():
    """Old schema: calendar is a DataFrame indexed by earnings timestamp."""
    ts = pd.Timestamp.now() + pd.Timedelta(days=7)
    df = pd.DataFrame({"foo": [1]}, index=[ts])
    df.index.name = "Earnings Date"
    fake = SimpleNamespace(calendar=df, earnings_dates=pd.DataFrame())
    result = _make_analyzer(fake).get_earnings_calendar()
    assert result["earnings_expected"] is True
    # Allow 1 day of slop since `days_to_earnings` is floor-of-now vs normalized today.
    assert 6 <= result["days_to_earnings"] <= 7


def test_empty_dataframe_falls_back_to_empty_result():
    fake = SimpleNamespace(calendar=pd.DataFrame(), earnings_dates=pd.DataFrame())
    result = _make_analyzer(fake).get_earnings_calendar()
    assert result["earnings_expected"] is False
    assert result["next_earnings_date"] is None


# ---------------------------------------------------------------------------
# earnings_dates fallback
# ---------------------------------------------------------------------------

def test_earnings_dates_fallback_when_calendar_empty():
    """If primary calendar is empty, use earnings_dates DataFrame."""
    future_ts = pd.Timestamp.now() + pd.Timedelta(days=14)
    past_ts = pd.Timestamp.now() - pd.Timedelta(days=90)
    edf = pd.DataFrame(
        {"EPS Estimate": [None, 1.2]},
        index=pd.DatetimeIndex([future_ts, past_ts]),
    )
    fake = SimpleNamespace(calendar={}, earnings_dates=edf)
    result = _make_analyzer(fake).get_earnings_calendar()
    assert result["earnings_expected"] is True
    assert 13 <= result["days_to_earnings"] <= 14


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_calendar_raising_exception_returns_safe_default():
    class ExplodingCalendar:
        @property
        def calendar(self):
            raise RuntimeError("yfinance is down")

        @property
        def earnings_dates(self):
            raise RuntimeError("also down")

    result = _make_analyzer(ExplodingCalendar()).get_earnings_calendar()
    assert result["earnings_expected"] is False
    assert result["days_to_earnings"] is None
