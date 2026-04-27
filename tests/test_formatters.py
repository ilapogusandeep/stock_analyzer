"""Tests for the price / percentage / ratio / big-money formatters used
throughout the UI.

These tiny helpers are called from every panel; a regression here would
silently break every metric display in the app.
"""

from __future__ import annotations

import pytest

from stockiq.ui.components import (
    _cls,
    fmt_big_money,
    fmt_pct,
    fmt_pct_ratio,
    fmt_price,
    fmt_ratio,
)


# ---------------------------------------------------------------------------
# fmt_price
# ---------------------------------------------------------------------------

def test_fmt_price_none():
    assert fmt_price(None) == "—"


def test_fmt_price_basic():
    assert fmt_price(123.45) == "$123.45"


def test_fmt_price_comma_thousands():
    assert fmt_price(1234567.89) == "$1,234,567.89"


def test_fmt_price_two_decimals_always():
    assert fmt_price(5.0) == "$5.00"
    assert fmt_price(0.001) == "$0.00"


# ---------------------------------------------------------------------------
# fmt_pct
# ---------------------------------------------------------------------------

def test_fmt_pct_none():
    assert fmt_pct(None) == "—"


def test_fmt_pct_positive_gets_sign():
    assert fmt_pct(5.2) == "+5.20%"


def test_fmt_pct_negative_keeps_sign():
    # "-5.20%" — the negative sign is intrinsic to the number, not
    # added by the signed branch.
    assert fmt_pct(-5.2) == "-5.20%"


def test_fmt_pct_unsigned_mode():
    assert fmt_pct(5.2, signed=False) == "5.20%"


def test_fmt_pct_decimal_control():
    assert fmt_pct(5.2345, decimals=1) == "+5.2%"
    assert fmt_pct(5.2345, decimals=3) == "+5.234%"


# ---------------------------------------------------------------------------
# fmt_ratio
# ---------------------------------------------------------------------------

def test_fmt_ratio_none():
    assert fmt_ratio(None) == "—"


def test_fmt_ratio_default_two_decimals():
    assert fmt_ratio(1.2345) == "1.23"


def test_fmt_ratio_custom_decimals():
    assert fmt_ratio(1.2345, decimals=3) == "1.234"


# ---------------------------------------------------------------------------
# fmt_pct_ratio (0..1 fraction -> percent string)
# ---------------------------------------------------------------------------

def test_fmt_pct_ratio_none():
    assert fmt_pct_ratio(None) == "—"


def test_fmt_pct_ratio_scales_to_percent():
    assert fmt_pct_ratio(0.50) == "50.0%"
    assert fmt_pct_ratio(0.1234) == "12.3%"


def test_fmt_pct_ratio_decimal_control():
    assert fmt_pct_ratio(0.1234, decimals=2) == "12.34%"


# ---------------------------------------------------------------------------
# fmt_big_money — auto unit selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, "—"),
    (0, "—"),
    (500, "$500"),
    (5_000, "$5.00K"),
    (5_200_000, "$5.20M"),
    (5_200_000_000, "$5.20B"),
    (5_200_000_000_000, "$5.20T"),
])
def test_fmt_big_money(value, expected):
    assert fmt_big_money(value) == expected


# ---------------------------------------------------------------------------
# _cls — direction helper used for up/down/flat CSS classes
# ---------------------------------------------------------------------------

def test_cls_none_is_flat():
    assert _cls(None) == "flat"


def test_cls_positive_is_up():
    assert _cls(0.01) == "up"
    assert _cls(100) == "up"


def test_cls_negative_is_down():
    assert _cls(-0.01) == "down"
    assert _cls(-100) == "down"


def test_cls_zero_is_flat():
    assert _cls(0) == "flat"
