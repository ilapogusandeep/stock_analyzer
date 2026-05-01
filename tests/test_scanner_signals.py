"""Tests for stockiq.scanner.signals._aggressor_net.

The bias the scanner shows for each ticker is downstream of this
function — getting the directional sign wrong here flips bullish vs
bearish across the whole Top movers table.
"""

from __future__ import annotations

from stockiq.scanner.signals import _aggressor_net


def _row(side: str, aggressor: str) -> dict:
    return {"side": side, "aggressor": aggressor}


def test_naive_sum_would_flip_sign_for_put_heavy_flow():
    """The bug we just fixed: 5 PUT BUYs + 1 CALL SELL is clearly bearish
    (long downside + short upside) but a naive sum-of-aggressor would
    score +4 because every PUT BUY counts as a 'BUY'. The fix pairs
    aggressor with option type so PUT BUYs count -1, not +1."""
    rows = [_row("PUT", "BUY")] * 5 + [_row("CALL", "SELL")]
    assert _aggressor_net(rows) == -6


def test_call_buys_are_bullish():
    rows = [_row("CALL", "BUY")] * 4
    assert _aggressor_net(rows) == 4


def test_put_sells_are_bullish():
    """Selling cash-secured puts expresses a bullish/neutral-to-up view."""
    rows = [_row("PUT", "SELL")] * 3
    assert _aggressor_net(rows) == 3


def test_call_sells_are_bearish():
    rows = [_row("CALL", "SELL")] * 2
    assert _aggressor_net(rows) == -2


def test_mid_and_unknown_contribute_zero():
    rows = [
        _row("CALL", "MID"),
        _row("PUT", "MID"),
        _row("CALL", "—"),
        {"side": "CALL"},          # missing aggressor
        {"aggressor": "BUY"},      # missing side
    ]
    assert _aggressor_net(rows) == 0


def test_mixed_bullish_dominates():
    """Strong call-buying with a stray put-buy still nets bullish."""
    rows = (
        [_row("CALL", "BUY")] * 4
        + [_row("PUT", "BUY")]      # one bearish hedge
        + [_row("PUT", "SELL")] * 2 # bullish income
    )
    # +4 (call buys) - 1 (put buy) + 2 (put sells) = +5
    assert _aggressor_net(rows) == 5


def test_empty_rows():
    assert _aggressor_net([]) == 0
