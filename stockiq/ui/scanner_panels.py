"""Render functions for the Scanner view (watchlist + universe scan)."""

from __future__ import annotations

from typing import Iterable

import streamlit as st

from stockiq.ui.components import _cls, fmt_pct, panel_close, panel_open


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_price(v) -> str:
    return f"${v:,.2f}" if isinstance(v, (int, float)) and v else "—"


def _fmt_signed(v, suffix: str = "") -> str:
    if v is None:
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v}{suffix}"


def _bias_pill(bias: str) -> str:
    cls = {
        "BULLISH": "rs-pill rs-bull",
        "BEARISH": "rs-pill rs-bear",
    }.get(bias, "rs-pill rs-neutral")
    return f'<span class="{cls}">{bias}</span>'


@st.cache_data(ttl=86400, show_spinner=False, max_entries=512)
def _ticker_exists(ticker: str) -> bool:
    """Quick yfinance sanity check that ``ticker`` resolves to real
    price data. Cached for a day per ticker so repeated Add attempts on
    the same symbol don't re-hit yfinance (and so a typo'd symbol stays
    rejected within a session)."""
    try:
        import yfinance as yf
    except ImportError:
        return True  # if yfinance is missing, don't block the add
    try:
        hist = yf.Ticker(ticker.strip().upper()).history(period="5d")
        return hist is not None and not hist.empty
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Scanner table — shared by watchlist and universe sections
# ---------------------------------------------------------------------------

SCANNER_HEADERS = [
    ("ticker", "Ticker"),
    ("price", "Price"),
    ("change_1d", "1d Δ"),
    ("unusual_count", "Unusual"),
    ("aggressor_net", "Agg net"),
    ("news_velocity", "News×"),
    ("score", "Score"),
    ("bias", "Bias"),
]


def _scanner_grid_html(rows: list[dict]) -> str:
    """Build the inner grid HTML for a scanner result set. Returns a
    string (caller embeds inside panel_open/panel_close in a single
    st.markdown call so Streamlit doesn't break the grid by wrapping
    each markdown invocation in its own container)."""
    if not rows:
        return '<div class="sent-label">No tickers to scan yet.</div>'

    parts: list[str] = []
    # Header row -- 8 cells matching SCANNER_HEADERS
    for _, label in SCANNER_HEADERS:
        parts.append(f'<div class="sc-th">{label}</div>')

    # Data rows -- 8 cells each, flat under the grid (no .sc-row wrapper).
    for r in rows:
        ticker = r.get("ticker", "—")
        price = r.get("price")
        is_invalid = price is None  # yfinance returned no price -> bad symbol
        chg = r.get("change_1d")
        chg_cls = _cls(chg) if chg is not None else "flat"
        chg_str = fmt_pct(chg * 100, decimals=2) if chg is not None else "—"
        unusual = r.get("unusual_count", 0) or 0
        agg = r.get("aggressor_net", 0) or 0
        agg_cls = _cls(agg) if agg else "flat"
        velocity = r.get("news_velocity") or 0
        score = r.get("score") or 0
        bias = r.get("bias", "NEUTRAL")

        # Mark broken rows so users know to remove them. yfinance
        # returns no price = the symbol doesn't resolve to real data.
        td_cls = "sc-td sc-td-invalid" if is_invalid else "sc-td"
        ticker_cell = (
            f'<a class="rs-pill" href="?ticker={ticker}" '
            f'target="_self">{ticker}</a>'
        )
        if is_invalid:
            ticker_cell += ' <span class="sc-invalid-tag">no data</span>'

        parts.append(f'<div class="{td_cls}">{ticker_cell}</div>')
        parts.append(f'<div class="{td_cls}">{_fmt_price(price)}</div>')
        parts.append(
            f'<div class="{td_cls}"><span class="{chg_cls}">{chg_str}</span></div>'
        )
        parts.append(f'<div class="{td_cls}">{unusual}</div>')
        parts.append(
            f'<div class="{td_cls}"><span class="{agg_cls}">{_fmt_signed(agg)}</span></div>'
        )
        parts.append(f'<div class="{td_cls}">{velocity:.1f}×</div>')
        parts.append(
            f'<div class="{td_cls}"><span class="sc-score">{score:.0f}</span></div>'
        )
        parts.append(f'<div class="{td_cls}">{_bias_pill(bias)}</div>')

    grid_template = "60px 70px 64px 64px 64px 60px 50px 80px"
    return (
        f'<div class="sc-grid" style="grid-template-columns: {grid_template};">'
        + "".join(parts) + '</div>'
    )


# ---------------------------------------------------------------------------
# Top-level view renderers
# ---------------------------------------------------------------------------

def render_watchlist_section(
    rows: list[dict],
    remove_callback,
    add_callback,
    last_refreshed_min: float | None,
) -> None:
    """Render the Watchlist panel: rows, refresh button, add input."""
    sub = (
        f"{len(rows)} tickers · last refreshed {last_refreshed_min:.0f}m ago"
        if rows and last_refreshed_min is not None
        else f"{len(rows)} tickers"
    )

    # Emit the full panel (open + grid + close) in ONE st.markdown call;
    # splitting these across separate calls makes Streamlit wrap each in
    # its own container which kills the CSS grid layout.
    st.markdown(
        panel_open("Watchlist", sub)
        + _scanner_grid_html(rows)
        + panel_close(),
        unsafe_allow_html=True,
    )

    # Per-ticker remove buttons -- Streamlit-native, so they can't live
    # inside the HTML grid above. Rendered as a compact row beneath.
    if rows:
        cols = st.columns(min(len(rows), 8) or 1)
        for i, r in enumerate(rows):
            with cols[i % 8]:
                if st.button(
                    f"× {r['ticker']}",
                    key=f"wl_rm_{r['ticker']}",
                    help=f"Remove {r['ticker']} from watchlist",
                ):
                    remove_callback(r["ticker"])
                    st.rerun()

    # Add input + Add button + Refresh button laid out in one row.
    add_col, btn_col, refresh_col = st.columns([0.55, 0.20, 0.25])
    with add_col:
        new_ticker = st.text_input(
            "Add ticker to watchlist",
            key="wl_add_input",
            placeholder="e.g. NBIS, BTC-USD, ^VIX",
            label_visibility="collapsed",
        )
    with btn_col:
        if st.button("➕ Add", key="wl_add_btn", width="stretch"):
            if new_ticker:
                if _ticker_exists(new_ticker):
                    add_callback(new_ticker)
                    st.session_state.pop("wl_add_input", None)
                    st.rerun()
                else:
                    st.error(
                        f"Couldn't find ticker '{new_ticker.upper()}'. "
                        "Check the symbol — yfinance uses '-USD' suffixes "
                        "for crypto (BTC-USD), '^' prefixes for indices "
                        "(^GSPC), and '=X' suffixes for forex (EURUSD=X)."
                    )
    with refresh_col:
        if st.button("🔄 Refresh signals", key="wl_refresh_btn", width="stretch"):
            st.session_state["wl_force_refresh"] = True
            st.rerun()


def render_universe_section(
    rows: list[dict],
    last_refreshed_min: float | None,
    universe_size: int,
) -> None:
    sub = (
        f"top {len(rows)} of {universe_size} · last scanned {last_refreshed_min:.0f}m ago"
        if last_refreshed_min is not None
        else f"top {len(rows)} of {universe_size}"
    )
    # Same single-st.markdown pattern as the watchlist section.
    st.markdown(
        panel_open("Top movers (curated universe)", sub)
        + _scanner_grid_html(rows)
        + panel_close(),
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh scan", key="universe_refresh_btn"):
        st.session_state["universe_force_refresh"] = True
        st.rerun()
