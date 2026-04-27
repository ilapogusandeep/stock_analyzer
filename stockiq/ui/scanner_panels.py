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


def render_scanner_table(rows: Iterable[dict], remove_callback=None) -> None:
    """Render a scanner result row set as an HTML grid.

    ``remove_callback`` (optional) — when provided, a small × button is
    rendered at the end of each row that calls remove_callback(ticker)
    and reruns. Used for the watchlist; the universe section omits it.

    Rows are also clickable: ticker pill links to ?ticker=<sym> which
    the top-of-script handler uses to swap the analyze view.
    """
    rows = list(rows)
    if not rows:
        st.markdown(
            '<div class="sent-label">No tickers to scan yet.</div>',
            unsafe_allow_html=True,
        )
        return

    # Column header row
    header_cells = "".join(
        f'<div class="sc-th">{label}</div>' for _, label in SCANNER_HEADERS
    )
    if remove_callback is not None:
        header_cells += '<div class="sc-th"></div>'  # × column

    body = ""
    for r in rows:
        ticker = r.get("ticker", "—")

        price = r.get("price")
        chg = r.get("change_1d")
        chg_cls = _cls(chg) if chg is not None else "flat"
        chg_str = fmt_pct(chg * 100, decimals=2) if chg is not None else "—"

        unusual = r.get("unusual_count", 0) or 0
        agg = r.get("aggressor_net", 0) or 0
        agg_cls = _cls(agg) if agg else "flat"
        velocity = r.get("news_velocity") or 0
        score = r.get("score") or 0
        bias = r.get("bias", "NEUTRAL")

        cells = [
            f'<a class="rs-pill" href="?ticker={ticker}" target="_self">{ticker}</a>',
            _fmt_price(price),
            f'<span class="{chg_cls}">{chg_str}</span>',
            str(unusual),
            f'<span class="{agg_cls}">{_fmt_signed(agg)}</span>',
            f"{velocity:.1f}×",
            f'<span class="sc-score">{score:.0f}</span>',
            _bias_pill(bias),
        ]
        row_html = "".join(f'<div class="sc-td">{c}</div>' for c in cells)
        if remove_callback is not None:
            row_html += '<div class="sc-td sc-rm-cell"></div>'
        body += f'<div class="sc-row">{row_html}</div>'

    cols_count = len(SCANNER_HEADERS) + (1 if remove_callback is not None else 0)
    grid_template = (
        "60px 70px 64px 64px 64px 60px 50px 80px"
        + (" 28px" if remove_callback is not None else "")
    )
    st.markdown(
        f'<div class="sc-grid" style="grid-template-columns: {grid_template};">'
        f'{header_cells}{body}</div>',
        unsafe_allow_html=True,
    )

    # Remove buttons rendered as a parallel block underneath -- Streamlit
    # buttons can't live inside an arbitrary HTML grid, so we surface
    # the removal control as a row of small × buttons keyed to the
    # ticker order. Compact and works.
    if remove_callback is not None:
        cols = st.columns(min(len(rows), 10) or 1)
        for i, r in enumerate(rows):
            with cols[i % 10]:
                if st.button(
                    f"× {r['ticker']}",
                    key=f"rm_{r['ticker']}",
                    help=f"Remove {r['ticker']} from watchlist",
                ):
                    remove_callback(r["ticker"])
                    st.rerun()


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

    st.markdown(panel_open("Watchlist", sub), unsafe_allow_html=True)
    render_scanner_table(rows, remove_callback=remove_callback)
    st.markdown(panel_close(), unsafe_allow_html=True)

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
                add_callback(new_ticker)
                st.session_state.pop("wl_add_input", None)
                st.rerun()
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
    st.markdown(panel_open("Top movers (curated universe)", sub), unsafe_allow_html=True)
    render_scanner_table(rows, remove_callback=None)
    st.markdown(panel_close(), unsafe_allow_html=True)

    if st.button("🔄 Refresh scan", key="universe_refresh_btn"):
        st.session_state["universe_force_refresh"] = True
        st.rerun()
