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


def _resolve_to_ticker(user_input: str) -> str | None:
    """Resolve free-text input to a real ticker symbol.

    Order of attempts:
      1. Already a ticker in our merged universe (POPULAR + SEC +
         searched)? Use it.
      2. yfinance accepts the upper-cased input directly? Use it.
      3. Fuzzy match the input against company names in the merged
         universe (case-insensitive substring + word-boundary).
         Return the unique match, or None if zero or ambiguous.
    """
    if not user_input:
        return None
    raw = user_input.strip()
    upper = raw.upper()
    if not raw:
        return None

    try:
        from stockiq.data.tickers import get_all_tickers
        universe = get_all_tickers()
    except Exception:
        universe = {}

    # Case 1: already a known ticker
    if upper in universe:
        return upper

    # Case 2: yfinance recognizes it (handles tickers we don't carry,
    # e.g. exotic foreign symbols)
    if _ticker_exists(upper):
        return upper

    # Case 3: company-name fuzzy match. Prefer exact-ish matches; allow
    # one ambiguous case to fail loudly.
    needle = raw.lower()
    word_matches: list[str] = []
    sub_matches: list[str] = []
    for tkr, name in universe.items():
        if not name:
            continue
        nm_lower = name.lower()
        if nm_lower == needle:
            return tkr
        # Word-boundary match: input is the start of a word in the name.
        # Cheap and avoids matching "ai" -> every "...ai..." substring.
        if any(w.startswith(needle) for w in nm_lower.split()):
            word_matches.append(tkr)
        elif needle in nm_lower:
            sub_matches.append(tkr)

    if len(word_matches) == 1:
        return word_matches[0]
    if not word_matches and len(sub_matches) == 1:
        return sub_matches[0]
    # Multiple word matches -- prefer the shortest ticker (often the
    # primary listing rather than a class-B share or wholly-owned sub).
    if word_matches:
        return min(word_matches, key=len)
    return None


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
    invalid_tickers = [r["ticker"] for r in rows if r.get("price") is None]
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

    # Bulk-remove all rows that came back without price data (typo'd
    # tickers, delisted symbols, etc.). Only render when there's
    # actually something to clean.
    if invalid_tickers:
        if st.button(
            f"🧹 Clean up {len(invalid_tickers)} invalid",
            key="wl_clean_btn",
            help="Remove every watchlist row whose symbol returned no price data",
        ):
            for tkr in invalid_tickers:
                remove_callback(tkr)
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
                resolved = _resolve_to_ticker(new_ticker)
                if resolved:
                    if resolved.upper() != new_ticker.strip().upper():
                        st.toast(
                            f"Resolved '{new_ticker}' → {resolved}",
                            icon="✅",
                        )
                    add_callback(resolved)
                    st.session_state.pop("wl_add_input", None)
                    st.rerun()
                else:
                    st.error(
                        f"Couldn't resolve '{new_ticker}' to a ticker. "
                        "Try a full symbol — '-USD' for crypto (BTC-USD), "
                        "'^' for indices (^GSPC), '=X' for forex (EURUSD=X)."
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
