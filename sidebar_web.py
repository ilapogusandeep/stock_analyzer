"""StockIQ · dense single-page dashboard.

One viewport-sized layout replacing the old classic and advanced views.
Every section lives on the same page; no tabs, no expanders.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def _import_stockiq_modules():
    """Import stockiq symbols with retries after purging stale sys.modules.

    Streamlit's hot-reload on Python 3.13 occasionally drops a subpackage
    like 'stockiq.data' from sys.modules between reruns while leaving
    'stockiq' itself, which causes KeyError at
    importlib._find_and_load_unlocked. Some retries also hit ImportError
    if a parent package is mid-teardown. Purge every stockiq* entry and
    try again — up to 3 attempts.
    """
    def _do_import():
        from stockiq.core.analyzer import UniversalStockAnalyzer
        from stockiq.core.prediction_log import PredictionLog
        from stockiq.data.options import get_options_flow, get_unusual_activity
        from stockiq.data.tickers import POPULAR_TICKERS
        from stockiq.ui.components import (
            earnings_history_block, external_links, fmt_big_money, fmt_pct,
            fmt_pct_ratio, fmt_price, fmt_ratio, header_band,
            institutional_holders_block, kv_block, news_feed_block,
            options_flow_block, panel_close, panel_open, performance_bars,
            performance_pills_html, probability_scenarios_combined,
            unusual_options_block,
        )
        from stockiq.ui.theme import inject_theme
        return locals()

    def _purge():
        for mod in [m for m in sys.modules
                    if m == "stockiq" or m.startswith("stockiq.")]:
            sys.modules.pop(mod, None)

    last_exc = None
    for attempt in range(3):
        try:
            return _do_import()
        except (KeyError, ImportError) as e:
            if isinstance(e, KeyError) and not str(e).startswith("'stockiq"):
                raise
            last_exc = e
            _purge()
    raise last_exc  # type: ignore[misc]


globals().update(_import_stockiq_modules())

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="StockIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_theme()

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#cbd5e1", size=10),
    margin=dict(l=8, r=8, t=8, b=8),
)

# ---------------------------------------------------------------------------
# Top bar — inline controls (sidebar is hidden via CSS)
# ---------------------------------------------------------------------------

st.markdown('<div class="topbar">', unsafe_allow_html=True)
tb_logo, tb_tkr, tb_fast, tb_run, tb_spacer = st.columns(
    [0.14, 0.36, 0.10, 0.12, 0.28],
    vertical_alignment="bottom",
)
with tb_logo:
    st.markdown(
        '<div class="topbar-logo">📈 StockIQ</div>',
        unsafe_allow_html=True,
    )
with tb_tkr:
    # Searchable dropdown: type a ticker or company name. accept_new_options
    # lets users pick any symbol not in POPULAR_TICKERS.
    ticker_options = sorted(POPULAR_TICKERS.keys())
    default_ticker = "AAPL" if "AAPL" in ticker_options else ticker_options[0]
    ticker_raw = st.selectbox(
        "Ticker / company",
        options=ticker_options,
        index=ticker_options.index(default_ticker),
        format_func=lambda t: f"{t}  —  {POPULAR_TICKERS.get(t.upper(), '')}".rstrip(" —"),
        accept_new_options=True,
        help="Type a ticker (AAPL) or company name (Apple) — any symbol works even if not listed.",
    )
    ticker = (ticker_raw or "").strip().upper()
with tb_fast:
    fast = st.checkbox("Fast mode", value=False,
                       help="Skip ML + backtest")
with tb_run:
    run = st.button("Analyze", type="primary", width="stretch")
st.markdown('</div>', unsafe_allow_html=True)


# Auto-analyze on selection change: once the user has run analysis at
# least once, picking a different ticker should fire off a new run
# without an extra button click. The button is still useful for
# forcing a re-run on the same ticker (e.g. refresh data).
if "last_analyzed_ticker" not in st.session_state:
    st.session_state.last_analyzed_ticker = None

selected_changed = (
    bool(ticker)
    and st.session_state.last_analyzed_ticker is not None
    and ticker != st.session_state.last_analyzed_ticker
)
should_analyze = bool(ticker) and (run or selected_changed)

if not should_analyze:
    st.markdown(
        '<div class="hb"><div><div class="hb-tkr">📈 StockIQ</div>'
        '<div class="hb-co">Pick a ticker above and hit Analyze — afterward, selecting a different ticker auto-runs.</div></div></div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

with st.spinner(f"Analyzing {ticker}…"):
    try:
        data = UniversalStockAnalyzer(ticker).analyze(
            show_charts=True,
            show_ml=not fast,
            show_fundamentals=True,
            show_sentiment=True,
        )
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

if not data:
    st.error(f"No data for {ticker}.")
    st.stop()

# Remember what we analyzed so the next selection change can auto-fire.
st.session_state.last_analyzed_ticker = ticker

# --- Prediction log: record this call + lazily resolve any old pending rows.
#    Stored as parquet under data/predictions.parquet at repo root.
_pred_log = PredictionLog(horizon_days=5)
try:
    _pred_log.log(
        ticker=ticker,
        ml=(data.get("ml_prediction") or {}),
        price=(data.get("tech_data") or {}).get("current_price"),
    )
    _pred_log.resolve_pending()
except Exception:
    # Logging failure should never block the UI.
    pass

# --- Options flow (nearest-expiry put/call ratios + ATM IV).
#    2-3s yfinance call; silently returns {} on failure.
options_flow = get_options_flow(
    ticker, spot=(data.get("tech_data") or {}).get("current_price")
)

# --- Unusual options activity — top strikes with V/OI ≥ 2× ranked by $ premium.
unusual_opts = get_unusual_activity(ticker, top_n=10, min_voi_ratio=2.0)

hist = data["hist"]
info = data.get("info", {}) or {}
tech = data.get("tech_data", {}) or {}
fund = data.get("fundamental_data", {}) or {}
sent = data.get("sentiment_data", {}) or {}
ml = data.get("ml_prediction") or {}
bt = data.get("backtest_results") or {}
inst = data.get("institutional_data", {}) or {}


# ---------------------------------------------------------------------------
# Header band + external links
# ---------------------------------------------------------------------------

header_band(ticker, data)
external_links(ticker, info)


# ---------------------------------------------------------------------------
# Main grid — 3 columns
# ---------------------------------------------------------------------------

c_left, c_mid, c_right = st.columns([0.22, 0.56, 0.22])

# ---- Left column: fundamentals + sentiment ---------------------------------

with c_left:
    kv_block(
        "Valuation",
        [
            ("P/E",        fmt_ratio(fund.get("pe_ratio"), 1)),
            ("Fwd P/E",    fmt_ratio(fund.get("forward_pe"), 1)),
            ("PEG",        fmt_ratio(fund.get("peg_ratio"), 2)),
            ("P/B",        fmt_ratio(fund.get("price_to_book"), 1)),
            ("P/S",        fmt_ratio(fund.get("price_to_sales"), 1)),
            ("Beta",       fmt_ratio(fund.get("beta"), 2)),
        ],
        sub="ratios",
    )

    kv_block(
        "Health & Growth",
        [
            ("Current Ratio", fmt_ratio(fund.get("current_ratio"), 2)),
            ("Debt / Equity", fmt_ratio(fund.get("debt_to_equity"), 2)),
            ("ROE",           fmt_pct_ratio(fund.get("return_on_equity"), 1)),
            ("Profit Margin", fmt_pct_ratio(fund.get("profit_margin"), 1)),
            ("Rev Growth",    fmt_pct_ratio(fund.get("revenue_growth"), 1)),
        ],
        sub="trailing",
    )

    hi = info.get("fiftyTwoWeekHigh")
    lo = info.get("fiftyTwoWeekLow")
    price = tech.get("current_price")
    pct_from_hi = ((price / hi - 1) * 100) if (price and hi) else None
    pct_from_lo = ((price / lo - 1) * 100) if (price and lo) else None
    kv_block(
        "52-Week Range",
        [
            ("High",      f"${hi:,.2f}" if hi else "—"),
            ("Low",       f"${lo:,.2f}" if lo else "—"),
            ("From High", fmt_pct(pct_from_hi, decimals=1) if pct_from_hi is not None else "—"),
            ("From Low",  fmt_pct(pct_from_lo, decimals=1) if pct_from_lo is not None else "—"),
        ],
    )

    # --- Short interest + ownership (positioning) ---
    short_pct = info.get("shortPercentOfFloat")
    short_ratio = info.get("shortRatio")  # days to cover
    shares_short = info.get("sharesShort")
    shares_short_prior = info.get("sharesShortPriorMonth")
    insider_pct = info.get("heldPercentInsiders")
    inst_pct = info.get("heldPercentInstitutions")

    short_delta_str = "—"
    if shares_short and shares_short_prior:
        delta_pct = (shares_short - shares_short_prior) / shares_short_prior * 100
        short_delta_str = f"{delta_pct:+.1f}%"

    kv_block(
        "Positioning",
        [
            ("Short % Float",  fmt_pct_ratio(short_pct, 2) if short_pct else "—"),
            ("Days to Cover",  fmt_ratio(short_ratio, 2) if short_ratio else "—"),
            ("Short MoM Δ",    short_delta_str),
            ("Insider Own",    fmt_pct_ratio(insider_pct, 2) if insider_pct else "—"),
            ("Institutional",  fmt_pct_ratio(inst_pct, 1) if inst_pct else "—"),
        ],
        sub="short · ownership",
        cols=2,
    )

    # Options flow — put/call ratios + ATM IV from nearest expiry
    options_flow_block(options_flow)


# ---- Middle column: price chart + technicals + performance -----------------

with c_mid:
    close = hist["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    sd20 = close.rolling(20).std()
    bb_up = sma20 + 2 * sd20
    bb_dn = sma20 - 2 * sd20

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.14, 0.24],
        vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name=ticker,
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=sma20, name="SMA20",
                             line=dict(color="#6366f1", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=sma50, name="SMA50",
                             line=dict(color="#a855f7", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=sma200, name="SMA200",
                             line=dict(color="#f59e0b", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=bb_up,
                             line=dict(color="rgba(168,85,247,0.35)", width=1),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=bb_dn,
                             line=dict(color="rgba(168,85,247,0.35)", width=1),
                             fill="tonexty", fillcolor="rgba(168,85,247,0.05)",
                             showlegend=False), row=1, col=1)

    vol_color = ["#22c55e" if c >= o else "#ef4444"
                 for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], marker_color=vol_color,
                         opacity=0.7, showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=hist.index, y=rsi,
                             line=dict(color="#f59e0b", width=1.2),
                             showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,68,68,0.5)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(34,197,94,0.5)", row=3, col=1)

    fig.update_layout(
        **DARK_LAYOUT, height=430, xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="top", y=1.04, xanchor="left", x=0.0,
                    font=dict(size=9)),
    )
    for r in (1, 2, 3):
        fig.update_xaxes(gridcolor="rgba(148,163,255,0.08)", row=r, col=1,
                         showgrid=True, tickfont=dict(size=9))
        fig.update_yaxes(gridcolor="rgba(148,163,255,0.08)", row=r, col=1,
                         showgrid=True, tickfont=dict(size=9))
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    st.markdown(
        panel_open(
            f"{ticker} · 1Y price · volume · RSI",
            right_html=performance_pills_html(tech),
        )
        + '<div style="margin:-2px 0 -4px;">',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("</div>" + panel_close(), unsafe_allow_html=True)

    mid_l, mid_r = st.columns(2)
    with mid_l:
        rsi_v = tech.get("rsi", 0) or 0
        rsi_state = ("🔴 Overbought" if rsi_v > 70 else
                     "🟢 Oversold" if rsi_v < 30 else "🟡 Neutral")
        macd_v = tech.get("macd")
        macd_state = (
            "🟢 Bullish" if (macd_v is not None and macd_v > 0)
            else "🔴 Bearish" if (macd_v is not None and macd_v < 0)
            else "—"
        )
        rsi_str = f"{rsi_v:.1f} {rsi_state.split(' ', 1)[1]}" if rsi_v else "—"
        macd_str = (
            f"{macd_v:+.2f} {macd_state.split(' ', 1)[1]}"
            if macd_v is not None else "—"
        )
        kv_block(
            "Technical",
            [
                ("RSI (14)",       rsi_str),
                ("MACD",           macd_str),
                ("SMA 20",         f"${tech.get('sma_20', 0):,.2f}" if tech.get('sma_20') else "—"),
                ("SMA 50",         f"${tech.get('sma_50', 0):,.2f}" if tech.get('sma_50') else "—"),
                ("SMA 200",        f"${tech.get('sma_200', 0):,.2f}" if tech.get('sma_200') else "—"),
                ("Volatility 20d", f"{tech.get('volatility_20d', 0):.1f}%" if tech.get('volatility_20d') else "—"),
            ],
            sub="14d RSI · 12/26/9 MACD",
            cols=2,
        )
    with mid_r:
        # Top hedge fund / institutional holders — latest 13F filings
        top_holders = (
            (inst.get("institutional_holders") or {}).get("top_holders", [])
            if isinstance(inst, dict) else []
        )
        institutional_holders_block(top_holders, max_items=8)

    # --- Analyst consensus + multi-source sentiment (merged) ----------------
    rec_key = info.get("recommendationKey", "") or ""
    rec_emoji = {
        "strong_buy": "🟢", "buy": "🟢",
        "hold": "🟡",
        "underperform": "🔴", "sell": "🔴", "strong_sell": "🔴",
    }.get(rec_key.lower(), "⚪")
    rec_mean = info.get("recommendationMean")
    n_analysts = info.get("numberOfAnalystOpinions")
    tgt_mean = info.get("targetMeanPrice")
    tgt_low = info.get("targetLowPrice")
    tgt_high = info.get("targetHighPrice")
    current_px = tech.get("current_price")

    implied_upside = "—"
    if tgt_mean and current_px:
        implied_upside = fmt_pct((tgt_mean / current_px - 1) * 100, decimals=1)

    sent_label = (sent or {}).get("sentiment_label", "NEUTRAL") if sent else "—"
    sent_emoji = {"POSITIVE": "🟢", "NEGATIVE": "🔴"}.get(sent_label, "⚪")

    rows = []
    if n_analysts or tgt_mean or rec_mean:
        rows += [
            ("Rating",         f"{rec_emoji} {rec_key.replace('_', ' ').upper() or '—'}"),
            ("Score (1–5)",    fmt_ratio(rec_mean, 2) if rec_mean else "—"),
            ("Analysts",       str(int(n_analysts)) if n_analysts else "—"),
            ("Target Mean",    fmt_price(tgt_mean) if tgt_mean else "—"),
            ("Target Range",   (f"{fmt_price(tgt_low)} – {fmt_price(tgt_high)}"
                                if (tgt_low and tgt_high) else "—")),
            ("Implied Upside", implied_upside),
        ]
    if sent:
        rows += [
            ("Sentiment",   f"{sent_emoji} {sent_label}"),
            ("Overall",     f"{sent.get('overall_sentiment', 0):+.3f}"),
            ("News score",  f"{sent.get('news_sentiment', 0):+.3f}"),
            ("Confidence",  fmt_pct_ratio(sent.get("confidence"), 0)),
        ]
    if rows:
        kv_block(
            "Analyst & sentiment", rows,
            sub="12m targets · multi-source",
            cols=2,
        )


# ---- Right column: AI predictions + backtest --------------------------------

with c_right:
    if ml:
        probability_scenarios_combined(
            ml.get("scenario_probabilities", {}) or {},
            ml.get("scenario_targets", {}) or {},
            tech.get("current_price"),
        )
    else:
        st.markdown(
            panel_open("AI scenario & targets")
            + "<div class='sent-label'>Fast mode — ML skipped.</div>"
            + panel_close(),
            unsafe_allow_html=True,
        )

    # Earnings history (from yfinance via institutional_data)
    earnings_hist = (
        (inst.get("earnings_data") or {}).get("history", [])
        if isinstance(inst, dict) else []
    )
    earnings_history_block(earnings_hist)

    # Unusual options — top strikes with volume >> open interest
    unusual_options_block(unusual_opts)

    # News feed — per-article headlines + sentiment that fed the model.
    news_feed_block(sent.get("articles") or [], max_items=20)


st.markdown(
    '<div class="footer">StockIQ · data via yfinance + RSS news · '
    'not investment advice</div>',
    unsafe_allow_html=True,
)
