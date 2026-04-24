"""StockIQ · dense single-page dashboard.

One viewport-sized layout replacing the old classic and advanced views.
Every section lives on the same page; no tabs, no expanders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stockiq.core.analyzer import UniversalStockAnalyzer
from stockiq.data.tickers import POPULAR_TICKERS
from stockiq.ui.components import (
    earnings_history_block,
    external_links,
    fmt_big_money,
    fmt_pct,
    fmt_pct_ratio,
    fmt_price,
    fmt_ratio,
    header_band,
    kv_block,
    panel_close,
    panel_open,
    performance_bars,
    probability_scenarios_combined,
)
from stockiq.ui.theme import inject_theme

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

    if sent:
        label = sent.get("sentiment_label", "NEUTRAL")
        label_emoji = {"POSITIVE": "🟢", "NEGATIVE": "🔴"}.get(label, "⚪")
        kv_block(
            "Sentiment",
            [
                ("Label",      f"{label_emoji} {label}"),
                ("Overall",    f"{sent.get('overall_sentiment', 0):+.3f}"),
                ("News",       f"{sent.get('news_sentiment', 0):+.3f}"),
                ("Social",     f"{sent.get('social_sentiment', 0):+.3f}"),
                ("News Count", str(sent.get("news_count", 0))),
                ("Confidence", fmt_pct_ratio(sent.get("confidence"), 0)),
            ],
            sub="multi-source",
        )


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
        panel_open(f"{ticker} · 1Y price · volume · RSI", "SMA20/50/200 · BB(20,2)")
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
        kv_block(
            "Technical",
            [
                ("RSI (14)",       f"{rsi_v:.1f}"),
                ("State",          rsi_state),
                ("MACD",           f"{macd_v:+.2f}" if macd_v is not None else "—"),
                ("Signal",         macd_state),
                ("SMA 20",         f"${tech.get('sma_20', 0):,.2f}" if tech.get('sma_20') else "—"),
                ("SMA 50",         f"${tech.get('sma_50', 0):,.2f}" if tech.get('sma_50') else "—"),
                ("SMA 200",        f"${tech.get('sma_200', 0):,.2f}" if tech.get('sma_200') else "—"),
                ("Volatility 20d", f"{tech.get('volatility_20d', 0):.1f}%" if tech.get('volatility_20d') else "—"),
            ],
            sub="14d RSI · 12/26/9 MACD",
        )
    with mid_r:
        performance_bars(tech)


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

    if ml:
        shap_exp = ml.get("shap_explanations") or {}
        top = shap_exp.get("top_features") or ml.get("feature_importance") or []
        if top:
            try:
                if isinstance(top[0], (list, tuple)):
                    pairs = [(str(f), float(v)) for f, v in top[:5]]
                else:
                    pairs = [(str(item["feature"]), float(item["importance"])) for item in top[:5]]
                rows = [(k, f"{v:.3f}") for k, v in pairs]
                kv_block("Top features", rows, sub="importance")
            except Exception:
                pass

    if bt:
        kv_block(
            "Backtest",
            [
                ("Win Rate",     fmt_pct_ratio(bt.get("win_rate"), 1)),
                ("Total Return", fmt_pct_ratio(bt.get("strategy_total_return"), 1)),
                ("Sharpe",       fmt_ratio(bt.get("sharpe_ratio"), 2)),
                ("Max DD",       fmt_pct_ratio(bt.get("max_drawdown"), 1)),
                ("Accuracy",     fmt_pct_ratio(bt.get("prediction_accuracy"), 1)),
                ("Trades",       str(bt.get("total_trades", 0))),
                ("HighConf Trd", str(bt.get("high_conf_trades", 0))),
                ("HC Win Rate",  fmt_pct_ratio(bt.get("high_conf_win_rate"), 1)),
            ],
            sub=f"{bt.get('backtest_period', '—')} · {bt.get('enhanced_features', 0)} feat",
        )
    else:
        st.markdown(
            panel_open("Backtest")
            + "<div class='sent-label'>—</div>"
            + panel_close(),
            unsafe_allow_html=True,
        )


st.markdown(
    '<div class="footer">StockIQ · data via yfinance + RSS news · '
    'not investment advice</div>',
    unsafe_allow_html=True,
)
