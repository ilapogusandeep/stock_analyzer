"""Advanced StockIQ view — dense multi-tab dashboard.

Streamlit auto-discovers files in pages/ relative to the entry script.
This page renders at /Advanced once the user hits sidebar_web.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stockiq.core.analyzer import UniversalStockAnalyzer
from stockiq.ui.components import (
    header_band,
    metric_card,
    probability_gauge,
    section,
)
from stockiq.ui.theme import inject_dark_theme

st.set_page_config(
    page_title="StockIQ · Advanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_dark_theme()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🚀 Advanced Analysis")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    run = st.button("Analyze", type="primary", width="stretch")

    st.markdown("---")
    st.markdown("### Peer Compare")
    peers_raw = st.text_input(
        "Compare with (comma-separated)",
        value="MSFT, GOOGL, NVDA",
        help="Used in the Compare tab",
    )

    st.markdown("---")
    st.caption(
        "Advanced view uses the same UniversalStockAnalyzer backend as the "
        "classic one-pager, with a denser Bloomberg-style layout."
    )


if not run:
    st.markdown(
        '<div class="adv-header"><div class="adv-ticker">🚀 StockIQ Advanced</div>'
        '<div class="adv-company">Enter a ticker on the left and hit Analyze.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

if not ticker:
    st.warning("Enter a ticker symbol.")
    st.stop()


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

with st.spinner(f"Running full analysis on {ticker}…"):
    try:
        data = UniversalStockAnalyzer(ticker).analyze(
            show_charts=True, show_ml=True, show_fundamentals=True, show_sentiment=True
        )
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

if not data:
    st.error(f"No data for {ticker}.")
    st.stop()

hist = data["hist"]
info = data.get("info", {}) or {}
tech = data.get("tech_data", {}) or {}
fund = data.get("fundamental_data", {}) or {}
sent = data.get("sentiment_data", {}) or {}
ml = data.get("ml_prediction") or {}
bt = data.get("backtest_results") or {}
inst = data.get("institutional_data", {}) or {}

header_band(ticker, data)

tabs = st.tabs(
    ["Overview", "Fundamentals", "Technicals", "AI Predictions", "Backtest", "Compare"]
)


# ---------------------------------------------------------------------------
# Helpers for Plotly dark styling
# ---------------------------------------------------------------------------

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#cbd5e1"),
    xaxis=dict(gridcolor="rgba(148,163,255,0.12)"),
    yaxis=dict(gridcolor="rgba(148,163,255,0.12)"),
    margin=dict(l=20, r=20, t=30, b=20),
)


def _safe(v, fmt: str = "{:.2f}", fallback: str = "N/A") -> str:
    if v is None:
        return fallback
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with tabs[0]:
    section("Price · 1 year")
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        )
    )
    for span, color in [(20, "#6366f1"), (50, "#a855f7"), (200, "#f59e0b")]:
        sma = hist["Close"].rolling(span).mean()
        fig.add_trace(
            go.Scatter(x=hist.index, y=sma, name=f"SMA{span}", line=dict(color=color, width=1.5))
        )
    fig.update_layout(**DARK_LAYOUT, height=420, xaxis_rangeslider_visible=False,
                      legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    section("Snapshot")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("P/E", _safe(fund.get("pe_ratio"), "{:.1f}"))
    with c2:
        metric_card("PEG", _safe(fund.get("peg_ratio"), "{:.2f}"))
    with c3:
        metric_card("Beta", _safe(fund.get("beta"), "{:.2f}"))
    with c4:
        rsi = tech.get("rsi") or 0
        rsi_delta = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        rsi_color = "down" if rsi > 70 else "up" if rsi < 30 else "neutral"
        metric_card("RSI", _safe(rsi, "{:.1f}"), delta=rsi_delta, delta_color=rsi_color)
    with c5:
        hi = info.get("fiftyTwoWeekHigh")
        lo = info.get("fiftyTwoWeekLow")
        metric_card("52W Range", f"${lo:,.0f} – ${hi:,.0f}" if hi and lo else "N/A")
    with c6:
        metric_card("Volatility (20d)", _safe(tech.get("volatility_20d"), "{:.1f}%"))

    section("Performance")
    perf_labels = ["1D", "5D", "1M", "3M", "1Y"]
    perf_values = [
        tech.get("performance_1d"), tech.get("performance_5d"),
        tech.get("performance_1m"), tech.get("performance_3m"),
        tech.get("performance_1y"),
    ]
    bar_colors = ["#22c55e" if (v or 0) >= 0 else "#ef4444" for v in perf_values]
    pfig = go.Figure(
        go.Bar(
            x=perf_labels,
            y=[v or 0 for v in perf_values],
            marker_color=bar_colors,
            text=[f"{v:+.2f}%" if v is not None else "—" for v in perf_values],
            textposition="outside",
        )
    )
    pfig.update_layout(**DARK_LAYOUT, height=260, yaxis_title="% return")
    st.plotly_chart(pfig, width="stretch")

    if sent:
        section("Sentiment")
        s1, s2, s3, s4 = st.columns(4)
        with s1: metric_card("Label", sent.get("sentiment_label", "—"))
        with s2: metric_card("Overall", _safe(sent.get("overall_sentiment"), "{:+.3f}"))
        with s3: metric_card("News", _safe(sent.get("news_sentiment"), "{:+.3f}"))
        with s4: metric_card("Confidence", _safe(sent.get("confidence"), "{:.1%}"))


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

with tabs[1]:
    section("Valuation")
    row = st.columns(6)
    items = [
        ("P/E",          fund.get("pe_ratio"),        "{:.1f}"),
        ("Fwd P/E",      fund.get("forward_pe"),      "{:.1f}"),
        ("PEG",          fund.get("peg_ratio"),       "{:.2f}"),
        ("P/B",          fund.get("price_to_book"),   "{:.1f}"),
        ("P/S",          fund.get("price_to_sales"),  "{:.1f}"),
        ("Beta",         fund.get("beta"),            "{:.2f}"),
    ]
    for col, (label, v, f) in zip(row, items):
        with col: metric_card(label, _safe(v, f))

    section("Health & Growth")
    row = st.columns(6)
    items = [
        ("Current Ratio", fund.get("current_ratio"),     "{:.2f}"),
        ("D/E",           fund.get("debt_to_equity"),    "{:.2f}"),
        ("ROE",           fund.get("return_on_equity"),  "{:.1%}"),
        ("Profit Margin", fund.get("profit_margin"),     "{:.1%}"),
        ("Rev Growth",    fund.get("revenue_growth"),    "{:.1%}"),
        ("Market Cap",    fund.get("market_cap"),        None),
    ]
    for col, (label, v, f) in zip(row, items):
        with col:
            if label == "Market Cap":
                mc = v
                mc_str = "N/A"
                if mc:
                    for unit, scale in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
                        if abs(mc) >= scale:
                            mc_str = f"${mc/scale:.2f}{unit}"
                            break
                metric_card(label, mc_str)
            else:
                metric_card(label, _safe(v, f))

    earnings_history = (inst.get("earnings_history") if isinstance(inst, dict) else None) or []
    if earnings_history:
        section("Earnings History")
        try:
            df = pd.DataFrame(earnings_history)
            st.dataframe(df, width="stretch", hide_index=True)
        except Exception:
            st.write(earnings_history)

    holders = (inst.get("top_holders") if isinstance(inst, dict) else None) or []
    if holders:
        section("Top Institutional Holders")
        try:
            df = pd.DataFrame(holders)
            st.dataframe(df, width="stretch", hide_index=True)
        except Exception:
            st.write(holders)


# ---------------------------------------------------------------------------
# Technicals
# ---------------------------------------------------------------------------

with tabs[2]:
    section("Price · SMA · Bollinger")
    close = hist["Close"]
    sma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    upper = sma20 + 2 * sd20
    lower = sma20 - 2 * sd20

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.58, 0.17, 0.25],
        vertical_spacing=0.04,
        subplot_titles=("Price", "Volume", "RSI"),
    )
    fig.add_trace(
        go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"],
                       low=hist["Low"], close=hist["Close"], name="OHLC",
                       increasing_line_color="#22c55e", decreasing_line_color="#ef4444"),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=hist.index, y=sma20, name="SMA20",
                             line=dict(color="#6366f1", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=upper, name="BB upper",
                             line=dict(color="rgba(168,85,247,0.4)", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=lower, name="BB lower",
                             line=dict(color="rgba(168,85,247,0.4)", width=1),
                             fill="tonexty", fillcolor="rgba(168,85,247,0.06)"), row=1, col=1)

    vol_color = ["#22c55e" if c >= o else "#ef4444"
                 for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume",
                         marker_color=vol_color, opacity=0.7), row=2, col=1)

    delta = close.diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    fig.add_trace(go.Scatter(x=hist.index, y=rsi, name="RSI",
                             line=dict(color="#f59e0b", width=1.4)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.6)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(34,197,94,0.6)", row=3, col=1)

    fig.update_layout(**DARK_LAYOUT, height=720, showlegend=True,
                      xaxis_rangeslider_visible=False,
                      legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.06))
    for r in (1, 2, 3):
        fig.update_xaxes(gridcolor="rgba(148,163,255,0.10)", row=r, col=1)
        fig.update_yaxes(gridcolor="rgba(148,163,255,0.10)", row=r, col=1)
    st.plotly_chart(fig, width="stretch")

    section("Key levels")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("SMA 20",  _safe(tech.get("sma_20"), "${:,.2f}"))
    with c2: metric_card("SMA 50",  _safe(tech.get("sma_50"), "${:,.2f}"))
    with c3: metric_card("SMA 200", _safe(tech.get("sma_200"), "${:,.2f}"))
    with c4: metric_card("MACD",    _safe(tech.get("macd"), "{:+.2f}"))


# ---------------------------------------------------------------------------
# AI Predictions
# ---------------------------------------------------------------------------

with tabs[3]:
    if not ml:
        st.info("ML prediction not available for this ticker.")
    else:
        section("Scenario probabilities")
        probs = ml.get("scenario_probabilities", {}) or {}
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(probability_gauge(probs.get("bullish", 0), "Bullish", "#22c55e"),
                            width="stretch")
        with g2:
            st.plotly_chart(probability_gauge(probs.get("neutral", 0), "Neutral", "#94a3b8"),
                            width="stretch")
        with g3:
            st.plotly_chart(probability_gauge(probs.get("bearish", 0), "Bearish", "#ef4444"),
                            width="stretch")

        section("Price scenarios")
        targets = ml.get("scenario_targets", {}) or {}
        current = tech.get("current_price")
        t1, t2, t3, t4 = st.columns(4)
        with t1: metric_card("Current", _safe(current, "${:,.2f}"))
        with t2:
            v = targets.get("bullish")
            delta = f"{(v/current-1)*100:+.1f}%" if v and current else None
            metric_card("Bullish", _safe(v, "${:,.2f}"), delta=delta, delta_color="up")
        with t3:
            v = targets.get("neutral")
            delta = f"{(v/current-1)*100:+.1f}%" if v and current else None
            metric_card("Neutral", _safe(v, "${:,.2f}"), delta=delta, delta_color="neutral")
        with t4:
            v = targets.get("bearish")
            delta = f"{(v/current-1)*100:+.1f}%" if v and current else None
            metric_card("Bearish", _safe(v, "${:,.2f}"), delta=delta, delta_color="down")

        shap_exp = ml.get("shap_explanations") or {}
        top_features = shap_exp.get("top_features") or ml.get("feature_importance") or []
        if top_features:
            section("Top feature contributions")
            try:
                if isinstance(top_features[0], (list, tuple)):
                    labels = [f for f, _ in top_features]
                    vals = [v for _, v in top_features]
                else:
                    labels = [item["feature"] for item in top_features]
                    vals = [item["importance"] for item in top_features]
                fig = go.Figure(go.Bar(
                    x=vals, y=labels, orientation="h",
                    marker=dict(color="#6366f1"),
                    text=[f"{v:.3f}" for v in vals],
                    textposition="outside",
                ))
                fig.update_layout(**DARK_LAYOUT, height=320)
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.caption(f"Could not render features: {e}")

        if shap_exp.get("explanation"):
            section("Model explanation")
            st.markdown(
                f'<div class="adv-metric">{shap_exp["explanation"]}</div>',
                unsafe_allow_html=True,
            )

        acc = ml.get("model_accuracies") or {}
        if acc:
            section("Model accuracies")
            cols = st.columns(len(acc) or 1)
            for col, (name, a) in zip(cols, acc.items()):
                with col: metric_card(name.replace("_", " ").title(),
                                      _safe(a, "{:.1%}"))


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

with tabs[4]:
    if not bt:
        st.info("Backtest not available for this ticker (needs >100 trading days).")
    else:
        section("Strategy performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: metric_card("Win Rate", _safe(bt.get("win_rate"), "{:.1%}"))
        with c2: metric_card("Total Return", _safe(bt.get("strategy_total_return"), "{:+.1%}"))
        with c3: metric_card("Sharpe", _safe(bt.get("sharpe_ratio"), "{:.2f}"))
        with c4: metric_card("Max Drawdown", _safe(bt.get("max_drawdown"), "{:+.1%}"))
        with c5: metric_card("Accuracy", _safe(bt.get("prediction_accuracy"), "{:.1%}"))

        section("High-conviction trades")
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Trades", str(bt.get("high_conf_trades", 0)))
        with c2: metric_card("Win Rate", _safe(bt.get("high_conf_win_rate"), "{:.1%}"))
        with c3: metric_card("Avg Return", _safe(bt.get("high_conf_avg_return"), "{:+.2%}"))
        with c4: metric_card("Total Trades", str(bt.get("total_trades", 0)))

        signals = bt.get("total_signals") or {}
        if signals:
            section("Signal distribution")
            fig = go.Figure(
                go.Pie(
                    labels=["Bullish", "Neutral", "Bearish"],
                    values=[signals.get("bullish", 0), signals.get("neutral", 0),
                            signals.get("bearish", 0)],
                    marker=dict(colors=["#22c55e", "#94a3b8", "#ef4444"]),
                    hole=0.55,
                    textinfo="label+percent",
                )
            )
            fig.update_layout(**DARK_LAYOUT, height=340,
                              legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, width="stretch")

        st.caption(
            f"Validation: {bt.get('validation_method', '—')} · "
            f"Period: {bt.get('backtest_period', '—')} · "
            f"Features: {bt.get('enhanced_features', 0)}"
        )


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

with tabs[5]:
    peers = [t.strip().upper() for t in (peers_raw or "").split(",") if t.strip()]
    peers = [p for p in peers if p and p != ticker]
    if not peers:
        st.info("Add peer tickers in the sidebar (comma-separated) to compare.")
    else:
        with st.spinner(f"Fetching peers: {', '.join(peers)}…"):
            import yfinance as yf
            peer_hist: dict[str, pd.Series] = {ticker: hist["Close"]}
            peer_info: dict[str, dict] = {ticker: info}
            for p in peers:
                try:
                    t = yf.Ticker(p)
                    h = t.history(period="1y")
                    if not h.empty:
                        peer_hist[p] = h["Close"]
                        peer_info[p] = t.info
                except Exception as e:
                    st.warning(f"Failed to load {p}: {e}")

        section("Normalized price · 1 year")
        fig = go.Figure()
        palette = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4"]
        for i, (sym, series) in enumerate(peer_hist.items()):
            norm = series / series.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm, name=sym,
                line=dict(color=palette[i % len(palette)], width=2 if sym == ticker else 1.5),
            ))
        fig.update_layout(**DARK_LAYOUT, height=400, yaxis_title="Indexed to 100",
                          legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

        section("Ratio comparison")
        rows = []
        for sym, i in peer_info.items():
            rows.append({
                "Ticker": sym,
                "Market Cap": f"${i.get('marketCap', 0)/1e9:.1f}B" if i.get("marketCap") else "N/A",
                "P/E":     f"{i.get('trailingPE'):.1f}" if i.get("trailingPE") else "N/A",
                "Fwd P/E": f"{i.get('forwardPE'):.1f}" if i.get("forwardPE") else "N/A",
                "PEG":     f"{i.get('pegRatio'):.2f}" if i.get("pegRatio") else "N/A",
                "P/S":     f"{i.get('priceToSalesTrailing12Months'):.2f}" if i.get("priceToSalesTrailing12Months") else "N/A",
                "P/B":     f"{i.get('priceToBook'):.1f}" if i.get("priceToBook") else "N/A",
                "Beta":    f"{i.get('beta'):.2f}" if i.get("beta") else "N/A",
                "Div Yld": f"{i.get('dividendYield', 0)*100:.2f}%" if i.get("dividendYield") else "—",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
