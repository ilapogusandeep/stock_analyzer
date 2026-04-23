"""Reusable UI components for the Advanced view."""

from __future__ import annotations

from typing import Optional

import streamlit as st


def _fmt_price(v: Optional[float]) -> str:
    return f"${v:,.2f}" if v is not None else "N/A"


def _fmt_pct(v: Optional[float], signed: bool = True) -> str:
    if v is None:
        return "N/A"
    sign = "+" if signed and v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _fmt_big_money(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    for unit, scale in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(v) >= scale:
            return f"${v/scale:.2f}{unit}"
    return f"${v:,.0f}"


def _change_class(delta: Optional[float]) -> str:
    if delta is None:
        return "adv-change-flat"
    if delta > 0:
        return "adv-change-up"
    if delta < 0:
        return "adv-change-down"
    return "adv-change-flat"


def _pill_class(direction: str) -> str:
    d = (direction or "").upper()
    if d == "BULLISH":
        return "pill-bull"
    if d == "BEARISH":
        return "pill-bear"
    return "pill-flat"


def header_band(ticker: str, data: dict) -> None:
    """Big top banner: ticker, price, change, recommendation pill, context row."""
    info = data.get("info", {}) or {}
    tech = data.get("tech_data", {}) or {}
    ml = data.get("ml_prediction") or {}
    fund = data.get("fundamental_data", {}) or {}
    earnings = data.get("earnings_data", {}) or {}

    price = tech.get("current_price")
    change_pct = tech.get("price_change_pct")
    change_abs = tech.get("price_change")
    direction = ml.get("direction", "NEUTRAL")

    change_cls = _change_class(change_pct)
    pill_cls = _pill_class(direction)

    market_cap = _fmt_big_money(fund.get("market_cap"))
    sector = info.get("sector") or "—"
    price_target = ml.get("price_target")
    target_txt = _fmt_price(price_target) if price_target else "—"

    earnings_txt = "—"
    if earnings.get("earnings_expected") and earnings.get("next_earnings_date"):
        d = earnings["next_earnings_date"]
        try:
            earnings_txt = f"{d.strftime('%b %d')} ({earnings.get('days_to_earnings', '?')}d)"
        except Exception:
            earnings_txt = "Scheduled"

    change_sign = "▲" if (change_pct or 0) > 0 else ("▼" if (change_pct or 0) < 0 else "■")

    html = f"""
    <div class="adv-header">
      <div style="flex: 1 1 200px;">
        <div class="adv-ticker">{ticker}</div>
        <div class="adv-company">{info.get('longName', '')}</div>
      </div>
      <div style="flex: 1 1 240px;">
        <div class="adv-price">{_fmt_price(price)}</div>
        <div class="{change_cls}">{change_sign} {_fmt_price(abs(change_abs)) if change_abs is not None else ''} &nbsp; {_fmt_pct(change_pct)}</div>
      </div>
      <div style="flex: 0 0 auto;">
        <div class="adv-pill {pill_cls}">{direction}</div>
        <div class="adv-sub">Model signal</div>
      </div>
      <div style="flex: 1 1 360px; display:flex; gap:22px; flex-wrap:wrap;">
        <div><div class="adv-metric-label">Market Cap</div><div class="adv-metric-value" style="font-size:1.05rem">{market_cap}</div></div>
        <div><div class="adv-metric-label">Sector</div><div class="adv-metric-value" style="font-size:1.05rem">{sector}</div></div>
        <div><div class="adv-metric-label">Price Target</div><div class="adv-metric-value" style="font-size:1.05rem">{target_txt}</div></div>
        <div><div class="adv-metric-label">Next Earnings</div><div class="adv-metric-value" style="font-size:1.05rem">{earnings_txt}</div></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: Optional[str] = None, delta_color: str = "neutral") -> None:
    """Dark-themed metric card rendered inside whatever column it's called in."""
    delta_html = ""
    if delta:
        color_cls = {"up": "adv-change-up", "down": "adv-change-down"}.get(delta_color, "adv-change-flat")
        delta_html = f'<div class="adv-metric-delta {color_cls}">{delta}</div>'
    st.markdown(
        f'<div class="adv-metric"><div class="adv-metric-label">{label}</div>'
        f'<div class="adv-metric-value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="adv-h3">{title}</div>', unsafe_allow_html=True)


def probability_gauge(prob: float, label: str, color: str) -> "object":
    """Return a Plotly gauge figure for a 0..1 probability."""
    import plotly.graph_objects as go

    pct = max(0.0, min(1.0, float(prob or 0))) * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"color": "#ffffff", "size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": "rgba(148,163,184,0.08)"},
                    {"range": [33, 66], "color": "rgba(148,163,184,0.14)"},
                    {"range": [66, 100], "color": "rgba(148,163,184,0.22)"},
                ],
            },
            title={"text": label, "font": {"color": "#e0e7ff", "size": 14}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e7ff"},
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def sparkline(series) -> "object":
    """Tiny price sparkline for the header (optional use)."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(y=list(series), mode="lines", line=dict(color="#6366f1", width=2)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    return fig
