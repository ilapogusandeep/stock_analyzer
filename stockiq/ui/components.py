"""Compact HTML building blocks for the dense single-page UI."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Tuple

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def fmt_price(v: Optional[float]) -> str:
    return f"${v:,.2f}" if v is not None else "—"


def fmt_pct(v: Optional[float], signed: bool = True, decimals: int = 2) -> str:
    if v is None:
        return "—"
    sign = "+" if signed and v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"


def fmt_ratio(v: Optional[float], decimals: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{v:.{decimals}f}"
    except Exception:
        return str(v)


def fmt_pct_ratio(v: Optional[float], decimals: int = 1) -> str:
    """Format a 0..1 fraction as a percent."""
    if v is None:
        return "—"
    return f"{v*100:.{decimals}f}%"


def fmt_big_money(v: Optional[float]) -> str:
    if v is None or v == 0:
        return "—"
    for unit, scale in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(v) >= scale:
            return f"${v/scale:.2f}{unit}"
    return f"${v:,.0f}"


def _cls(v: Optional[float]) -> str:
    if v is None:
        return "flat"
    return "up" if v > 0 else ("down" if v < 0 else "flat")


# ---------------------------------------------------------------------------
# Header band
# ---------------------------------------------------------------------------

def header_band(ticker: str, data: dict) -> None:
    info = data.get("info", {}) or {}
    tech = data.get("tech_data", {}) or {}
    fund = data.get("fundamental_data", {}) or {}
    ml = data.get("ml_prediction") or {}
    earnings = data.get("earnings_data", {}) or {}

    price = tech.get("current_price")
    change_pct = tech.get("price_change_pct")
    change_abs = tech.get("price_change")
    direction = (ml.get("direction") or "NEUTRAL").upper()

    pill_cls = {"BULLISH": "pill-bull", "BEARISH": "pill-bear"}.get(direction, "pill-flat")
    change_cls = _cls(change_pct)
    arrow = "▲" if (change_pct or 0) > 0 else ("▼" if (change_pct or 0) < 0 else "■")

    target = ml.get("price_target")
    target_pct = None
    if target and price:
        target_pct = (target / price - 1) * 100

    earnings_txt = "—"
    if earnings.get("earnings_expected") and earnings.get("next_earnings_date"):
        try:
            earnings_txt = f"{earnings['next_earnings_date'].strftime('%b %d')} · {earnings.get('days_to_earnings', '?')}d"
        except Exception:
            earnings_txt = "scheduled"

    company = info.get("longName") or info.get("shortName") or ""
    sector = info.get("sector") or "—"

    # One-line business description. yfinance returns a long paragraph
    # in longBusinessSummary -- trim to the first "real" sentence (or a
    # word-boundary truncation at ~180 chars) so it fits a single line
    # under the company name. Skip short "sentences" that are just the
    # company name ending in "Inc.", "Corp.", etc.
    summary = (info.get("longBusinessSummary") or "").strip()
    desc = ""
    if summary:
        # Avoid splitting on common abbreviations' periods: require the
        # candidate sentence to be at least 60 chars long before accepting.
        search_from = 0
        while search_from < len(summary):
            i = -1
            for pos in range(search_from, len(summary) - 1):
                if summary[pos] in ".!?" and summary[pos + 1] == " ":
                    i = pos
                    break
            if i < 0:
                break
            candidate = summary[: i + 1]
            if len(candidate) >= 60:
                desc = candidate
                break
            search_from = i + 2
        if not desc:
            desc = summary
        if len(desc) > 180:
            cut = desc[:177].rsplit(" ", 1)[0]
            desc = cut.rstrip(",.;: ") + "…"
        desc = desc.replace("<", "&lt;").replace(">", "&gt;")

    desc_html = f'<div class="hb-desc">{desc}</div>' if desc else ""

    html = f"""
    <div class="hb">
      <div>
        <div class="hb-tkr">{ticker}</div>
        <div class="hb-co">{company}</div>
        {desc_html}
      </div>
      <div>
        <div class="hb-px">{fmt_price(price)}</div>
        <div class="hb-chg {change_cls}">{arrow} {fmt_price(abs(change_abs)) if change_abs is not None else '—'} &nbsp; {fmt_pct(change_pct)}</div>
      </div>
      <div>
        <span class="pill {pill_cls}">{direction}</span>
        <div class="hb-sub">model signal</div>
      </div>
      <div class="hb-ctx">
        <div>
          <div class="hb-ctx-l">Market Cap</div>
          <div class="hb-ctx-v">{fmt_big_money(fund.get('market_cap'))}</div>
        </div>
        <div>
          <div class="hb-ctx-l">Sector</div>
          <div class="hb-ctx-v" style="font-size:0.8rem;font-weight:500;">{sector}</div>
        </div>
        <div>
          <div class="hb-ctx-l">Price Target</div>
          <div class="hb-ctx-v">{fmt_price(target)}{f' <span class="{_cls(target_pct)}" style="font-size:0.7rem;font-weight:500;">({fmt_pct(target_pct, decimals=1)})</span>' if target_pct is not None else ''}</div>
        </div>
        <div>
          <div class="hb-ctx-l">Next Earnings</div>
          <div class="hb-ctx-v" style="font-size:0.8rem;">{earnings_txt}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# External links row
# ---------------------------------------------------------------------------

def external_links(ticker: str, info: dict) -> None:
    """Small row of click-through research links for the selected ticker."""
    website = info.get("website") or ""
    website_link = (
        f'<a class="ext-link" href="{website}" target="_blank" rel="noopener">🌐 Website</a>'
        if website else ""
    )
    links = [
        ("Yahoo Finance",  f"https://finance.yahoo.com/quote/{ticker}"),
        ("Finviz",         f"https://finviz.com/quote.ashx?t={ticker}"),
        ("TradingView",    f"https://www.tradingview.com/symbols/{ticker}/"),
        ("Stock Analysis", f"https://stockanalysis.com/stocks/{ticker.lower()}/"),
        ("SEC Filings",    f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K"),
        ("Google News",    f"https://www.google.com/search?q={ticker}+stock&tbm=nws"),
        ("Seeking Alpha",  f"https://seekingalpha.com/symbol/{ticker}"),
    ]
    anchors = "".join(
        f'<a class="ext-link" href="{url}" target="_blank" rel="noopener">{label}</a>'
        for label, url in links
    )
    st.markdown(
        f'<div class="ext-row"><span class="ext-label">Sources</span>{anchors}{website_link}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Panels / stats
# ---------------------------------------------------------------------------

def panel_open(title: str, sub: str = "", right_html: str = "") -> str:
    """Open a panel container with a header row.

    ``right_html`` (when provided) renders in the header's right slot
    instead of the text ``sub``, letting callers inject rich content like
    performance pills alongside the title.
    """
    if right_html:
        right = right_html
    else:
        right = f'<span class="panel-h-sub">{sub}</span>' if sub else ""
    return f'<div class="panel"><div class="panel-h"><span>{title}</span>{right}</div>'


def panel_close() -> str:
    return "</div>"


def kv_block(
    title: str,
    rows: Iterable[Tuple[str, str]],
    sub: str = "",
    cols: int = 1,
) -> None:
    """Render a titled panel with a key:value list.

    ``cols=1`` is the default single-column layout (label | value).
    ``cols=2`` packs rows into a four-column grid (label | value | label
    | value) so tall panels use roughly half the vertical space.
    """
    body = "".join(
        f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in rows
    )
    grid_cls = "kv kv-2" if cols == 2 else "kv"
    html = panel_open(title, sub) + f'<div class="{grid_cls}">{body}</div>' + panel_close()
    st.markdown(html, unsafe_allow_html=True)


def probability_bars(probs: dict, title: str = "Scenario probability") -> None:
    """Horizontal bullish/neutral/bearish probability bars."""
    bull = max(0.0, min(1.0, float(probs.get("bullish", 0) or 0)))
    neut = max(0.0, min(1.0, float(probs.get("neutral", 0) or 0)))
    bear = max(0.0, min(1.0, float(probs.get("bearish", 0) or 0)))

    rows = [
        ("Bullish", bull, "pb-bull"),
        ("Neutral", neut, "pb-neut"),
        ("Bearish", bear, "pb-bear"),
    ]
    body = ""
    for label, v, cls in rows:
        pct = v * 100
        body += (
            f'<div class="pb-row">'
            f'<div class="pb-label">{label}</div>'
            f'<div class="pb-bar"><div class="pb-fill {cls}" style="width:{pct:.1f}%"></div></div>'
            f'<div class="pb-val">{pct:.1f}%</div>'
            f'</div>'
        )
    st.markdown(panel_open(title) + body + panel_close(), unsafe_allow_html=True)


def probability_scenarios_combined(
    probs: dict,
    targets: dict,
    current: Optional[float],
    title: str = "AI scenario & targets",
    sub: str = "12m horizon",
    **_extra,  # tolerate forward-compat kwargs without breaking old callers
) -> None:
    """One panel that shows probability bar + target price + % delta per
    direction, saving a slot in the right column for other content.

    ``title``/``sub`` let callers reuse the panel for different horizons
    (e.g. "AI · 1 week" / "5d horizon · accuracy 58%")."""
    rows_spec = [
        ("Bullish", "bullish", "pb-bull"),
        ("Neutral", "neutral", "pb-neut"),
        ("Bearish", "bearish", "pb-bear"),
    ]
    body = ""
    for label, key, cls in rows_spec:
        p = max(0.0, min(1.0, float(probs.get(key, 0) or 0)))
        tgt = targets.get(key)
        if tgt is None or current is None:
            tgt_str, delta_str, delta_cls = "—", "—", "flat"
        else:
            tgt_str = fmt_price(tgt)
            delta = (tgt / current - 1) * 100
            delta_str = fmt_pct(delta, decimals=1)
            delta_cls = _cls(delta)
        body += (
            f'<div class="pbx-row">'
            f'<div class="pbx-label">{label}</div>'
            f'<div class="pb-bar"><div class="pb-fill {cls}" style="width:{p*100:.1f}%"></div></div>'
            f'<div class="pbx-pct">{p*100:.1f}%</div>'
            f'<div class="pbx-tgt">{tgt_str}</div>'
            f'<div class="pbx-delta {delta_cls}">{delta_str}</div>'
            f'</div>'
        )
    st.markdown(
        panel_open(title, sub) + body + panel_close(),
        unsafe_allow_html=True,
    )


def regime_3m_block(regime: dict) -> None:
    """One-line 3-month regime indicator. Sparse on purpose — at this
    horizon a single price target on a single ticker is mostly noise,
    so we just label the regime and show the model's accuracy.
    """
    if not regime:
        return

    label = regime.get("regime", "—")
    cls = {"BULLISH": "up", "BEARISH": "down", "SIDEWAYS": "flat"}.get(label, "flat")
    emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "SIDEWAYS": "🟡"}.get(label, "⚪")
    conf = regime.get("confidence")
    conf_str = f"{conf*100:.0f}%" if isinstance(conf, (int, float)) else "—"

    accs = list((regime.get("model_accuracies") or {}).values())
    accs = [a for a in accs if a is not None]
    sub = "63d horizon"
    if accs:
        sub += f" · accuracy {sum(accs)/len(accs)*100:.0f}%"

    probs = regime.get("probabilities") or {}
    bull_p = probs.get("BULLISH", 0) * 100
    side_p = probs.get("SIDEWAYS", 0) * 100
    bear_p = probs.get("BEARISH", 0) * 100

    body = (
        f'<div class="rg-row">'
        f'<span class="rg-label {cls}">{emoji} {label}</span>'
        f'<span class="rg-conf">conf {conf_str}</span>'
        f'</div>'
        f'<div class="rg-mix">'
        f'<span class="rg-mix-l">B {bull_p:.0f}%</span>'
        f'<span class="rg-mix-l">S {side_p:.0f}%</span>'
        f'<span class="rg-mix-l">D {bear_p:.0f}%</span>'
        f'</div>'
    )
    st.markdown(
        panel_open("AI · 3 month regime", sub) + body + panel_close(),
        unsafe_allow_html=True,
    )


def news_feed_block(articles: list, max_items: int = 20) -> None:
    """Show the actual headlines + per-article sentiment score that fed the
    model. Transparency so users can see the 'evidence'. Rendered inside a
    scrollable container so many headlines fit without blowing up height.

    Articles are sorted latest-first so the visible rows (before scrolling)
    are the most recent.
    """
    if not articles:
        return

    def _sort_key(a):
        ts = a.get("published_at")
        try:
            return -float(ts) if ts else 0.0
        except (TypeError, ValueError):
            return 0.0

    articles = sorted(articles, key=_sort_key)

    rows = ""
    for a in articles[:max_items]:
        score = a.get("score") or 0.0
        cls = _cls(score)
        emoji = "🟢" if score > 0.1 else "🔴" if score < -0.1 else "⚪"
        src = (a.get("source") or "").strip()
        src_html = f' <span class="nf-src">{src[:22]}</span>' if src else ""
        headline = (a.get("headline") or "").strip()[:140]
        # Escape minimal HTML to avoid broken markup from headlines
        headline = (headline.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;"))
        rows += (
            f'<div class="nf-row">'
            f'<span class="nf-dot">{emoji}</span>'
            f'<span class="nf-h">{headline}{src_html}</span>'
            f'<span class="nf-score {cls}">{score:+.2f}</span>'
            f'</div>'
        )
    shown = min(len(articles), max_items)
    # Count unique source names so users see the breadth at a glance.
    sources_seen = len({(a.get("source") or "").strip() for a in articles[:max_items]
                        if (a.get("source") or "").strip()})
    # Freshness: how old is the newest article?
    import time as _time
    now = _time.time()
    recent_ts = [a.get("published_at") for a in articles[:max_items]
                 if a.get("published_at")]
    freshness = ""
    if recent_ts:
        age_s = now - max(recent_ts)
        if age_s < 3600:
            freshness = f" · freshest {int(age_s/60)}m"
        elif age_s < 86400:
            freshness = f" · freshest {int(age_s/3600)}h"
        else:
            freshness = f" · freshest {int(age_s/86400)}d"
    sub = (f"{shown} headlines · {sources_seen} sources{freshness}" if sources_seen
           else f"{shown} headlines{freshness}")
    st.markdown(
        panel_open("News feed", sub)
        + f'<div class="nf-scroll">{rows}</div>'
        + panel_close(),
        unsafe_allow_html=True,
    )


def options_flow_block(flow: dict) -> None:
    """Near-term options summary as a horizontal pill strip.

    Dropping the 6-row kv layout in favor of pills saves ~100px of
    column height without losing any of the aggregates.
    """
    if not flow:
        return

    tilt = flow.get("tilt") or "—"
    tilt_cls = {"BULLISH": "up", "BEARISH": "down"}.get(tilt, "flat")

    def _fmt_vol(v: Optional[float]) -> str:
        if v is None or v == 0:
            return "—"
        if v >= 1e6:
            return f"{v/1e6:.2f}M"
        if v >= 1e3:
            return f"{v/1e3:.1f}K"
        return f"{int(v)}"

    dte = flow.get("days_to_expiry")
    expiry_label = flow.get("expiry") or "—"
    if dte is not None:
        expiry_label = f"{flow.get('expiry')} · {dte}d"

    atm_iv = flow.get("atm_iv")
    iv_str = f"{atm_iv*100:.1f}%" if atm_iv else "—"

    pills = [
        ("P/C VOL", fmt_ratio(flow.get("put_call_vol_ratio"), 2), ""),
        ("P/C OI",  fmt_ratio(flow.get("put_call_oi_ratio"), 2), ""),
        ("CALL",    _fmt_vol(flow.get("call_volume")), ""),
        ("PUT",     _fmt_vol(flow.get("put_volume")), ""),
        ("ATM IV",  iv_str, ""),
        ("TILT",    tilt, tilt_cls),
    ]
    body = '<div class="of-pills">'
    for lbl, val, cls in pills:
        body += (
            f'<span class="ps-pill {cls}">'
            f'<span class="ps-lbl">{lbl}</span>'
            f'<span class="ps-val">{val}</span></span>'
        )
    body += '</div>'

    st.markdown(
        panel_open("Options flow", f"expiry {expiry_label}")
        + body + panel_close(),
        unsafe_allow_html=True,
    )


def unusual_options_block(rows: list) -> None:
    """Top unusual-activity strikes across near expiries, ranked by $ premium."""
    if not rows:
        return

    def _fmt_flow(v: float) -> str:
        if v >= 1e6:
            return f"${v/1e6:.1f}M"
        if v >= 1e3:
            return f"${v/1e3:.0f}K"
        return f"${int(v)}"

    def _fmt_exp(e: str) -> str:
        try:
            dt = datetime.strptime(e, "%Y-%m-%d")
            return dt.strftime("%b%d")
        except Exception:
            return e

    def _agg_badge(agg: str) -> str:
        cls = {"BUY": "up", "SELL": "down", "MID": "flat"}.get(agg, "flat")
        label = {"BUY": "B", "SELL": "S", "MID": "M"}.get(agg, "·")
        return f'<span class="uo-agg {cls}">{label}</span>'

    html = ""
    for r in rows:
        side = r["side"]
        side_cls = "up" if side == "C" else "down"
        side_lbl = "CALL" if side == "C" else "PUT"
        cluster_mark = '<span class="uo-cluster">🔗</span>' if r.get("cluster") else ""
        html += (
            f'<div class="uo-row">'
            f'<span class="uo-side {side_cls}">{side_lbl}</span>'
            f'<span class="uo-strike">{cluster_mark}${r["strike"]:.0f}</span>'
            f'<span class="uo-exp">{_fmt_exp(r["expiry"])}</span>'
            f'<span class="uo-voi">{r["voi_ratio"]:.1f}×</span>'
            f'<span class="uo-flow">{_fmt_flow(r["premium_flow"])}</span>'
            f'{_agg_badge(r.get("aggressor", "—"))}'
            f'</div>'
        )
    st.markdown(
        panel_open("Unusual options",
                   "V/OI ≥ 2× · B/S/M = bid-ask aggressor · 🔗 = cluster")
        + f'<div class="uo-scroll">{html}</div>'
        + panel_close(),
        unsafe_allow_html=True,
    )


def institutional_holders_block(holders: list, max_items: int = 8) -> None:
    """Top hedge funds / institutions holding the stock, latest filing first."""
    if not holders:
        return

    def _fmt_value(v) -> str:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "—"
        if v >= 1e9:
            return f"${v/1e9:.1f}B"
        if v >= 1e6:
            return f"${v/1e6:.0f}M"
        if v >= 1e3:
            return f"${v/1e3:.0f}K"
        return f"${v:.0f}"

    def _fmt_pct_held(v) -> str:
        try:
            v = float(v)
            return f"{v*100:.1f}%" if v < 1 else f"{v:.1f}%"
        except (TypeError, ValueError):
            return "—"

    def _fmt_change(v) -> str:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "—"
        sign = "+" if v > 0 else ""
        pct = v * 100 if abs(v) < 1 else v
        return f"{sign}{pct:.1f}%"

    def _fmt_date(d) -> str:
        try:
            import pandas as pd_
            ts = pd_.to_datetime(d)
            return ts.strftime("%b%Y")
        except Exception:
            return str(d)[:10] if d else "—"

    def _sort_key(h):
        try:
            import pandas as pd_
            return pd_.to_datetime(h.get("date_reported"))
        except Exception:
            return pd.Timestamp.min

    sorted_holders = sorted(holders, key=_sort_key, reverse=True)[:max_items]

    html = ""
    for h in sorted_holders:
        name = str(h.get("name") or "—")[:28]
        change = h.get("pct_change")
        try:
            change_val = float(change)
            change_cls = "up" if change_val > 0 else "down" if change_val < 0 else "flat"
        except (TypeError, ValueError):
            change_cls = "flat"
        html += (
            f'<div class="ih-row">'
            f'<span class="ih-name">{name}</span>'
            f'<span class="ih-val">{_fmt_value(h.get("value"))}</span>'
            f'<span class="ih-pct">{_fmt_pct_held(h.get("percent_out"))}</span>'
            f'<span class="ih-chg {change_cls}">{_fmt_change(h.get("pct_change"))}</span>'
            f'<span class="ih-date">{_fmt_date(h.get("date_reported"))}</span>'
            f'</div>'
        )

    st.markdown(
        panel_open("Top holders", f"latest 13F · {min(len(holders), max_items)} funds")
        + f'<div class="ih-scroll">{html}</div>'
        + panel_close(),
        unsafe_allow_html=True,
    )


def track_record_block(summary: dict) -> None:
    """Render overall hit-rate + calibration buckets + recent predictions."""
    total = summary.get("total", 0)
    if total == 0:
        st.markdown(
            panel_open("Prediction track record", "builds over time")
            + '<div class="sent-label">No predictions logged yet. '
              'Run analysis on a few tickers — each call is tracked and '
              'scored automatically after the resolution horizon.</div>'
            + panel_close(),
            unsafe_allow_html=True,
        )
        return

    resolved = summary.get("resolved", 0)
    pending = summary.get("pending", 0)
    hit_rate = summary.get("hit_rate")
    hits = summary.get("hits", 0)

    overall = (
        f'<div class="tr-headline">'
        f'<div><div class="tr-big">{hits}/{resolved}</div>'
        f'<div class="tr-sub">Directional hits</div></div>'
        f'<div><div class="tr-big">{fmt_pct_ratio(hit_rate, 1) if hit_rate is not None else "—"}</div>'
        f'<div class="tr-sub">Hit rate</div></div>'
        f'<div><div class="tr-big">{pending}</div>'
        f'<div class="tr-sub">Pending</div></div>'
        f'</div>'
    )

    calibration_html = ""
    cal = summary.get("calibration") or []
    if cal:
        rows = ""
        for row in cal:
            # colour the hit_rate green/red relative to avg_confidence
            delta = row["hit_rate"] - row["avg_confidence"]
            cls = _cls(delta)
            rows += (
                f'<div class="cal-row">'
                f'<div>{row["range"]}</div>'
                f'<div class="eh-v">n={row["n"]}</div>'
                f'<div class="eh-v">{fmt_pct_ratio(row["hit_rate"], 1)}</div>'
                f'<div class="eh-v {cls}">{fmt_pct(delta*100, decimals=1)}</div>'
                f'</div>'
            )
        calibration_html = (
            '<div class="tr-section">Calibration by confidence</div>'
            '<div class="cal-row cal-head">'
            '<div>Range</div><div class="eh-v">N</div>'
            '<div class="eh-v">Hit</div><div class="eh-v">vs Conf</div>'
            '</div>'
            + rows
        )

    recent = summary.get("recent") or []
    recent_html = ""
    if recent:
        rows = ""
        for r in recent[:5]:
            hit = r.get("hit")
            if hit is True:
                pill = '<span class="tr-pill tr-pill-hit">HIT</span>'
            elif hit is False:
                pill = '<span class="tr-pill tr-pill-miss">MISS</span>'
            else:
                pill = '<span class="tr-pill tr-pill-pending">PEND</span>'
            ts = r.get("timestamp")
            try:
                ts_str = pd.to_datetime(ts).strftime("%m/%d %H:%M")
            except Exception:
                ts_str = "—"
            dir_cls = {"BULLISH": "up", "BEARISH": "down"}.get(
                (r.get("direction") or "").upper(), "flat"
            )
            rows += (
                f'<div class="tr-row">'
                f'<div>{ts_str}</div>'
                f'<div class="tr-tkr">{r.get("ticker", "—")}</div>'
                f'<div class="{dir_cls}">{r.get("direction", "—")}</div>'
                f'<div class="eh-v">{fmt_pct_ratio(r.get("confidence"), 0)}</div>'
                f'<div>{pill}</div>'
                f'</div>'
            )
        recent_html = (
            '<div class="tr-section">Recent predictions</div>' + rows
        )

    st.markdown(
        panel_open(
            "Prediction track record",
            f"{resolved} resolved · {pending} pending",
        )
        + overall + calibration_html + recent_html + panel_close(),
        unsafe_allow_html=True,
    )


def earnings_history_block(history: list) -> None:
    """Render past earnings — quarter, actual vs estimate EPS, surprise %."""
    if not history:
        return

    def _safe_num(v, fmt: str = "{:.2f}") -> str:
        try:
            if v is None or v == "N/A":
                return "—"
            return fmt.format(float(v))
        except Exception:
            return "—"

    header = (
        '<div class="eh-row eh-head">'
        '<div>Quarter</div><div>Est</div><div>Actual</div><div>Surprise</div>'
        '</div>'
    )
    body = ""
    for row in history[:4]:  # last 4 quarters
        # yfinance returns surprise_percent as a 0..1 fraction (e.g. 0.0169
        # for a 1.69% beat), so scale to percent before formatting.
        surprise = row.get("surprise_percent")
        try:
            spct = float(surprise) * 100 if surprise not in (None, "N/A") else None
        except Exception:
            spct = None
        surprise_cls = _cls(spct) if spct is not None else "flat"
        surprise_str = fmt_pct(spct, decimals=1) if spct is not None else "—"
        body += (
            f'<div class="eh-row">'
            f'<div class="eh-q">{row.get("quarter", "—")}</div>'
            f'<div class="eh-v">{_safe_num(row.get("estimate_eps"))}</div>'
            f'<div class="eh-v">{_safe_num(row.get("actual_eps"))}</div>'
            f'<div class="eh-v {surprise_cls}">{surprise_str}</div>'
            f'</div>'
        )
    st.markdown(
        panel_open("Earnings history", f"last {min(len(history), 4)} Q")
        + header + body + panel_close(),
        unsafe_allow_html=True,
    )


def scenario_block(targets: dict, current: Optional[float]) -> None:
    def _row(label, v, cls):
        if v is None or current is None:
            return f'<div class="scen"><div class="l {cls}">{label}</div><div class="p">—</div><div class="p">—</div></div>'
        pct = (v / current - 1) * 100
        pct_cls = _cls(pct)
        return (
            f'<div class="scen">'
            f'<div class="l {cls}">{label}</div>'
            f'<div class="p">{fmt_price(v)}</div>'
            f'<div class="p {pct_cls}">{fmt_pct(pct, decimals=1)}</div>'
            f'</div>'
        )

    body = "".join([
        _row("Bullish", targets.get("bullish"), "up"),
        _row("Neutral", targets.get("neutral"), "flat"),
        _row("Bearish", targets.get("bearish"), "down"),
    ])
    st.markdown(panel_open("Price scenarios", "12m horizon") + body + panel_close(),
                unsafe_allow_html=True)


def performance_bars(tech: dict) -> None:
    periods = [
        ("1D", tech.get("performance_1d")),
        ("5D", tech.get("performance_5d")),
        ("1M", tech.get("performance_1m")),
        ("3M", tech.get("performance_3m")),
        ("1Y", tech.get("performance_1y")),
    ]
    vals = [v for _, v in periods if v is not None]
    extent = max((abs(v) for v in vals), default=10)
    extent = max(extent, 5)  # at least 5% for visual scale

    body = ""
    for label, v in periods:
        if v is None:
            body += (
                f'<div class="perf">'
                f'<div class="pl">{label}</div>'
                f'<div class="perf-track"></div>'
                f'<div class="pv">—</div>'
                f'</div>'
            )
            continue
        width_pct = min(abs(v) / extent * 50, 50)  # 50% = half the track
        if v >= 0:
            left = 50
            cls = "pos"
        else:
            left = 50 - width_pct
            cls = "neg"
        body += (
            f'<div class="perf">'
            f'<div class="pl">{label}</div>'
            f'<div class="perf-track">'
            f'<div class="perf-fill {cls}" style="left:{left}%;width:{width_pct}%"></div>'
            f'</div>'
            f'<div class="pv {_cls(v)}">{fmt_pct(v, decimals=1)}</div>'
            f'</div>'
        )
    st.markdown(panel_open("Performance", "vs today") + body + panel_close(),
                unsafe_allow_html=True)


def performance_pills_html(tech: dict) -> str:
    """Build the inline pills HTML string (reusable in headers)."""
    periods = [
        ("1D", tech.get("performance_1d")),
        ("5D", tech.get("performance_5d")),
        ("1M", tech.get("performance_1m")),
        ("3M", tech.get("performance_3m")),
        ("1Y", tech.get("performance_1y")),
    ]
    pills = ""
    for label, v in periods:
        if v is None:
            pills += (
                f'<span class="ps-pill">'
                f'<span class="ps-lbl">{label}</span>'
                f'<span class="ps-val">—</span></span>'
            )
            continue
        cls = _cls(v)
        pills += (
            f'<span class="ps-pill {cls}">'
            f'<span class="ps-lbl">{label}</span>'
            f'<span class="ps-val">{fmt_pct(v, decimals=1)}</span></span>'
        )
    return f'<span class="ps-row">{pills}</span>'


def performance_strip(tech: dict) -> None:
    """Thin horizontal strip of 1D/5D/1M/3M/1Y returns as colored pills.

    Standalone variant — most callers use performance_pills_html() to
    embed pills inside another panel's header.
    """
    st.markdown(performance_pills_html(tech), unsafe_allow_html=True)
