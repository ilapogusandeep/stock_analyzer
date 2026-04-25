"""Dark terminal theme for the dense single-page UI."""

DENSE_DARK_CSS = """
<style>
  /* ===== Global ===== */
  html, body, [data-testid="stAppViewContainer"] {
      font-size: 15px;
  }
  .stApp {
      background: linear-gradient(180deg, #0a0f24 0%, #0f1530 100%);
      color: #e6ebff;
  }
  section.main > div.block-container {
      padding: 0.1rem 1.1rem 1rem 1.1rem !important;
      max-width: 1900px;
  }
  .block-container h1, .block-container h2, .block-container h3 {
      margin: 0;
  }
  /* Tighten default Streamlit gaps */
  .block-container [data-testid="stVerticalBlock"] { gap: 0.45rem; }
  .block-container [data-testid="stHorizontalBlock"] { gap: 0.6rem; }

  /* ===== Squash the top — pull content to the very top of the viewport ===== */
  header[data-testid="stHeader"],
  .stApp > header,
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"] {
      display: none !important;
      height: 0 !important;
      min-height: 0 !important;
  }
  #MainMenu, footer { visibility: hidden; }
  /* Streamlit wraps the app in a <section data-testid="stMain"> that has its
     own top padding via an emotion class — zero it out. */
  section[data-testid="stMain"] { padding-top: 0 !important; }
  section[data-testid="stMain"] > div:first-child { padding-top: 0 !important; }
  .stApp { padding-top: 0 !important; }

  /* ===== Kill the sidebar ===== */
  [data-testid="stSidebar"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="collapsedControl"] { display: none !important; }
  [data-testid="stAppViewContainer"] > section:first-child { margin-left: 0 !important; }

  /* ===== Top controls bar ===== */
  .topbar-logo {
      font-size: 1.15rem; font-weight: 800; color: #fff;
      letter-spacing: 0.04em;
      padding-top: 6px;
  }
  .topbar-logo span { color: #a5b4fc; font-weight: 600; font-size: 0.75rem; margin-left: 4px; letter-spacing: 0.06em; text-transform: uppercase; }

  /* Compact Streamlit input controls inside the top bar.
     Force every label + widget-text to a light slate so the dark
     background never eats it. */
  .topbar label,
  .topbar [data-testid="stWidgetLabel"],
  .topbar [data-testid="stTextInput"] label,
  .topbar [data-testid="stSelectbox"] label,
  .topbar [data-testid="stCheckbox"] label {
      font-size: 0.72rem !important;
      color: #cbd5e1 !important;
      text-transform: uppercase; letter-spacing: 0.08em;
      margin-bottom: 2px !important;
  }
  .topbar [data-testid="stTextInput"] input,
  .topbar [data-testid="stSelectbox"] [data-baseweb="select"] *,
  .topbar [data-testid="stSelectbox"] [role="combobox"],
  .topbar [data-testid="stSelectbox"] input {
      color: #ffffff !important;
  }
  .topbar [data-testid="stTextInput"] input {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(148,163,255,0.2); padding: 6px 10px;
      font-size: 0.85rem; height: 34px;
  }
  .topbar [data-testid="stTextInput"] input:focus { border-color: #6366f1; box-shadow: none; }
  .topbar [data-testid="stSelectbox"] [data-baseweb="select"] > div {
      background: rgba(255,255,255,0.04) !important;
      border: 1px solid rgba(148,163,255,0.2) !important;
  }
  .topbar [data-testid="stButton"] button {
      height: 34px; margin-top: 18px;
      font-weight: 700; letter-spacing: 0.04em;
  }
  .topbar [data-testid="stCheckbox"] { margin-top: 20px; }
  .topbar [data-testid="stCheckbox"] label > div:last-child { color: #cbd5e1 !important; }

  /* ===== Header band ===== */
  .hb {
      display: grid;
      grid-template-columns: 1.1fr 1.2fr 0.8fr 2fr;
      gap: 18px;
      align-items: center;
      background: linear-gradient(135deg, rgba(37, 99, 235, 0.18), rgba(168, 85, 247, 0.14));
      border: 1px solid rgba(148, 163, 255, 0.25);
      border-radius: 10px;
      padding: 10px 16px;
      margin-bottom: 8px;
  }
  .hb-tkr { font-size: 2.1rem; font-weight: 800; color: #fff; letter-spacing: 0.03em; line-height: 1; }
  .hb-co  { color: #a5b4fc; font-size: 0.85rem; margin-top: 2px; }
  .hb-desc-row {
      grid-column: 1 / -1;
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid rgba(148,163,255,0.12);
      color: #cbd5e1;
      font-size: 0.76rem;
      line-height: 1.35;
      font-style: italic;
      opacity: 0.9;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
  }
  .hb-px  { font-size: 1.8rem; font-weight: 700; color: #fff; font-variant-numeric: tabular-nums; line-height: 1; }
  .hb-chg { font-size: 0.9rem; font-weight: 600; margin-top: 3px; font-variant-numeric: tabular-nums; }
  .up     { color: #22c55e; }
  .down   { color: #ef4444; }
  .flat   { color: #cbd5e1; }

  .pill {
      display: inline-block; padding: 6px 14px; border-radius: 999px;
      font-weight: 700; font-size: 0.9rem; letter-spacing: 0.04em;
  }
  .pill-bull { background: rgba(34,197,94,0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.5); }
  .pill-bear { background: rgba(239,68,68,0.18); color: #f87171; border: 1px solid rgba(239,68,68,0.5); }
  .pill-flat { background: rgba(148,163,184,0.18); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.5); }
  .hb-sub  { font-size: 0.7rem; color: #94a3b8; margin-top: 2px; letter-spacing: 0.08em; text-transform: uppercase; }

  .hb-ctx {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
  }
  .hb-ctx-l { font-size: 0.68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
  .hb-ctx-v { font-size: 1rem; color: #fff; font-weight: 600; font-variant-numeric: tabular-nums; }

  /* ===== External links row ===== */
  .ext-row {
      display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
      padding: 8px 2px 2px 2px;
      margin-bottom: 8px;
  }
  .ext-label {
      font-size: 0.72rem; color: #94a3b8;
      text-transform: uppercase; letter-spacing: 0.1em;
      margin-right: 4px;
  }
  a.ext-link {
      display: inline-block;
      padding: 5px 10px;
      font-size: 0.82rem;
      color: #cbd5e1 !important;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(148,163,255,0.18);
      border-radius: 6px;
      text-decoration: none !important;
      transition: background 0.1s ease, border-color 0.1s ease;
  }
  a.ext-link:hover {
      background: rgba(99, 102, 241, 0.15);
      border-color: rgba(148,163,255,0.4);
      color: #fff !important;
  }

  /* ===== Panel ===== */
  .panel {
      background: rgba(255,255,255,0.025);
      border: 1px solid rgba(148,163,255,0.18);
      border-radius: 8px;
      padding: 12px 14px;
  }
  .panel-h {
      display: flex; align-items: center; justify-content: space-between;
      font-size: 0.75rem; font-weight: 700; color: #a5b4fc;
      text-transform: uppercase; letter-spacing: 0.1em;
      margin: 0 0 6px 0;
      padding-bottom: 4px;
      border-bottom: 1px solid rgba(148,163,255,0.12);
  }
  .panel-h .panel-h-sub { color: #64748b; font-size: 0.68rem; letter-spacing: 0.08em; font-weight: 500; }

  /* Two-column key:value stats list */
  .kv {
      display: grid; grid-template-columns: 1fr auto;
      gap: 8px 16px;
      font-size: 0.95rem;
  }
  .kv.kv-2 {
      grid-template-columns: minmax(0, auto) minmax(0, 1fr) minmax(0, auto) minmax(0, 1fr);
      gap: 6px 12px;
      font-size: 0.88rem;
  }
  .kv .k { color: #94a3b8; }
  .kv .v { color: #f1f5f9; font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }
  .kv.kv-2 .v { text-align: left; }
  .kv .v.good { color: #4ade80; }
  .kv .v.bad  { color: #f87171; }

  /* Probability bars */
  .pb-row {
      display: grid; grid-template-columns: 74px 1fr 58px; gap: 10px;
      align-items: center; font-size: 0.88rem; margin: 6px 0;
  }
  .pb-label { color: #cbd5e1; font-weight: 600; }
  .pb-bar {
      height: 12px; border-radius: 999px; background: rgba(148,163,255,0.08);
      overflow: hidden;
  }
  .pb-fill { height: 100%; border-radius: 999px; }
  .pb-val { color: #fff; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .pb-bull { background: linear-gradient(90deg, #16a34a, #22c55e); }
  .pb-neut { background: linear-gradient(90deg, #64748b, #94a3b8); }
  .pb-bear { background: linear-gradient(90deg, #dc2626, #ef4444); }

  /* Combined AI scenario + target row
     cols: label | bar | prob%  | target$ | delta% */
  .pbx-row {
      display: grid;
      grid-template-columns: 64px 1fr 48px 68px 52px;
      gap: 8px; align-items: center;
      font-size: 0.88rem; margin: 7px 0;
  }
  .pbx-label { color: #cbd5e1; font-weight: 600; }
  .pbx-pct   { color: #fff; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .pbx-tgt   { color: #f1f5f9; text-align: right; font-variant-numeric: tabular-nums; }
  .pbx-delta { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }

  /* News feed panel */
  .nf-scroll {
      max-height: 340px;
      overflow-y: auto;
      padding-right: 4px;
  }
  .nf-scroll::-webkit-scrollbar { width: 6px; }
  .nf-scroll::-webkit-scrollbar-thumb {
      background: rgba(148,163,255,0.15);
      border-radius: 3px;
  }
  .nf-row {
      display: grid;
      grid-template-columns: 18px 1fr 48px;
      column-gap: 8px;
      font-size: 0.82rem;
      padding: 5px 0;
      border-bottom: 1px dashed rgba(148,163,255,0.07);
      align-items: start;
  }
  .nf-row:last-child { border-bottom: none; }
  .nf-dot { color: #94a3b8; }
  .nf-h { color: #e6ebff; line-height: 1.3; }
  .nf-src {
      display: inline-block;
      color: #64748b;
      font-size: 0.68rem;
      margin-left: 6px;
      text-transform: uppercase; letter-spacing: 0.05em;
  }
  .nf-score {
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-weight: 600;
  }

  /* Unusual options panel */
  .uo-scroll {
      max-height: 220px;
      overflow-y: auto;
      padding-right: 4px;
  }
  .uo-scroll::-webkit-scrollbar { width: 6px; }
  .uo-scroll::-webkit-scrollbar-thumb {
      background: rgba(148,163,255,0.15);
      border-radius: 3px;
  }
  .uo-row {
      display: grid;
      grid-template-columns: 44px 58px 48px 44px 1fr 22px;
      column-gap: 6px;
      font-size: 0.82rem;
      padding: 6px 0;
      border-bottom: 1px dashed rgba(148,163,255,0.07);
      align-items: center;
  }
  .uo-row:last-child { border-bottom: none; }
  .uo-side {
      font-size: 0.66rem; font-weight: 700;
      text-align: center; padding: 2px 0; border-radius: 4px;
      letter-spacing: 0.05em;
  }
  .uo-side.up   { background: rgba(34,197,94,0.18); color: #4ade80; }
  .uo-side.down { background: rgba(239,68,68,0.18); color: #f87171; }
  .uo-strike {
      color: #fff; font-weight: 600;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
  }
  .uo-cluster {
      font-size: 0.68rem;
      margin-right: 2px;
      opacity: 0.85;
  }
  .uo-exp {
      color: #94a3b8; font-size: 0.74rem;
      text-transform: uppercase; letter-spacing: 0.03em;
  }
  .uo-voi {
      color: #fbbf24; font-weight: 600;
      font-variant-numeric: tabular-nums; text-align: right;
  }
  .uo-flow {
      color: #e6ebff; font-weight: 600;
      text-align: right;
      font-variant-numeric: tabular-nums;
  }
  .uo-agg {
      font-size: 0.62rem; font-weight: 700;
      text-align: center;
      padding: 1px 0;
      border-radius: 3px;
      letter-spacing: 0;
      background: rgba(148,163,255,0.08);
      color: #94a3b8;
  }
  .uo-agg.up   { background: rgba(34,197,94,0.20); color: #4ade80; }
  .uo-agg.down { background: rgba(239,68,68,0.20); color: #f87171; }
  .uo-agg.flat { background: rgba(148,163,255,0.08); color: #94a3b8; }

  /* 3-month regime panel */
  .rg-row {
      display: flex; justify-content: space-between; align-items: center;
      margin: 4px 0 6px;
      font-size: 0.95rem;
  }
  .rg-label { font-weight: 700; letter-spacing: 0.05em; }
  .rg-label.up   { color: #4ade80; }
  .rg-label.down { color: #f87171; }
  .rg-label.flat { color: #fbbf24; }
  .rg-conf { color: #94a3b8; font-size: 0.78rem; }
  .rg-mix {
      display: flex; gap: 14px; flex-wrap: wrap;
      font-size: 0.74rem; color: #94a3b8;
      font-variant-numeric: tabular-nums;
      letter-spacing: 0.02em;
  }

  /* Options flow pills (panel body variant — wraps) */
  .of-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 4px;
  }

  /* Performance pills (embedded in panel header) */
  .ps-row {
      display: inline-flex;
      gap: 4px;
      flex-wrap: nowrap;
      align-items: center;
  }
  .ps-pill {
      display: inline-flex; gap: 4px; align-items: baseline;
      padding: 2px 7px;
      background: rgba(148,163,255,0.05);
      border: 1px solid rgba(148,163,255,0.10);
      border-radius: 10px;
      font-size: 0.70rem;
      font-variant-numeric: tabular-nums;
      letter-spacing: 0;
      text-transform: none;
  }
  .ps-pill .ps-lbl {
      color: #94a3b8; font-size: 0.62rem;
      text-transform: uppercase; letter-spacing: 0.05em;
      font-weight: 600;
  }
  .ps-pill .ps-val { color: #f1f5f9; font-weight: 600; }
  .ps-pill.up   { border-color: rgba(34,197,94,0.30); background: rgba(34,197,94,0.08); }
  .ps-pill.up   .ps-val { color: #4ade80; }
  .ps-pill.down { border-color: rgba(239,68,68,0.30); background: rgba(239,68,68,0.08); }
  .ps-pill.down .ps-val { color: #f87171; }

  /* Top holders panel (13F) */
  .ih-scroll {
      max-height: 220px;
      overflow-y: auto;
      padding-right: 4px;
  }
  .ih-scroll::-webkit-scrollbar { width: 6px; }
  .ih-scroll::-webkit-scrollbar-thumb {
      background: rgba(148,163,255,0.15);
      border-radius: 3px;
  }
  .ih-row {
      display: grid;
      grid-template-columns: 1fr 56px 44px 48px 52px;
      column-gap: 6px;
      font-size: 0.78rem;
      padding: 5px 0;
      border-bottom: 1px dashed rgba(148,163,255,0.07);
      align-items: center;
  }
  .ih-row:last-child { border-bottom: none; }
  .ih-name {
      color: #e6ebff;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
  }
  .ih-val {
      color: #fff; font-weight: 600;
      text-align: right;
      font-variant-numeric: tabular-nums;
  }
  .ih-pct {
      color: #94a3b8;
      text-align: right;
      font-variant-numeric: tabular-nums;
  }
  .ih-chg {
      text-align: right;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
  }
  .ih-date {
      color: #64748b;
      text-align: right;
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
  }

  /* Track record panel */
  .tr-headline {
      display: grid; grid-template-columns: repeat(3, 1fr);
      gap: 8px; margin: 2px 0 10px;
  }
  .tr-big {
      font-size: 1.35rem; font-weight: 700; color: #fff;
      font-variant-numeric: tabular-nums;
  }
  .tr-sub {
      font-size: 0.65rem; color: #94a3b8;
      text-transform: uppercase; letter-spacing: 0.08em;
  }
  .tr-section {
      font-size: 0.68rem; color: #a5b4fc;
      text-transform: uppercase; letter-spacing: 0.1em;
      margin: 10px 0 4px;
      padding-top: 6px;
      border-top: 1px dashed rgba(148,163,255,0.12);
  }
  .tr-row {
      display: grid;
      grid-template-columns: 70px 54px 78px 44px 58px;
      gap: 6px; align-items: center;
      font-size: 0.78rem; padding: 3px 0;
      border-bottom: 1px dashed rgba(148,163,255,0.07);
      font-variant-numeric: tabular-nums;
  }
  .tr-row:last-child { border-bottom: none; }
  .tr-tkr { color: #fff; font-weight: 700; }
  .tr-pill {
      display: inline-block; font-size: 0.65rem; font-weight: 700;
      padding: 2px 6px; border-radius: 4px; letter-spacing: 0.05em;
      text-align: center;
  }
  .tr-pill-hit     { background: rgba(34,197,94,0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
  .tr-pill-miss    { background: rgba(239,68,68,0.18); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
  .tr-pill-pending { background: rgba(148,163,184,0.14); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.35); }

  .cal-row {
      display: grid; grid-template-columns: 80px 1fr 1fr 1fr; gap: 6px;
      font-size: 0.8rem; padding: 3px 0;
      border-bottom: 1px dashed rgba(148,163,255,0.07);
      font-variant-numeric: tabular-nums;
  }
  .cal-row:last-child { border-bottom: none; }
  .cal-row.cal-head {
      font-size: 0.62rem; color: #94a3b8;
      text-transform: uppercase; letter-spacing: 0.08em;
      border-bottom: 1px solid rgba(148,163,255,0.12);
  }

  /* Earnings history table */
  .eh-row {
      display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr; gap: 6px;
      font-size: 0.85rem; padding: 5px 0;
      border-bottom: 1px dashed rgba(148,163,255,0.08);
      font-variant-numeric: tabular-nums;
  }
  .eh-row:last-child { border-bottom: none; }
  .eh-row.eh-head {
      font-size: 0.68rem; color: #94a3b8;
      text-transform: uppercase; letter-spacing: 0.08em;
      border-bottom: 1px solid rgba(148,163,255,0.15);
      padding-bottom: 4px; margin-bottom: 2px;
  }
  .eh-row.eh-head > div:not(:first-child),
  .eh-row .eh-v { text-align: right; }
  .eh-q { color: #cbd5e1; font-weight: 600; }
  .eh-v { color: #f1f5f9; }
  .eh-v.up   { color: #4ade80; font-weight: 600; }
  .eh-v.down { color: #f87171; font-weight: 600; }

  /* Scenario row (legacy, kept for backward compat) */
  .scen {
      display: grid; grid-template-columns: 80px 1fr 1fr; gap: 10px;
      font-size: 0.9rem; padding: 6px 0; border-bottom: 1px dashed rgba(148,163,255,0.08);
  }
  .scen:last-child { border-bottom: none; }
  .scen .l { color: #cbd5e1; font-weight: 600; }
  .scen .p { color: #f1f5f9; text-align: right; font-variant-numeric: tabular-nums; }

  /* Performance bars */
  .perf {
      display: grid; grid-template-columns: 42px 1fr 64px; gap: 10px;
      align-items: center; font-size: 0.88rem; margin: 6px 0;
  }
  .perf .pl { color: #94a3b8; font-weight: 600; }
  .perf .pv { color: #f1f5f9; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .perf-track { position: relative; height: 10px; background: rgba(148,163,255,0.06); border-radius: 999px; }
  .perf-track::before {
      content: ""; position: absolute; left: 50%; top: 0; bottom: 0; width: 1px;
      background: rgba(148,163,255,0.25);
  }
  .perf-fill { position: absolute; top: 0; bottom: 0; border-radius: 999px; }
  .perf-fill.pos { background: #22c55e; }
  .perf-fill.neg { background: #ef4444; }

  /* Sentiment gauge (compact) */
  .sent-label { font-size: 0.78rem; color: #94a3b8; }
  .sent-value { font-size: 1.05rem; font-weight: 700; color: #fff; }

  /* Footer strip */
  .footer {
      margin-top: 10px;
      font-size: 0.78rem; color: #64748b;
      text-align: center;
      padding: 6px 0;
      border-top: 1px solid rgba(148,163,255,0.1);
  }

  /* Plotly background transparent in cards */
  .js-plotly-plot .plotly .main-svg { background: transparent !important; }

  /* Kill default Streamlit label padding on inputs inside sidebar */
  [data-testid="stSidebar"] label { font-size: 0.72rem; color: #a5b4fc; }
</style>
"""


def inject_theme() -> None:
    import streamlit as st
    st.markdown(DENSE_DARK_CSS, unsafe_allow_html=True)
