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

  /* Compact Streamlit input controls inside the top bar */
  .topbar [data-testid="stTextInput"] label,
  .topbar [data-testid="stCheckbox"] label { font-size: 0.68rem !important; color: #94a3b8 !important;
      text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px !important; }
  .topbar [data-testid="stTextInput"] input {
      background: rgba(255,255,255,0.04); color: #fff;
      border: 1px solid rgba(148,163,255,0.2); padding: 6px 10px;
      font-size: 0.85rem; height: 34px;
  }
  .topbar [data-testid="stTextInput"] input:focus { border-color: #6366f1; box-shadow: none; }
  .topbar [data-testid="stButton"] button {
      height: 34px; margin-top: 18px;
      font-weight: 700; letter-spacing: 0.04em;
  }
  .topbar [data-testid="stCheckbox"] { margin-top: 20px; }

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
  .kv .k { color: #94a3b8; }
  .kv .v { color: #f1f5f9; font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }
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

  /* Scenario row */
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
