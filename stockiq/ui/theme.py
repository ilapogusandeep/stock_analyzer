"""Dark terminal theme for the dense single-page UI."""

DENSE_DARK_CSS = """
<style>
  /* ===== Global ===== */
  html, body, [data-testid="stAppViewContainer"] {
      font-size: 12px;
  }
  .stApp {
      background: linear-gradient(180deg, #0a0f24 0%, #0f1530 100%);
      color: #e6ebff;
  }
  section.main > div.block-container {
      padding: 0.6rem 1rem 1rem 1rem;
      max-width: 1800px;
  }
  .block-container h1, .block-container h2, .block-container h3 {
      margin: 0;
  }
  /* Tighten default Streamlit gaps */
  .block-container [data-testid="stVerticalBlock"] { gap: 0.4rem; }
  .block-container [data-testid="stHorizontalBlock"] { gap: 0.5rem; }

  /* Hide Streamlit's "deploy" band and default page header */
  header[data-testid="stHeader"] { background: transparent; height: 0; }
  #MainMenu, footer { visibility: hidden; }

  /* ===== Sidebar ===== */
  [data-testid="stSidebar"] {
      background: #070b1c;
      border-right: 1px solid rgba(148, 163, 255, 0.15);
  }
  [data-testid="stSidebar"] .block-container { padding: 1rem 0.75rem; }

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
  .hb-tkr { font-size: 2rem; font-weight: 800; color: #fff; letter-spacing: 0.03em; line-height: 1; }
  .hb-co  { color: #a5b4fc; font-size: 0.78rem; margin-top: 2px; }
  .hb-px  { font-size: 1.7rem; font-weight: 700; color: #fff; font-variant-numeric: tabular-nums; line-height: 1; }
  .hb-chg { font-size: 0.82rem; font-weight: 600; margin-top: 3px; font-variant-numeric: tabular-nums; }
  .up     { color: #22c55e; }
  .down   { color: #ef4444; }
  .flat   { color: #cbd5e1; }

  .pill {
      display: inline-block; padding: 5px 12px; border-radius: 999px;
      font-weight: 700; font-size: 0.82rem; letter-spacing: 0.04em;
  }
  .pill-bull { background: rgba(34,197,94,0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.5); }
  .pill-bear { background: rgba(239,68,68,0.18); color: #f87171; border: 1px solid rgba(239,68,68,0.5); }
  .pill-flat { background: rgba(148,163,184,0.18); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.5); }
  .hb-sub  { font-size: 0.65rem; color: #94a3b8; margin-top: 2px; letter-spacing: 0.08em; text-transform: uppercase; }

  .hb-ctx {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;
  }
  .hb-ctx-l { font-size: 0.62rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
  .hb-ctx-v { font-size: 0.92rem; color: #fff; font-weight: 600; font-variant-numeric: tabular-nums; }

  /* ===== Panel ===== */
  .panel {
      background: rgba(255,255,255,0.025);
      border: 1px solid rgba(148,163,255,0.18);
      border-radius: 8px;
      padding: 8px 10px;
  }
  .panel-h {
      display: flex; align-items: center; justify-content: space-between;
      font-size: 0.68rem; font-weight: 700; color: #a5b4fc;
      text-transform: uppercase; letter-spacing: 0.1em;
      margin: 0 0 6px 0;
      padding-bottom: 4px;
      border-bottom: 1px solid rgba(148,163,255,0.12);
  }
  .panel-h .panel-h-sub { color: #64748b; font-size: 0.6rem; letter-spacing: 0.08em; font-weight: 500; }

  /* Two-column key:value stats list */
  .kv {
      display: grid; grid-template-columns: 1fr auto;
      gap: 2px 10px;
      font-size: 0.78rem;
  }
  .kv .k { color: #94a3b8; }
  .kv .v { color: #f1f5f9; font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }
  .kv .v.good { color: #4ade80; }
  .kv .v.bad  { color: #f87171; }

  /* Probability bars */
  .pb-row {
      display: grid; grid-template-columns: 64px 1fr 52px; gap: 8px;
      align-items: center; font-size: 0.75rem; margin: 2px 0;
  }
  .pb-label { color: #cbd5e1; font-weight: 600; }
  .pb-bar {
      height: 10px; border-radius: 999px; background: rgba(148,163,255,0.08);
      overflow: hidden;
  }
  .pb-fill { height: 100%; border-radius: 999px; }
  .pb-val { color: #fff; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .pb-bull { background: linear-gradient(90deg, #16a34a, #22c55e); }
  .pb-neut { background: linear-gradient(90deg, #64748b, #94a3b8); }
  .pb-bear { background: linear-gradient(90deg, #dc2626, #ef4444); }

  /* Scenario row */
  .scen {
      display: grid; grid-template-columns: 70px 1fr 1fr; gap: 8px;
      font-size: 0.78rem; padding: 3px 0; border-bottom: 1px dashed rgba(148,163,255,0.08);
  }
  .scen:last-child { border-bottom: none; }
  .scen .l { color: #cbd5e1; font-weight: 600; }
  .scen .p { color: #f1f5f9; text-align: right; font-variant-numeric: tabular-nums; }

  /* Performance bars */
  .perf {
      display: grid; grid-template-columns: 36px 1fr 56px; gap: 8px;
      align-items: center; font-size: 0.74rem; margin: 2px 0;
  }
  .perf .pl { color: #94a3b8; font-weight: 600; }
  .perf .pv { color: #f1f5f9; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
  .perf-track { position: relative; height: 8px; background: rgba(148,163,255,0.06); border-radius: 999px; }
  .perf-track::before {
      content: ""; position: absolute; left: 50%; top: 0; bottom: 0; width: 1px;
      background: rgba(148,163,255,0.25);
  }
  .perf-fill { position: absolute; top: 0; bottom: 0; border-radius: 999px; }
  .perf-fill.pos { background: #22c55e; }
  .perf-fill.neg { background: #ef4444; }

  /* Sentiment gauge (compact) */
  .sent-label { font-size: 0.7rem; color: #94a3b8; }
  .sent-value { font-size: 1rem; font-weight: 700; color: #fff; }

  /* Footer strip */
  .footer {
      margin-top: 10px;
      font-size: 0.7rem; color: #64748b;
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
