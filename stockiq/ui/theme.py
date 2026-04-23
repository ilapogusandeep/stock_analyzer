"""Dark terminal theme for the Advanced UI."""

ADVANCED_DARK_CSS = """
<style>
  /* Global dark background + monospace feel */
  .stApp {
      background: linear-gradient(180deg, #0b1020 0%, #121832 100%);
      color: #e6ebff;
  }
  section.main > div.block-container {
      padding-top: 1rem;
      padding-bottom: 2rem;
      max-width: 1500px;
  }

  /* Header band */
  .adv-header {
      background: linear-gradient(135deg, rgba(37, 99, 235, 0.18), rgba(168, 85, 247, 0.14));
      border: 1px solid rgba(148, 163, 255, 0.25);
      border-radius: 14px;
      padding: 18px 24px;
      margin-bottom: 16px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 24px;
  }
  .adv-ticker {
      font-size: 2.6rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      color: #ffffff;
      line-height: 1;
  }
  .adv-company {
      color: #a5b4fc;
      font-size: 0.95rem;
      margin-top: 4px;
  }
  .adv-price {
      font-size: 2.6rem;
      font-weight: 700;
      color: #ffffff;
      font-variant-numeric: tabular-nums;
      line-height: 1;
  }
  .adv-change-up    { color: #22c55e; font-weight: 600; }
  .adv-change-down  { color: #ef4444; font-weight: 600; }
  .adv-change-flat  { color: #9ca3af; font-weight: 600; }

  .adv-pill {
      display: inline-block;
      padding: 6px 14px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.95rem;
      letter-spacing: 0.05em;
  }
  .pill-bull  { background: rgba(34, 197, 94, 0.18); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.5); }
  .pill-bear  { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5); }
  .pill-flat  { background: rgba(148, 163, 184, 0.18); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.5); }

  .adv-sub {
      font-size: 0.85rem;
      color: #94a3b8;
      margin-top: 2px;
  }

  /* Metric cards */
  .adv-metric {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(148, 163, 255, 0.18);
      border-radius: 10px;
      padding: 14px 16px;
  }
  .adv-metric-label {
      font-size: 0.72rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
  }
  .adv-metric-value {
      font-size: 1.4rem;
      font-weight: 700;
      color: #ffffff;
      font-variant-numeric: tabular-nums;
  }
  .adv-metric-delta {
      font-size: 0.82rem;
      margin-top: 4px;
      font-variant-numeric: tabular-nums;
  }

  /* Tab bar */
  .stTabs [data-baseweb="tab-list"] {
      gap: 4px;
      border-bottom: 1px solid rgba(148, 163, 255, 0.15);
  }
  .stTabs [data-baseweb="tab"] {
      color: #94a3b8;
      background: transparent;
      height: 44px;
      padding: 0 18px;
      font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
      color: #ffffff !important;
      border-bottom: 2px solid #6366f1 !important;
  }

  /* Section headings inside tabs */
  .adv-h3 {
      color: #e0e7ff;
      font-size: 1.05rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin: 18px 0 10px 0;
      padding-bottom: 6px;
      border-bottom: 1px solid rgba(148, 163, 255, 0.15);
  }

  /* Dataframes on dark */
  [data-testid="stDataFrame"] { background: rgba(255,255,255,0.02); border-radius: 8px; }

  /* Sidebar on dark */
  [data-testid="stSidebar"] {
      background: #0a0f24;
      border-right: 1px solid rgba(148, 163, 255, 0.15);
  }
</style>
"""


def inject_dark_theme():
    import streamlit as st
    st.markdown(ADVANCED_DARK_CSS, unsafe_allow_html=True)
