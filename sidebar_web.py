"""Streamlit Cloud entry point.

Thin shim — real UI lives in stockiq.ui.sidebar_web. Importing the module
executes its top-level Streamlit calls (st.set_page_config, sidebar, etc.),
which is exactly what Streamlit needs when it runs this file.
"""

from stockiq.ui import sidebar_web  # noqa: F401
