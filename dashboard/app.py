"""Streamlit entry point: ``streamlit run dashboard/app.py``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mpl"))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Agentic Demand Forecasting",
    page_icon=":material/inventory_2:",
    layout="wide",
)

from dashboard.components.state import render_sidebar  # noqa: E402

pages = {
    "Summary": [
        st.Page("views/overview.py", title="Overview", icon=":material/dashboard:",
                default=True),
    ],
    "Agents": [
        st.Page("views/forecast.py", title="1. Forecast Agent", icon=":material/insights:"),
        st.Page("views/risk.py", title="2. Risk & Uncertainty", icon=":material/warning:"),
        st.Page("views/inventory.py", title="3. Inventory Planning",
                icon=":material/inventory:"),
        st.Page("views/critic.py", title="4. Critic & Approval", icon=":material/gavel:"),
    ],
    "Insights": [
        st.Page("views/trace.py", title="Agent Trace", icon=":material/account_tree:"),
        st.Page("views/metrics.py", title="Performance Metrics", icon=":material/monitoring:"),
    ],
}

navigation = st.navigation(pages)
render_sidebar()
navigation.run()
