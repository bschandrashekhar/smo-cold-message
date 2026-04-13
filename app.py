"""SMO Workbench — top-level Streamlit entry point.

Hosts multiple pipelines via sidebar navigation:
  - Prospect Outreach: research + message generation
  - Event Prospecting: exhibitor scraping + prospect enrichment

Run with: streamlit run app.py
"""

import sys
import os

# Allow absolute imports for sub-packages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(page_title="SMO Workbench", layout="wide")


def render_settings_sidebar():
    """Render settings in the sidebar."""
    pass


def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("SMO Workbench")
        pipeline = st.radio(
            "Pipeline",
            ["Prospect Outreach", "Event Prospecting", "Client Referencing"],
            label_visibility="collapsed",
        )
        st.divider()
        render_settings_sidebar()

    # ── Main area ────────────────────────────────────────────────────────
    if pipeline == "Prospect Outreach":
        from prospect_outreach.ui import render
        render()
    elif pipeline == "Event Prospecting":
        from event_prospecting.ui import render
        render()
    else:
        from client_referencing.ui import render
        render()


if __name__ == "__main__":
    main()
