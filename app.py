"""SMO Workbench — top-level Streamlit entry point.

Hosts multiple pipelines via sidebar navigation:
  - Prospect Outreach: research + message generation
  - Event Prospecting: exhibitor scraping + prospect enrichment

Run with: streamlit run app.py
"""

import sys
import os
from datetime import datetime

# Allow absolute imports for sub-packages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(page_title="SMO Workbench", layout="wide")


def render_settings_sidebar():
    """Render brand knowledge settings in the sidebar."""
    from prospect_outreach import brand_knowledge, config

    with st.expander("Brand Knowledge"):
        status = brand_knowledge.are_brand_files_present()

        for name, exists in status.items():
            path = config.BRAND_JSONS[name]
            if exists:
                mod_time = datetime.fromtimestamp(path.stat().st_mtime)
                st.caption(f"**{name}** — {mod_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.caption(f"**{name}** — Not generated")

        if st.button("Refresh Brand Knowledge", key="sidebar_refresh_brands"):
            with st.spinner("Scraping brand websites..."):
                try:
                    brand_knowledge.refresh_brand_profiles()
                    st.success("Refreshed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")


def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("SMO Workbench")
        pipeline = st.radio(
            "Pipeline",
            ["Prospect Outreach", "Event Prospecting"],
            label_visibility="collapsed",
        )
        st.divider()
        render_settings_sidebar()

    # ── Main area ────────────────────────────────────────────────────────
    if pipeline == "Prospect Outreach":
        from prospect_outreach.ui import render
        render()
    else:
        from event_prospecting.ui import render
        render()


if __name__ == "__main__":
    main()
