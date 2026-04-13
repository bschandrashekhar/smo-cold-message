"""SMO Workbench — top-level Streamlit entry point.

Hosts multiple pipelines via sidebar navigation:
  - Prospect Outreach: research + message generation
  - MR & CC Knowledge Tools: client/casestudy/brand matching

Run with: streamlit run app.py
"""

import sys
import os

# Allow absolute imports for sub-packages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(page_title="SMO Workbench", layout="wide")


def render_settings_sidebar():
    """Render brand knowledge refresh in the sidebar."""
    from prospect_outreach import brand_knowledge, config
    from datetime import datetime

    with st.expander("Brand Knowledge"):
        status = brand_knowledge.are_brand_files_present()

        for name, exists in status.items():
            path = config.BRAND_JSONS[name]
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                if exists:
                    mod_time = datetime.fromtimestamp(path.stat().st_mtime)
                    st.caption(f"**{name}** -- {mod_time.strftime('%Y-%m-%d %H:%M')}")
                else:
                    st.caption(f"**{name}** -- Not generated")
            with col_btn:
                if st.button("\u27f3", key=f"refresh_{name}"):
                    with st.spinner(f"Scraping {name}..."):
                        try:
                            brand_knowledge.refresh_single_brand(name)
                            st.success(f"{name} refreshed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")


def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("SMO Workbench")
        pipeline = st.radio(
            "Pipeline",
            ["Prospect Outreach", "MR & CC Knowledge Tools"],
            label_visibility="collapsed",
        )
        st.divider()
        render_settings_sidebar()

    # ── Main area ────────────────────────────────────────────────────────
    if pipeline == "Prospect Outreach":
        from prospect_outreach.ui import render
        render()
    elif pipeline == "MR & CC Knowledge Tools":
        from client_referencing.ui import render
        render()


if __name__ == "__main__":
    main()
