"""SMO Workbench — top-level Streamlit entry point.

Hosts multiple pipelines via sidebar navigation:
  - Prospect Outreach: research + message generation
  - Admin/Debug Tools: client/casestudy/brand matching

Run with: streamlit run app.py
"""

import sys
import os

# Allow absolute imports for sub-packages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(page_title="SMO Workbench", layout="wide")


def main():
    # Supabase keepalive — cron-job.org hits ?keepalive=1
    if st.query_params.get("keepalive"):
        from supabase import create_client
        from prospect_outreach.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        sb.table("industry_embeddings").select("term").limit(1).execute()
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("assets/cloudchillies_logo.svg", width=200)
        st.title("SMO Workbench")
        pipeline = st.radio(
            "Pipeline",
            ["Prospect Outreach", "Master Casestudy Finder", "Admin/Debug Tools"],
            label_visibility="collapsed",
        )

    # ── Main area ────────────────────────────────────────────────────────
    if pipeline == "Prospect Outreach":
        from prospect_outreach.ui import render
        render()
    elif pipeline == "Master Casestudy Finder":
        from client_referencing.ui import render_master
        render_master()
    elif pipeline == "Admin/Debug Tools":
        from client_referencing.ui import render
        render()


if __name__ == "__main__":
    main()
