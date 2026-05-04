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


def main():
    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("SMO Workbench")
        pipeline = st.radio(
            "Pipeline",
            ["Prospect Outreach", "MR & CC Knowledge Tools"],
            label_visibility="collapsed",
        )

    # ── Main area ────────────────────────────────────────────────────────
    if pipeline == "Prospect Outreach":
        from prospect_outreach.ui import render
        render()
    elif pipeline == "MR & CC Knowledge Tools":
        from client_referencing.ui import render
        render()


if __name__ == "__main__":
    main()
