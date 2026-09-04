"""SMO Workbench — top-level Streamlit entry point.

Hosts multiple pipelines via sidebar navigation:
  - Prospect Outreach: research + message generation
  - Master Casestudy Finder: client + casestudy matching
  - Admin/Debug Tools: client/casestudy/brand matching (admin only)

Run with: streamlit run app.py
"""

import sys
import os

# Allow absolute imports for sub-packages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(page_title="SMO Workbench", layout="wide")


def _get_supabase():
    from supabase import create_client
    from prospect_outreach.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def _login_form():
    st.markdown(
        "<h1 style='text-align:center; margin-bottom: 1rem;'>Casestudy Finder</h1>",
        unsafe_allow_html=True,
    )
    logo_col1, logo_col2, logo_col3 = st.columns([2, 1, 2])
    with logo_col2:
        st.image("assets/cloudchillies_logo.svg", width=160)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("Please enter username and password.")
                return
            try:
                sb = _get_supabase()
                result = sb.table("user_credentials").select("password").eq("username", username.lower()).execute()
                if result.data and result.data[0]["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username.lower()
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            except Exception as e:
                st.error(f"Login failed: {e}")



def main():
    # Supabase keepalive — cron-job.org hits ?keepalive=1
    if st.query_params.get("keepalive"):
        from supabase import create_client
        from prospect_outreach.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        sb.table("industry_embeddings").select("term").limit(1).execute()
        st.stop()

    # Auth gate
    if not st.session_state.get("logged_in"):
        _login_form()
        return

    username = st.session_state.username

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("assets/cloudchillies_logo.svg", width=200)
        if username == "admin":
            st.title("SMO Workbench")

        if username == "admin":
            pipeline = st.radio(
                "Pipeline",
                ["Prospect Outreach", "Master Casestudy Finder", "Admin/Debug Tools"],
                label_visibility="collapsed",
            )
        else:
            # marcom sees only Master Casestudy Finder, auto-selected
            pipeline = "Master Casestudy Finder"

        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

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
