"""Prospect Outreach UI — 3-tab Streamlit interface.

Tab 1: Pass 1 — Research prospects
Tab 2: Pass 2 — Generate messages
Tab 3: Settings — Brand knowledge management
"""

import io
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = [
    "First_Name", "Last_Name", "Designation", "Company_Name",
    "Email", "City", "State", "Country", "Industry", "Website",
]


def render():
    """Render the Prospect Outreach pipeline."""
    tabs = st.tabs(["Pass 1: Research", "Pass 2: Generate Messages", "Settings"])

    with tabs[0]:
        _render_pass1_tab()
    with tabs[1]:
        _render_pass2_tab()
    with tabs[2]:
        _render_settings_tab()


# ── Pass 1: Research ──────────────────────────────────────────────────────

def _render_pass1_tab():
    st.subheader("Pass 1: Research Prospects")
    st.caption(
        "Upload data.xlsx with a 'prospects' sheet (and optional 'dates' sheet). "
        "The pipeline will research each prospect, match case studies and clients, "
        "and produce data_output.xlsx for your review."
    )

    uploaded = st.file_uploader("Upload data.xlsx", type=["xlsx"], key="pass1_upload")

    if uploaded is None:
        return

    file_bytes = uploaded.read()

    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="prospects")
    except Exception as e:
        st.error(f"Could not read 'prospects' sheet: {e}")
        return

    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    # Check for dates sheet
    try:
        dates_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="dates")
        has_dates = True
    except Exception:
        dates_df = pd.DataFrame()
        has_dates = False

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Prospects", len(df))
    col2.metric("Unique Companies", df["Company_Name"].nunique())
    col3.metric("Dates Sheet", "Yes" if has_dates else "No")

    st.dataframe(df[REQUIRED_COLUMNS].head(10), use_container_width=True)

    st.divider()

    # UI options
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        generate_intent = st.toggle("Generate Intent Score", value=True)
    with col_b:
        max_case_studies = st.number_input("Max Case Studies", min_value=1, max_value=20, value=5)
    with col_c:
        max_clients = st.number_input("Max Client Matches", min_value=5, max_value=20, value=5)

    if st.button("Run Research", type="primary", key="run_research_btn"):
        from prospect_outreach.research_pipeline import research_workbook

        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(current, total, msg):
            pct = current / total if total > 0 else 0
            progress_bar.progress(pct)
            status_text.text(msg)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            output_path = tmp.name

        try:
            research_workbook(
                file_bytes=file_bytes,
                output_path=output_path,
                generate_intent_score=generate_intent,
                max_case_studies=int(max_case_studies),
                max_client_matches=int(max_clients),
                progress_callback=on_progress,
            )
            progress_bar.progress(1.0)
            status_text.text("Done!")

            result_df = pd.read_excel(output_path, sheet_name="prospects")
            st.success(f"Research complete. {len(result_df)} prospects processed.")
            st.dataframe(result_df.head(10), use_container_width=True)

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download data_output.xlsx",
                    data=f.read(),
                    file_name="data_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.error(f"Research failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ── Pass 2: Generate Messages ─────────────────────────────────────────────

def _render_pass2_tab():
    st.subheader("Pass 2: Generate Messages")
    st.caption(
        "Upload the reviewed data_output.xlsx. Rows with a Research_Summary and no existing "
        "Message_to_send will have messages generated. Rows with empty Research_Summary are skipped."
    )

    uploaded = st.file_uploader("Upload reviewed data_output.xlsx", type=["xlsx"], key="pass2_upload")

    if uploaded is None:
        return

    file_bytes = uploaded.read()

    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="prospects")
    except Exception as e:
        st.error(f"Could not read 'prospects' sheet: {e}")
        return

    # Preview ready/skipped counts
    def _is_ready(row):
        msg = str(row.get("Message_to_send", "")).strip()
        summary = str(row.get("Research_Summary", "")).strip()
        return not msg and bool(summary)

    ready = sum(1 for _, row in df.iterrows() if _is_ready(row))
    skipped = len(df) - ready

    col1, col2 = st.columns(2)
    col1.metric("Ready to Generate", ready)
    col2.metric("Skipped", skipped)

    if ready == 0:
        st.info("No prospects to generate messages for — all rows either already have messages or have empty Research_Summary.")
        return

    if st.button("Generate Messages", type="primary", key="run_gen_btn"):
        from prospect_outreach.message_generator import generate_messages

        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(current, total, msg):
            pct = current / total if total > 0 else 0
            progress_bar.progress(pct)
            status_text.text(msg)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            output_path = tmp.name

        try:
            stats = generate_messages(
                file_bytes=file_bytes,
                output_path=output_path,
                progress_callback=on_progress,
            )
            progress_bar.progress(1.0)
            status_text.text("Done!")

            st.success(f"Generated {stats['ready']} messages. {stats['skipped']} skipped.")

            result_df = pd.read_excel(output_path, sheet_name="prospects")
            preview_cols = ["First_Name", "Last_Name", "Company_Name", "Message_to_send"]
            available = [c for c in preview_cols if c in result_df.columns]
            st.dataframe(result_df[available].head(10), use_container_width=True)

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download data_final.xlsx",
                    data=f.read(),
                    file_name="data_final.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.error(f"Message generation failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ── Settings: Brand Knowledge ─────────────────────────────────────────────

def _render_settings_tab():
    st.subheader("Settings — Brand Knowledge")
    st.caption("View and refresh brand profiles scraped from company websites.")

    from prospect_outreach import brand_knowledge, config

    status = brand_knowledge.are_brand_files_present()

    for name, exists in status.items():
        path = config.BRAND_JSONS[name]
        st.markdown(f"**{name}**")

        col_info, col_btn = st.columns([4, 1])
        with col_info:
            if exists:
                mod_time = datetime.fromtimestamp(path.stat().st_mtime)
                st.caption(f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.caption("Not generated yet.")
        with col_btn:
            if st.button("\u27f3", key=f"settings_refresh_{name}"):
                with st.spinner(f"Scraping {name}..."):
                    try:
                        brand_knowledge.refresh_single_brand(name)
                        st.success(f"{name} refreshed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

        if exists:
            with st.expander(f"View {name} profile", expanded=False):
                try:
                    profile = brand_knowledge.load_brand_profile(name)
                    st.json(profile)
                except Exception as e:
                    st.error(f"Could not load profile: {e}")

        st.divider()
