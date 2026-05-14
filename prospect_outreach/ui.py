"""Prospect Outreach UI — 2-tab Streamlit interface.

Tab 1: Pass 1 — Research prospects
Tab 2: Pass 2 — Generate messages
"""

import io
import tempfile

import pandas as pd
import streamlit as st

REQUIRED_COLUMNS = [
    "First_Name", "Designation", "Company_Name",
    "Email", "City", "State", "Country", "Industry", "Website",
]


def render():
    """Render the Prospect Outreach pipeline."""
    tabs = st.tabs(["Pass 1: Research", "Pass 2: Generate Messages"])

    with tabs[0]:
        _render_pass1_tab()
    with tabs[1]:
        _render_pass2_tab()


# ── Pass 1: Research ──────────────────────────────────────────────────────

def _render_pass1_tab():
    st.subheader("Pass 1: Research Prospects")
    st.caption(
        "Upload data.xlsx with a 'prospects' sheet (and optional 'dates' sheet). "
        "The pipeline will research each prospect, match case studies and clients, "
        "and produce Pass-1-Output.xlsx for your review."
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
        use_cache = st.toggle("Use Research Cache", value=True)
    with col_b:
        max_case_studies = st.number_input("Max Case Studies", min_value=5, max_value=8, value=5)
    with col_c:
        max_clients = st.number_input("Max Client Matches", min_value=6, max_value=10, value=6)

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
                max_case_studies=int(max_case_studies),
                max_client_matches=int(max_clients),
                use_cache=use_cache,
                progress_callback=on_progress,
            )
            progress_bar.progress(1.0)
            status_text.text("Done!")

            result_df = pd.read_excel(output_path, sheet_name="prospects")
            with open(output_path, "rb") as f:
                st.session_state["pass1_output_bytes"] = f.read()
            st.session_state["pass1_result_df"] = result_df
        except Exception as e:
            st.error(f"Research failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    if "pass1_output_bytes" in st.session_state:
        result_df = st.session_state["pass1_result_df"]
        st.success(f"Research complete. {len(result_df)} prospects processed.")
        st.dataframe(result_df.head(10), use_container_width=True)
        st.download_button(
            "Download Pass-1-Output.xlsx",
            data=st.session_state["pass1_output_bytes"],
            file_name="Pass-1-Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ── Pass 2: Generate Messages ─────────────────────────────────────────────

def _render_pass2_tab():
    st.subheader("Pass 2: Generate Messages")
    st.caption(
        "Upload the reviewed Pass-1-Output.xlsx. Rows with Prospect_Technologies and no existing "
        "WARM_MESSAGE will have messages generated. Rows with empty Prospect_Technologies are skipped."
    )

    uploaded = st.file_uploader("Upload reviewed Pass-1-Output.xlsx", type=["xlsx"], key="pass2_upload")

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
        # Check both old and new column names for backwards compatibility
        msg = row.get("WARM_MESSAGE") or row.get("Message_to_send") or ""
        msg = str(msg).strip()
        if msg and msg.lower() != "nan":
            return False
        tech_research = str(row.get("Prospect_Technologies", "")).strip()
        return bool(tech_research) and tech_research.lower() != "nan"

    ready = sum(1 for _, row in df.iterrows() if _is_ready(row))
    skipped = len(df) - ready

    col1, col2 = st.columns(2)
    col1.metric("Ready to Generate", ready)
    col2.metric("Skipped", skipped)

    if ready == 0:
        st.info("No prospects to generate messages for — all rows either already have messages or have empty Prospect_Technologies.")
        with st.expander("Debug: skip reasons per row"):
            for idx, row in df.iterrows():
                msg = str(row.get("WARM_MESSAGE", "")).strip()
                tech = str(row.get("Prospect_Technologies", "")).strip()
                name = f"{row.get('First_Name', '')} {row.get('Last_Name', '')}".strip()
                has_msg = "YES" if msg else "NO"
                has_tech = "YES" if (tech and tech != "nan") else "NO"
                st.text(f"Row {idx}: {name} | WARM_MESSAGE={has_msg} | Prospect_Technologies={has_tech}")
            st.caption(f"Columns in file: {list(df.columns)}")
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

            result_df = pd.read_excel(output_path, sheet_name="prospects")
            with open(output_path, "rb") as f:
                st.session_state["pass2_output_bytes"] = f.read()
            st.session_state["pass2_result_df"] = result_df
            st.session_state["pass2_stats"] = stats
        except Exception as e:
            st.error(f"Message generation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    if "pass2_output_bytes" in st.session_state:
        stats = st.session_state["pass2_stats"]
        result_df = st.session_state["pass2_result_df"]
        st.success(f"Generated {stats['ready']} messages. {stats['skipped']} skipped.")
        preview_cols = ["First_Name", "Last_Name", "Company_Name", "WARM_MESSAGE"]
        available = [c for c in preview_cols if c in result_df.columns]
        st.dataframe(result_df[available].head(10), use_container_width=True)
        st.download_button(
            "Download Pass-2-Output.xlsx",
            data=st.session_state["pass2_output_bytes"],
            file_name="Pass-2-Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
