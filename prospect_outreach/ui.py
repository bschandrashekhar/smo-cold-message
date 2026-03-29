"""Prospect Outreach pipeline UI — Research + Generate Messages tabs."""

import os
import tempfile

import pandas as pd
import streamlit as st

from prospect_outreach import prospect_research, message_generator


def render():
    """Render the Prospect Outreach pipeline tabs."""
    tabs = st.tabs(["Research", "Generate Messages"])

    # ── Tab 1: Research ──────────────────────────────────────────────────
    with tabs[0]:
        st.header("Pass 1: Research Prospects")
        st.caption("Upload your data.xlsx, run AI research, then download and review before generating messages.")

        uploaded = st.file_uploader("Upload data.xlsx", type=["xlsx"], key="po_research_upload")

        if uploaded is not None:
            try:
                xls = pd.ExcelFile(uploaded)
                if "prospects" not in xls.sheet_names:
                    st.error("Workbook must contain a 'prospects' sheet.")
                    return

                df = pd.read_excel(xls, sheet_name="prospects")

                if not prospect_research.validate_prospects_sheet(df):
                    st.error("Missing required columns: Prospect Name, Designation, Company Name, Website, Location of Prospect")
                    return

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Prospects", len(df))
                col2.metric("Unique Companies", df["Company Name"].nunique())
                has_dates = "dates" in xls.sheet_names
                col3.metric("Dates Sheet", "Found" if has_dates else "Missing")

                st.subheader("Preview")
                st.dataframe(df, use_container_width=True)

                if st.button("Run Research", type="primary"):
                    st.session_state.pop("po_research_output", None)
                    st.session_state.pop("po_research_count", None)

                    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    tmp_in.write(uploaded.getbuffer())
                    tmp_in.flush()
                    tmp_in.close()

                    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    tmp_out.close()

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def on_progress(current, total, text):
                        if total > 0:
                            progress_bar.progress(current / total)
                        status_text.text(text)

                    try:
                        prospect_research.research_workbook(
                            tmp_in.name, tmp_out.name,
                            progress_callback=on_progress,
                        )
                        progress_bar.progress(1.0)
                        status_text.text("Research complete!")

                        with open(tmp_out.name, "rb") as f:
                            st.session_state["po_research_output"] = f.read()
                        st.session_state["po_research_count"] = len(df)
                    except Exception as e:
                        st.error(f"Research failed: {e}")
                    finally:
                        os.unlink(tmp_in.name)

                if "po_research_output" in st.session_state:
                    st.success(f"Research complete for {st.session_state.get('po_research_count', '?')} prospects. Download the file below.")

                    result_df = pd.read_excel(st.session_state["po_research_output"], sheet_name="prospects")
                    st.subheader("Results Preview")
                    st.dataframe(result_df, use_container_width=True)

                    st.download_button(
                        "Download data_output.xlsx",
                        st.session_state["po_research_output"],
                        file_name="data_output.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )

            except Exception as e:
                st.error(f"Failed to read workbook: {e}")

    # ── Tab 2: Generate Messages ─────────────────────────────────────────
    with tabs[1]:
        st.header("Pass 2: Generate Messages")
        st.caption("Upload the reviewed data_output.xlsx to generate personalized outreach messages.")

        uploaded2 = st.file_uploader("Upload reviewed data_output.xlsx", type=["xlsx"], key="po_gen_upload")

        if uploaded2 is not None:
            try:
                df2 = pd.read_excel(uploaded2, sheet_name="prospects")
            except Exception as e:
                st.error(f"Failed to read prospects sheet: {e}")
                df2 = None

            if df2 is not None:
                ready = 0
                skip_has_message = 0
                skip_no_research = 0
                skip_no_score = 0

                for _, row in df2.iterrows():
                    msg = row.get("Message to send")
                    if pd.notna(msg) and str(msg).strip() != "":
                        skip_has_message += 1
                        continue
                    research = row.get("Research Summary")
                    if pd.isna(research) or str(research).strip() == "":
                        skip_no_research += 1
                        continue
                    intent = row.get("Intent Score")
                    if pd.isna(intent) or str(intent).strip() == "":
                        skip_no_score += 1
                        continue
                    ready += 1

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ready to Generate", ready)
                col2.metric("Already Has Message", skip_has_message)
                col3.metric("No Research (skipped)", skip_no_research)
                col4.metric("No Score (skipped)", skip_no_score)

                st.subheader("Preview")
                st.dataframe(df2, use_container_width=True)

                if ready == 0:
                    st.warning("No prospects ready for message generation.")
                elif st.button("Generate Messages", type="primary"):
                    st.session_state.pop("po_messages_output", None)
                    st.session_state.pop("po_messages_count", None)

                    tmp_in2 = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    tmp_in2.write(uploaded2.getbuffer())
                    tmp_in2.flush()
                    tmp_in2.close()

                    tmp_out2 = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    tmp_out2.close()

                    progress_bar2 = st.progress(0)
                    status_text2 = st.empty()

                    def on_progress2(current, total, text):
                        if total > 0:
                            progress_bar2.progress(current / total)
                        status_text2.text(text)

                    try:
                        message_generator.generate_messages(
                            tmp_in2.name, tmp_out2.name,
                            progress_callback=on_progress2,
                        )
                        progress_bar2.progress(1.0)
                        status_text2.text("Generation complete!")

                        with open(tmp_out2.name, "rb") as f:
                            st.session_state["po_messages_output"] = f.read()
                        st.session_state["po_messages_count"] = ready
                    except Exception as e:
                        st.error(f"Message generation failed: {e}")
                    finally:
                        os.unlink(tmp_in2.name)

                if "po_messages_output" in st.session_state:
                    st.success(f"Messages generated for {st.session_state.get('po_messages_count', '?')} prospects. Download the file below.")

                    result_df2 = pd.read_excel(st.session_state["po_messages_output"], sheet_name="prospects")
                    st.subheader("Results Preview")
                    st.dataframe(result_df2, use_container_width=True)

                    st.download_button(
                        "Download data_final.xlsx",
                        st.session_state["po_messages_output"],
                        file_name="data_final.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )
