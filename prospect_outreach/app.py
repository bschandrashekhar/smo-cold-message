import sys
import os

# Allow absolute imports when run directly by Streamlit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime

from prospect_outreach import brand_knowledge, prospect_research, message_generator, config

st.set_page_config(page_title="Prospect Outreach Generator", layout="wide")


def main():
    st.title("Prospect Outreach Generator")

    tabs = st.tabs(["Research", "Generate Messages", "Settings"])

    # ── Tab 1: Research ──────────────────────────────────────────────────
    with tabs[0]:
        st.header("Pass 1: Research Prospects")
        st.caption("Upload your data.xlsx, run AI research, then download and review before generating messages.")

        uploaded = st.file_uploader("Upload data.xlsx", type=["xlsx"], key="research_upload")

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

                # Show summary stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Prospects", len(df))
                col2.metric("Unique Companies", df["Company Name"].nunique())
                has_dates = "dates" in xls.sheet_names
                col3.metric("Dates Sheet", "Found" if has_dates else "Missing")

                st.subheader("Preview")
                st.dataframe(df, use_container_width=True)

                if st.button("Run Research", type="primary"):
                    # Save uploaded file to temp
                    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    tmp_in.write(uploaded.getbuffer())
                    tmp_in.flush()
                    tmp_in.close()

                    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                    tmp_out.close()

                    # Progress tracking
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
                        st.success(f"Research completed for {len(df)} prospects.")

                        # Show results preview
                        result_df = pd.read_excel(tmp_out.name, sheet_name="prospects")
                        st.subheader("Results Preview")
                        st.dataframe(result_df, use_container_width=True)

                        with open(tmp_out.name, "rb") as f:
                            st.download_button(
                                "Download data_output.xlsx",
                                f.read(),
                                file_name="data_output.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary",
                            )
                    except Exception as e:
                        st.error(f"Research failed: {e}")
                    finally:
                        os.unlink(tmp_in.name)

            except Exception as e:
                st.error(f"Failed to read workbook: {e}")

    # ── Tab 2: Generate Messages ─────────────────────────────────────────
    with tabs[1]:
        st.header("Pass 2: Generate Messages")
        st.caption("Upload the reviewed data_output.xlsx to generate personalized outreach messages.")

        uploaded2 = st.file_uploader("Upload reviewed data_output.xlsx", type=["xlsx"], key="gen_upload")

        if uploaded2 is not None:
            try:
                df2 = pd.read_excel(uploaded2, sheet_name="prospects")
            except Exception as e:
                st.error(f"Failed to read prospects sheet: {e}")
                df2 = None

            if df2 is not None:
                # Calculate skip logic summary
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

                # Show summary
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
                    # Save to temp
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
                        st.success(f"Messages generated for {ready} prospects.")

                        # Show results
                        result_df2 = pd.read_excel(tmp_out2.name, sheet_name="prospects")
                        st.subheader("Results Preview")
                        st.dataframe(result_df2, use_container_width=True)

                        with open(tmp_out2.name, "rb") as f:
                            st.download_button(
                                "Download data_final.xlsx",
                                f.read(),
                                file_name="data_final.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary",
                            )
                    except Exception as e:
                        st.error(f"Message generation failed: {e}")
                    finally:
                        os.unlink(tmp_in2.name)

    # ── Tab 3: Settings ──────────────────────────────────────────────────
    with tabs[2]:
        st.header("Settings")

        st.subheader("Brand Knowledge Status")
        status = brand_knowledge.are_brand_files_present()

        for name, exists in status.items():
            path = config.BRAND_JSONS[name]
            col1, col2 = st.columns([3, 1])

            if exists:
                mod_time = datetime.fromtimestamp(path.stat().st_mtime)
                col1.success(f"**{name}** — Last updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                col1.warning(f"**{name}** — Not generated yet")

        for name, exists in status.items():
            if exists:
                with st.expander(f"View {name} Profile"):
                    try:
                        profile = brand_knowledge.load_brand_profile(name)
                        st.json(profile)
                    except Exception as e:
                        st.error(f"Failed to load: {e}")

        st.divider()

        if st.button("Refresh Brand Knowledge", type="primary"):
            with st.spinner("Scraping brand websites with Claude web search..."):
                try:
                    brand_knowledge.refresh_brand_profiles()
                    st.success("Brand profiles refreshed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to refresh brand profiles: {e}")



if __name__ == "__main__":
    main()
