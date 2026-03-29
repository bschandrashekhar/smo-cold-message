"""Event Prospecting pipeline UI — Scrape Exhibitors + Enrich Prospects tabs."""

import os
import tempfile

import pandas as pd
import streamlit as st


def render():
    """Render the Event Prospecting pipeline tabs."""
    tabs = st.tabs(["Scrape Exhibitors", "Enrich Prospects"])

    # ── Tab 1: Scrape Exhibitors (Step 1) ────────────────────────────────
    with tabs[0]:
        st.header("Step 1: Scrape Exhibitors")
        st.caption("Upload an Excel file with exhibition names and exhibitor page links. The tool scrapes company lists and enriches them.")

        uploaded = st.file_uploader("Upload data.xlsx", type=["xlsx"], key="ep_scrape_upload")

        if uploaded is not None:
            try:
                df = pd.read_excel(uploaded)

                required = ["Exhibition", "Exhibitor Link"]
                missing = [c for c in required if c not in df.columns]
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                    return

                col1, col2 = st.columns(2)
                col1.metric("Exhibitions", df["Exhibition"].nunique())
                col2.metric("Exhibitor Links", len(df))

                st.subheader("Preview")
                st.dataframe(df, use_container_width=True)

                st.info("Shortlist criteria: Not IT service providers, revenue under $2B, preferably financial services, using Salesforce/Dell Boomi/Snowflake/.NET/Tableau/MS Fabric.")

                if st.button("Scrape Exhibitors", type="primary"):
                    st.session_state.pop("ep_scrape_output", None)
                    st.session_state.pop("ep_scrape_count", None)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def on_progress(current, total, text):
                        if total > 0:
                            progress_bar.progress(current / total)
                        status_text.text(text)

                    try:
                        from event_prospecting import exhibitor_scraper
                        result_df = exhibitor_scraper.scrape_exhibitors(df, progress_callback=on_progress)

                        progress_bar.progress(1.0)
                        status_text.text("Scraping complete!")

                        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                        tmp_out.close()
                        result_df.to_excel(tmp_out.name, index=False)

                        with open(tmp_out.name, "rb") as f:
                            st.session_state["ep_scrape_output"] = f.read()
                        st.session_state["ep_scrape_count"] = len(result_df)
                        os.unlink(tmp_out.name)
                    except NotImplementedError:
                        st.warning("Exhibitor scraping is not yet implemented.")
                    except Exception as e:
                        st.error(f"Scraping failed: {e}")

                if "ep_scrape_output" in st.session_state:
                    st.success(f"Found {st.session_state.get('ep_scrape_count', '?')} companies. Download the file below.")

                    result_df = pd.read_excel(st.session_state["ep_scrape_output"])
                    st.subheader("Results Preview")
                    st.dataframe(result_df, use_container_width=True)

                    st.download_button(
                        "Download companies.xlsx",
                        st.session_state["ep_scrape_output"],
                        file_name="companies.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )

            except Exception as e:
                st.error(f"Failed to read workbook: {e}")

    # ── Tab 2: Enrich Prospects (Step 2) ─────────────────────────────────
    with tabs[1]:
        st.header("Step 2: Enrich Prospects")
        st.caption("Upload the companies.xlsx from Step 1. The tool finds prospects by role and enriches contact details via Apollo.io.")

        uploaded2 = st.file_uploader("Upload companies.xlsx", type=["xlsx"], key="ep_enrich_upload")

        if uploaded2 is not None:
            try:
                df2 = pd.read_excel(uploaded2)

                required = ["Company Name"]
                missing = [c for c in required if c not in df2.columns]
                if missing:
                    st.error(f"Missing required columns: {', '.join(missing)}")
                    return

                col1, col2 = st.columns(2)
                col1.metric("Companies", len(df2))
                industries = df2.get("Industry Vertical")
                if industries is not None:
                    col2.metric("Industries", industries.nunique())

                st.subheader("Preview")
                st.dataframe(df2, use_container_width=True)

                st.info("Targeting: Large companies → Director-level, Small/mid → C-level, excluding CFOs.")

                if st.button("Find Prospects", type="primary"):
                    st.session_state.pop("ep_enrich_output", None)
                    st.session_state.pop("ep_enrich_count", None)

                    progress_bar2 = st.progress(0)
                    status_text2 = st.empty()

                    def on_progress2(current, total, text):
                        if total > 0:
                            progress_bar2.progress(current / total)
                        status_text2.text(text)

                    try:
                        from event_prospecting import prospect_finder
                        result_df2 = prospect_finder.find_prospects(df2, progress_callback=on_progress2)

                        progress_bar2.progress(1.0)
                        status_text2.text("Enrichment complete!")

                        tmp_out2 = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                        tmp_out2.close()
                        result_df2.to_excel(tmp_out2.name, index=False)

                        with open(tmp_out2.name, "rb") as f:
                            st.session_state["ep_enrich_output"] = f.read()
                        st.session_state["ep_enrich_count"] = len(result_df2)
                        os.unlink(tmp_out2.name)
                    except NotImplementedError:
                        st.warning("Prospect enrichment is not yet implemented.")
                    except Exception as e:
                        st.error(f"Enrichment failed: {e}")

                if "ep_enrich_output" in st.session_state:
                    st.success(f"Found {st.session_state.get('ep_enrich_count', '?')} prospects. Download the file below.")

                    result_df2 = pd.read_excel(st.session_state["ep_enrich_output"])
                    st.subheader("Results Preview")
                    st.dataframe(result_df2, use_container_width=True)

                    st.download_button(
                        "Download prospects.xlsx",
                        st.session_state["ep_enrich_output"],
                        file_name="prospects.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )

            except Exception as e:
                st.error(f"Failed to read workbook: {e}")
