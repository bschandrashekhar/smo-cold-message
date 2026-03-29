"""Event Prospecting pipeline UI — Scrape Exhibitors + Enrich Prospects tabs."""

import os
import tempfile

import pandas as pd
import streamlit as st


def _write_sheets_to_excel(sheets: dict, path: str) -> None:
    """Write a dict of {sheet_name: DataFrame} to an Excel file."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            # Excel sheet names max 31 chars
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)


def _read_sheets_from_excel(data: bytes) -> dict:
    """Read all sheets from Excel bytes into {sheet_name: DataFrame}."""
    xls = pd.ExcelFile(data)
    return {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}


def render():
    """Render the Event Prospecting pipeline tabs."""
    tabs = st.tabs(["Scrape Exhibitors", "Enrich Prospects"])

    # ── Tab 1: Scrape Exhibitors (Step 1) ────────────────────────────────
    with tabs[0]:
        st.header("Step 1: Scrape Exhibitors")
        st.caption(
            "Upload an Excel file with exhibition names and exhibitor page links. "
            "The tool scrapes company lists into per-exhibition worksheets."
        )

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

                st.info(
                    "Shortlist criteria: Not IT service providers, revenue under $2B, "
                    "preferably financial services, using Salesforce/Dell Boomi/Snowflake/.NET/Tableau/MS Fabric."
                )

                if st.button("Scrape Exhibitors", type="primary"):
                    st.session_state.pop("ep_scrape_output", None)
                    st.session_state.pop("ep_scrape_sheets", None)

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def on_progress(current, total, text):
                        if total > 0:
                            progress_bar.progress(current / total)
                        status_text.text(text)

                    try:
                        from event_prospecting import exhibitor_scraper

                        sheets = exhibitor_scraper.scrape_exhibitors(df, progress_callback=on_progress)

                        progress_bar.progress(1.0)
                        status_text.text("Scraping complete!")

                        if not sheets:
                            st.warning("No companies passed the shortlist criteria. Try different exhibitions or adjust criteria.")
                        else:
                            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                            tmp_out.close()
                            _write_sheets_to_excel(sheets, tmp_out.name)

                            with open(tmp_out.name, "rb") as f:
                                st.session_state["ep_scrape_output"] = f.read()
                            st.session_state["ep_scrape_sheets"] = list(sheets.keys())
                            os.unlink(tmp_out.name)
                    except Exception as e:
                        st.error(f"Scraping failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                if "ep_scrape_output" in st.session_state:
                    sheet_names = st.session_state.get("ep_scrape_sheets", [])
                    st.success(f"Scraped {len(sheet_names)} exhibition(s). Download the file below.")

                    # Preview each exhibition sheet in its own expander
                    all_sheets = _read_sheets_from_excel(st.session_state["ep_scrape_output"])
                    for sheet_name, sheet_df in all_sheets.items():
                        with st.expander(f"{sheet_name} ({len(sheet_df)} companies)"):
                            st.dataframe(sheet_df, use_container_width=True)

                    st.download_button(
                        "Download exhibitors.xlsx",
                        st.session_state["ep_scrape_output"],
                        file_name="exhibitors.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )

            except Exception as e:
                st.error(f"Failed to read workbook: {e}")

    # ── Tab 2: Enrich Prospects (Step 2) ─────────────────────────────────
    with tabs[1]:
        st.header("Step 2: Enrich Prospects")
        st.caption(
            "Upload the exhibitors.xlsx from Step 1. The tool finds prospects by role "
            "and adds contact details (via Apollo.io) to the right of each company row."
        )

        uploaded2 = st.file_uploader("Upload exhibitors.xlsx", type=["xlsx"], key="ep_enrich_upload")

        if uploaded2 is not None:
            try:
                xls = pd.ExcelFile(uploaded2)
                sheets_in = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}

                total_companies = sum(len(df) for df in sheets_in.values())
                col1, col2 = st.columns(2)
                col1.metric("Exhibition Sheets", len(sheets_in))
                col2.metric("Total Companies", total_companies)

                # Preview each sheet
                for sheet_name, sheet_df in sheets_in.items():
                    with st.expander(f"{sheet_name} ({len(sheet_df)} companies)"):
                        st.dataframe(sheet_df, use_container_width=True)

                st.info("Targeting: Large companies → Director-level, Small/mid → C-level, excluding CFOs.")

                if st.button("Find Prospects", type="primary"):
                    st.session_state.pop("ep_enrich_output", None)
                    st.session_state.pop("ep_enrich_sheets", None)

                    progress_bar2 = st.progress(0)
                    status_text2 = st.empty()

                    def on_progress2(current, total, text):
                        if total > 0:
                            progress_bar2.progress(current / total)
                        status_text2.text(text)

                    try:
                        from event_prospecting import prospect_finder

                        enriched_sheets = prospect_finder.find_prospects(
                            sheets_in, progress_callback=on_progress2
                        )

                        progress_bar2.progress(1.0)
                        status_text2.text("Enrichment complete!")

                        tmp_out2 = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                        tmp_out2.close()
                        _write_sheets_to_excel(enriched_sheets, tmp_out2.name)

                        with open(tmp_out2.name, "rb") as f:
                            st.session_state["ep_enrich_output"] = f.read()
                        st.session_state["ep_enrich_sheets"] = list(enriched_sheets.keys())
                        os.unlink(tmp_out2.name)
                    except Exception as e:
                        st.error(f"Enrichment failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                if "ep_enrich_output" in st.session_state:
                    sheet_names = st.session_state.get("ep_enrich_sheets", [])
                    st.success(f"Enriched {len(sheet_names)} exhibition(s). Download the file below.")

                    all_sheets = _read_sheets_from_excel(st.session_state["ep_enrich_output"])
                    for sheet_name, sheet_df in all_sheets.items():
                        with st.expander(f"{sheet_name} ({len(sheet_df)} rows)"):
                            st.dataframe(sheet_df, use_container_width=True)

                    st.download_button(
                        "Download enriched.xlsx",
                        st.session_state["ep_enrich_output"],
                        file_name="enriched.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )

            except Exception as e:
                st.error(f"Failed to read workbook: {e}")
