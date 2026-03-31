"""Event Prospecting pipeline UI — Scrape Exhibitors + Enrich Prospects tabs."""

import os
import tempfile

import pandas as pd
import streamlit as st


def _write_sheets_to_excel(sheets: dict, path: str) -> None:
    """Write a dict of {sheet_name: DataFrame} to an Excel file."""
    if not sheets:
        raise ValueError("No sheets to write — result was empty")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            # Excel sheet names max 31 chars
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)


def _read_sheets_from_excel(data: bytes) -> dict:
    """Read all sheets from Excel bytes into {sheet_name: DataFrame}."""
    xls = pd.ExcelFile(data)
    return {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}


_BUILD = "v3.0"  # bump to verify deployments


def render():
    """Render the Event Prospecting pipeline tabs."""
    st.caption(f"Build: {_BUILD}")
    tabs = st.tabs(["Enrich Exhibitors", "Find Prospects"])

    # ── Tab 1: Enrich Exhibitors (Step 1) ────────────────────────────────
    with tabs[0]:
        st.header("Step 1: Enrich Exhibitors")
        st.caption(
            "Upload an Excel file with sheets named `<ExhibitionName>-Exhibitors`. "
            "Each sheet needs at minimum a **Company Name** column (and optionally **Website**). "
            "The tool enriches each company via Apollo.io and applies shortlist filtering."
        )

        uploaded = st.file_uploader("Upload exhibitors.xlsx", type=["xlsx"], key="ep_scrape_upload")

        if uploaded is not None:
            try:
                xls = pd.ExcelFile(uploaded)
                exhibitor_sheets = {
                    name: pd.read_excel(xls, sheet_name=name)
                    for name in xls.sheet_names
                }

                # Validate at least one sheet has Company Name
                valid_sheets = {
                    name: df for name, df in exhibitor_sheets.items()
                    if "Company Name" in df.columns
                }

                if not valid_sheets:
                    st.error("No sheets found with a 'Company Name' column. Each sheet needs at least a 'Company Name' column.")
                    return

                total_companies = sum(len(df) for df in valid_sheets.values())
                col1, col2 = st.columns(2)
                col1.metric("Exhibition Sheets", len(valid_sheets))
                col2.metric("Total Companies", total_companies)

                for sheet_name, sheet_df in valid_sheets.items():
                    with st.expander(f"{sheet_name} ({len(sheet_df)} companies)"):
                        st.dataframe(sheet_df, use_container_width=True)

                st.info(
                    "Shortlist criteria: Not IT service providers, revenue under $2B, "
                    "preferably financial services, using Salesforce/Dell Boomi/Snowflake/.NET/Tableau/MS Fabric."
                )

                if st.button("Enrich & Shortlist", type="primary"):
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

                        sheets, logs = exhibitor_scraper.enrich_exhibitors(
                            valid_sheets, progress_callback=on_progress
                        )

                        progress_bar.progress(1.0)
                        status_text.text("Enrichment complete!")

                        # Persist logs and results in session state
                        st.session_state["ep_scrape_logs"] = logs

                        if sheets:
                            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                            tmp_out.close()
                            _write_sheets_to_excel(sheets, tmp_out.name)

                            with open(tmp_out.name, "rb") as f:
                                st.session_state["ep_scrape_output"] = f.read()
                            st.session_state["ep_scrape_sheets"] = list(sheets.keys())
                            os.unlink(tmp_out.name)
                        else:
                            st.session_state.pop("ep_scrape_output", None)
                            st.session_state["ep_scrape_sheets"] = []

                    except Exception as e:
                        st.error(f"Enrichment failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())

                # Show enrichment log if available
                if "ep_scrape_logs" in st.session_state:
                    scrape_logs = st.session_state["ep_scrape_logs"]
                    has_output = "ep_scrape_output" in st.session_state
                    with st.expander("Enrichment Log", expanded=not has_output):
                        st.code("\n".join(scrape_logs), language=None)

                if "ep_scrape_output" in st.session_state:
                    sheet_names = st.session_state.get("ep_scrape_sheets", [])
                    shortlist_count = sum(1 for s in sheet_names if "Shortlist" in s)
                    all_count = sum(1 for s in sheet_names if "-All" in s)

                    if shortlist_count > 0:
                        st.success(f"Enriched {all_count} exhibition(s), {shortlist_count} with shortlisted companies. Download below.")
                    else:
                        st.warning(
                            f"Enriched {all_count} exhibition(s) but no companies passed the shortlist. "
                            "Download the All sheets below to review what was found."
                        )

                    # Preview each exhibition sheet in its own expander
                    all_sheets = _read_sheets_from_excel(st.session_state["ep_scrape_output"])
                    for sheet_name, sheet_df in all_sheets.items():
                        with st.expander(f"{sheet_name} ({len(sheet_df)} companies)"):
                            st.dataframe(sheet_df, use_container_width=True)

                    st.download_button(
                        "Download enriched_exhibitors.xlsx",
                        st.session_state["ep_scrape_output"],
                        file_name="enriched_exhibitors.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                    )
                elif "ep_scrape_sheets" in st.session_state and not st.session_state["ep_scrape_sheets"]:
                    st.warning("No companies could be enriched. Check the Enrichment Log above for details.")

            except Exception as e:
                st.error(f"Failed to read workbook: {e}")

    # ── Tab 2: Find Prospects (Step 2) ───────────────────────────────────
    with tabs[1]:
        st.header("Step 2: Find Prospects")
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
