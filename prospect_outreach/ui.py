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
    tabs = st.tabs(["Pass 1: Research", "Pass 2: Generate Messages", "Settings", "Test Tab"])

    with tabs[0]:
        _render_pass1_tab()
    with tabs[1]:
        _render_pass2_tab()
    with tabs[2]:
        _render_settings_tab()
    with tabs[3]:
        _render_test_tab()


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
        max_clients = st.number_input("Max Client Matches", min_value=5, max_value=10, value=5)

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


# ── Test Tab: Verbose Pass 1 Pipeline ─────────────────────────────────────

def _test_cache_badge(source: str) -> str:
    badges = {
        "CACHE HIT": "🟢 CACHE HIT",
        "CACHE STALE": "🟡 CACHE STALE (re-fetched via Serper)",
        "CACHE MISS": "🔴 CACHE MISS (fetched via Serper)",
        "CACHE SKIPPED": "⚪ CACHE SKIPPED (fetched via Serper)",
        "DEDUPED": "⚪ DEDUPED (reused from earlier row)",
    }
    return badges.get(source, source)


def _test_get_company_research(company_name: str, website: str, use_cache: bool):
    import json
    from prospect_outreach.research_pipeline import (
        _get_supabase, _is_cache_stale, _run_company_research,
    )
    source = None
    data = None
    if use_cache:
        try:
            sb = _get_supabase()
            result = sb.table("Cache_Prospect_Company_Research").select("*").eq("Website", website).execute()
            if result.data:
                row = result.data[0]
                if not _is_cache_stale(row.get("Date_of_Research")):
                    source = "CACHE HIT"
                    cr = row["Company_Research"]
                    data = json.loads(cr) if isinstance(cr, str) else cr
                else:
                    source = "CACHE STALE"
            else:
                source = "CACHE MISS"
        except Exception as e:
            source = f"CACHE ERROR: {e}"
    else:
        source = "CACHE SKIPPED"
    if data is None:
        data = _run_company_research(company_name, website)
    return data, source


def _test_get_prospect_research(prospect_name: str, designation: str, company_name: str,
                                city: str, country: str, email: str,
                                linkedin_url: str, use_cache: bool):
    import json
    from prospect_outreach.research_pipeline import (
        _get_supabase, _is_cache_stale, _run_prospect_research,
    )
    source = None
    data = None
    if use_cache:
        try:
            sb = _get_supabase()
            result = sb.table("Cache_Prospect_Contact_Research").select("*").eq("Email", email).execute()
            if result.data:
                row = result.data[0]
                if not _is_cache_stale(row.get("Date_of_Research")):
                    source = "CACHE HIT"
                    pr = row["Prospect_Research"]
                    data = json.loads(pr) if isinstance(pr, str) else pr
                else:
                    source = "CACHE STALE"
            else:
                source = "CACHE MISS"
        except Exception as e:
            source = f"CACHE ERROR: {e}"
    else:
        source = "CACHE SKIPPED"
    if data is None:
        data = _run_prospect_research(prospect_name, designation, company_name,
                                      city, country, linkedin_url)
    return data, source


def _render_test_tab():
    st.subheader("Test Tab: Verbose Pass 1 Pipeline")
    st.caption(
        "Runs the full Pass 1 research pipeline row-by-row with verbose output. "
        "Use cache toggles to control whether cached data is used for company and prospect research."
    )

    uploaded = st.file_uploader("Upload data.xlsx", type=["xlsx"], key="test_upload")
    if uploaded is None:
        return

    file_bytes = uploaded.read()

    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="prospects")
    except Exception as e:
        st.error(f"Could not read 'prospects' sheet: {e}")
        return

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.divider()

    col_ca, col_cb = st.columns(2)
    with col_ca:
        use_company_cache = st.toggle("Use Company Cache", value=False, key="test_company_cache")
    with col_cb:
        use_prospect_cache = st.toggle("Use Prospect Cache", value=False, key="test_prospect_cache")

    col_ta, col_tb, col_tc = st.columns(3)
    with col_ta:
        generate_intent = st.toggle("Generate Intent Score", value=False, key="test_intent")
    with col_tb:
        max_case_studies = st.number_input("Max Case Studies", min_value=1, max_value=20, value=5, key="test_cs")
    with col_tc:
        max_clients = st.number_input("Max Client Matches", min_value=5, max_value=10, value=5, key="test_cl")

    if not st.button("Run Test", type="primary", key="run_test_btn"):
        return

    from prospect_outreach.research_pipeline import (
        _extract_technologies, _build_research_summary, _score_intent, _apply_emea_coding,
    )
    from client_referencing.brand_matcher import find_brand_match
    from client_referencing.casestudy_matcher import find_casestudy_matches
    from client_referencing.matcher import find_matches

    company_cache: dict = {}  # website -> (research_dict, source_label)
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        prospect_name = f"{row.get('First_Name', '')} {row.get('Last_Name', '')}".strip()
        designation = str(row.get("Designation", ""))
        company_name = str(row.get("Company_Name", ""))
        email = str(row.get("Email", ""))
        website = str(row.get("Website", ""))
        industry = str(row.get("Industry", ""))
        country = str(row.get("Country", ""))
        city = str(row.get("City", ""))
        state = str(row.get("State", ""))
        _li = row.get("LinkedIn", "") if "LinkedIn" in df.columns else ""
        linkedin_url = "" if not _li or pd.isna(_li) else str(_li).strip()

        with st.expander(f"Prospect {i+1}/{total}: {prospect_name} | {designation} | {company_name}", expanded=(i == 0)):

            # ── Step 1: Brand Match ──────────────────────────────────────
            st.markdown("#### Step 1 — Brand Match")
            st.write(f"**Params:** `prospect_industry={industry!r}`")
            try:
                brand_result = find_brand_match(prospect_industry=industry)
                c1, c2 = st.columns(2)
                c1.metric("Brand", brand_result.get("brand", "—"))
                c2.metric("Similarity", f"{brand_result.get('similarity_score', 0):.3f}")
                st.write(f"**Matched term:** `{brand_result.get('matched_industry_term')}`")
                with st.expander("Full return"):
                    st.json({
                        "brand": brand_result.get("brand"),
                        "matched_industry_term": brand_result.get("matched_industry_term"),
                        "similarity_score": brand_result.get("similarity_score"),
                        "all_matches": brand_result.get("all_matches", []),
                        "debug_log": brand_result.get("debug_log", []),
                    })
            except Exception as e:
                st.error(f"Brand match failed: {e}")
                brand_result = {"brand": "CloudChillies"}

            st.divider()

            # ── Step 2: Company Research ─────────────────────────────────
            st.markdown("#### Step 2 — Company Research")
            st.write(f"**Params:** `company_name={company_name!r}`, `website={website!r}`")
            try:
                if website in company_cache:
                    company_research, co_source = company_cache[website]
                    co_source = "DEDUPED"
                else:
                    company_research, co_source = _test_get_company_research(
                        company_name, website, use_company_cache
                    )
                    company_cache[website] = (company_research, co_source)
                st.info(_test_cache_badge(co_source))
                st.caption("↓ COMPANY_RESEARCH — synthesized by Claude from Serper results. Cached in Supabase keyed by Website.")
                with st.expander("COMPANY_RESEARCH dict", expanded=True):
                    st.json(company_research)
            except Exception as e:
                st.error(f"Company research failed: {e}")
                company_research = {"company_name": company_name}

            st.divider()

            # ── Step 3: Technology Extraction ────────────────────────────
            st.markdown("#### Step 3 — Technology Extraction")
            st.write("**Params:** `company_research` (dict above)")
            try:
                prospect_technologies = _extract_technologies(company_research)
                st.write(f"**Return:** `{prospect_technologies}`")
            except Exception as e:
                st.error(f"Tech extraction failed: {e}")
                prospect_technologies = ""

            st.divider()

            # ── Step 4: Prospect Research ────────────────────────────────
            st.markdown("#### Step 4 — Prospect Research")
            st.write(
                f"**Params:** `prospect_name={prospect_name!r}`, `designation={designation!r}`, "
                f"`company_name={company_name!r}`, `city={city!r}`, `country={country!r}`, "
                f"`email={email!r}`, `linkedin_url={linkedin_url!r}`"
            )
            try:
                prospect_research, pr_source = _test_get_prospect_research(
                    prospect_name, designation, company_name,
                    city, country, email, linkedin_url, use_prospect_cache
                )
                st.info(_test_cache_badge(pr_source))
                st.caption("↓ PROSPECT_RESEARCH — synthesized by Claude from Serper/LinkedIn results. Cached in Supabase keyed by Email.")
                with st.expander("PROSPECT_RESEARCH dict", expanded=True):
                    st.json(prospect_research)
            except Exception as e:
                st.error(f"Prospect research failed: {e}")
                prospect_research = {}

            st.divider()

            # ── Step 5: Research Summary ─────────────────────────────────
            st.markdown("#### Step 5 — Research Summary")
            st.write("**Params:** `company_research` (Step 2) + `prospect_research` (Step 4) merged")
            st.caption("Claude merges both dicts and produces 5-6 plain-text bullets. This is stored in the `Research_Summary` column in the output Excel.")
            try:
                research_summary = _build_research_summary(company_research, prospect_research)
                st.markdown("**→ `Research_Summary` (stored in Excel):**")
                st.text(research_summary)
                st.caption(f"Length: {len(research_summary)} chars")
            except Exception as e:
                st.error(f"Research summary failed: {e}")
                research_summary = ""

            st.divider()

            # ── Step 6: Case Study Matching ──────────────────────────────
            st.markdown("#### Step 6 — Case Study Matching")
            cs_params = {
                "prospect_context": research_summary[:200] + "..." if len(research_summary) > 200 else research_summary,
                "prospect_industry": industry,
                "prospect_technologies": prospect_technologies,
                "prospect_country": "",
                "max_matches": int(max_case_studies),
            }
            st.write("**Params:**")
            st.json(cs_params)
            try:
                cs_result = find_casestudy_matches(
                    prospect_context=research_summary,
                    prospect_industry=industry,
                    prospect_technologies=prospect_technologies,
                    prospect_country="",
                    max_matches=int(max_case_studies),
                )
                cs_matches = cs_result.get("matches", [])
                st.write(f"**{len(cs_matches)} match(es) returned**")
                for j, m in enumerate(cs_matches):
                    md = m.to_dict() if hasattr(m, "to_dict") else (m if isinstance(m, dict) else vars(m))
                    with st.expander(f"Match {j+1}: {md.get('casestudy_name', '—')} | {md.get('client_name', '—')} | score={md.get('final_score', 0):.3f} | tier={md.get('tier', '—')}"):
                        st.json(md)
                with st.expander("debug_log"):
                    st.json(cs_result.get("debug_log", []))
            except Exception as e:
                st.error(f"Case study matching failed: {e}")

            st.divider()

            # ── Step 7: Client Reference Matching ────────────────────────
            st.markdown("#### Step 7 — Client Reference Matching")
            cl_params = {
                "prospect_industry": industry,
                "prospect_technologies": prospect_technologies,
                "prospect_country": country,
                "max_matches": int(max_clients),
            }
            st.write("**Params:**")
            st.json(cl_params)
            try:
                cl_result = find_matches(
                    prospect_industry=industry,
                    prospect_technologies=prospect_technologies,
                    prospect_country=country,
                    max_matches=int(max_clients),
                )
                cl_matches = cl_result.get("matches", [])
                st.write(f"**{len(cl_matches)} match(es) returned**")
                col_if, col_tc2 = st.columns(2)
                col_if.write(f"**industry_filter_applied:** {cl_result.get('industry_filter_applied')}")
                col_tc2.write(f"**total_candidates:** {cl_result.get('total_candidates')}")
                for j, m in enumerate(cl_matches):
                    md = m.to_dict() if hasattr(m, "to_dict") else (m if isinstance(m, dict) else vars(m))
                    with st.expander(f"Match {j+1}: {md.get('client_name', '—')} | {md.get('client_industry', '—')} | {md.get('client_geography', '—')} | score={md.get('final_score', 0):.3f} | source={md.get('match_source', '—')}"):
                        st.json(md)
                with st.expander("debug_log"):
                    st.json(cl_result.get("debug_log", []))
            except Exception as e:
                st.error(f"Client reference matching failed: {e}")

            st.divider()

            # ── Step 8: Intent Score ─────────────────────────────────────
            if generate_intent:
                st.markdown("#### Step 8 — Intent Score")
                st.write(f"**Params:** `prospect_name={prospect_name!r}`, `designation={designation!r}`, `company_name={company_name!r}`, `research_summary` (above)")
                try:
                    score = _score_intent(prospect_name, designation, company_name, research_summary)
                    st.metric("Intent Score", score)
                except Exception as e:
                    st.error(f"Intent scoring failed: {e}")
                st.divider()

            # ── Step 9: EMEA Coding ──────────────────────────────────────
            st.markdown("#### Step 9 — EMEA Coding")
            st.write(f"**Params:** `country={country!r}`")
            try:
                coded_country = _apply_emea_coding(country)
                st.write(f"**Return:** `{country}` → `{coded_country}`")
            except Exception as e:
                st.error(f"EMEA coding failed: {e}")
