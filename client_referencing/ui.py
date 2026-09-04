"""Client Referencing pipeline UI — Data Sync + VectorMatch + VectorCasestudyMatch tabs."""

import io

import pandas as pd
import streamlit as st


def render():
    """Render the Admin/Debug Tools pipeline with tabs."""
    tabs = st.tabs(["Data Sync", "Client Matcher", "Casestudy Matcher", "Brand Matcher", "Pass 1: Research (Test Mode)"])

    with tabs[0]:
        _render_sync_tab()

    with tabs[1]:
        _render_vector_match_tab()

    with tabs[2]:
        _render_casestudy_match_tab()

    with tabs[3]:
        _render_brand_match_tab()

    with tabs[4]:
        _render_test_tab()


# ── Data Sync Tab ────────────────────────────────────────────────────────

def _render_sync_tab():
    """Render the sync UI (upload Excel, preview diff, apply)."""
    st.caption(
        "This contains controls to refresh the data about clients, casestudies and brand JSONs. "
        "This is an admin tool to be used carefully. "
        "The data is complete delete and reinsert, so exercise extreme caution."
    )
    with st.expander("Client Reference Data Refresh", expanded=False):
        _render_client_referencing_sync()
    with st.expander("Industry Reference Data Refresh", expanded=False):
        _render_industry_reference_sync()
    with st.expander("Casestudies Reference Data Refresh", expanded=False):
        _render_casestudy_sync()
    with st.expander("Brand Reference Refresh", expanded=False):
        _render_brand_reference_refresh()


# ── Client Reference Data Refresh ────────────────────────────────────────

def _render_client_referencing_sync():
    """Client Reference Data Refresh section."""
    st.caption(
        "Upload the client-referencing-data.xlsx file (with Clients + Technologies worksheets) "
        "to sync with Supabase. The tool compares your Excel against the database and shows "
        "what will be created, updated, or deleted before applying changes."
    )

    uploaded = st.file_uploader(
        "Upload client-referencing-data.xlsx", type=["xlsx"], key="cr_upload"
    )

    if uploaded is None:
        _show_current_data()
        return

    from client_referencing.sync import read_excel, fetch_existing, compute_diff, apply_sync

    try:
        excel_data = read_excel(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to parse Excel: {e}")
        return

    st.success(
        f"Parsed **{len(excel_data['clients'])}** clients and "
        f"**{len(excel_data['technologies'])}** technologies from Excel."
    )

    with st.spinner("Fetching current data from Supabase..."):
        try:
            db_data = fetch_existing()
        except Exception as e:
            st.error(f"Failed to fetch from Supabase: {e}")
            return

    st.info(
        f"**{len(db_data['clients'])}** clients and "
        f"**{len(db_data['technologies'])}** technologies currently in Supabase."
    )

    diff = compute_diff(excel_data, db_data)

    # Client metrics
    nc_create = len(diff["client_create"])
    nc_update = len(diff["client_update"])
    nc_delete = len(diff["client_delete"])

    # Tech metrics
    nt_create = len(diff["tech_create"])
    nt_update = len(diff["tech_update"])
    nt_delete = len(diff["tech_delete"])

    total_changes = nc_create + nc_update + nc_delete + nt_create + nt_update + nt_delete

    st.markdown("**Clients**")
    col1, col2, col3 = st.columns(3)
    col1.metric("To Create", nc_create, delta=f"+{nc_create}" if nc_create else None)
    col2.metric("To Update", nc_update)
    col3.metric("To Delete", nc_delete, delta=f"-{nc_delete}" if nc_delete else None, delta_color="inverse")

    st.markdown("**Technologies**")
    col4, col5, col6 = st.columns(3)
    col4.metric("To Create", nt_create, delta=f"+{nt_create}" if nt_create else None)
    col5.metric("To Update", nt_update)
    col6.metric("To Delete", nt_delete, delta=f"-{nt_delete}" if nt_delete else None, delta_color="inverse")

    if total_changes == 0:
        st.success("Everything is in sync. No changes needed.")
        return

    # Client change details
    if nc_create > 0:
        with st.expander(f"New clients to create ({nc_create})"):
            st.dataframe(pd.DataFrame(diff["client_create"]), use_container_width=True)

    if nc_update > 0:
        with st.expander(f"Clients to update ({nc_update})"):
            update_rows = []
            for u in diff["client_update"]:
                changed_fields = [k for k in u["old"] if u["old"].get(k) != u["new"].get(k)]
                update_rows.append({
                    "client_name": u["new"]["client_name"],
                    "changed_fields": ", ".join(changed_fields),
                })
            st.dataframe(pd.DataFrame(update_rows), use_container_width=True)

    if nc_delete > 0:
        with st.expander(f"Clients to delete ({nc_delete})"):
            st.dataframe(pd.DataFrame(diff["client_delete"]), use_container_width=True)

    # Tech change details
    if nt_create > 0:
        with st.expander(f"New tech rows to create ({nt_create})"):
            st.dataframe(pd.DataFrame(diff["tech_create"]), use_container_width=True)

    if nt_update > 0:
        with st.expander(f"Tech rows to update ({nt_update})"):
            update_rows = []
            for u in diff["tech_update"]:
                changed_fields = [k for k in u["old"] if u["old"].get(k) != u["new"].get(k)]
                update_rows.append({
                    "client_name": u["new"]["client_name"],
                    "exact_key": u["new"]["exact_key"],
                    "changed_fields": ", ".join(changed_fields),
                    "re_embed": "Yes" if u.get("embed_text_changed") else "No",
                })
            st.dataframe(pd.DataFrame(update_rows), use_container_width=True)

    if nt_delete > 0:
        with st.expander(f"Tech rows to delete ({nt_delete})"):
            st.dataframe(pd.DataFrame(diff["tech_delete"]), use_container_width=True)

    texts_needing_embed = nt_create + sum(1 for u in diff["tech_update"] if u.get("embed_text_changed"))
    if texts_needing_embed > 0:
        st.caption(f"Voyage AI embeddings will be generated for **{texts_needing_embed}** tech rows (deduplicated where possible).")

    if st.button("Apply Sync", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(current, total, text):
            if total > 0:
                progress_bar.progress(min(current / total, 1.0))
            status_text.text(text)

        try:
            result = apply_sync(diff, progress_callback=on_progress)
            progress_bar.progress(1.0)
            status_text.text("Sync complete!")
            st.success(
                f"Sync complete: "
                f"Clients — {result['clients_created']} created, "
                f"{result['clients_updated']} updated, {result['clients_deleted']} deleted. "
                f"Technologies — {result['techs_created']} created, "
                f"{result['techs_updated']} updated, {result['techs_deleted']} deleted."
            )
            from client_referencing.matcher import invalidate_cache
            invalidate_cache()
        except Exception as e:
            st.error(f"Sync failed: {e}")
            import traceback
            st.code(traceback.format_exc())


# ── Industry Reference Data Sync ────────────────────────────────────────

def _render_industry_reference_sync():
    """Industry Reference Data Sync section."""
    st.caption(
        "Upload industry-reference-data.xlsx to sync industry terms with Supabase. "
        "Each term gets a Voyage embedding for semantic industry matching during VectorMatch."
    )

    uploaded = st.file_uploader(
        "Upload industry-reference-data.xlsx", type=["xlsx"], key="ind_upload"
    )

    if uploaded is None:
        _show_current_industry_data()
        return

    from client_referencing.industry_sync import read_excel, fetch_existing, compute_diff, apply_sync

    try:
        excel_rows = read_excel(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to parse Excel: {e}")
        return

    st.success(f"Parsed **{len(excel_rows)}** industry terms from Excel.")

    with st.spinner("Fetching current industry terms from Supabase..."):
        try:
            db_rows = fetch_existing()
        except Exception as e:
            st.error(f"Failed to fetch from Supabase: {e}")
            return

    st.info(f"**{len(db_rows)}** industry terms currently in Supabase.")

    diff = compute_diff(excel_rows, db_rows)
    n_create = len(diff["create"])
    n_delete = len(diff["delete"])

    col1, col2 = st.columns(2)
    col1.metric("To Create", n_create, delta=f"+{n_create}" if n_create else None)
    col2.metric("To Delete", n_delete, delta=f"-{n_delete}" if n_delete else None, delta_color="inverse")

    if n_create == 0 and n_delete == 0:
        st.success("Industry terms are in sync. No changes needed.")
        return

    if n_create > 0:
        with st.expander(f"New terms to create ({n_create})"):
            st.dataframe(pd.DataFrame(diff["create"]), use_container_width=True)

    if n_delete > 0:
        with st.expander(f"Terms to delete ({n_delete})"):
            delete_display = [{"term": r["term"]} for r in diff["delete"]]
            st.dataframe(pd.DataFrame(delete_display), use_container_width=True)

    if n_create > 0:
        st.caption(f"Voyage AI embeddings will be generated for **{n_create}** new terms.")

    if st.button("Apply Industry Sync", type="primary", key="ind_sync_btn"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(current, total, text):
            if total > 0:
                progress_bar.progress(min(current / total, 1.0))
            status_text.text(text)

        try:
            result = apply_sync(diff, progress_callback=on_progress)
            progress_bar.progress(1.0)
            status_text.text("Industry sync complete!")
            st.success(
                f"Industry sync complete: **{result['created']}** created, "
                f"**{result['deleted']}** deleted."
            )
            from client_referencing.matcher import invalidate_cache
            invalidate_cache()
        except Exception as e:
            st.error(f"Industry sync failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def _show_current_industry_data():
    """Show current industry embeddings when no file is uploaded."""
    try:
        from client_referencing.industry_sync import fetch_existing
        with st.spinner("Loading industry terms..."):
            db_rows = fetch_existing()

        if not db_rows:
            st.info("No industry terms in Supabase yet. Upload an Excel file to get started.")
            return

        st.metric("Total Terms", len(db_rows))
        df = pd.DataFrame(db_rows)
        if "embedding" in df.columns:
            df["embedding"] = df["embedding"].apply(
                lambda v: str(v)[:50] + "..." if v else None
            )
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load industry data: {e}")


def _show_current_data():
    """Show current Supabase data when no file is uploaded."""
    st.divider()
    st.subheader("Current Data in Supabase")

    try:
        from client_referencing.sync import fetch_existing
        with st.spinner("Loading..."):
            db_data = fetch_existing()

        clients = db_data.get("clients", [])
        techs = db_data.get("technologies", [])

        if not clients and not techs:
            st.info("No data in Supabase yet. Upload an Excel file to get started.")
            return

        col1, col2 = st.columns(2)
        col1.metric("Clients", len(clients))
        col2.metric("Technologies", len(techs))

        if clients:
            with st.expander(f"Clients ({len(clients)})", expanded=False):
                st.dataframe(pd.DataFrame(clients), use_container_width=True)

        if techs:
            with st.expander(f"Technologies ({len(techs)})", expanded=False):
                df = pd.DataFrame(techs)
                if "embedding" in df.columns:
                    df["embedding"] = df["embedding"].apply(
                        lambda v: str(v)[:50] + "..." if v else None
                    )
                st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load current data: {e}")


# ── Casestudies Referencing Data Sync ──────────────────────────────────

def _render_casestudy_sync():
    """Case Studies Data Sync section."""
    st.caption(
        "Upload the case studies Excel file (with 'Case Studies Summary' + "
        "'Case Studies Technology' worksheets) to sync with Supabase. "
        "New case studies with PDFs will be summarized by Claude and embedded with Voyage AI."
    )

    uploaded = st.file_uploader(
        "Upload client-case-studies-mapping.xlsx", type=["xlsx"], key="cs_upload"
    )

    if uploaded is None:
        _show_current_casestudy_data()
        return

    from client_referencing.casestudy_sync import read_excel, fetch_existing, compute_diff, apply_sync

    try:
        excel_data = read_excel(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to parse Excel: {e}")
        return

    st.success(
        f"Parsed **{len(excel_data['summaries'])}** case studies and "
        f"**{len(excel_data['technologies'])}** technology mappings from Excel."
    )

    with st.spinner("Fetching current case studies from Supabase..."):
        try:
            db_data = fetch_existing()
        except Exception as e:
            st.error(f"Failed to fetch from Supabase: {e}")
            return

    st.info(
        f"**{len(db_data['case_studies'])}** case studies and "
        f"**{len(db_data['technologies'])}** technology mappings currently in Supabase."
    )

    diff = compute_diff(excel_data, db_data)

    ncs_create = len(diff["cs_create"])
    ncs_update = len(diff["cs_update"])
    ncs_delete = len(diff["cs_delete"])
    nt_create = len(diff["tech_create"])
    nt_update = len(diff.get("tech_update", []))
    nt_delete = len(diff["tech_delete"])

    total_changes = ncs_create + ncs_update + ncs_delete + nt_create + nt_update + nt_delete

    st.markdown("**Case Studies**")
    col1, col2, col3 = st.columns(3)
    col1.metric("To Create", ncs_create, delta=f"+{ncs_create}" if ncs_create else None)
    col2.metric("To Update", ncs_update)
    col3.metric("To Delete", ncs_delete, delta=f"-{ncs_delete}" if ncs_delete else None, delta_color="inverse")

    st.markdown("**Technology Mappings**")
    col4, col5, col6 = st.columns(3)
    col4.metric("To Create", nt_create, delta=f"+{nt_create}" if nt_create else None)
    col5.metric("To Update", nt_update)
    col6.metric("To Delete", nt_delete, delta=f"-{nt_delete}" if nt_delete else None, delta_color="inverse")

    if total_changes == 0:
        st.success("Everything is in sync. No changes needed.")
        return

    if ncs_create > 0:
        with st.expander(f"New case studies to create ({ncs_create})"):
            st.dataframe(pd.DataFrame(diff["cs_create"]), use_container_width=True)

    if ncs_update > 0:
        with st.expander(f"Case studies to update ({ncs_update})"):
            rows = [{"casestudy_name": u["casestudy_name"], "change": "Add summary from PDF"} for u in diff["cs_update"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if ncs_delete > 0:
        with st.expander(f"Case studies to delete ({ncs_delete})"):
            rows = [{"casestudy_name": cs["casestudy_name"]} for cs in diff["cs_delete"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if nt_create > 0:
        with st.expander(f"New tech mappings to create ({nt_create})"):
            st.dataframe(pd.DataFrame(diff["tech_create"]), use_container_width=True)

    if nt_update > 0:
        with st.expander(f"Tech mappings to update ({nt_update})"):
            st.dataframe(pd.DataFrame(diff["tech_update"]), use_container_width=True)

    if nt_delete > 0:
        with st.expander(f"Tech mappings to delete ({nt_delete})"):
            rows = [{"casestudy_technology": t["casestudy_technology"]} for t in diff["tech_delete"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if ncs_create + ncs_update > 0:
        st.caption(
            f"Claude summaries and Voyage embeddings will be generated for "
            f"**{ncs_create + ncs_update}** case studies. "
            f"PDFs must be in the local folder."
        )

    if st.button("Apply Case Study Sync", type="primary", key="cs_sync_btn"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(current, total, text):
            if total > 0:
                progress_bar.progress(min(current / total, 1.0))
            status_text.text(text)

        try:
            result = apply_sync(diff, progress_callback=on_progress)
            progress_bar.progress(1.0)
            status_text.text("Case study sync complete!")
            st.success(
                f"Case study sync complete: "
                f"**{result['cs_created']}** created, "
                f"**{result['cs_updated']}** updated, "
                f"**{result['cs_deleted']}** deleted. "
                f"Tech mappings: **{result['tech_created']}** created, "
                f"**{result.get('tech_updated', 0)}** updated, "
                f"**{result['tech_deleted']}** deleted."
            )
        except Exception as e:
            st.error(f"Case study sync failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def _show_current_casestudy_data():
    """Show current case studies data when no file is uploaded."""
    try:
        from client_referencing.casestudy_sync import fetch_existing
        with st.spinner("Loading case studies..."):
            db_data = fetch_existing()

        cs = db_data.get("case_studies", [])
        techs = db_data.get("technologies", [])

        if not cs and not techs:
            st.info("No case studies in Supabase yet. Upload an Excel file to get started.")
            return

        col1, col2 = st.columns(2)
        col1.metric("Case Studies", len(cs))
        col2.metric("Technology Mappings", len(techs))

        if cs:
            with st.expander(f"Case Studies ({len(cs)})", expanded=False):
                df = pd.DataFrame(cs)
                display_cols = [c for c in df.columns if c not in ("summary_embedding",  "summary_problem", "summary_solution", "summary_outcomes")]
                st.dataframe(df[display_cols], use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load case study data: {e}")


# ── Brand Reference Refresh ──────────────────────────────────────────────

def _render_brand_reference_refresh():
    """Render brand knowledge refresh UI with per-brand refresh buttons and profile viewing."""
    from prospect_outreach import brand_knowledge, config
    from datetime import datetime

    st.caption(
        "Brand profiles are scraped from company websites using Claude web search. "
        "Click the refresh button next to each brand to update its profile."
    )

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
            if st.button("\u27f3", key=f"sync_refresh_{name}"):
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


# ── Match Explanation ────────────────────────────────────────────────────

def _render_match_explanation(exp: dict, prospect_technologies: str, prospect_country: str):
    """Render a step-by-step explanation of the matching process."""
    with st.expander("Match Explanation (Step-by-Step)", expanded=True):
        # Step 1: Tier Selection
        st.markdown("### Step 1: Tier Selection")
        tier_used = exp.get("tier_used", "Unknown")
        tier_reason = exp.get("tier_reason", "")
        st.info(f"**{tier_used}** — {tier_reason}")

        tier1 = exp.get("tier1_clients", [])
        tier2 = exp.get("tier2_clients", [])
        if tier1:
            st.markdown(f"**Tier 1 (Industry + Geography):** {len(tier1)} clients")
            st.caption(", ".join(tier1))
        else:
            st.markdown("**Tier 1 (Industry + Geography):** 0 clients")

        if len(tier1) <= 4 and tier2:
            st.markdown(f"**Tier 2 (Industry Only):** {len(tier2)} clients")
            st.caption(", ".join(tier2))

        st.markdown("---")

        # Step 2: Technology Matching on Shortlist
        st.markdown("### Step 2: Technology Matching")
        prospect_techs = exp.get("prospect_techs", [])
        unmatched = exp.get("unmatched_techs", [])
        matched_input = [t for t in prospect_techs if t not in unmatched]

        if matched_input:
            st.markdown(f"**Exact-matched input techs:** {', '.join(f'`{t}`' for t in matched_input)}")
        if unmatched:
            st.markdown(f"**Unmatched (used for semantic):** {', '.join(f'`{t}`' for t in unmatched)}")

        core_exact = exp.get("core_exact_by_client", {})
        core_semantic = exp.get("core_semantic_by_client", {})
        core_clients = exp.get("core_match_clients", [])

        if core_clients:
            # Deduplicate client names preserving order
            seen = set()
            unique_core = []
            for c in core_clients:
                if c not in seen:
                    seen.add(c)
                    unique_core.append(c)

            st.markdown(f"**Clients with tech matches:** {len(unique_core)}")
            rows = []
            for c in unique_core:
                exact_list = core_exact.get(c, [])
                sem_list = core_semantic.get(c, [])
                sem_strs = [f"{pt}→{et} ({sim:.2f})" for pt, et, sim in sem_list] if sem_list else []
                rows.append({
                    "Client": c,
                    "Exact Matches": ", ".join(exact_list) if exact_list else "—",
                    "Semantic Matches": ", ".join(sem_strs) if sem_strs else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No clients matched any technologies from the shortlist.")

        st.markdown("---")

        # Step 3: Top-5 Truncation
        st.markdown("### Step 3: Top-5 Truncation")
        top5 = exp.get("top5_clients", [])
        top5_count = exp.get("top5_count", 0)
        st.markdown(f"**{top5_count} unique client(s)** kept from core matching (max 5)")
        if top5:
            st.caption(", ".join(top5))

        st.markdown("---")

        # Step 4: Backfill
        st.markdown("### Step 4: Backfill")
        flag_tier_3 = exp.get("flag_tier_3", False)

        generic_picked = exp.get("generic_backfill_picked", [])
        # Deduplicate
        seen_generic = set()
        generic_picked_unique = []
        for c in generic_picked:
            if c not in seen_generic:
                seen_generic.add(c)
                generic_picked_unique.append(c)

        geo_clients = exp.get("geo_backfill_clients", [])

        if flag_tier_3:
            st.markdown("**Branch B (Tier 3):** Tech already used for shortlisting → skip tech backfill, go straight to geo.")
        else:
            st.markdown("**Branch A (Tier 1/2):** Tech backfill first, then geo.")
            generic_all = exp.get("generic_backfill_all", [])
            generic_deficit = exp.get("generic_deficit", 0)
            bf_exact = exp.get("bf_exact_by_client", {})
            bf_semantic = exp.get("bf_semantic_by_client", {})

            if generic_all:
                st.markdown(f"**Tech backfill candidates:** {len(generic_all)} clients (ordered by industry score)")
                st.markdown(f"**Deficit to fill:** {generic_deficit} slot(s)")

                if generic_picked_unique:
                    st.markdown(f"**Picked ({len(generic_picked_unique)}):** {', '.join(generic_picked_unique)}")
                    rows = []
                    for c in generic_picked_unique:
                        exact_list = bf_exact.get(c, [])
                        sem_list = bf_semantic.get(c, [])
                        sem_strs = [f"{pt}→{et} ({sim:.2f})" for pt, et, sim in sem_list] if sem_list else []
                        rows.append({
                            "Client": c,
                            "Exact Matches": ", ".join(exact_list) if exact_list else "—",
                            "Semantic Matches": ", ".join(sem_strs) if sem_strs else "—",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Show why top candidates were picked over others
                if len(generic_all) > len(generic_picked_unique):
                    not_picked = [c for c in generic_all if c not in generic_picked_unique][:5]
                    st.caption(f"Not picked (next in line): {', '.join(not_picked)}")
            else:
                st.caption("No tech backfill candidates found.")

        if geo_clients:
            geo_deficit = exp.get("geo_deficit", 0)
            st.markdown(f"**Geo backfill ({prospect_country}):** {len(geo_clients)} client(s) picked (deficit: {geo_deficit})")
            st.caption(", ".join(geo_clients))
        else:
            st.caption("No geo backfill needed or no geo clients available.")

        st.markdown("---")

        # Step 5: Final Composition
        st.markdown("### Step 5: Final Result")
        final = exp.get("final_clients", [])
        if final:
            rows = []
            for i, c in enumerate(final, 1):
                rows.append({
                    "#": i,
                    "Client": c["name"],
                    "Final Score": f"{c['score']:.3f}",
                    "Industry Score": f"{c['industry_score']:.4f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── VectorMatch Tab ──────────────────────────────────────────────────────

def _render_vector_match_tab():
    """Render the Vector Existing Client Match tab."""
    st.caption(
        "This test tool contains full verbose for testing prospects with existing clients."
    )
    _render_vectormatch_section()


# ── Hyper Personalized Message Auto Generator ────────────────────────────


# ── VectorMatch — Prospect Client Matching ───────────────────────────────

def _render_vectormatch_section():
    """Render the VectorMatch prospect matching UI."""
    st.caption(
        "Enter a prospect's industry and technologies to find the best-matching "
        "existing clients from the reference database."
    )

    with st.form("vectormatch_form"):
        col1, col2, col3, col4 = st.columns([3, 3, 3, 2])
        with col1:
            prospect_industry = st.text_input(
                "Prospect Industry",
                placeholder="e.g. Banking, Healthcare, Retail",
            )
        with col2:
            prospect_technologies = st.text_input(
                "Prospect Technologies (CSV)",
                placeholder="e.g. Salesforce, Mulesoft, Snowflake",
            )
        with col3:
            prospect_country = st.text_input(
                "Prospect Country",
                placeholder="e.g. USA, Australia, India",
            )
        with col4:
            max_matches = st.number_input(
                "Max Matches", min_value=6, max_value=10, value=6,
            )
        submitted = st.form_submit_button("Find Matches", type="primary")

    if not submitted:
        return

    if not prospect_technologies.strip():
        st.warning("Please enter at least one technology.")
        return

    from client_referencing.matcher import find_matches

    with st.spinner("Matching prospect against client database..."):
        try:
            results = find_matches(prospect_industry, prospect_technologies, prospect_country, max_matches)
        except Exception as e:
            st.error(f"Matching failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    if not results["matches"]:
        st.info("No matching clients found.")
        return

    # Industry filter status
    st.subheader(f"Top {len(results['matches'])} Matching Clients")
    if results["industry_filter_applied"]:
        st.success(f"Industry filter applied: \"{prospect_industry}\"")
    else:
        st.info("Industry filter was not applied (too few matches or no industry provided).")

    # Results
    for i, match in enumerate(results["matches"], 1):
        with st.expander(
            f"#{i} — {match.client_name} (Score: {match.final_score:.3f})",
            expanded=(i <= 3),
        ):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Industry:** {match.client_industry}")
            c2.markdown(f"**Geography:** {match.client_geography}")
            c3.markdown(f"**URL:** {match.client_url}")

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Final Score", f"{match.final_score:.3f}")
            sc2.metric("Exact Match Ratio", f"{match.match_ratio:.3f}",
                       help="Exact matches / total prospect technologies")
            sc3.metric("Semantic Similarity", f"{match.similarity_score:.3f}",
                       help="Average semantic similarity for unmatched techs")

            if match.exact_techs:
                st.markdown(f"**Exact Matches ({len(match.exact_techs)}):** "
                           + ", ".join(f"`{t}`" for t in match.exact_techs))
            else:
                st.caption("No exact technology matches.")

            if match.semantic_techs:
                st.markdown("**Semantic Matches:**")
                for (ptech, embed_text, sim) in match.semantic_techs:
                    st.caption(f"  {ptech} \u2192 {embed_text} (similarity: {sim:.3f})")
            else:
                st.caption("No semantic technology matches.")

            if match.industry_match:
                st.caption(f"Industry relevance: {match.industry_score:.3f} (passed filter)")
            elif match.industry_score > 0:
                st.caption(f"Industry relevance: {match.industry_score:.3f}")

    # Industry-only clients
    if results["industry_filter_applied"] and results["industry_filtered_only"]:
        st.divider()
        st.subheader("Other Clients Passing Industry Filter")
        st.caption("These clients matched the industry filter but did not rank in the top 5.")
        for name in results["industry_filtered_only"]:
            st.text(f"  • {name}")
    elif not results["industry_filter_applied"]:
        st.divider()
        st.caption("Industry filtering was not applied, so no separate industry-only list is shown.")

    # Match Explanation
    if results.get("explanation"):
        st.divider()
        _render_match_explanation(results["explanation"], prospect_technologies, prospect_country)

    # Debug log
    if results.get("debug_log"):
        st.divider()
        with st.expander("Debug Log", expanded=False):
            for label, content in results["debug_log"]:
                st.markdown(f"**{label}:**")
                st.text(content)


# ── VectorCasestudyMatch Tab ────────────────────────────────────────────

def _render_casestudy_match_tab():
    """Render the case-study matching UI."""
    st.subheader("Case Study Matcher")
    st.caption(
        "This test tool contains full verbose for testing prospects with "
        "technologies to existing casestudies."
    )

    with st.form("cs_match_form"):
        prospect_context = st.text_area(
            "Prospect Context / Signals",
            placeholder="e.g. client wants to implement a system to automate prescription management from Salesforce CRM",
            height=100,
        )
        col1, col2 = st.columns(2)
        with col1:
            prospect_industry = st.text_input("Industry", placeholder="e.g. Healthcare")
            prospect_technologies = st.text_input(
                "Technologies (comma-separated)",
                placeholder="e.g. Salesforce, Mulesoft, Snowflake",
            )
        with col2:
            max_matches = st.number_input("Max Matches", min_value=6, max_value=10, value=8)

        submitted = st.form_submit_button("Find Matching Case Studies", type="primary")

    if not submitted:
        return

    if not prospect_technologies.strip() and not prospect_context.strip():
        st.warning("Please provide at least technologies or prospect context.")
        return

    from client_referencing.casestudy_matcher import find_casestudy_matches

    with st.spinner("Matching case studies..."):
        results = find_casestudy_matches(
            prospect_context=prospect_context,
            prospect_industry=prospect_industry,
            prospect_technologies=prospect_technologies,
            max_matches=max_matches,
        )

    matches = results.get("matches", [])

    if not matches:
        st.info("No matching case studies found.")
    else:
        st.success(f"Found **{len(matches)}** matching case studies.")

        for i, m in enumerate(matches, 1):
            tier_badge = "🟢 Tier 1" if m.tier == "tier_1" else "🔵 Tier 2"
            with st.expander(
                f"#{i} — {m.casestudy_name} ({m.client_name}) — {tier_badge} — Score: {m.final_score:.3f}",
                expanded=(i <= 3),
            ):
                # Metrics row
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric("Final Score", f"{m.final_score:.3f}")
                mc2.metric("Exact Ratio", f"{m.match_ratio:.3f}")
                mc3.metric("Semantic", f"{m.similarity_score:.3f}")
                mc4.metric("Context", f"{m.context_score:.3f}")
                mc5.metric("Industry", f"{m.industry_score:.3f}")

                st.markdown(f"**Client:** {m.client_name} | **Industry:** {m.client_industry}")

                if m.url:
                    st.markdown(f"**Case Study Link:** {m.url}")

                # Tech matches
                if m.exact_techs:
                    st.markdown(f"**Exact Tech Matches:** {', '.join(m.exact_techs)}")
                if m.semantic_techs:
                    sem_parts = [
                        f"{t[0]} → {t[1]} ({t[2]:.2f})" for t in m.semantic_techs
                    ]
                    st.markdown(f"**Semantic Tech Matches:** {', '.join(sem_parts)}")

                # Summary
                if m.summary_problem or m.summary_solution or m.summary_outcomes:
                    st.markdown("---")
                    if m.summary_problem:
                        st.markdown(f"**Problem:** {m.summary_problem}")
                    if m.summary_solution:
                        st.markdown(f"**Solution:** {m.summary_solution}")
                    if m.summary_outcomes:
                        st.markdown(f"**Outcomes:** {m.summary_outcomes}")

    # Excel JSON output (6-field format)
    if matches:
        import json
        st.divider()
        with st.expander("Excel JSON Output (6-field format)", expanded=False):
            excel_json = [m.to_excel_dict() for m in matches]
            st.code(json.dumps(excel_json, indent=2), language="json")

    # Match Explanation (Step-by-Step)
    if results.get("explanation"):
        st.divider()
        st.subheader("Match Explanation (Step-by-Step)")
        for step_label, step_detail in results["explanation"]:
            with st.expander(step_label, expanded=True):
                st.text(step_detail)

    # Debug log
    if results.get("debug_log"):
        st.divider()
        with st.expander("Debug Log", expanded=False):
            for label, content in results["debug_log"]:
                st.markdown(f"**{label}:**")
                st.text(content)


def render_master():
    """Render the Master Casestudy Finder — non-verbose Client Matcher + Casestudy Matcher."""
    tabs = st.tabs(["Client Matcher", "Casestudy Matcher"])
    with tabs[0]:
        _render_client_matcher_simple()
    with tabs[1]:
        _render_casestudy_matcher_simple()


def _render_client_matcher_simple():
    """Non-verbose client matcher — clean results only, no scores or debug."""
    st.subheader("Client Matcher")
    st.caption("Match a prospect against existing clients.")

    with st.form("master_clientmatch_form"):
        col1, col2, col3, col4 = st.columns([3, 3, 3, 2])
        with col1:
            prospect_industry = st.text_input("Prospect Industry", placeholder="e.g. Banking, Healthcare")
        with col2:
            prospect_technologies = st.text_input("Prospect Technologies (CSV)", placeholder="e.g. Salesforce, Boomi")
        with col3:
            prospect_country = st.text_input("Prospect Country", placeholder="e.g. Australia, USA")
        with col4:
            max_matches = st.number_input("Max Matches", min_value=6, max_value=10, value=6)
        submitted = st.form_submit_button("Find Matches", type="primary")

    if not submitted:
        return

    if not prospect_technologies.strip():
        st.warning("Please enter at least one technology.")
        return

    from client_referencing.matcher import find_matches

    with st.spinner("Matching..."):
        try:
            results = find_matches(prospect_industry, prospect_technologies, prospect_country, max_matches)
        except Exception as e:
            st.error(f"Matching failed: {e}")
            return

    matches = results.get("matches", [])
    if not matches:
        st.info("No matching clients found.")
        return

    st.success(f"Found **{len(matches)}** matching clients.")
    rows = [
        {
            "Client": m.client_name,
            "Industry": m.client_industry,
            "Geography": m.client_geography,
            "URL": m.client_url,
            "Exact Tech Matches": ", ".join(m.exact_techs) if m.exact_techs else "—",
            "Score": round(m.final_score, 3),
        }
        for m in matches
    ]
    st.dataframe(rows, use_container_width=True)


def _render_casestudy_matcher_simple():
    """Non-verbose casestudy matcher — clean results only, no scores or debug."""
    st.subheader("Casestudy Matcher")
    st.caption("Match a prospect against existing case studies.")

    with st.form("master_csmatch_form"):
        prospect_context = st.text_area(
            "Prospect Context",
            placeholder="e.g. client wants to automate prescription management from Salesforce CRM",
            height=80,
        )
        col1, col2 = st.columns(2)
        with col1:
            prospect_industry = st.text_input("Industry", placeholder="e.g. Healthcare")
            prospect_technologies = st.text_input("Technologies (CSV)", placeholder="e.g. Salesforce, Mulesoft")
        with col2:
            max_matches = st.number_input("Max Matches", min_value=6, max_value=10, value=8)
        submitted = st.form_submit_button("Find Matching Case Studies", type="primary")

    if not submitted:
        return

    if not prospect_technologies.strip() and not prospect_context.strip():
        st.warning("Please provide at least technologies or prospect context.")
        return

    from client_referencing.casestudy_matcher import find_casestudy_matches

    with st.spinner("Matching case studies..."):
        results = find_casestudy_matches(
            prospect_context=prospect_context,
            prospect_industry=prospect_industry,
            prospect_technologies=prospect_technologies,
            max_matches=max_matches,
        )

    matches = results.get("matches", [])
    if not matches:
        st.info("No matching case studies found.")
        return

    st.success(f"Found **{len(matches)}** matching case studies.")
    for i, m in enumerate(matches, 1):
        download_link = f"[Download]({m.url})" if m.url else "—"
        st.markdown(f"**{i}. {m.casestudy_name}** ({m.client_name}) — Score: {m.final_score:.3f} &nbsp; {download_link}")


def _render_brand_match_tab():
    """Render the Brand Matcher UI."""
    st.subheader("Brand Matcher")
    st.caption(
        "This test tool contains full verbose for testing prospects that are "
        "matched to a respective brand i.e. LendingLogik & CloudChillies."
    )

    with st.form("brand_match_form"):
        prospect_industry = st.text_input(
            "Industry (optional)",
            placeholder="e.g. Healthcare, Banking, Fintech",
        )
        submitted = st.form_submit_button("Find Brand Match")

    if not submitted:
        return

    from client_referencing.brand_matcher import find_brand_match

    with st.spinner("Matching brand..."):
        results = find_brand_match(
            prospect_industry=prospect_industry,
        )

    # Prominent brand result
    brand = results["brand"]
    if brand == "LendingLogik":
        st.success(f"Recommended Brand: **{brand}**")
    else:
        st.info(f"Recommended Brand: **{brand}**")

    # Match details
    if results["matched_industry_term"]:
        col1, col2 = st.columns(2)
        col1.metric("Matched Industry Term", results["matched_industry_term"])
        col2.metric("Similarity Score", f"{results['similarity_score']:.4f}")

    # All matches above threshold
    if results["all_matches"]:
        st.markdown("**All matches above threshold:**")
        match_df = pd.DataFrame(results["all_matches"], columns=["Industry Term", "Similarity"])
        st.dataframe(match_df, use_container_width=True, hide_index=True)

    # Debug log
    with st.expander("Debug Log", expanded=False):
        for label, content in results["debug_log"]:
            st.markdown(f"**{label}:**")
            st.text(content)


# ── Pass 1: Research (Test Mode) ─────────────────────────────────────────

REQUIRED_COLUMNS = [
    "First_Name", "Designation", "Company_Name",
    "Email", "City", "State", "Country", "Industry", "Website",
]


def _test_cache_badge(source: str) -> str:
    badges = {
        "CACHE HIT": "🟢 CACHE HIT",
        "CACHE STALE": "🟡 CACHE STALE (re-fetched via Serper)",
        "CACHE MISS": "🔴 CACHE MISS (fetched via Serper)",
        "CACHE SKIPPED": "⚪ CACHE SKIPPED (fetched via Serper)",
        "DEDUPED": "⚪ DEDUPED (reused from earlier row)",
    }
    return badges.get(source, source)


def _test_get_technology_research(company_name: str, website: str, use_cache: bool):
    """Returns (data, source, verbose_info) where verbose_info has per-query snippets and Claude's raw response."""
    from prospect_outreach.research_pipeline import (
        _get_supabase, _is_cache_stale, _run_technology_research,
    )
    source = None
    data = None
    verbose_info = None
    if use_cache:
        try:
            sb = _get_supabase()
            result = sb.table("Cache_Prospect_Company_Research").select("*").eq("Website", website).execute()
            if result.data:
                row = result.data[0]
                if not _is_cache_stale(row.get("Date_of_Research")):
                    cr = row.get("Technology_Research")
                    if cr is not None:
                        source = "CACHE HIT"
                        data = str(cr)
                    else:
                        source = "CACHE NULL"
                else:
                    source = "CACHE STALE"
            else:
                source = "CACHE MISS"
        except Exception as e:
            source = f"CACHE ERROR: {e}"
    else:
        source = "CACHE SKIPPED"
    if data is None:
        from datetime import datetime, timezone
        verbose_result = _run_technology_research(company_name, website, verbose=True)
        data = verbose_result["result"]
        verbose_info = {
            "per_query_snippets": verbose_result["per_query_snippets"],
            "claude_raw_response": verbose_result["claude_raw_response"],
        }
        # Write back to cache (same as Pass 1 pipeline)
        try:
            sb = _get_supabase()
            sb.table("Cache_Prospect_Company_Research").upsert({
                "Website": website,
                "Technology_Research": data,
                "Date_of_Research": datetime.now(timezone.utc).isoformat(),
            }).execute()
        except Exception:
            pass  # cache write failure is non-critical
    return data, source, verbose_info


def _render_test_tab():
    st.subheader("Pass 1: Research (Test Mode)")
    st.caption(
        "Runs the full Pass 1 research pipeline row-by-row with verbose output. "
        "Use cache toggle to control whether cached data is used for technology research."
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

    st.write(f"**{len(df)} prospect(s) loaded**")
    with st.expander("Preview uploaded data", expanded=False):
        st.dataframe(df, use_container_width=True)

    st.divider()

    use_company_cache = st.toggle("Use Technology Research Cache", value=False, key="test_company_cache")

    col_ta, col_tb = st.columns(2)
    with col_ta:
        max_case_studies = st.number_input("Max Case Studies", min_value=5, max_value=8, value=5, key="test_cs")
    with col_tb:
        max_clients = st.number_input("Max Client Matches", min_value=6, max_value=10, value=6, key="test_cl")

    if not st.button("Run Test", type="primary", key="run_test_btn"):
        return

    from prospect_outreach.brand_knowledge import ensure_brand_profiles
    from prospect_outreach.research_pipeline import (
        _apply_emea_coding,
    )
    from client_referencing.brand_matcher import find_brand_match
    from client_referencing.casestudy_matcher import find_casestudy_matches
    from client_referencing.matcher import find_matches

    ensure_brand_profiles()

    company_cache: dict = {}  # website -> (research_dict, source_label, verbose_info)
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        prospect_name = f"{row.get('First_Name', '')} {row.get('Last_Name', '')}".strip()
        designation = str(row.get("Designation", ""))
        company_name = str(row.get("Company_Name", ""))
        website = str(row.get("Website", ""))
        industry = str(row.get("Industry", ""))
        country = str(row.get("Country", ""))
        city = str(row.get("City", ""))
        state = str(row.get("State", ""))

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

            # ── Step 2: EMEA Coding ──────────────────────────────────────
            st.markdown("#### Step 2 — EMEA Coding")
            st.write(f"**Params:** `country={country!r}`")
            try:
                coded_country = _apply_emea_coding(country)
                st.write(f"**Return:** `{country}` → `{coded_country}`")
                country = coded_country
            except Exception as e:
                st.error(f"EMEA coding failed: {e}")

            st.divider()

            # ── Step 3: Technology Research ───────────────────────────────
            st.markdown("#### Step 3 — Technology Research")
            st.write(f"**Params:** `company_name={company_name!r}`, `website={website!r}`")
            tech_research = ""
            verbose_info = None
            try:
                if website in company_cache:
                    tech_research, co_source, verbose_info = company_cache[website]
                    co_source = "DEDUPED"
                else:
                    tech_research, co_source, verbose_info = _test_get_technology_research(
                        company_name, website, use_company_cache
                    )
                    company_cache[website] = (tech_research, co_source, verbose_info)
                st.info(_test_cache_badge(co_source))

                # Show per-query Serper results
                if verbose_info and verbose_info.get("per_query_snippets"):
                    st.caption("↓ Individual Serper query results (input to Claude)")
                    for qi, qs in enumerate(verbose_info["per_query_snippets"], 1):
                        count = qs.get("result_count", "?")
                        with st.expander(f"Query {qi} ({count} results): {qs['query']}", expanded=False):
                            if qs.get("serper_meta"):
                                st.json(qs["serper_meta"])
                            st.text(qs["results"] if qs["results"] else "(no results)")

                # Show Claude's raw response
                if verbose_info and verbose_info.get("claude_raw_response"):
                    with st.expander("Claude raw response (before JSON parsing)", expanded=False):
                        st.code(verbose_info["claude_raw_response"], language="json")

                st.caption("↓ TECHNOLOGY_RESEARCH — synthesized by Claude from Serper results. Cached in Supabase keyed by Website.")
                with st.expander("TECHNOLOGY_RESEARCH", expanded=True):
                    st.write(tech_research)
                tech_names_csv = tech_research
                st.write(f"**Tech names (for matchers):** `{tech_names_csv}`")
            except Exception as e:
                st.error(f"Technology research failed: {e}")
                tech_names_csv = ""

            st.divider()

            # ── Step 4: Case Study Matching ──────────────────────────────
            st.markdown("#### Step 4 — Case Study Matching")
            cs_params = {
                "prospect_context": "",
                "prospect_industry": industry,
                "prospect_technologies": tech_names_csv,
                "max_matches": int(max_case_studies),
            }
            st.write("**Params:**")
            st.json(cs_params)
            try:
                cs_result = find_casestudy_matches(
                    prospect_context="",
                    prospect_industry=industry,
                    prospect_technologies=tech_names_csv,
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

            # ── Step 5: Client Reference Matching ────────────────────────
            st.markdown("#### Step 5 — Client Reference Matching")
            cl_params = {
                "prospect_industry": industry,
                "prospect_technologies": tech_names_csv,
                "prospect_country": country,
                "max_matches": int(max_clients),
            }
            st.write("**Params:**")
            st.json(cl_params)
            try:
                cl_result = find_matches(
                    prospect_industry=industry,
                    prospect_technologies=tech_names_csv,
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
                    with st.expander(f"Match {j+1}: {md.get('client_name', '—')} | {md.get('client_industry', '—')} | {md.get('client_geography', '—')} | score={md.get('final_score', 0):.3f}"):
                        st.json(md)
                with st.expander("debug_log"):
                    st.json(cl_result.get("debug_log", []))
            except Exception as e:
                st.error(f"Client reference matching failed: {e}")

            st.divider()
