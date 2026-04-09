"""Client Referencing pipeline UI — Data Sync + VectorMatch tabs."""

import pandas as pd
import streamlit as st


def render():
    """Render the Client Referencing pipeline with tabs."""
    tabs = st.tabs(["Data Sync", "VectorMatch"])

    with tabs[0]:
        _render_sync_tab()

    with tabs[1]:
        _render_vector_match_tab()


# ── Data Sync Tab ────────────────────────────────────────────────────────

def _render_sync_tab():
    """Render the sync UI (upload Excel, preview diff, apply)."""
    with st.expander("Client Referencing Data Sync", expanded=False):
        _render_client_referencing_sync()
    with st.expander("Industry Reference Data Sync", expanded=False):
        _render_industry_reference_sync()


# ── Client Referencing Data Sync ────────────────────────────────────────

def _render_client_referencing_sync():
    """Client Referencing Data Sync section."""
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
            _SOURCE_LABELS = {
                "industry_exact": "Industry + Exact",
                "industry_semantic": "Industry + Semantic",
                "exact": "Exact Match",
                "semantic": "Semantic Match",
                "backfill_exact": "Tech Backfill (Exact)",
                "backfill_semantic": "Tech Backfill (Semantic)",
                "geography": "Geo Backfill",
            }
            rows = []
            for i, c in enumerate(final, 1):
                rows.append({
                    "#": i,
                    "Client": c["name"],
                    "Source": _SOURCE_LABELS.get(c["source"], c["source"]),
                    "Final Score": f"{c['score']:.3f}",
                    "Industry Score": f"{c['industry_score']:.4f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── VectorMatch Tab ──────────────────────────────────────────────────────

def _render_vector_match_tab():
    """Render the VectorMatch tab with two accordion sections."""
    # Section 1: Hyper Personalized Message Auto Generator (open by default)
    with st.expander("Hyper Personalized Message Auto Generator", expanded=True):
        _render_message_generator_section()

    # Section 2: VectorMatch — Prospect Client Matching (closed by default)
    with st.expander("VectorMatch — Prospect Client Matching", expanded=False):
        _render_vectormatch_section()


# ── Hyper Personalized Message Auto Generator ────────────────────────────

def _render_message_generator_section():
    """Render the prospect upload and message generation UI."""
    st.caption(
        "Upload a Prospect Data Excel sheet to auto-generate hyper-personalized "
        "outreach messages using client references, case studies, and AI research."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload Prospect Data Excel", type=["xlsx"], key="prospect_upload"
        )
    with col2:
        st.write("")  # spacer
        st.write("")  # spacer
        if st.button("Clear Cache", type="secondary", key="clear_cache_btn"):
            from client_referencing.matcher import invalidate_cache
            invalidate_cache()
            st.success("Cache cleared.")

    st.caption(
        "**Clear Cache** refreshes client data and industry embeddings from Supabase. "
        "Prospect industry and technology embeddings (Voyage API lookups) are preserved "
        "across clears since the same input always produces the same output."
    )

    if uploaded is not None:
        st.info("Prospect data upload received. Processing pipeline coming soon.")
        # TODO: parse Excel, validate columns, run pipeline


# ── VectorMatch — Prospect Client Matching ───────────────────────────────

def _render_vectormatch_section():
    """Render the VectorMatch prospect matching UI."""
    st.caption(
        "Enter a prospect's industry and technologies to find the best-matching "
        "existing clients from the reference database."
    )

    with st.form("vectormatch_form"):
        col1, col2, col3, col4 = st.columns([3, 3, 3, 1])
        with col1:
            prospect_industry = st.text_input(
                "Prospect Industry",
                placeholder="e.g. Banking, Healthcare, Retail",
            )
        with col2:
            prospect_technologies = st.text_input(
                "Prospect Technologies (comma-separated)",
                placeholder="e.g. Salesforce, Mulesoft, Snowflake",
            )
        with col3:
            prospect_country = st.text_input(
                "Prospect Country",
                placeholder="e.g. USA, Australia, India",
            )
        with col4:
            max_matches = st.number_input(
                "Max Matches", min_value=6, max_value=20, value=6,
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

    # Match source labels
    _SOURCE_LABELS = {
        "industry_exact": "Industry + Exact Match",
        "industry_semantic": "Industry + Semantic Match",
        "exact": "Exact Match",
        "semantic": "Semantic Match",
        "backfill_exact": "Backfill — Exact Match",
        "backfill_semantic": "Backfill — Semantic Match",
        "geography": "Geography Backfill",
    }

    # Results
    for i, match in enumerate(results["matches"], 1):
        source_label = _SOURCE_LABELS.get(match.match_source, match.match_source)
        with st.expander(
            f"#{i} — {match.client_name} (Score: {match.final_score:.3f}) | {source_label}",
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

            if match.case_studies:
                st.markdown(f"**Case Studies ({len(match.case_studies)}):**")
                for cs in match.case_studies:
                    st.markdown(f"- [{cs['title']}]({cs['url']})")

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
