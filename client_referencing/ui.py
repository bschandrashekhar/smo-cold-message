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
        "Upload the client-referencing-data.xlsx file to sync with Supabase. "
        "The tool compares your Excel against the database and shows what will be "
        "created, updated, or deleted before applying changes."
    )

    uploaded = st.file_uploader(
        "Upload client-referencing-data.xlsx", type=["xlsx"], key="cr_upload"
    )

    if uploaded is None:
        _show_current_data()
        return

    from client_referencing.sync import read_excel, fetch_existing, compute_diff, apply_sync

    try:
        excel_rows = read_excel(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to parse Excel: {e}")
        return

    st.success(f"Parsed **{len(excel_rows)}** rows from Excel.")

    with st.spinner("Fetching current data from Supabase..."):
        try:
            db_rows = fetch_existing()
        except Exception as e:
            st.error(f"Failed to fetch from Supabase: {e}")
            return

    st.info(f"**{len(db_rows)}** rows currently in Supabase.")

    diff = compute_diff(excel_rows, db_rows)
    n_create = len(diff["create"])
    n_update = len(diff["update"])
    n_delete = len(diff["delete"])

    col1, col2, col3 = st.columns(3)
    col1.metric("To Create", n_create, delta=f"+{n_create}" if n_create else None)
    col2.metric("To Update", n_update)
    col3.metric("To Delete", n_delete, delta=f"-{n_delete}" if n_delete else None, delta_color="inverse")

    if n_create == 0 and n_update == 0 and n_delete == 0:
        st.success("Everything is in sync. No changes needed.")
        return

    if n_create > 0:
        with st.expander(f"New rows to create ({n_create})"):
            st.dataframe(pd.DataFrame(diff["create"]), use_container_width=True)

    if n_update > 0:
        with st.expander(f"Rows to update ({n_update})"):
            update_rows = []
            for u in diff["update"]:
                changed_fields = [
                    k for k in u["old"]
                    if u["old"].get(k) != u["new"].get(k)
                ]
                update_rows.append({
                    "client_name": u["new"]["client_name"],
                    "exact_key": u["new"]["exact_key"],
                    "changed_fields": ", ".join(changed_fields),
                    "re_embed": "Yes" if u["embed_text_changed"] else "No",
                })
            st.dataframe(pd.DataFrame(update_rows), use_container_width=True)

    if n_delete > 0:
        with st.expander(f"Rows to delete ({n_delete})"):
            st.dataframe(pd.DataFrame(diff["delete"]), use_container_width=True)

    texts_needing_embed = n_create + sum(1 for u in diff["update"] if u["embed_text_changed"])
    if texts_needing_embed > 0:
        st.caption(f"Voyage AI embeddings will be generated for **{texts_needing_embed}** rows (deduplicated where possible).")

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
                f"Sync complete: **{result['created']}** created, "
                f"**{result['updated']}** updated, **{result['deleted']}** deleted."
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
            db_rows = fetch_existing()

        if not db_rows:
            st.info("No data in Supabase yet. Upload an Excel file to get started.")
            return

        st.metric("Total Rows", len(db_rows))
        df = pd.DataFrame(db_rows)
        if "embedding" in df.columns:
            df["embedding"] = df["embedding"].apply(
                lambda v: str(v)[:50] + "..." if v else None
            )
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load current data: {e}")


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
                "Max Matches", min_value=1, max_value=20, value=6,
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

    # Debug log
    if results.get("debug_log"):
        st.divider()
        with st.expander("Debug Log", expanded=False):
            for label, content in results["debug_log"]:
                st.markdown(f"**{label}:**")
                st.text(content)
