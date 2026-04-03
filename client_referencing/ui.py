"""Client Referencing Data Sync UI — upload Excel, preview diff, apply sync."""

import pandas as pd
import streamlit as st


def render():
    """Render the Client Referencing Data sync tab."""
    st.header("Client Referencing Data Sync")
    st.caption(
        "Upload the client-referencing-data.xlsx file to sync with Supabase. "
        "The tool compares your Excel against the database and shows what will be "
        "created, updated, or deleted before applying changes."
    )

    uploaded = st.file_uploader(
        "Upload client-referencing-data.xlsx", type=["xlsx"], key="cr_upload"
    )

    if uploaded is None:
        # Show current DB state
        _show_current_data()
        return

    from client_referencing.sync import read_excel, fetch_existing, compute_diff, apply_sync

    # Parse Excel
    try:
        excel_rows = read_excel(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to parse Excel: {e}")
        return

    st.success(f"Parsed **{len(excel_rows)}** rows from Excel.")

    # Fetch existing DB data
    with st.spinner("Fetching current data from Supabase..."):
        try:
            db_rows = fetch_existing()
        except Exception as e:
            st.error(f"Failed to fetch from Supabase: {e}")
            return

    st.info(f"**{len(db_rows)}** rows currently in Supabase.")

    # Compute diff
    diff = compute_diff(excel_rows, db_rows)
    n_create = len(diff["create"])
    n_update = len(diff["update"])
    n_delete = len(diff["delete"])

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("To Create", n_create, delta=f"+{n_create}" if n_create else None)
    col2.metric("To Update", n_update)
    col3.metric("To Delete", n_delete, delta=f"-{n_delete}" if n_delete else None, delta_color="inverse")

    if n_create == 0 and n_update == 0 and n_delete == 0:
        st.success("Everything is in sync. No changes needed.")
        return

    # Detail expandable sections
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

    # Embed cost note
    texts_needing_embed = n_create + sum(1 for u in diff["update"] if u["embed_text_changed"])
    if texts_needing_embed > 0:
        st.caption(f"Voyage AI embeddings will be generated for **{texts_needing_embed}** rows (deduplicated where possible).")

    # Apply button
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
        except Exception as e:
            st.error(f"Sync failed: {e}")
            import traceback
            st.code(traceback.format_exc())


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
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load current data: {e}")
