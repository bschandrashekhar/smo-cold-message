"""Sync logic for industry reference embeddings.

Mirrors the pattern in sync.py but for a simpler single-column table:
  industry_embeddings(id, term, embedding)
"""

import pandas as pd
import voyageai
from supabase import create_client

from client_referencing.config import (
    INDUSTRY_TABLE_NAME,
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    VOYAGE_API_KEY,
    VOYAGE_BATCH_SIZE,
    VOYAGE_MODEL,
)


def read_excel(file_bytes: bytes) -> list[dict]:
    """Parse uploaded Excel bytes into a list of industry term dicts."""
    df = pd.read_excel(file_bytes, engine="openpyxl")

    expected = ["IndustryTerm"]
    if len(df.columns) >= 1 and list(df.columns) != expected:
        df.columns = expected[: len(df.columns)]

    rows = []
    for _, r in df.iterrows():
        term = str(r["IndustryTerm"]).strip().lower()
        if term:
            rows.append({"term": term})
    return rows


def fetch_existing() -> list[dict]:
    """Fetch all current rows from the industry_embeddings table."""
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    result = sb.table(INDUSTRY_TABLE_NAME).select("id,term,embedding").execute()
    return result.data


def compute_diff(excel_rows: list[dict], db_rows: list[dict]) -> dict:
    """Compare Excel terms to DB terms and return creates, updates, deletes."""
    excel_by_term = {r["term"]: r for r in excel_rows}
    db_by_term = {r["term"]: r for r in db_rows}

    excel_terms = set(excel_by_term.keys())
    db_terms = set(db_by_term.keys())

    to_create = [excel_by_term[t] for t in sorted(excel_terms - db_terms)]
    to_delete = [db_by_term[t] for t in sorted(db_terms - excel_terms)]

    # No "update" concept — terms are immutable strings. If a term exists,
    # its embedding stays the same. To re-embed, delete and re-create.

    return {"create": to_create, "update": [], "delete": to_delete}


def generate_embeddings(terms: list[str]) -> dict[str, list[float]]:
    """Generate Voyage embeddings for industry terms."""
    if not terms:
        return {}

    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    unique = list(set(terms))
    result_map = {}

    for i in range(0, len(unique), VOYAGE_BATCH_SIZE):
        batch = unique[i : i + VOYAGE_BATCH_SIZE]
        result = vo.embed(batch, model=VOYAGE_MODEL, input_type="document")
        for text, emb in zip(batch, result.embeddings):
            result_map[text] = emb

    return result_map


def apply_sync(diff: dict, progress_callback=None) -> dict:
    """Apply the diff to Supabase. Returns summary counts."""
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    created = 0
    deleted = 0
    total_ops = len(diff["create"]) + len(diff["delete"])
    done = 0

    def _progress(msg):
        if progress_callback:
            progress_callback(done, total_ops, msg)

    # Generate embeddings for new terms
    new_terms = [r["term"] for r in diff["create"]]
    _progress("Generating embeddings for new industry terms...")
    embedding_map = generate_embeddings(new_terms)

    # Creates
    if diff["create"]:
        records = []
        for r in diff["create"]:
            records.append({
                "term": r["term"],
                "embedding": embedding_map[r["term"]],
            })
        for i in range(0, len(records), 50):
            chunk = records[i : i + 50]
            sb.table(INDUSTRY_TABLE_NAME).insert(chunk).execute()
            created += len(chunk)
            done += len(chunk)
            _progress(f"Created {created} terms...")

    # Deletes
    for d_row in diff["delete"]:
        sb.table(INDUSTRY_TABLE_NAME).delete().eq("id", d_row["id"]).execute()
        deleted += 1
        done += 1
        _progress(f"Deleted {deleted} terms...")

    return {"created": created, "deleted": deleted}
