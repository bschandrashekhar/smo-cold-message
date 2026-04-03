"""Sync logic: diff Excel rows against Supabase and apply create/update/delete."""

import json

import pandas as pd
import voyageai
from supabase import create_client

from client_referencing.config import (
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    TABLE_NAME,
    VOYAGE_API_KEY,
    VOYAGE_BATCH_SIZE,
    VOYAGE_MODEL,
)

# Composite key used to match Excel rows to Supabase rows
KEY_COLS = ("client_name", "exact_key")


def _row_key(row: dict) -> tuple:
    return (row["client_name"], row["exact_key"])


def _comparable(row: dict) -> dict:
    """Return the fields used to detect changes (excludes id, embedding, created_at)."""
    return {
        "client_name": row["client_name"],
        "client_industry": row["client_industry"],
        "client_geography": row["client_geography"],
        "client_url": row["client_url"],
        "industry_array": row["industry_array"],
        "industry_primary": row.get("industry_primary"),
        "industry_group": row.get("industry_group"),
        "embed_text": row["embed_text"],
        "exact_key": row["exact_key"],
    }


def read_excel(file_bytes: bytes) -> list[dict]:
    """Parse uploaded Excel bytes into a list of row dicts."""
    df = pd.read_excel(file_bytes, engine="openpyxl")

    # Normalize column names to match expected order
    expected = [
        "Client Name", "Client Industry", "Client Geography", "Client URL",
        "ExactKey", "EmbedText", "IndustryArray", "IndustryPrimary", "IndustryGroup",
    ]
    if list(df.columns) != expected:
        # Try positional mapping if headers don't match exactly
        if len(df.columns) >= 9:
            df.columns = expected[:len(df.columns)]

    rows = []
    for _, r in df.iterrows():
        industry_array_raw = r.get("IndustryArray", "[]")
        try:
            industry_array = json.loads(industry_array_raw) if pd.notna(industry_array_raw) else []
        except (json.JSONDecodeError, TypeError):
            industry_array = []

        rows.append({
            "client_name": r["Client Name"],
            "client_industry": r["Client Industry"],
            "client_geography": r["Client Geography"],
            "client_url": r["Client URL"],
            "exact_key": r["ExactKey"],
            "embed_text": r["EmbedText"],
            "industry_array": industry_array,
            "industry_primary": r.get("IndustryPrimary") if pd.notna(r.get("IndustryPrimary")) else None,
            "industry_group": r.get("IndustryGroup") if pd.notna(r.get("IndustryGroup")) else None,
        })
    return rows


def fetch_existing() -> list[dict]:
    """Fetch all current rows from Supabase (without embedding vectors)."""
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    # Fetch all columns except the large embedding vector for diffing
    cols = "id,client_name,client_industry,client_geography,client_url,industry_array,industry_primary,industry_group,embed_text,exact_key,embedding"
    result = sb.table(TABLE_NAME).select(cols).execute()
    return result.data


def compute_diff(excel_rows: list[dict], db_rows: list[dict]) -> dict:
    """Compare Excel rows to DB rows and return creates, updates, deletes."""
    excel_by_key = {}
    for r in excel_rows:
        excel_by_key[_row_key(r)] = r

    db_by_key = {}
    for r in db_rows:
        db_by_key[_row_key(r)] = r

    excel_keys = set(excel_by_key.keys())
    db_keys = set(db_by_key.keys())

    to_create = [excel_by_key[k] for k in (excel_keys - db_keys)]
    to_delete = [db_by_key[k] for k in (db_keys - excel_keys)]

    to_update = []
    for k in (excel_keys & db_keys):
        excel_comp = _comparable(excel_by_key[k])
        db_comp = _comparable(db_by_key[k])
        if excel_comp != db_comp:
            to_update.append({
                "db_id": db_by_key[k]["id"],
                "old": db_comp,
                "new": excel_by_key[k],
                "embed_text_changed": excel_comp["embed_text"] != db_comp["embed_text"],
            })

    return {"create": to_create, "update": to_update, "delete": to_delete}


def generate_embeddings(texts: list[str]) -> dict[str, list[float]]:
    """Generate embeddings for a list of texts using Voyage AI."""
    if not texts:
        return {}

    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    unique_texts = list(set(texts))
    embedding_map = {}

    for i in range(0, len(unique_texts), VOYAGE_BATCH_SIZE):
        batch = unique_texts[i : i + VOYAGE_BATCH_SIZE]
        result = vo.embed(batch, model=VOYAGE_MODEL, input_type="document")
        for text, emb in zip(batch, result.embeddings):
            embedding_map[text] = emb

    return embedding_map


def apply_sync(diff: dict, progress_callback=None) -> dict:
    """Apply the diff to Supabase. Returns summary counts."""
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    created = 0
    updated = 0
    deleted = 0
    total_ops = len(diff["create"]) + len(diff["update"]) + len(diff["delete"])
    done = 0

    def _progress(msg):
        if progress_callback:
            progress_callback(done, total_ops, msg)

    # Collect all embed_texts that need embeddings
    texts_needing_embeddings = [r["embed_text"] for r in diff["create"]]
    texts_needing_embeddings += [u["new"]["embed_text"] for u in diff["update"] if u["embed_text_changed"]]

    _progress("Generating embeddings...")
    embedding_map = generate_embeddings(texts_needing_embeddings)

    # For updates where embed_text didn't change, we need the existing embedding
    # We'll fetch those from DB
    ids_needing_existing_emb = [
        u["db_id"] for u in diff["update"] if not u["embed_text_changed"]
    ]
    existing_embeddings = {}
    if ids_needing_existing_emb:
        for db_id in ids_needing_existing_emb:
            result = sb.table(TABLE_NAME).select("id,embedding").eq("id", db_id).execute()
            if result.data:
                existing_embeddings[db_id] = result.data[0]["embedding"]

    # --- Creates ---
    if diff["create"]:
        records = []
        for r in diff["create"]:
            records.append({
                **{k: v for k, v in r.items()},
                "embedding": embedding_map[r["embed_text"]],
            })
        for i in range(0, len(records), 50):
            chunk = records[i : i + 50]
            sb.table(TABLE_NAME).insert(chunk).execute()
            created += len(chunk)
            done += len(chunk)
            _progress(f"Created {created} rows...")

    # --- Updates ---
    for u in diff["update"]:
        row_data = {k: v for k, v in u["new"].items()}
        if u["embed_text_changed"]:
            row_data["embedding"] = embedding_map[u["new"]["embed_text"]]
        else:
            row_data["embedding"] = existing_embeddings.get(u["db_id"])
        sb.table(TABLE_NAME).update(row_data).eq("id", u["db_id"]).execute()
        updated += 1
        done += 1
        _progress(f"Updated {updated} rows...")

    # --- Deletes ---
    for d_row in diff["delete"]:
        sb.table(TABLE_NAME).delete().eq("id", d_row["id"]).execute()
        deleted += 1
        done += 1
        _progress(f"Deleted {deleted} rows...")

    return {"created": created, "updated": updated, "deleted": deleted}
