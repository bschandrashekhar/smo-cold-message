"""Sync logic: diff Excel rows against Supabase and apply create/update/delete.

Supports two-table schema:
  - client_referencing_data (clients): metadata only
  - client_tech_data (technologies): exact_key, embed_text, embedding with FK to client
"""

import json

import pandas as pd
import voyageai
from supabase import create_client

from client_referencing.config import (
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    TABLE_NAME,
    TECH_TABLE_NAME,
    VOYAGE_API_KEY,
    VOYAGE_BATCH_SIZE,
    VOYAGE_MODEL,
)


# ---------------------------------------------------------------------------
# Comparable helpers
# ---------------------------------------------------------------------------

def _client_comparable(row: dict) -> dict:
    """Fields used to detect client metadata changes."""
    return {
        "client_name": row["client_name"],
        "client_industry": row["client_industry"],
        "client_geography": row["client_geography"],
        "client_url": row["client_url"],
        "industry_array": row["industry_array"],
        "industry_primary": row.get("industry_primary"),
        "industry_group": row.get("industry_group"),
        "geo_priority": row.get("geo_priority"),
    }


def _tech_comparable(row: dict) -> dict:
    """Fields used to detect tech changes (excludes id, embedding, created_at)."""
    return {
        "client_name": row["client_name"],
        "exact_key": row["exact_key"],
        "embed_text": row["embed_text"],
    }


# ---------------------------------------------------------------------------
# Excel parsing (two worksheets)
# ---------------------------------------------------------------------------

def read_excel(file_bytes: bytes) -> dict:
    """Parse uploaded Excel bytes with two worksheets into client + tech dicts.

    Returns {"clients": [...], "technologies": [...]}.
    """
    # --- Clients sheet ---
    df_clients = pd.read_excel(file_bytes, sheet_name="Clients", engine="openpyxl")
    expected_client_cols = [
        "Client Name", "Client Industry", "Client Geography", "Client URL",
        "IndustryArray", "IndustryPrimary", "IndustryGroup", "GeoPriority",
    ]
    if list(df_clients.columns) != expected_client_cols:
        if len(df_clients.columns) >= len(expected_client_cols):
            df_clients.columns = expected_client_cols[:len(df_clients.columns)]

    clients = []
    for _, r in df_clients.iterrows():
        industry_array_raw = r.get("IndustryArray", "[]")
        try:
            industry_array = json.loads(industry_array_raw) if pd.notna(industry_array_raw) else []
        except (json.JSONDecodeError, TypeError):
            industry_array = []

        geo_priority = r.get("GeoPriority")
        if pd.notna(geo_priority):
            geo_priority = int(geo_priority)
        else:
            geo_priority = 999

        clients.append({
            "client_name": r["Client Name"],
            "client_industry": r["Client Industry"],
            "client_geography": r["Client Geography"],
            "client_url": r["Client URL"],
            "industry_array": industry_array,
            "industry_primary": r.get("IndustryPrimary") if pd.notna(r.get("IndustryPrimary")) else None,
            "industry_group": r.get("IndustryGroup") if pd.notna(r.get("IndustryGroup")) else None,
            "geo_priority": geo_priority,
        })

    # --- Technologies sheet ---
    df_techs = pd.read_excel(file_bytes, sheet_name="Technologies", engine="openpyxl")
    expected_tech_cols = ["Client Name", "ExactKey", "EmbedText"]
    if list(df_techs.columns) != expected_tech_cols:
        if len(df_techs.columns) >= len(expected_tech_cols):
            df_techs.columns = expected_tech_cols[:len(df_techs.columns)]

    technologies = []
    for _, r in df_techs.iterrows():
        technologies.append({
            "client_name": r["Client Name"],
            "exact_key": r["ExactKey"],
            "embed_text": r["EmbedText"],
        })

    return {"clients": clients, "technologies": technologies}


# ---------------------------------------------------------------------------
# Fetch existing from DB
# ---------------------------------------------------------------------------

def fetch_existing() -> dict:
    """Fetch current clients and technologies from Supabase.

    Returns {"clients": [...], "technologies": [...]}.
    """
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    client_cols = ("id,client_name,client_industry,client_geography,client_url,"
                   "industry_array,industry_primary,industry_group,geo_priority")
    client_result = sb.table(TABLE_NAME).select(client_cols).execute()

    tech_cols = "id,client_id,exact_key,embed_text"
    tech_result = sb.table(TECH_TABLE_NAME).select(tech_cols).execute()

    # Resolve client_name for each tech row via client lookup
    client_lookup = {r["id"]: r["client_name"] for r in (client_result.data or [])}
    tech_rows = []
    for t in (tech_result.data or []):
        cname = client_lookup.get(t["client_id"], "")
        tech_rows.append({
            "id": t["id"],
            "client_id": t["client_id"],
            "client_name": cname,
            "exact_key": t["exact_key"],
            "embed_text": t["embed_text"],
        })

    return {"clients": client_result.data or [], "technologies": tech_rows}


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------

def compute_diff(excel_data: dict, db_data: dict) -> dict:
    """Compare Excel data to DB data for both clients and technologies.

    Returns dict with keys:
        client_create, client_update, client_delete,
        tech_create, tech_update, tech_delete
    """
    # --- Client diff (keyed on client_name) ---
    excel_clients = {r["client_name"]: r for r in excel_data["clients"]}
    db_clients = {r["client_name"]: r for r in db_data["clients"]}

    excel_cnames = set(excel_clients.keys())
    db_cnames = set(db_clients.keys())

    client_create = [excel_clients[k] for k in (excel_cnames - db_cnames)]
    client_delete = [db_clients[k] for k in (db_cnames - excel_cnames)]

    client_update = []
    for k in (excel_cnames & db_cnames):
        excel_comp = _client_comparable(excel_clients[k])
        db_comp = _client_comparable(db_clients[k])
        if excel_comp != db_comp:
            client_update.append({
                "db_id": db_clients[k]["id"],
                "old": db_comp,
                "new": excel_clients[k],
            })

    # --- Tech diff (keyed on client_name + exact_key) ---
    def _tech_key(row):
        return (row["client_name"], row["exact_key"])

    excel_techs = {_tech_key(r): r for r in excel_data["technologies"]}
    db_techs = {_tech_key(r): r for r in db_data["technologies"]}

    excel_tkeys = set(excel_techs.keys())
    db_tkeys = set(db_techs.keys())

    tech_create = [excel_techs[k] for k in (excel_tkeys - db_tkeys)]
    tech_delete = [db_techs[k] for k in (db_tkeys - excel_tkeys)]

    tech_update = []
    for k in (excel_tkeys & db_tkeys):
        excel_comp = _tech_comparable(excel_techs[k])
        db_comp = _tech_comparable(db_techs[k])
        if excel_comp != db_comp:
            tech_update.append({
                "db_id": db_techs[k]["id"],
                "old": db_comp,
                "new": excel_techs[k],
                "embed_text_changed": excel_comp["embed_text"] != db_comp["embed_text"],
            })

    return {
        "client_create": client_create,
        "client_update": client_update,
        "client_delete": client_delete,
        "tech_create": tech_create,
        "tech_update": tech_update,
        "tech_delete": tech_delete,
    }


# ---------------------------------------------------------------------------
# Embedding generation (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Apply sync (two-table)
# ---------------------------------------------------------------------------

def apply_sync(diff: dict, progress_callback=None) -> dict:
    """Apply the diff to Supabase (clients + technologies). Returns summary counts."""
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    counts = {
        "clients_created": 0, "clients_updated": 0, "clients_deleted": 0,
        "techs_created": 0, "techs_updated": 0, "techs_deleted": 0,
    }
    total_ops = (
        len(diff["client_create"]) + len(diff["client_update"]) + len(diff["client_delete"])
        + len(diff["tech_create"]) + len(diff["tech_update"]) + len(diff["tech_delete"])
    )
    done = 0

    def _progress(msg):
        if progress_callback:
            progress_callback(done, total_ops, msg)

    # ── 1. Create new clients first (need IDs for tech FKs) ──
    new_client_ids = {}  # client_name -> new db id
    for c in diff["client_create"]:
        record = {k: v for k, v in c.items()}
        result = sb.table(TABLE_NAME).insert(record).execute()
        if result.data:
            new_client_ids[c["client_name"]] = result.data[0]["id"]
        counts["clients_created"] += 1
        done += 1
        _progress(f"Created {counts['clients_created']} clients...")

    # ── 2. Update existing clients ──
    for u in diff["client_update"]:
        row_data = {k: v for k, v in u["new"].items()}
        sb.table(TABLE_NAME).update(row_data).eq("id", u["db_id"]).execute()
        counts["clients_updated"] += 1
        done += 1
        _progress(f"Updated {counts['clients_updated']} clients...")

    # ── 3. Build client_name -> id lookup for tech operations ──
    # Fetch all current clients to resolve names to IDs
    client_result = sb.table(TABLE_NAME).select("id,client_name").execute()
    name_to_id = {r["client_name"]: r["id"] for r in (client_result.data or [])}

    # ── 4. Tech creates (with embeddings) ──
    if diff["tech_create"]:
        texts_for_embed = [t["embed_text"] for t in diff["tech_create"]]
        texts_for_embed += [
            u["new"]["embed_text"] for u in diff["tech_update"]
            if u.get("embed_text_changed")
        ]
        _progress("Generating embeddings...")
        embedding_map = generate_embeddings(texts_for_embed)

        records = []
        for t in diff["tech_create"]:
            client_id = name_to_id.get(t["client_name"])
            if client_id is None:
                continue
            records.append({
                "client_id": client_id,
                "exact_key": t["exact_key"],
                "embed_text": t["embed_text"],
                "embedding": embedding_map[t["embed_text"]],
            })
        for i in range(0, len(records), 50):
            chunk = records[i : i + 50]
            sb.table(TECH_TABLE_NAME).insert(chunk).execute()
            counts["techs_created"] += len(chunk)
            done += len(chunk)
            _progress(f"Created {counts['techs_created']} tech rows...")
    else:
        # Still generate embeddings for tech updates even if no creates
        texts_for_embed = [
            u["new"]["embed_text"] for u in diff["tech_update"]
            if u.get("embed_text_changed")
        ]
        embedding_map = generate_embeddings(texts_for_embed) if texts_for_embed else {}

    # ── 5. Tech updates ──
    # For updates where embed_text didn't change, fetch existing embedding
    ids_needing_existing_emb = [
        u["db_id"] for u in diff["tech_update"] if not u.get("embed_text_changed")
    ]
    existing_embeddings = {}
    if ids_needing_existing_emb:
        for db_id in ids_needing_existing_emb:
            result = sb.table(TECH_TABLE_NAME).select("id,embedding").eq("id", db_id).execute()
            if result.data:
                existing_embeddings[db_id] = result.data[0]["embedding"]

    for u in diff["tech_update"]:
        client_id = name_to_id.get(u["new"]["client_name"])
        if client_id is None:
            continue
        row_data = {
            "client_id": client_id,
            "exact_key": u["new"]["exact_key"],
            "embed_text": u["new"]["embed_text"],
        }
        if u.get("embed_text_changed"):
            row_data["embedding"] = embedding_map[u["new"]["embed_text"]]
        else:
            row_data["embedding"] = existing_embeddings.get(u["db_id"])
        sb.table(TECH_TABLE_NAME).update(row_data).eq("id", u["db_id"]).execute()
        counts["techs_updated"] += 1
        done += 1
        _progress(f"Updated {counts['techs_updated']} tech rows...")

    # ── 6. Tech deletes ──
    for t in diff["tech_delete"]:
        sb.table(TECH_TABLE_NAME).delete().eq("id", t["id"]).execute()
        counts["techs_deleted"] += 1
        done += 1
        _progress(f"Deleted {counts['techs_deleted']} tech rows...")

    # ── 7. Client deletes (CASCADE handles orphan techs, but we deleted explicitly above) ──
    for c in diff["client_delete"]:
        sb.table(TABLE_NAME).delete().eq("id", c["id"]).execute()
        counts["clients_deleted"] += 1
        done += 1
        _progress(f"Deleted {counts['clients_deleted']} clients...")

    return counts
