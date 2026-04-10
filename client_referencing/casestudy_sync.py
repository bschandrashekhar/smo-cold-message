"""Case Studies sync: Excel -> Supabase (client_case_studies + technology mapping).

Handles reading the Excel file, comparing against DB, generating Claude summaries
and Voyage embeddings, uploading PDFs to storage, and applying changes.
"""

import io
import os
import time

import pandas as pd
from supabase import create_client

from prospect_outreach.config import (
    ANTHROPIC_API_KEY,
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    VOYAGE_API_KEY,
)

from client_referencing.config import VOYAGE_MODEL, VOYAGE_BATCH_SIZE

CASE_STUDIES_TABLE = "client_case_studies"
TECH_MAPPING_TABLE = "client_case_studies_technology_mapping"
STORAGE_BUCKET = "case-studies-relevant"

PDF_FOLDER = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "allclients-industry-tech-mapping",
        "data",
        "client-case-studies-mapping",
    )
)

_sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Lazy-initialized clients (only needed during apply_sync)
_claude = None
_vo = None


def _get_claude():
    global _claude
    if _claude is None:
        import anthropic
        _claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _claude


def _get_vo():
    global _vo
    if _vo is None:
        import voyageai
        _vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _vo


# ---------------------------------------------------------------------------
# Read Excel
# ---------------------------------------------------------------------------

def read_excel(file_bytes) -> dict:
    """Read the case studies Excel file (two worksheets).

    Returns {"summaries": [...], "technologies": [...]}.
    """
    buf = io.BytesIO(file_bytes) if isinstance(file_bytes, bytes) else file_bytes

    df_summary = pd.read_excel(buf, sheet_name="Case Studies Summary", engine="openpyxl")
    buf.seek(0)
    df_tech = pd.read_excel(buf, sheet_name="Case Studies Technology", engine="openpyxl")

    summaries = []
    for _, row in df_summary.iterrows():
        entry = {
            "client_id": int(row["client_id"]),
            "casestudy_name": str(row["casestudy_name"]).strip(),
            "case_study_file": str(row.get("case_study_file", "")).strip()
            if pd.notna(row.get("case_study_file"))
            else None,
        }
        # Capture pre-filled summary columns if present
        for col in ("summary_problem", "summary_solution", "summary_outcomes"):
            val = row.get(col)
            entry[col] = str(val).strip() if pd.notna(val) and str(val).strip() else None
        summaries.append(entry)

    technologies = []
    for _, row in df_tech.iterrows():
        technologies.append({
            "casestudy_name": str(row["casestudy_name"]).strip(),
            "casestudy_technology": str(row["casestudy_technology"]).strip(),
        })

    return {"summaries": summaries, "technologies": technologies}


# ---------------------------------------------------------------------------
# Fetch existing from DB
# ---------------------------------------------------------------------------

def fetch_existing() -> dict:
    """Fetch existing case studies and tech mappings from Supabase."""
    cs_result = (
        _sb.table(CASE_STUDIES_TABLE)
        .select("casestudy_id,client_id,casestudy_name,url,summary_problem,summary_solution,summary_outcomes")
        .execute()
    )
    tech_result = (
        _sb.table(TECH_MAPPING_TABLE)
        .select("id,casestudy_id,casestudy_technology")
        .execute()
    )
    return {
        "case_studies": cs_result.data,
        "technologies": tech_result.data,
    }


# ---------------------------------------------------------------------------
# Compute diff
# ---------------------------------------------------------------------------

def compute_diff(excel_data: dict, db_data: dict) -> dict:
    """Compare Excel data against DB and return changes needed."""
    # DB lookups
    db_cases_by_name = {r["casestudy_name"]: r for r in db_data["case_studies"]}

    # Build reverse map: casestudy_id -> casestudy_name
    id_to_name = {r["casestudy_id"]: r["casestudy_name"] for r in db_data["case_studies"]}

    db_techs_by_key = {}
    for t in db_data["technologies"]:
        cs_name = id_to_name.get(t["casestudy_id"], "")
        key = (cs_name, t["casestudy_technology"])
        db_techs_by_key[key] = t

    # Excel lookups
    excel_names = {s["casestudy_name"] for s in excel_data["summaries"]}
    excel_tech_keys = {(t["casestudy_name"], t["casestudy_technology"]) for t in excel_data["technologies"]}

    # Case study creates: in Excel with PDF, not in DB
    cs_create = []
    cs_update = []
    for s in excel_data["summaries"]:
        name = s["casestudy_name"]
        if name not in db_cases_by_name:
            if s.get("case_study_file"):
                cs_create.append(s)
        else:
            db_row = db_cases_by_name[name]
            # Check if PDF was added where it wasn't before
            if s.get("case_study_file") and not db_row.get("summary_problem"):
                cs_update.append({
                    "casestudy_id": db_row["casestudy_id"],
                    "casestudy_name": name,
                    "case_study_file": s["case_study_file"],
                    "summary_problem": s.get("summary_problem"),
                    "summary_solution": s.get("summary_solution"),
                    "summary_outcomes": s.get("summary_outcomes"),
                    "needs_summary": True,
                })

    # Case study deletes: in DB but not in Excel
    cs_delete = []
    for name, row in db_cases_by_name.items():
        if name not in excel_names:
            cs_delete.append(row)

    # Tech creates
    tech_create = []
    for t in excel_data["technologies"]:
        key = (t["casestudy_name"], t["casestudy_technology"])
        if key not in db_techs_by_key:
            tech_create.append(t)

    # Tech deletes
    tech_delete = []
    for key, row in db_techs_by_key.items():
        if key not in excel_tech_keys:
            tech_delete.append(row)

    return {
        "cs_create": cs_create,
        "cs_update": cs_update,
        "cs_delete": cs_delete,
        "tech_create": tech_create,
        "tech_delete": tech_delete,
    }


# ---------------------------------------------------------------------------
# PDF processing helpers
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: str) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _summarize_with_claude(pdf_text: str, client_name: str) -> dict:
    """Return {"problem": ..., "solution": ..., "outcomes": ...}."""
    response = _get_claude().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Below is the full text of a case study for the client '{client_name}'.\n\n"
                    f"Generate a structured summary with exactly these three sections:\n"
                    f"- **Problem**: What challenge or need did the client face?\n"
                    f"- **Solution**: What approach was taken to address it?\n"
                    f"- **Outcomes**: What results or benefits were achieved?\n\n"
                    f"Include specific technology names, product names, and platforms "
                    f"mentioned in the case study where relevant.\n\n"
                    f"Keep the total summary under 300 words.\n\n"
                    f"---\n{pdf_text}\n---"
                ),
            }
        ],
    )
    return _parse_summary(response.content[0].text)


def _parse_summary(text: str) -> dict:
    """Parse a structured summary into problem/solution/outcomes."""
    import re
    sections = {"problem": "", "solution": "", "outcomes": ""}
    patterns = [
        (r"\*\*Problem\*\*[:\s]*(.*?)(?=\*\*Solution\*\*)", "problem"),
        (r"\*\*Solution\*\*[:\s]*(.*?)(?=\*\*Outcomes\*\*)", "solution"),
        (r"\*\*Outcomes\*\*[:\s]*(.*?)$", "outcomes"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            sections[key] = m.group(1).strip().strip("-").strip()
    return sections


def _embed_texts(texts: list[str]) -> dict[str, list[float]]:
    if not texts:
        return {}
    unique = list(set(texts))
    mapping = {}
    for i in range(0, len(unique), VOYAGE_BATCH_SIZE):
        batch = unique[i : i + VOYAGE_BATCH_SIZE]
        result = _get_vo().embed(batch, model=VOYAGE_MODEL, input_type="document")
        for text, emb in zip(batch, result.embeddings):
            mapping[text] = emb
    return mapping


def _upload_pdf(pdf_path: str, filename: str) -> str:
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    _sb.storage.from_(STORAGE_BUCKET).upload(
        path=filename,
        file=pdf_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )
    return _sb.storage.from_(STORAGE_BUCKET).get_public_url(filename)


# ---------------------------------------------------------------------------
# Apply sync
# ---------------------------------------------------------------------------

def apply_sync(diff: dict, progress_callback=None) -> dict:
    """Apply the computed diff to Supabase."""
    total_ops = (
        len(diff["cs_create"])
        + len(diff["cs_update"])
        + len(diff["cs_delete"])
        + len(diff["tech_create"])
        + len(diff["tech_delete"])
    )
    current = 0

    def _progress(text):
        nonlocal current
        current += 1
        if progress_callback:
            progress_callback(current, total_ops + 3, text)

    # Verify client IDs
    all_client_ids = list({s["client_id"] for s in diff["cs_create"]})
    if all_client_ids:
        result = (
            _sb.table("client_referencing_data")
            .select("id")
            .in_("id", all_client_ids)
            .execute()
        )
        valid_ids = {r["id"] for r in result.data}
    else:
        valid_ids = set()

    # Get existing case study name->id map for tech inserts
    existing = _sb.table(CASE_STUDIES_TABLE).select("casestudy_id,casestudy_name").execute()
    name_to_id = {r["casestudy_name"]: r["casestudy_id"] for r in existing.data}

    # 1. Process new case studies (extract PDF, summarize, embed, upload, insert)
    cs_created = 0
    for cs in diff["cs_create"]:
        _progress(f"Processing {cs['casestudy_name']}...")

        if cs["client_id"] not in valid_ids:
            continue

        filename = cs["case_study_file"]
        pdf_path = os.path.join(PDF_FOLDER, filename)

        if not os.path.exists(pdf_path):
            continue

        # Use pre-filled summaries from Excel if all 3 are present; otherwise generate via Claude
        prefilled = (
            cs.get("summary_problem")
            and cs.get("summary_solution")
            and cs.get("summary_outcomes")
        )
        if prefilled:
            parts = {
                "problem": cs["summary_problem"],
                "solution": cs["summary_solution"],
                "outcomes": cs["summary_outcomes"],
            }
        else:
            pdf_text = _extract_pdf_text(pdf_path)
            if not pdf_text.strip():
                continue
            parts = _summarize_with_claude(pdf_text, cs["casestudy_name"])

        url = _upload_pdf(pdf_path, filename)
        embed_text = parts["problem"] + " " + parts["solution"]
        emb_map = _embed_texts([embed_text])

        record = {
            "client_id": cs["client_id"],
            "casestudy_name": cs["casestudy_name"],
            "url": url,
            "summary_problem": parts["problem"],
            "summary_solution": parts["solution"],
            "summary_outcomes": parts["outcomes"],
        }
        if embed_text in emb_map:
            record["summary_embedding"] = emb_map[embed_text]

        result = _sb.table(CASE_STUDIES_TABLE).insert(record).execute()
        if result.data:
            name_to_id[cs["casestudy_name"]] = result.data[0]["casestudy_id"]
            cs_created += 1

        time.sleep(0.5)

    # 2. Update existing case studies (add summary where missing)
    cs_updated = 0
    for cs in diff["cs_update"]:
        _progress(f"Updating {cs['casestudy_name']}...")

        filename = cs["case_study_file"]
        pdf_path = os.path.join(PDF_FOLDER, filename)

        if not os.path.exists(pdf_path):
            continue

        # Use pre-filled summaries from Excel if all 3 are present
        prefilled = (
            cs.get("summary_problem")
            and cs.get("summary_solution")
            and cs.get("summary_outcomes")
        )
        if prefilled:
            parts = {
                "problem": cs["summary_problem"],
                "solution": cs["summary_solution"],
                "outcomes": cs["summary_outcomes"],
            }
        else:
            pdf_text = _extract_pdf_text(pdf_path)
            if not pdf_text.strip():
                continue
            parts = _summarize_with_claude(pdf_text, cs["casestudy_name"])

        url = _upload_pdf(pdf_path, filename)
        embed_text = parts["problem"] + " " + parts["solution"]
        emb_map = _embed_texts([embed_text])

        update_data = {
            "summary_problem": parts["problem"],
            "summary_solution": parts["solution"],
            "summary_outcomes": parts["outcomes"],
            "url": url,
        }
        if embed_text in emb_map:
            update_data["summary_embedding"] = emb_map[embed_text]

        _sb.table(CASE_STUDIES_TABLE).update(update_data).eq(
            "casestudy_id", cs["casestudy_id"]
        ).execute()
        cs_updated += 1
        time.sleep(0.5)

    # 3. Delete case studies (CASCADE handles tech mappings)
    cs_deleted = 0
    for cs in diff["cs_delete"]:
        _progress(f"Deleting {cs['casestudy_name']}...")
        _sb.table(CASE_STUDIES_TABLE).delete().eq(
            "casestudy_id", cs["casestudy_id"]
        ).execute()
        cs_deleted += 1

    # 4. Create tech mappings
    tech_created = 0
    if diff["tech_create"]:
        _progress("Generating technology embeddings...")
        tech_texts = list({t["casestudy_technology"] for t in diff["tech_create"]})
        tech_emb_map = _embed_texts(tech_texts)

        for t in diff["tech_create"]:
            cs_id = name_to_id.get(t["casestudy_name"])
            if not cs_id:
                continue

            record = {
                "casestudy_id": cs_id,
                "casestudy_technology": t["casestudy_technology"],
            }
            tech_name = t["casestudy_technology"]
            if tech_name in tech_emb_map:
                record["casestudy_technology_embedding"] = tech_emb_map[tech_name]

            _sb.table(TECH_MAPPING_TABLE).insert(record).execute()
            tech_created += 1
            _progress(f"Tech: {t['casestudy_technology']}")

    # 5. Delete tech mappings
    tech_deleted = 0
    for t in diff["tech_delete"]:
        _progress(f"Deleting tech: {t['casestudy_technology']}...")
        _sb.table(TECH_MAPPING_TABLE).delete().eq("id", t["id"]).execute()
        tech_deleted += 1

    return {
        "cs_created": cs_created,
        "cs_updated": cs_updated,
        "cs_deleted": cs_deleted,
        "tech_created": tech_created,
        "tech_deleted": tech_deleted,
    }
