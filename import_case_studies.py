"""One-time import: case study PDFs -> Claude summaries -> Voyage embeddings -> Supabase.

Reads the Excel mapping file, extracts text from PDFs, generates structured
summaries via Claude, embeds summaries + technologies via Voyage AI, uploads
PDFs to Supabase Storage, and inserts everything into:
  - client_case_studies  (summary + summary_embedding)
  - client_case_studies_technology_mapping  (tech + tech_embedding)
"""

import os
import sys
import time

import anthropic
import pandas as pd
import pdfplumber
import voyageai
from supabase import create_client

# ── Config ───────────────────────────────────────────────────────────────
from prospect_outreach.config import (
    ANTHROPIC_API_KEY,
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    VOYAGE_API_KEY,
)

VOYAGE_MODEL = "voyage-4-large"
VOYAGE_BATCH_SIZE = 128
STORAGE_BUCKET = "case-studies-relevant"

PDF_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "..",
    "allclients-industry-tech-mapping",
    "data",
    "client-case-studies-mapping",
)

EXCEL_FILE = os.path.join(PDF_FOLDER, "client-case-studies-mapping.xlsx")

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
vo = voyageai.Client(api_key=VOYAGE_API_KEY)


# ── Helpers ──────────────────────────────────────────────────────────────


def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from a PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def summarize_with_claude(pdf_text: str, client_name: str) -> dict:
    """Generate a structured summary (problem, solution, outcomes) via Claude.

    Returns {"problem": ..., "solution": ..., "outcomes": ...}.
    The summary includes specific technology names, product names, and
    platforms mentioned in the case study where relevant.
    """
    response = claude.messages.create(
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
    return parse_summary(response.content[0].text)


def parse_summary(text: str) -> dict:
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


def embed_texts(texts: list[str]) -> dict[str, list[float]]:
    """Embed a list of texts with Voyage AI in batches."""
    if not texts:
        return {}
    unique = list(set(texts))
    mapping = {}
    for i in range(0, len(unique), VOYAGE_BATCH_SIZE):
        batch = unique[i : i + VOYAGE_BATCH_SIZE]
        result = vo.embed(batch, model=VOYAGE_MODEL, input_type="document")
        for text, emb in zip(batch, result.embeddings):
            mapping[text] = emb
    return mapping


def upload_pdf_to_storage(pdf_path: str, filename: str) -> str:
    """Upload PDF to Supabase Storage and return the public URL."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    sb.storage.from_(STORAGE_BUCKET).upload(
        path=filename,
        file=pdf_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )
    public_url = sb.storage.from_(STORAGE_BUCKET).get_public_url(filename)
    return public_url


def verify_client_ids(df_summary: pd.DataFrame) -> dict[int, str]:
    """Verify client_ids against client_referencing_data. Returns {id: name} map."""
    client_ids = df_summary["client_id"].dropna().astype(int).unique().tolist()
    result = (
        sb.table("client_referencing_data")
        .select("id,client_name")
        .in_("id", client_ids)
        .execute()
    )
    return {r["id"]: r["client_name"] for r in result.data}


# ── Main import ──────────────────────────────────────────────────────────


def run_import():
    print("=" * 60)
    print("Case Study Import")
    print("=" * 60)

    # 1. Read Excel
    print("\n[1/6] Reading Excel file...")
    df_summary = pd.read_excel(EXCEL_FILE, sheet_name="Case Studies Summary")
    df_tech = pd.read_excel(EXCEL_FILE, sheet_name="Case Studies Technology")
    print(f"  Case studies: {len(df_summary)}")
    print(f"  Technology mappings: {len(df_tech)}")

    # 2. Verify client IDs
    print("\n[2/6] Verifying client IDs...")
    valid_clients = verify_client_ids(df_summary)
    print(f"  Verified {len(valid_clients)} client IDs")

    mismatches = []
    for _, row in df_summary.iterrows():
        cid = int(row["client_id"])
        if cid not in valid_clients:
            mismatches.append(f"  [X] client_id={cid} ({row['casestudy_name']}) - NOT FOUND in DB")
        else:
            db_name = valid_clients[cid]
            if db_name.strip().lower() != str(row["casestudy_name"]).strip().lower():
                mismatches.append(
                    f"  [!] client_id={cid}: Excel='{row['casestudy_name']}' vs DB='{db_name}'"
                )
    if mismatches:
        print("  Mismatches/missing:")
        for m in mismatches:
            print(m)
    else:
        print("  All client IDs verified OK")

    # 3. Process PDFs: extract text, summarize, upload
    print("\n[3/6] Processing PDFs (extract -> Claude summary -> upload)...")
    has_pdf = df_summary[df_summary["case_study_file"].notna()]
    summaries = {}  # casestudy_name -> summary text
    urls = {}  # casestudy_name -> public URL

    for idx, row in has_pdf.iterrows():
        name = row["casestudy_name"]
        filename = row["case_study_file"].strip()
        pdf_path = os.path.join(PDF_FOLDER, filename)

        if not os.path.exists(pdf_path):
            print(f"  [X] PDF not found: {filename} - skipping {name}")
            continue

        print(f"  [{idx+1}/{len(df_summary)}] {name}...")

        # Extract text
        pdf_text = extract_pdf_text(pdf_path)
        if not pdf_text.strip():
            print(f"    [X] No text extracted from {filename}")
            continue

        # Summarize with Claude
        try:
            parts = summarize_with_claude(pdf_text, name)
            summaries[name] = parts
            print(f"    [OK] Summary generated (problem={len(parts['problem'])} solution={len(parts['solution'])} outcomes={len(parts['outcomes'])} chars)")
        except Exception as e:
            print(f"    [X] Claude error: {e}")
            continue

        # Upload PDF to storage
        try:
            url = upload_pdf_to_storage(pdf_path, filename)
            urls[name] = url
            print(f"    [OK] Uploaded to storage")
        except Exception as e:
            print(f"    [X] Upload error: {e}")

        # Rate limit courtesy
        time.sleep(0.5)

    print(f"\n  Summaries generated: {len(summaries)}")
    print(f"  PDFs uploaded: {len(urls)}")

    # 4. Generate embeddings for summaries (Problem + Solution only)
    print("\n[4/6] Generating summary embeddings (Voyage AI)...")
    embed_input = {name: parts["problem"] + " " + parts["solution"] for name, parts in summaries.items()}
    summary_embeddings = embed_texts(list(embed_input.values()))
    print(f"  Embedded {len(summary_embeddings)} summaries")

    # 5. Insert into client_case_studies
    print("\n[5/6] Inserting into client_case_studies...")
    inserted_cases = {}  # casestudy_name -> casestudy_id (UUID)
    insert_count = 0
    skip_count = 0

    for _, row in df_summary.iterrows():
        name = row["casestudy_name"]
        cid = int(row["client_id"])

        if cid not in valid_clients:
            skip_count += 1
            continue

        parts = summaries.get(name)
        record = {
            "client_id": cid,
            "casestudy_name": name,
            "url": urls.get(name),
            "summary_problem": parts["problem"] if parts else None,
            "summary_solution": parts["solution"] if parts else None,
            "summary_outcomes": parts["outcomes"] if parts else None,
        }

        # Add summary embedding if we have a summary (based on Problem + Solution)
        if name in embed_input and embed_input[name] in summary_embeddings:
            record["summary_embedding"] = summary_embeddings[embed_input[name]]

        try:
            result = sb.table("client_case_studies").insert(record).execute()
            if result.data:
                inserted_cases[name] = result.data[0]["casestudy_id"]
                insert_count += 1
        except Exception as e:
            print(f"  [X] Insert failed for {name}: {e}")
            skip_count += 1

    print(f"  Inserted: {insert_count}, Skipped: {skip_count}")

    # 6. Insert technology mappings + embeddings
    print("\n[6/6] Processing technology mappings...")

    # Generate tech embeddings
    tech_texts = df_tech["casestudy_technology"].unique().tolist()
    print(f"  Generating embeddings for {len(tech_texts)} unique technologies...")
    tech_embeddings = embed_texts(tech_texts)

    tech_insert_count = 0
    tech_skip_count = 0

    for _, row in df_tech.iterrows():
        name = row["casestudy_name"]
        tech = row["casestudy_technology"]

        if name not in inserted_cases:
            tech_skip_count += 1
            continue

        record = {
            "casestudy_id": inserted_cases[name],
            "casestudy_technology": tech,
        }
        if tech in tech_embeddings:
            record["casestudy_technology_embedding"] = tech_embeddings[tech]

        try:
            sb.table("client_case_studies_technology_mapping").insert(record).execute()
            tech_insert_count += 1
        except Exception as e:
            print(f"  [X] Tech insert failed for {name}/{tech}: {e}")
            tech_skip_count += 1

    print(f"  Inserted: {tech_insert_count}, Skipped: {tech_skip_count}")

    # Done
    print("\n" + "=" * 60)
    print("Import complete!")
    print(f"  Case studies inserted: {insert_count}")
    print(f"  Technology mappings inserted: {tech_insert_count}")
    print(f"  PDFs uploaded: {len(urls)}")
    print("=" * 60)


if __name__ == "__main__":
    run_import()
