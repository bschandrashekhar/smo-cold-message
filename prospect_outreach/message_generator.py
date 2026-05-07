"""Prospect Outreach — Pass 2 Message Generation.

For each prospect row that is ready (has Prospect_Technologies, no existing Message_to_send):
  1. Load brand profile JSON
  2. Look up date range from dates sheet by city
  3. Generate 4-paragraph outreach message via Claude
"""

import io
import json
from typing import Callable, Optional

import anthropic
import pandas as pd

from prospect_outreach import config
from prospect_outreach.brand_knowledge import load_brand_profile

# Approved Salesforce products — do NOT mention anything outside this list
APPROVED_SF_PRODUCTS = [
    "Service Cloud", "Sales Cloud", "Non Profit", "NPSP", "Experience Cloud",
    "Commerce Cloud", "Agentforce", "LWC", "Appexchange Product Development",
    "Marketing Cloud", "Pardot", "Tableau", "CPQ", "Data Cloud",
]

MESSAGE_PROMPT = """You are writing a concise, personalized sales outreach email on behalf of {brand_name}.

Brand tone and positioning: {tone_and_positioning}
Brand services: {services}

Prospect details:
- First name: {first_name}
- Designation: {designation}
- Company: {company_name}
- Location: {location}

Technology research findings:
{technology_research}

Case studies (use for Para 2, anonymized as "a leading [industry] company"):
{case_studies}

Industry reference clients (use for Para 3 — mention by name):
{client_references}

Date window for meeting (Para 4): {date_window}

WRITING RULES — follow every rule strictly:
1. Address prospect by first name only (e.g. "Hi {first_name},")
2. Do NOT use hyphens or dashes anywhere in the message
3. Do NOT reference compliance standards by name (APRA, SOC2, PCI DSS, etc.) — say "compliance standards" instead
4. Total message must be UNDER 150 words
5. Case study references are brand-neutral — do NOT mention CloudChillies or LendingLogik within case study descriptions
6. Industry reference client names CAN be mentioned by name
7. No subject line, no signature
8. No specific numbers, metrics, or statistics in Para 1

STRUCTURE — write exactly 4 paragraphs:

Para 1: Show understanding of the prospect's situation. Reference a specific initiative or technology from the research, tailored to their role ({designation}). Naturally mention the specific technology names found (sets up Para 2). Keep to 1-2 sentences.

Para 2: Pick up the GENERAL-PURPOSE technology names from Para 1 (e.g. Boomi, Snowflake, MuleSoft) — NOT proprietary systems. Weave them into a case study narrative. Anonymize as "a leading [industry] company". Show how similar challenges were solved using Salesforce. Only reference these approved Salesforce products (and ONLY if clearly relevant): {approved_sf_products}. If no specific product fits, just say "Salesforce".

Para 3: In a SEPARATE sentence (not joined to Para 2 narrative), list ALL industry reference client names: e.g. "Some of our clients in this space include [Client A], [Client B], and [Client C]."

Para 4: {cta}

Return ONLY the message text, no labels, no "Para 1:" prefixes."""


def _get_date_window(dates_df: pd.DataFrame, city: str, state: str) -> str:
    """Look up meeting date window from dates sheet for a given city/state."""
    if dates_df.empty:
        return None

    city_lower = city.strip().lower()
    state_lower = state.strip().lower()

    for _, row in dates_df.iterrows():
        row_city = str(row.get("City", "")).strip().lower()
        row_state = str(row.get("State", "")).strip().lower()
        if row_city == city_lower and (not state_lower or row_state == state_lower):
            start = row.get("Start_Date", "")
            end = row.get("End_Date", "")
            if pd.notna(start) and pd.notna(end):
                if hasattr(start, "strftime"):
                    start = f"{start.strftime('%B')} {start.day}, {start.year}"
                if hasattr(end, "strftime"):
                    end = f"{end.strftime('%B')} {end.day}, {end.year}"
                return f"{start} and {end}"
    return None


def _format_case_studies(case_studies_json: str) -> str:
    """Format case studies JSON into readable text for the prompt."""
    try:
        data = json.loads(case_studies_json) if case_studies_json else []
    except (json.JSONDecodeError, TypeError):
        return "No case studies available."

    if isinstance(data, dict) and "error" in data:
        return "No case studies available."

    if not data:
        return "No case studies available."

    lines = []
    for cs in data[:5]:
        if isinstance(cs, dict):
            name = cs.get("casestudy_name", cs.get("name", ""))
            industry = cs.get("client_industry", "")
            problem = cs.get("summary_problem", "")
            solution = cs.get("summary_solution", "")
            outcomes = cs.get("summary_outcomes", "")
            exact_techs = cs.get("exact_techs", [])
            tech_str = ", ".join(exact_techs) if isinstance(exact_techs, list) else str(exact_techs)
            parts = [f"- {name} ({industry})"]
            if problem:
                parts.append(f"  Problem: {problem}")
            if solution:
                parts.append(f"  Solution: {solution}")
            if outcomes:
                parts.append(f"  Outcomes: {outcomes}")
            if tech_str:
                parts.append(f"  Technologies: {tech_str}")
            lines.append("\n".join(parts))
    return "\n".join(lines) if lines else "No case studies available."


def _format_client_references(refs_json: str) -> str:
    """Format client references JSON into a list of names."""
    try:
        data = json.loads(refs_json) if refs_json else []
    except (json.JSONDecodeError, TypeError):
        return "No client references available."

    if isinstance(data, dict) and "error" in data:
        return "No client references available."

    if not data:
        return "No client references available."

    names = []
    for ref in data:
        if isinstance(ref, dict):
            name = ref.get("client_name", ref.get("name", ""))
            if name:
                names.append(name)
    return ", ".join(names) if names else "No client references available."


def _generate_single_message(
    client: anthropic.Anthropic,
    row: pd.Series,
    dates_df: pd.DataFrame,
    brand_profile: dict,
) -> str:
    """Generate message for a single prospect. Returns message text."""
    first_name = str(row.get("First_Name", "")).strip()
    designation = str(row.get("Designation", "")).strip()
    company_name = str(row.get("Company_Name", "")).strip()
    city = str(row.get("City", "")).strip()
    state = str(row.get("State", "")).strip()
    location = f"{city}, {row.get('Country', '')}".strip(", ")
    technology_research = str(row.get("Prospect_Technologies", "")).strip()
    case_studies_json = str(row.get("Case_Studies", ""))
    refs_json = str(row.get("Industry_Client_References", ""))

    # Date window
    date_window = _get_date_window(dates_df, city, state)
    if date_window:
        cta = f"It will be really good to discuss this over a brief call between {date_window}. Please let me know what works best for you."
    else:
        cta = "It would be really good to discuss this over a brief call. Please let me know when we can connect."

    prompt = MESSAGE_PROMPT.format(
        brand_name=brand_profile.get("name", ""),
        tone_and_positioning=brand_profile.get("tone_and_positioning", ""),
        services=", ".join(brand_profile.get("services", [])),
        first_name=first_name,
        designation=designation,
        company_name=company_name,
        location=location,
        technology_research=technology_research,
        case_studies=_format_case_studies(case_studies_json),
        client_references=_format_client_references(refs_json),
        date_window=date_window or "no specific dates — use generic phrasing",
        approved_sf_products=", ".join(APPROVED_SF_PRODUCTS),
        cta=cta,
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def generate_messages(
    file_bytes: bytes,
    output_path: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Run Pass 2 message generation on reviewed data_output.xlsx.

    Args:
        file_bytes: Raw bytes of the uploaded .xlsx file.
        output_path: Path to write data_final.xlsx.
        progress_callback: Optional fn(current, total, status_msg).

    Returns:
        dict with "ready" and "skipped" counts.
    """
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="prospects")

    try:
        dates_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="dates")
    except Exception:
        dates_df = pd.DataFrame()

    if "Message_to_send" not in df.columns:
        df["Message_to_send"] = ""
    df["Message_to_send"] = df["Message_to_send"].astype(object).fillna("")

    # Determine ready vs skipped
    def _is_ready(row):
        msg = str(row.get("Message_to_send", "")).strip()
        tech_research = str(row.get("Prospect_Technologies", "")).strip()
        return not msg and bool(tech_research)

    ready_indices = [idx for idx, row in df.iterrows() if _is_ready(row)]
    skipped = len(df) - len(ready_indices)
    total = len(ready_indices)

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Brand profile cache
    brand_cache: dict[str, dict] = {}

    for i, idx in enumerate(ready_indices):
        row = df.loc[idx]
        brand_name = str(row.get("Suggested_Brand_Name_to_use", "CloudChillies")).strip()
        if not brand_name:
            brand_name = "CloudChillies"

        if progress_callback:
            first_name = str(row.get("First_Name", "")).strip()
            progress_callback(i + 1, total, f"Generating message for {first_name}...")

        if brand_name not in brand_cache:
            try:
                brand_cache[brand_name] = load_brand_profile(brand_name)
            except FileNotFoundError:
                brand_cache[brand_name] = {"name": brand_name, "tone_and_positioning": "", "services": []}

        message = _generate_single_message(client, row, dates_df, brand_cache[brand_name])
        df.at[idx, "Message_to_send"] = message

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="prospects", index=False)
        if not dates_df.empty:
            dates_df.to_excel(writer, sheet_name="dates", index=False)

    if progress_callback:
        progress_callback(total, total, "Message generation complete.")

    return {"ready": total, "skipped": skipped}
