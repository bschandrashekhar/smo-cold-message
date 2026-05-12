"""Prospect Outreach — Pass 2 Message Generation.

For each prospect row that is ready (has Prospect_Technologies, no existing WARM_MESSAGE):
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

WRITING RULES — follow every rule strictly:
1. Address prospect by first name only (e.g. "Hi {first_name},")
2. Do NOT use hyphens or dashes anywhere in the message
3. Do NOT reference compliance standards by name (APRA, SOC2, PCI DSS, etc.) — say "compliance standards" instead
4. Total message must be UNDER 150 words
5. Case study references are brand-neutral — do NOT mention CloudChillies or LendingLogik within case study descriptions
6. No subject line, no signature
7. No specific numbers, metrics, or statistics in Para 1

STRUCTURE — write exactly 2 paragraphs:

Para 1: Show understanding of the prospect's technological situation, tailored to their role ({designation}). Naturally mention specific technology names found in the research (sets up Para 2). Do NOT use prospect-specific or proprietary systems (e.g. "NextGen ApplyOnline"). Keep to 1-2 sentences.

Para 2: Pick up the GENERAL-PURPOSE technology names from Para 1 (e.g. Boomi, Snowflake, MuleSoft) — NOT proprietary systems. Weave them into a case study narrative. Anonymize as "a leading [industry] company". Show how similar challenges were solved using Salesforce. Only reference these approved Salesforce products (and ONLY if clearly relevant): {approved_sf_products}. If no specific product fits, just say "Salesforce".

Return ONLY the 2 paragraphs of message text, no labels, no "Para 1:" prefixes."""

MESSAGE_PROMPT_PARA1_ONLY = """You are writing a concise, personalized sales outreach email on behalf of {brand_name}.

Brand tone and positioning: {tone_and_positioning}
Brand services: {services}

Prospect details:
- First name: {first_name}
- Designation: {designation}
- Company: {company_name}
- Location: {location}

Technology research findings:
{technology_research}

WRITING RULES — follow every rule strictly:
1. Address prospect by first name only (e.g. "Hi {first_name},")
2. Do NOT use hyphens or dashes anywhere in the message
3. Do NOT reference compliance standards by name (APRA, SOC2, PCI DSS, etc.) — say "compliance standards" instead
4. No subject line, no signature
5. No specific numbers, metrics, or statistics

Write exactly 1 paragraph: Show understanding of the prospect's technological situation, tailored to their role ({designation}). Naturally mention specific technology names found in the research. Do NOT use prospect-specific or proprietary systems (e.g. "NextGen ApplyOnline"). Keep to 1-2 sentences.

Return ONLY the paragraph text, no labels."""


def _get_date_window(dates_df: pd.DataFrame, city: str, state: str) -> str:
    """Look up meeting date window from dates sheet for a given city/state."""
    if dates_df.empty:
        return None

    city_lower = city.strip().lower()
    state_lower = state.strip().lower()

    for _, row in dates_df.iterrows():
        row_city = str(row.get("City", "")).strip().lower()
        row_state = str(row.get("State", "")).strip().lower()
        if row_city == city_lower and row_state == state_lower:
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
            name = cs.get("casestudy_name", "")
            solution = cs.get("summary_solution", "")
            exact_techs = cs.get("exact_techs", [])
            tech_str = ", ".join(exact_techs) if isinstance(exact_techs, list) else str(exact_techs)
            parts = [f"- {name}"]
            if solution:
                parts.append(f"  Solution: {solution}")
            if tech_str:
                parts.append(f"  Technologies: {tech_str}")
            lines.append("\n".join(parts))
    return "\n".join(lines) if lines else "No case studies available."


def _format_client_references(refs_json: str) -> str:
    """Format client references JSON into a list of names."""
    try:
        data = json.loads(refs_json) if refs_json else {}
    except (json.JSONDecodeError, TypeError):
        return "No client references available."

    if not data or (isinstance(data, dict) and "error" in data):
        return "No client references available."

    if isinstance(data, dict) and "client_names" in data:
        return data["client_names"] if data["client_names"] else "No client references available."

    return "No client references available."


def _has_case_studies(case_studies_json: str) -> bool:
    """Check if case studies data is non-empty and valid."""
    formatted = _format_case_studies(case_studies_json)
    return formatted != "No case studies available."


def _has_client_references(refs_json: str) -> bool:
    """Check if client references data is non-empty and valid."""
    formatted = _format_client_references(refs_json)
    return formatted != "No client references available."


def _generate_single_message(
    client: anthropic.Anthropic,
    row: pd.Series,
    dates_df: pd.DataFrame,
    brand_profile: dict,
) -> tuple[str, str]:
    """Generate message for a single prospect.

    Returns (message_text, flag_value).
    """
    first_name = str(row.get("First_Name", "")).strip()
    designation = str(row.get("Designation", "")).strip()
    company_name = str(row.get("Company_Name", "")).strip()
    city = str(row.get("City", "")).strip()
    state = str(row.get("State", "")).strip()
    location = f"{city}, {row.get('Country', '')}".strip(", ")
    technology_research = str(row.get("Prospect_Technologies", "")).strip()
    case_studies_json = str(row.get("Case_Studies", ""))
    refs_json = str(row.get("Industry_Client_References", ""))

    flag = ""
    has_cs = _has_case_studies(case_studies_json)
    has_refs = _has_client_references(refs_json)

    if not has_cs or not has_refs:
        flag = "TO BE DECIDED MANUALLY"

    # Para 1+2: GenAI via Claude (skip Para 2 if no case studies)
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
        approved_sf_products=", ".join(APPROVED_SF_PRODUCTS),
    )

    if not has_cs:
        # Only generate Para 1 — replace prompt to ask for 1 paragraph only
        prompt = MESSAGE_PROMPT_PARA1_ONLY.format(
            brand_name=brand_profile.get("name", ""),
            tone_and_positioning=brand_profile.get("tone_and_positioning", ""),
            services=", ".join(brand_profile.get("services", [])),
            first_name=first_name,
            designation=designation,
            company_name=company_name,
            location=location,
            technology_research=technology_research,
        )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    warm_message = response.content[0].text.strip()

    # Para 3: Fixed template (skip if no client references)
    if has_refs:
        client_refs = _format_client_references(refs_json)
        warm_message += f"\n\nSome of our existing clients in similar space such as yours include: {client_refs}."

    # Para 4: Fixed template
    date_window = _get_date_window(dates_df, city, state)
    if date_window:
        warm_message += f"\n\nIt will be really good to discuss this over a brief call between {date_window}. Please let me know what works best for you."
    else:
        warm_message += "\n\nIt would be really good to discuss this over a brief call. Please let me know when we can connect."

    return warm_message, flag


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

    # Handle old column name: rename Message_to_send → WARM_MESSAGE
    if "Message_to_send" in df.columns and "WARM_MESSAGE" not in df.columns:
        df.rename(columns={"Message_to_send": "WARM_MESSAGE"}, inplace=True)
    if "WARM_MESSAGE" not in df.columns:
        df["WARM_MESSAGE"] = ""
    df["WARM_MESSAGE"] = df["WARM_MESSAGE"].astype(object).fillna("")
    if "FLAG" not in df.columns:
        df["FLAG"] = ""
    df["FLAG"] = df["FLAG"].astype(object).fillna("")

    # Determine ready vs skipped
    def _is_ready(row):
        msg = str(row.get("WARM_MESSAGE", "")).strip()
        if msg and msg.lower() != "nan":
            return False
        tech_research = str(row.get("Prospect_Technologies", "")).strip()
        return bool(tech_research) and tech_research.lower() != "nan"

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

        message, flag = _generate_single_message(client, row, dates_df, brand_cache[brand_name])
        df.at[idx, "WARM_MESSAGE"] = message
        if flag:
            df.at[idx, "FLAG"] = flag

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="prospects", index=False)
        if not dates_df.empty:
            dates_df.to_excel(writer, sheet_name="dates", index=False)

    if progress_callback:
        progress_callback(total, total, "Message generation complete.")

    return {"ready": total, "skipped": skipped}
