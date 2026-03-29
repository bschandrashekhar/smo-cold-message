"""Message generation: Claude synthesizes personalized 3-paragraph outreach messages."""

import json
import time
import pandas as pd
from datetime import datetime
from typing import Optional, Callable, List, Dict

import anthropic
from . import config, brand_knowledge

MAX_RETRIES = 3
RETRY_DELAY = 65
BATCH_SIZE = 5
BATCH_DELAY = 3


def _call_claude_with_retry(client, **kwargs):
    """Call Claude API with automatic retry on rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s before retry {attempt + 2}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise


def format_date(date_value):
    if pd.isna(date_value):
        return None
    if isinstance(date_value, datetime):
        return date_value.strftime("%B %d, %Y")
    return str(date_value)


def _get_date_range(prospect_location: str, dates_df: pd.DataFrame) -> str:
    """Look up the meeting date range for a prospect's location."""
    if dates_df.empty or "Location of Prospect" not in dates_df.columns:
        return "in the coming weeks"

    matching = dates_df[dates_df["Location of Prospect"] == prospect_location]
    if not matching.empty:
        start = format_date(matching.iloc[0].get("Start Date"))
        end = format_date(matching.iloc[0].get("End Date"))
        if start and end:
            return f"between {start} and {end}"
    return "in the coming weeks"


def _compress_research(client, research: str, designation: str) -> str:
    """Compress research summary into key signals, preserving all important details."""
    if len(research) <= 1500:
        return research

    prompt = f"""Compress the following prospect research into a dense briefing of key signals. Keep ALL specific facts, numbers, technology names, project names, and strategic initiatives. Remove filler, redundancy, and formatting overhead.

Prioritize signals relevant to a {designation}'s concerns.

RESEARCH:
{research}

Return ONLY the compressed briefing — no preamble."""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def _generate_single_message(
    prospect: pd.Series,
    dates_df: pd.DataFrame,
    brand_profile: dict,
    case_studies: List[Dict],
    industry_references: List[str],
) -> str:
    """Use Claude to generate a personalized 3-paragraph outreach message."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    full_name = str(prospect.get("Prospect Name", "")).strip()
    first_name = full_name.split()[0] if full_name else full_name
    company = str(prospect.get("Company Name", "")).strip()
    designation = str(prospect.get("Designation", "")).strip()
    brand_name = str(prospect.get("Suggested Brand Name to use", "")).strip()
    research = str(prospect.get("Research Summary", "")).strip()
    location = str(prospect.get("Location of Prospect", "")).strip()

    date_range = _get_date_range(location, dates_df)

    # Compress research summary to preserve all signals without truncation
    research = _compress_research(client, research, designation)

    # Build case study reference text (brand-neutral)
    cs_text = ""
    if case_studies:
        cs_lines = []
        for cs in case_studies[:3]:
            cs_lines.append(
                f"- Helped a leading {cs.get('industry', 'industry')} company with {cs.get('use_case', 'their project')}. "
                f"{cs.get('summary', '')[:200]}"
            )
        cs_text = "\n".join(cs_lines)

    # Build industry references text
    refs_text = ""
    if industry_references:
        refs_text = ", ".join(industry_references)

    brand_tone = brand_profile.get("tone_and_positioning", "professional")
    brand_services = json.dumps(brand_profile.get("services", []))

    prompt = f"""Generate a personalized cold outreach message from {brand_name} to {first_name}.

PROSPECT DETAILS:
- Name: {first_name}
- Designation: {designation}
- Company: {company}
- Location: {location}

RESEARCH FINDINGS:
{research}

BRAND SENDING THE MESSAGE: {brand_name}
- Tone/Positioning: {brand_tone}
- Services: {brand_services}

RELEVANT WORK WE'VE DONE (reference these brand-neutrally — do NOT mention "{brand_name}" when citing case studies):
{cs_text if cs_text else "No specific case studies available — reference general experience."}

INDUSTRY REFERENCE CLIENTS (clients we've worked with in the same vertical as {company}):
{refs_text if refs_text else "No specific industry references available."}

MEETING WINDOW: {date_range}

Write exactly 3 short paragraphs. Keep the total message under 150 words.

**Paragraph 1**: Start with "Hi {first_name}," on its own line. One or two concise sentences showing you understand their situation. Reference a specific initiative or strategic direction from the research, and naturally mention the specific technology names found in the research (e.g. Boomi, Snowflake, NextGen ApplyOnline, MuleSoft, AWS — whatever technologies the research mentions). Do NOT cite specific numbers, metrics, or statistics (e.g. avoid "reduced from X to Y", "13% improvement"). It should read like a knowledgeable human wrote it, not a data report.

**Paragraph 2**: Pick up the specific technology names you mentioned in Paragraph 1 and weave them into the case study narrative to create a cohesive story — but only carry over general-purpose/common technologies (e.g. Boomi, Snowflake, MuleSoft), NOT prospect-specific or proprietary systems that wouldn't credibly appear in "work we've done for others." For example, if Para 1 mentions Boomi and Snowflake, Para 2 should reference how we've helped clients with Boomi integration or Snowflake data pipelines leveraging Salesforce. Briefly reference similar work done for clients (anonymized as "a leading [industry] company"), showing how we solved challenges around those same technologies leveraging Salesforce. ONLY reference these Salesforce clouds/products: Service Cloud, Sales Cloud, Non Profit, NPSP, Experience Cloud, Commerce Cloud, Agentforce, LWC, Appexchange Product Development, Marketing Cloud, Pardot, Tableau, CPQ, Data Cloud. Do NOT mention any Salesforce products outside this list (e.g. do NOT use "Financial Services Cloud", "Health Cloud", "Education Cloud" — these are NOT in our list). If no specific cloud is a clear match for the prospect's needs, just say "Salesforce" without naming a specific cloud. You may combine multiple clouds if relevant. Then, in a SEPARATE sentence, mention the industry reference clients. Do NOT join the client names with the description of work — keep them apart. For example: "We've helped leading [industry] companies [do X] leveraging Salesforce. Some of our clients in this space include {refs_text if refs_text else "similar companies"}." You MUST mention ALL client names. Do not drop any.

**Paragraph 3**: Close with something like: "It will be really good to discuss this over a brief call between [start date] and [end date]. Please let me know what works best for you." Use the exact dates from the meeting window above. Keep it warm and simple, not salesy.

IMPORTANT STYLE RULES:
- Write in {brand_name}'s tone: {brand_tone}
- Every sentence must reference something specific from the research
- Do NOT mention the brand name when citing case studies (keep them brand neutral)
- Industry reference client names CAN be mentioned by name (they are real clients)
- Do NOT use hyphens (no dashes like - or em dashes) anywhere in the message
- Do NOT reference specific compliance standards or regulations by name (e.g. no APRA, SOC2, PCI DSS). Use generic terms like "compliance standards" or "regulatory requirements" instead
- Keep paragraphs short and punchy, not verbose or flowery
- Do NOT include a subject line
- Do NOT add a sign-off or signature"""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def generate_messages(
    input_path: str,
    output_path: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """Read reviewed workbook, generate messages via Claude, and save to new file."""
    xls = pd.ExcelFile(input_path)
    if "prospects" not in xls.sheet_names:
        raise ValueError("Workbook must contain a 'prospects' sheet")

    df = pd.read_excel(xls, sheet_name="prospects")
    dates_df = pd.DataFrame()
    if "dates" in xls.sheet_names:
        dates_df = pd.read_excel(xls, sheet_name="dates")

    # Ensure message column exists with object dtype
    if "Message to send" not in df.columns:
        df["Message to send"] = ""
    df["Message to send"] = df["Message to send"].astype(object)

    # Determine which rows need messages
    rows_to_process = []
    for idx, row in df.iterrows():
        msg = row.get("Message to send")
        if pd.notna(msg) and str(msg).strip() != "":
            continue  # Already has a message
        research = row.get("Research Summary")
        if pd.isna(research) or str(research).strip() == "":
            continue  # No research — skip
        intent = row.get("Intent Score")
        if pd.isna(intent) or str(intent).strip() == "":
            continue  # Not scored — skip
        rows_to_process.append(idx)

    total = len(rows_to_process)
    if total == 0:
        # Nothing to process, just save as-is
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="prospects", index=False)
            if not dates_df.empty:
                dates_df.to_excel(writer, sheet_name="dates", index=False)
        return

    # Load brand profiles
    brand_profiles = {}
    for name in config.BRAND_JSONS:
        try:
            brand_profiles[name] = brand_knowledge.load_brand_profile(name)
        except FileNotFoundError:
            brand_profiles[name] = {"name": name, "services": [], "verticals": []}

    for i, idx in enumerate(rows_to_process):
        row = df.loc[idx]
        prospect_name = str(row.get("Prospect Name", "")).strip()
        brand_name = str(row.get("Suggested Brand Name to use", "LendingLogik")).strip()

        if progress_callback:
            progress_callback(i, total, f"Generating message for: {prospect_name}")

        # Get brand profile
        brand_profile = brand_profiles.get(brand_name, brand_profiles.get("LendingLogik", {}))

        # Read case studies from spreadsheet (persisted by Pass 1)
        cs_raw = row.get("Case Studies", "")
        case_studies = []
        if pd.notna(cs_raw) and str(cs_raw).strip():
            try:
                case_studies = json.loads(str(cs_raw))
            except json.JSONDecodeError:
                pass

        # Read industry references from spreadsheet (persisted by Pass 1)
        refs_raw = row.get("Industry References", "")
        industry_references = []
        if pd.notna(refs_raw) and str(refs_raw).strip():
            try:
                industry_references = json.loads(str(refs_raw))
            except json.JSONDecodeError:
                pass

        # Generate message
        message = _generate_single_message(row, dates_df, brand_profile, case_studies, industry_references)
        df.at[idx, "Message to send"] = message

        if progress_callback:
            progress_callback(i + 1, total, f"Completed: {prospect_name}")

        # Rate limiting
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)

    # Save output
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="prospects", index=False)
        if not dates_df.empty:
            dates_df.to_excel(writer, sheet_name="dates", index=False)
