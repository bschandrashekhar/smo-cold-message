"""Message generation: Claude synthesizes personalized 3-paragraph outreach messages."""

import json
import time
import pandas as pd
from datetime import datetime
from typing import Optional, Callable, List, Dict

import anthropic
from . import config, brand_knowledge, case_study_match

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


def _generate_single_message(
    prospect: pd.Series,
    dates_df: pd.DataFrame,
    brand_profile: dict,
    case_studies: List[Dict],
) -> str:
    """Use Claude to generate a personalized 3-paragraph outreach message."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    name = str(prospect.get("Prospect Name", "")).strip()
    company = str(prospect.get("Company Name", "")).strip()
    designation = str(prospect.get("Designation", "")).strip()
    brand_name = str(prospect.get("Suggested Brand Name to use", "")).strip()
    research = str(prospect.get("Research Summary", "")).strip()
    location = str(prospect.get("Location of Prospect", "")).strip()

    date_range = _get_date_range(location, dates_df)

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

    brand_tone = brand_profile.get("tone_and_positioning", "professional")
    brand_services = json.dumps(brand_profile.get("services", []))

    prompt = f"""Generate a personalized cold outreach message from {brand_name} to {name}.

PROSPECT DETAILS:
- Name: {name}
- Designation: {designation}
- Company: {company}
- Location: {location}

RESEARCH FINDINGS:
{research[:1500]}

BRAND SENDING THE MESSAGE: {brand_name}
- Tone/Positioning: {brand_tone}
- Services: {brand_services}

RELEVANT WORK WE'VE DONE (reference these brand-neutrally — do NOT mention "{brand_name}" when citing case studies):
{cs_text if cs_text else "No specific case studies available — reference general experience."}

MEETING WINDOW: {date_range}

Write exactly 3 paragraphs:

**Paragraph 1**: Show understanding of the prospect's situation. Reference specific findings from the research. Tailor to their role/designation — e.g., for a CTO focus on technical depth, for a VP Sales focus on ROI. Start with "Hi {name}," on its own line.

**Paragraph 2**: Reference similar work done for other clients (from the case studies above, anonymized as "a leading [industry] company"). Connect it to the prospect's specific needs and relevant services.

**Paragraph 3**: Request a meeting during the specific date range ({date_range}). Keep it brief and confident.

IMPORTANT:
- Write in {brand_name}'s tone: {brand_tone}
- Do NOT use generic filler — every sentence should reference something specific from the research
- Do NOT mention the brand name when citing case studies (they should be brand-neutral)
- Keep the total message under 250 words
- Do NOT include a subject line — just the message body
- Do NOT add a sign-off/signature — just the 3 paragraphs"""

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

    # Cache case studies per company
    company_case_studies: Dict[str, List[Dict]] = {}

    for i, idx in enumerate(rows_to_process):
        row = df.loc[idx]
        prospect_name = str(row.get("Prospect Name", "")).strip()
        company = str(row.get("Company Name", "")).strip()
        brand_name = str(row.get("Suggested Brand Name to use", "LendingLogik")).strip()

        if progress_callback:
            progress_callback(i, total, f"Generating message for: {prospect_name}")

        # Get brand profile
        brand_profile = brand_profiles.get(brand_name, brand_profiles.get("LendingLogik", {}))

        # Get case studies (cached per company)
        if company not in company_case_studies:
            research = str(row.get("Research Summary", ""))
            company_case_studies[company] = case_study_match.match_case_studies(research[:2000])
        case_studies = company_case_studies[company]

        # Generate message
        message = _generate_single_message(row, dates_df, brand_profile, case_studies)
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
