"""Prospect research pipeline: Claude web search per company/prospect,
case study matching, brand selection, and intent scoring."""

import json
import time
import pandas as pd
import anthropic
from typing import Dict, List, Tuple, Callable, Optional

from . import config, brand_knowledge, case_study_match


def validate_prospects_sheet(df: pd.DataFrame) -> bool:
    required = [
        "Prospect Name",
        "Designation",
        "Company Name",
        "Website",
        "Location of Prospect",
    ]
    return all(col in df.columns for col in required)


# ---------------------------------------------------------------------------
# Step 2a: Company-level research (deduplicated)
# ---------------------------------------------------------------------------

def _research_company(company_name: str, website: str) -> str:
    """Use Claude with web search to research a company."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    prompt = f"""Research the company "{company_name}" (website: {website}).

Find and summarize:
1. Company overview — what they do, size, industry vertical
2. Recent digital transformation or technology initiatives
3. Job openings related to: Salesforce, Snowflake, custom development, data engineering, cloud migration
4. Recent news, press releases, funding rounds, or partnerships
5. Technology stack signals (CRM, cloud platforms, data tools)

Be specific and cite real findings. If you can't find information on a topic, say so.
Return a structured summary with clear bullet points."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
    return "\n".join(text_parts).strip()


# ---------------------------------------------------------------------------
# Step 2b: Prospect-level personalization
# ---------------------------------------------------------------------------

def _research_prospect(prospect_name: str, designation: str, company_name: str, company_research: str) -> str:
    """Use Claude with web search to research an individual prospect."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    prompt = f"""Research "{prospect_name}", who is {designation} at {company_name}.

Context — here's what we already know about the company:
{company_research[:2000]}

Find and summarize:
1. Their professional background and experience
2. Their role scope and responsibilities
3. Recent LinkedIn activity, speaking engagements, or publications
4. Persona-specific signals relevant to their designation
   - If they're a CTO/VP Engineering → focus on technical initiatives
   - If they're a VP Sales/CMO → focus on growth/revenue signals
   - If they're a CEO/COO → focus on strategic direction

Return 3-4 concise bullet points tailored to this person's role."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
    return "\n".join(text_parts).strip()


# ---------------------------------------------------------------------------
# Step 3: Matching, brand selection, intent scoring
# ---------------------------------------------------------------------------

def _select_brand_and_score(
    prospect_name: str,
    designation: str,
    company_name: str,
    company_research: str,
    prospect_research: str,
    case_studies: List[Dict],
    brand_profiles: Dict[str, dict],
) -> Tuple[str, int]:
    """Use Claude to select the best brand and assign an intent score."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    case_study_text = ""
    if case_studies:
        cs_lines = []
        for cs in case_studies[:5]:
            cs_lines.append(f"- {cs['company_name']}: {cs['use_case']} ({cs['industry']})")
            if cs.get("summary"):
                cs_lines.append(f"  Summary: {cs['summary'][:200]}")
        case_study_text = "\n".join(cs_lines)

    brand_summaries = ""
    for name, profile in brand_profiles.items():
        brand_summaries += f"\n--- {name} ---\n"
        brand_summaries += f"Services: {json.dumps(profile.get('services', []))}\n"
        brand_summaries += f"Verticals: {json.dumps(profile.get('verticals', []))}\n"
        brand_summaries += f"Tech Stack: {json.dumps(profile.get('tech_stack', []))}\n"
        brand_summaries += f"Tone: {profile.get('tone_and_positioning', '')}\n"
        brand_summaries += f"Target Audience: {profile.get('target_audience', '')}\n"

    prompt = f"""You are helping select which brand to use for outreach and scoring intent.

PROSPECT: {prospect_name}, {designation} at {company_name}

COMPANY RESEARCH:
{company_research[:1500]}

PROSPECT RESEARCH:
{prospect_research[:500]}

MATCHING CASE STUDIES:
{case_study_text if case_study_text else "No matching case studies found."}

BRAND PROFILES:
{brand_summaries}

Based on the above, respond with ONLY a JSON object:
{{
  "brand": "LendingLogik" or "CloudChillies",
  "brand_reasoning": "1-sentence explanation of why this brand fits better",
  "intent_score": <number 1-10>,
  "intent_reasoning": "1-sentence explanation of the score"
}}

Brand selection criteria:
- Match the prospect's industry vertical to the brand's focus areas
- Consider which brand's case studies are more relevant
- Consider which brand's services align with the prospect's needs

Intent scoring criteria:
- 8-10: Strong signals (active job postings for relevant tech, recent RFPs, explicit digital transformation announcements)
- 5-7: Moderate signals (general growth indicators, industry trends, some tech adoption)
- 1-4: Weak signals (no specific indicators found, generic company)

Return ONLY the JSON object."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Parse JSON
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(text)
        brand = result.get("brand", "LendingLogik")
        score = int(result.get("intent_score", 5))
        # Validate brand name
        if brand not in ("LendingLogik", "CloudChillies"):
            brand = "LendingLogik"
        return brand, max(1, min(10, score))
    except (json.JSONDecodeError, ValueError):
        return "LendingLogik", 5


# ---------------------------------------------------------------------------
# Main pipeline: research_workbook
# ---------------------------------------------------------------------------

BATCH_SIZE = 5
BATCH_DELAY = 3  # seconds between batches


def run_research(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Run the full research pipeline on a prospects DataFrame.

    Args:
        df: DataFrame with prospect data.
        progress_callback: Optional callback(current, total, status_text) for progress updates.
    """
    df = df.copy()

    # Ensure output columns exist
    for col in ["Research Summary", "Suggested Brand Name to use", "Intent Score"]:
        if col not in df.columns:
            df[col] = ""

    # Ensure brand profiles exist
    brand_knowledge.ensure_brand_profiles()
    brand_profiles = {}
    for name in config.BRAND_JSONS:
        try:
            brand_profiles[name] = brand_knowledge.load_brand_profile(name)
        except FileNotFoundError:
            brand_profiles[name] = {"name": name, "services": [], "verticals": []}

    # Group prospects by company for deduplication
    company_groups: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        company = str(row.get("Company Name", "")).strip()
        if company not in company_groups:
            company_groups[company] = []
        company_groups[company].append(idx)

    total = len(df)
    processed = 0

    # Research company-by-company
    company_research_cache: Dict[str, str] = {}
    company_case_studies_cache: Dict[str, List[Dict]] = {}

    for company, indices in company_groups.items():
        # Company-level research (done once per company)
        website = str(df.at[indices[0], "Website"]).strip()
        if progress_callback:
            progress_callback(processed, total, f"Researching company: {company}")

        company_research = _research_company(company, website)
        company_research_cache[company] = company_research

        # Case study matching for this company
        search_query = company_research[:2000]
        case_studies = case_study_match.match_case_studies(search_query)
        company_case_studies_cache[company] = case_studies

        # Per-prospect research within this company
        for idx in indices:
            row = df.loc[idx]
            prospect_name = str(row.get("Prospect Name", "")).strip()
            designation = str(row.get("Designation", "")).strip()

            if progress_callback:
                progress_callback(processed, total, f"Researching: {prospect_name} ({designation})")

            # Prospect-level research
            prospect_research = _research_prospect(
                prospect_name, designation, company, company_research
            )

            # Build research summary (company + prospect findings)
            summary_parts = []
            summary_parts.append(f"Company: {company_research[:500]}")
            summary_parts.append(f"Prospect: {prospect_research[:500]}")
            if case_studies:
                cs_brief = "; ".join(
                    f"{cs['use_case']} ({cs['industry']})" for cs in case_studies[:3]
                )
                summary_parts.append(f"Relevant case studies: {cs_brief}")
            research_summary = "\n".join(summary_parts)

            # Brand selection + intent scoring
            brand, intent_score = _select_brand_and_score(
                prospect_name, designation, company,
                company_research, prospect_research,
                case_studies, brand_profiles,
            )

            df.at[idx, "Research Summary"] = research_summary
            df.at[idx, "Suggested Brand Name to use"] = brand
            df.at[idx, "Intent Score"] = intent_score

            processed += 1
            if progress_callback:
                progress_callback(processed, total, f"Completed: {prospect_name}")

            # Rate limiting within batches
            if processed % BATCH_SIZE == 0:
                time.sleep(BATCH_DELAY)

    return df


def research_workbook(
    input_path: str,
    output_path: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """Load an Excel file, run research, and save results."""
    xls = pd.ExcelFile(input_path)
    if "prospects" not in xls.sheet_names:
        raise ValueError("Workbook must contain a 'prospects' sheet")

    df = pd.read_excel(xls, sheet_name="prospects")

    if not validate_prospects_sheet(df):
        raise ValueError("prospects sheet is missing required columns")

    result = run_research(df, progress_callback=progress_callback)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="prospects", index=False)
        if "dates" in xls.sheet_names:
            dates_df = pd.read_excel(xls, sheet_name="dates")
            dates_df.to_excel(writer, sheet_name="dates", index=False)
