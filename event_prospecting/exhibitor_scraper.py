"""Step 1: Scrape exhibition websites for exhibitor companies and enrich with company details.

For each exhibition row, uses Claude web search to extract exhibitor names from the
exhibitor link, then enriches each company via Claude web search, applying
shortlist criteria to filter results.
"""

import json
import time
from typing import Dict, List, Optional, Callable

import pandas as pd
import anthropic

from event_prospecting import config

MAX_RETRIES = 3
RETRY_DELAY = 65

# Company columns produced by Step 1
COMPANY_COLUMNS = [
    "Company Name",
    "Website",
    "Location of Company",
    "Country",
    "Industry Vertical",
    "Sub-industry",
    "Revenue Range",
]

SHORTLIST_CRITERIA = """
Shortlist criteria — only include companies that meet ALL of these:
- NOT an IT service provider, IT consulting firm, or systems integrator
- Revenue under $2 billion (exclude if clearly above)
- Preferably in financial services, banking, insurance, lending, or payments (but other industries are acceptable if they meet the tech criteria)
- Uses or is likely to use technologies in our expertise area: Salesforce, Dell Boomi, Snowflake, .NET, Tableau, MS Fabric
- Do NOT cast a wide net on generic data engineering tools — stick to the specific tools listed above

If you cannot determine whether a company meets a criterion, include it with a note.
"""


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


def _extract_json(text: str) -> str:
    """Extract JSON from Claude response text, handling code fences."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return text


def _extract_exhibitor_names(client, exhibitor_link: str) -> List[str]:
    """Use Claude with web search to extract company names from an exhibitor page.

    Returns a list of company names found on the page.
    """
    prompt = f"""Visit this exhibition exhibitor page and extract ALL company names listed:
{exhibitor_link}

Search for this page and extract the complete list of exhibiting companies.
Return ONLY a JSON array of company name strings, no other text.

Example: ["Company A", "Company B", "Company C"]

If you cannot access the page or find no exhibitors, return an empty array: []"""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text from response (after last web search result)
    last_search_idx = -1
    for i, block in enumerate(response.content):
        if block.type == "web_search_tool_result":
            last_search_idx = i

    text_parts = []
    for i, block in enumerate(response.content):
        if block.type == "text" and i > last_search_idx:
            text_parts.append(block.text)

    text = "".join(text_parts).strip()
    text = _extract_json(text)

    try:
        names = json.loads(text)
        if isinstance(names, list):
            return [str(n).strip() for n in names if str(n).strip()]
    except json.JSONDecodeError:
        pass

    return []


def _enrich_companies(
    client,
    company_names: List[str],
    exhibition_name: str,
    progress_callback: Optional[Callable] = None,
    progress_offset: int = 0,
    progress_total: int = 0,
) -> tuple:
    """Enrich a list of company names with details, then apply shortlist criteria.

    Uses Claude web search to research each company, then synthesizes results
    into structured data. Returns both the full list and the shortlisted subset.

    Returns:
        Tuple of (all_companies, shortlisted_companies) — both are List[Dict].
    """
    # Use Claude with web search to enrich all companies in one call
    company_list_text = "\n".join(f"- {name}" for name in company_names)

    if progress_callback:
        progress_callback(
            progress_offset, progress_total,
            f"Researching {len(company_names)} companies for {exhibition_name}..."
        )

    enrich_prompt = f"""Search the web and research each of the following companies from the exhibition "{exhibition_name}".

COMPANIES TO RESEARCH:
{company_list_text}

For each company, find and extract:
- "company_name": Company name
- "website": Company website URL (homepage only)
- "location": City where the company is based
- "country": Country
- "industry_vertical": Main industry (e.g. Financial Services, Healthcare, Automotive, Retail)
- "sub_industry": Sub-vertical if applicable (e.g. Banking, Insurance, Lending)
- "revenue_range": Estimated revenue range (e.g. "$50M-$100M", "$500M-$1B", "Unknown")

Include ALL companies — do NOT filter any out. If information is unknown, use "Unknown".

Return ONLY a JSON array of objects, no other text."""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": enrich_prompt}],
    )

    # Extract text from response (after last web search result)
    last_search_idx = -1
    for i, block in enumerate(response.content):
        if block.type == "web_search_tool_result":
            last_search_idx = i

    text_parts = []
    for i, block in enumerate(response.content):
        if block.type == "text" and i > last_search_idx:
            text_parts.append(block.text)

    text = "".join(text_parts).strip()
    text = _extract_json(text)

    try:
        all_companies = json.loads(text)
        if not isinstance(all_companies, list):
            all_companies = []
    except json.JSONDecodeError:
        all_companies = []

    if not all_companies:
        return [], []

    # Step 2: Apply shortlist criteria
    filter_prompt = f"""Given this list of companies, apply the shortlist criteria and return ONLY the companies that pass.

COMPANIES:
{json.dumps(all_companies, indent=2)}

{SHORTLIST_CRITERIA}

Return ONLY a JSON array of the companies that pass ALL criteria. Keep the exact same fields.
If no companies pass, return an empty array: []

Return ONLY the JSON array, no other text."""

    response2 = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": filter_prompt}],
    )

    text2 = response2.content[0].text.strip()
    text2 = _extract_json(text2)

    try:
        shortlisted = json.loads(text2)
        if not isinstance(shortlisted, list):
            shortlisted = []
    except json.JSONDecodeError:
        shortlisted = []

    return all_companies, shortlisted


def scrape_exhibitors(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple:
    """Scrape exhibitor links and return enriched company lists per exhibition.

    Args:
        df: Input DataFrame with 'Exhibition' and 'Exhibitor Link' columns.
        progress_callback: Optional callback(current, total, status_text).

    Returns:
        Tuple of (sheets_dict, logs_list).
        sheets_dict: Dict mapping sheet names ('<ExhibitionName>-Exhibitors') to DataFrames.
        logs_list: List of diagnostic log strings for UI display.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    total_exhibitions = len(df)
    sheets = {}
    logs = []

    for idx, row in df.iterrows():
        exhibition_name = str(row["Exhibition"]).strip()
        exhibitor_link = str(row["Exhibitor Link"]).strip()
        logs.append(f"--- {exhibition_name} ---")
        logs.append(f"Link: {exhibitor_link}")

        if progress_callback:
            progress_callback(idx, total_exhibitions, f"Extracting exhibitors: {exhibition_name}")

        # Step 1a: Extract company names from exhibitor page
        company_names = _extract_exhibitor_names(client, exhibitor_link)
        logs.append(f"Exhibitors extracted: {len(company_names)}")
        if company_names:
            logs.append(f"Names: {', '.join(company_names[:20])}")
            if len(company_names) > 20:
                logs.append(f"  ... and {len(company_names) - 20} more")

        if not company_names:
            logs.append("⚠ No exhibitors found — skipping")
            if progress_callback:
                progress_callback(idx, total_exhibitions, f"No exhibitors found for {exhibition_name}")
            continue

        if progress_callback:
            progress_callback(
                idx, total_exhibitions,
                f"Found {len(company_names)} companies for {exhibition_name}. Enriching..."
            )

        # Step 1b: Enrich companies with details + apply shortlist
        all_enriched, shortlisted = _enrich_companies(
            client,
            company_names,
            exhibition_name,
            progress_callback=progress_callback,
            progress_offset=idx,
            progress_total=total_exhibitions,
        )
        logs.append(f"Companies enriched: {len(all_enriched)}")
        logs.append(f"Companies after shortlist: {len(shortlisted)}")

        if not all_enriched:
            logs.append("⚠ Enrichment returned no results — skipping")
            if progress_callback:
                progress_callback(idx, total_exhibitions, f"No enrichment results for {exhibition_name}")
            continue

        def _build_rows(companies):
            return [{
                "Company Name": c.get("company_name", ""),
                "Website": c.get("website", ""),
                "Location of Company": c.get("location", ""),
                "Country": c.get("country", ""),
                "Industry Vertical": c.get("industry_vertical", ""),
                "Sub-industry": c.get("sub_industry", ""),
                "Revenue Range": c.get("revenue_range", "Unknown"),
            } for c in companies]

        # Sheet with ALL enriched companies (before filtering)
        all_sheet_name = f"{exhibition_name}-All"[:31]
        sheets[all_sheet_name] = pd.DataFrame(_build_rows(all_enriched), columns=COMPANY_COLUMNS)
        logs.append(f"✓ Sheet '{all_sheet_name}': {len(all_enriched)} companies")

        # Sheet with shortlisted companies only
        if shortlisted:
            short_sheet_name = f"{exhibition_name}-Shortlist"[:31]
            sheets[short_sheet_name] = pd.DataFrame(_build_rows(shortlisted), columns=COMPANY_COLUMNS)
            logs.append(f"✓ Sheet '{short_sheet_name}': {len(shortlisted)} companies")
        else:
            logs.append("⚠ No companies passed shortlist — only All sheet created")

        if progress_callback:
            progress_callback(
                idx + 1, total_exhibitions,
                f"Completed {exhibition_name}: {len(all_enriched)} total, {len(shortlisted)} shortlisted"
            )

    return sheets, logs
