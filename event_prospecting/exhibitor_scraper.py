"""Step 1: Scrape exhibition websites for exhibitor companies and enrich with company details.

For each exhibition row, uses Claude web search to extract exhibitor names from the
exhibitor link, then enriches each company via Serper + Claude synthesis, applying
shortlist criteria to filter results.
"""

import json
import time
from typing import Dict, List, Optional, Callable

import requests
import pandas as pd
import anthropic

from event_prospecting import config

MAX_RETRIES = 3
RETRY_DELAY = 65
BATCH_SIZE = 5
BATCH_DELAY = 3

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


def _serper_search(query: str, num_results: int = 10) -> List[Dict]:
    """Call Serper.dev Google Search API and return organic results."""
    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": config.SERPER_API_KEY,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": num_results},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("organic", [])


def _serper_search_results_to_text(results: List[Dict]) -> str:
    """Format Serper search results into text for Claude synthesis."""
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        lines.append(f"{i}. {title}\n   {snippet}\n   URL: {link}")
    return "\n\n".join(lines)


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
) -> List[Dict]:
    """Enrich a list of company names with details and apply shortlist criteria.

    Uses Serper to search for each company, then Claude to synthesize results
    into structured data and apply filtering.
    """
    # Batch companies for enrichment — search for all, then synthesize in one Claude call
    all_search_results = []
    for i, name in enumerate(company_names):
        if progress_callback:
            progress_callback(
                progress_offset + i, progress_total,
                f"Searching: {name}"
            )

        query = f'"{name}" company website industry revenue'
        results = _serper_search(query, num_results=5)
        result_text = _serper_search_results_to_text(results) if results else "No results found."
        all_search_results.append(f"Company: {name}\n{result_text}")

        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(BATCH_DELAY)

    search_context = "\n\n---\n\n".join(all_search_results)

    prompt = f"""Based on the search results below, produce a JSON array of enriched company profiles for the exhibition "{exhibition_name}".

SEARCH RESULTS:
{search_context}

For each company, extract:
- "company_name": Company name
- "website": Company website URL (homepage only)
- "location": City where the company is based
- "country": Country
- "industry_vertical": Main industry (e.g. Financial Services, Healthcare, Automotive, Retail)
- "sub_industry": Sub-vertical if applicable (e.g. Banking, Insurance, Lending)
- "revenue_range": Estimated revenue range (e.g. "$50M-$100M", "$500M-$1B", "Unknown")

{SHORTLIST_CRITERIA}

Return ONLY a JSON array of objects for companies that pass the shortlist criteria.
Companies that are clearly IT service providers or have revenue over $2B should be excluded.
If unsure about a company, include it with revenue_range "Unknown".

Return ONLY the JSON array, no other text."""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    text = _extract_json(text)

    try:
        companies = json.loads(text)
        if isinstance(companies, list):
            return companies
    except json.JSONDecodeError:
        pass

    return []


def scrape_exhibitors(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Scrape exhibitor links and return enriched company lists per exhibition.

    Args:
        df: Input DataFrame with 'Exhibition' and 'Exhibitor Link' columns.
        progress_callback: Optional callback(current, total, status_text).

    Returns:
        Dict mapping sheet names ('<ExhibitionName>-Exhibitors') to DataFrames.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    total_exhibitions = len(df)
    sheets = {}

    for idx, row in df.iterrows():
        exhibition_name = str(row["Exhibition"]).strip()
        exhibitor_link = str(row["Exhibitor Link"]).strip()

        if progress_callback:
            progress_callback(idx, total_exhibitions, f"Extracting exhibitors: {exhibition_name}")

        # Step 1a: Extract company names from exhibitor page
        company_names = _extract_exhibitor_names(client, exhibitor_link)

        if not company_names:
            if progress_callback:
                progress_callback(idx, total_exhibitions, f"No exhibitors found for {exhibition_name}")
            continue

        if progress_callback:
            progress_callback(
                idx, total_exhibitions,
                f"Found {len(company_names)} companies for {exhibition_name}. Enriching..."
            )

        # Step 1b: Enrich companies with details + apply shortlist
        enriched = _enrich_companies(
            client,
            company_names,
            exhibition_name,
            progress_callback=progress_callback,
            progress_offset=idx,
            progress_total=total_exhibitions,
        )

        if not enriched:
            if progress_callback:
                progress_callback(idx, total_exhibitions, f"No companies passed shortlist for {exhibition_name}")
            continue

        # Build DataFrame from enriched data
        rows = []
        for company in enriched:
            rows.append({
                "Company Name": company.get("company_name", ""),
                "Website": company.get("website", ""),
                "Location of Company": company.get("location", ""),
                "Country": company.get("country", ""),
                "Industry Vertical": company.get("industry_vertical", ""),
                "Sub-industry": company.get("sub_industry", ""),
                "Revenue Range": company.get("revenue_range", "Unknown"),
            })

        sheet_name = f"{exhibition_name}-Exhibitors"[:31]  # Excel 31-char limit
        sheets[sheet_name] = pd.DataFrame(rows, columns=COMPANY_COLUMNS)

        if progress_callback:
            progress_callback(
                idx + 1, total_exhibitions,
                f"Completed {exhibition_name}: {len(rows)} companies"
            )

    return sheets
