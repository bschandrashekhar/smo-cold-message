"""Step 1: Enrich exhibitor companies with Apollo.io and apply shortlist filtering.

Input: Excel with sheets named <ExhibitionName>-Exhibitors, each containing
at minimum "Company Name" and optionally "Website" columns.

For each exhibition sheet:
1. Reads exhibitor list from the sheet
2. Uses Apollo.io Organization Enrichment to get company details
3. Applies shortlist criteria via Claude to produce a filtered subset

Data source tracking: each company row includes a "Data Source" field
indicating where the enrichment data came from (e.g. "Apollo", "Manual only").

NOTE: HTML scraping logic is commented out below for future use (Playwright fallback).
"""

import json
import re
import time
from typing import Dict, List, Optional, Callable

import requests
import pandas as pd
import anthropic

from event_prospecting import config

MAX_RETRIES = 3
RETRY_DELAY = 65
APOLLO_BATCH_DELAY = 1  # seconds between Apollo API calls

# Company columns produced by Step 1
COMPANY_COLUMNS = [
    "Company Name",
    "Website",
    "Location of Company",
    "Country",
    "Industry Vertical",
    "Sub-industry",
    "Revenue Range",
    "Data Source",
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


# ── Apollo Enrichment ─────────────────────────────────────────────────────────


def _apollo_org_enrichment(domain: str) -> Optional[Dict]:
    """Enrich a company via Apollo.io Organization Enrichment API."""
    if not config.APOLLO_API_KEY or not domain:
        return None

    try:
        response = requests.get(
            "https://api.apollo.io/api/v1/organizations/enrich",
            params={
                "api_key": config.APOLLO_API_KEY,
                "domain": domain,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        org = data.get("organization")
        return org if org else None
    except requests.RequestException as e:
        print(f"Apollo org enrichment error for {domain}: {e}")
        return None


def _extract_domain(url: str) -> str:
    """Extract domain from a URL."""
    if not url:
        return ""
    url = url.strip().lower()
    if not url.startswith("http"):
        url = "https://" + url
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else ""


def _format_revenue(estimated_revenue: Optional[float]) -> str:
    """Format Apollo's annual_revenue into a human-readable string."""
    if not estimated_revenue:
        return "Unknown"
    if estimated_revenue >= 1_000_000_000:
        return f"${estimated_revenue / 1_000_000_000:.1f}B"
    if estimated_revenue >= 1_000_000:
        return f"${estimated_revenue / 1_000_000:.0f}M"
    if estimated_revenue >= 1_000:
        return f"${estimated_revenue / 1_000:.0f}K"
    return f"${estimated_revenue:.0f}"


def _build_company_row(exhibitor: Dict, apollo_org: Optional[Dict]) -> Dict:
    """Build a company row dict from exhibitor info + Apollo enrichment data."""
    if apollo_org:
        revenue_printed = apollo_org.get("annual_revenue_printed") or ""
        if not revenue_printed:
            raw_revenue = apollo_org.get("annual_revenue")
            revenue_printed = _format_revenue(raw_revenue) if raw_revenue else "Unknown"

        return {
            "company_name": apollo_org.get("name") or exhibitor["company_name"],
            "website": apollo_org.get("website_url") or exhibitor["website"],
            "location": apollo_org.get("city") or "",
            "country": apollo_org.get("country") or "",
            "industry_vertical": apollo_org.get("industry") or "Unknown",
            "sub_industry": apollo_org.get("subindustry") or "",
            "revenue_range": revenue_printed,
            "data_source": "Apollo",
        }
    else:
        return {
            "company_name": exhibitor["company_name"],
            "website": exhibitor.get("website", ""),
            "location": "",
            "country": "",
            "industry_vertical": "Unknown",
            "sub_industry": "",
            "revenue_range": "Unknown",
            "data_source": "Manual only",
        }


# ── Shortlist Filtering ──────────────────────────────────────────────────────


def _apply_shortlist(client, all_companies: List[Dict]) -> List[Dict]:
    """Use Claude to apply shortlist criteria to the enriched company list."""
    filter_prompt = f"""Given this list of companies, apply the shortlist criteria and return ONLY the companies that pass.

COMPANIES:
{json.dumps(all_companies, indent=2)}

{SHORTLIST_CRITERIA}

Return ONLY a JSON array of the companies that pass ALL criteria. Keep the exact same fields.
If no companies pass, return an empty array: []

Return ONLY the JSON array, no other text."""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": filter_prompt}],
    )

    text = response.content[0].text.strip()
    text = _extract_json(text)

    try:
        shortlisted = json.loads(text)
        if isinstance(shortlisted, list):
            return shortlisted
    except json.JSONDecodeError:
        pass

    return []


# ── Main Orchestrator ─────────────────────────────────────────────────────────


def enrich_exhibitors(
    exhibitor_sheets: Dict[str, pd.DataFrame],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple:
    """Enrich exhibitor lists with Apollo.io and apply shortlist filtering.

    Input: dict of {sheet_name: DataFrame} where sheet names end with "-Exhibitors".
    Each DataFrame must have at minimum a "Company Name" column, optionally "Website".

    Pipeline per exhibition:
    1. Read exhibitor list from sheet
    2. Enrich each company via Apollo.io
    3. Apply shortlist criteria via Claude

    Returns:
        Tuple of (sheets_dict, logs_list).
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    total_exhibitions = len(exhibitor_sheets)
    sheets = {}
    logs = []

    for idx, (sheet_name, sheet_df) in enumerate(exhibitor_sheets.items()):
        # Derive exhibition name from sheet name (strip "-Exhibitors" suffix)
        exhibition_name = sheet_name
        if exhibition_name.lower().endswith("-exhibitors"):
            exhibition_name = exhibition_name[: -len("-Exhibitors")]
        exhibition_name = exhibition_name.strip()

        logs.append(f"--- {exhibition_name} ---")
        logs.append(f"Sheet: {sheet_name} ({len(sheet_df)} companies)")

        if "Company Name" not in sheet_df.columns:
            logs.append("⚠ Missing 'Company Name' column — skipping")
            continue

        # Build exhibitor list from the sheet
        exhibitors = []
        for _, row in sheet_df.iterrows():
            name = str(row.get("Company Name", "")).strip()
            website = str(row.get("Website", "")).strip() if "Website" in sheet_df.columns else ""
            if name and name.lower() != "nan":
                exhibitors.append({"company_name": name, "website": website if website.lower() != "nan" else ""})

        if not exhibitors:
            logs.append("⚠ No valid company names found — skipping")
            continue

        logs.append(f"Companies to enrich: {len(exhibitors)}")

        if progress_callback:
            progress_callback(
                idx, total_exhibitions,
                f"Enriching {len(exhibitors)} companies via Apollo: {exhibition_name}"
            )

        # Step 1: Enrich each company via Apollo.io
        all_enriched = []
        for i, exhibitor in enumerate(exhibitors):
            domain = _extract_domain(exhibitor["website"])
            apollo_org = None

            if domain:
                apollo_org = _apollo_org_enrichment(domain)
                time.sleep(APOLLO_BATCH_DELAY)

            row_data = _build_company_row(exhibitor, apollo_org)
            all_enriched.append(row_data)

            if progress_callback and (i + 1) % 5 == 0:
                progress_callback(
                    idx, total_exhibitions,
                    f"Enriched {i + 1}/{len(exhibitors)} companies for {exhibition_name}"
                )

        apollo_hits = sum(1 for c in all_enriched if c["data_source"] == "Apollo")
        logs.append(f"Companies enriched: {len(all_enriched)} (Apollo: {apollo_hits}, Manual only: {len(all_enriched) - apollo_hits})")

        # Step 2: Apply shortlist criteria
        if progress_callback:
            progress_callback(idx, total_exhibitions, f"Applying shortlist: {exhibition_name}")

        shortlisted = _apply_shortlist(client, all_enriched)
        logs.append(f"Companies after shortlist: {len(shortlisted)}")

        def _build_df_rows(companies):
            return [{
                "Company Name": c.get("company_name", ""),
                "Website": c.get("website", ""),
                "Location of Company": c.get("location", ""),
                "Country": c.get("country", ""),
                "Industry Vertical": c.get("industry_vertical", ""),
                "Sub-industry": c.get("sub_industry", ""),
                "Revenue Range": c.get("revenue_range", "Unknown"),
                "Data Source": c.get("data_source", ""),
            } for c in companies]

        # Sheet with ALL enriched companies (before filtering)
        all_sheet_name = f"{exhibition_name}-All"[:31]
        sheets[all_sheet_name] = pd.DataFrame(_build_df_rows(all_enriched), columns=COMPANY_COLUMNS)
        logs.append(f"✓ Sheet '{all_sheet_name}': {len(all_enriched)} companies")

        # Sheet with shortlisted companies only
        if shortlisted:
            short_sheet_name = f"{exhibition_name}-Shortlist"[:31]
            sheets[short_sheet_name] = pd.DataFrame(_build_df_rows(shortlisted), columns=COMPANY_COLUMNS)
            logs.append(f"✓ Sheet '{short_sheet_name}': {len(shortlisted)} companies")
        else:
            logs.append("⚠ No companies passed shortlist — only All sheet created")

        if progress_callback:
            progress_callback(
                idx + 1, total_exhibitions,
                f"Completed {exhibition_name}: {len(all_enriched)} total, {len(shortlisted)} shortlisted"
            )

    return sheets, logs
