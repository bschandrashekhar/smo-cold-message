"""Step 1: Scrape exhibition websites for exhibitor companies and enrich with company details.

For each exhibition row:
1. Fetches the exhibitor page HTML via requests
2. Uses Claude to parse the HTML and extract company names + website URLs
3. If HTML parsing fails (JS-heavy page), falls back to Claude Web Search
4. Uses Apollo.io Organization Enrichment to get company details (industry, revenue, location)
5. Applies shortlist criteria to produce a filtered subset

Data source tracking: each company row includes a "Data Source" field
indicating where the enrichment data came from (e.g. "Apollo", "HTML only").
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

# Minimum meaningful HTML length — below this, the page is likely JS-rendered
MIN_HTML_CONTENT_LENGTH = 500

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


def _fetch_page_html(url: str) -> str:
    """Fetch a web page and return raw HTML.

    Returns empty string if the page can't be fetched.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return ""


def _clean_html_for_parsing(html: str, max_chars: int = 80000) -> str:
    """Strip scripts, styles, and excess whitespace from HTML to reduce token usage."""
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    html = re.sub(r'\s+', ' ', html)
    if len(html) > max_chars:
        html = html[:max_chars] + "\n... [truncated]"
    return html.strip()


def _is_js_heavy_page(html: str) -> bool:
    """Detect if a page is likely JavaScript-rendered with minimal server-side content.

    Checks if the cleaned HTML (minus scripts/styles) has very little text content,
    which suggests the real content is loaded via JS.
    """
    cleaned = _clean_html_for_parsing(html)
    # Strip all remaining HTML tags to get just text
    text_only = re.sub(r'<[^>]+>', ' ', cleaned)
    text_only = re.sub(r'\s+', ' ', text_only).strip()
    return len(text_only) < MIN_HTML_CONTENT_LENGTH


def _extract_exhibitors_from_html(client, html: str, exhibitor_link: str) -> List[Dict]:
    """Use Claude to parse HTML and extract exhibitor company names + website URLs.

    Returns a list of dicts with 'company_name' and 'website' keys.
    """
    cleaned = _clean_html_for_parsing(html)

    prompt = f"""Parse this HTML from an exhibition exhibitor listing page ({exhibitor_link}) and extract ALL exhibiting company names and their website URLs.

HTML CONTENT:
{cleaned}

For each exhibitor, extract:
- "company_name": The company/organization name
- "website": Their website URL if available in the HTML (href links), otherwise empty string ""

Return ONLY a JSON array of objects. Example:
[
  {{"company_name": "Acme Corp", "website": "https://acme.com"}},
  {{"company_name": "Beta Inc", "website": ""}}
]

If you find no exhibitors, return an empty array: []
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
        exhibitors = json.loads(text)
        if isinstance(exhibitors, list):
            return [
                {
                    "company_name": str(e.get("company_name", "")).strip(),
                    "website": str(e.get("website", "")).strip(),
                }
                for e in exhibitors
                if str(e.get("company_name", "")).strip()
            ]
    except json.JSONDecodeError:
        pass

    return []


def _extract_exhibitors_via_web_search(client, exhibitor_link: str) -> List[Dict]:
    """Fallback: Use Claude Web Search to extract exhibitors when HTML parsing fails.

    Used for JS-heavy pages that can't be scraped with requests.get().
    Returns a list of dicts with 'company_name' and 'website' keys.
    """
    prompt = f"""Visit this exhibition exhibitor page and extract ALL company names listed:
{exhibitor_link}

Search for this page and extract the complete list of exhibiting companies.
For each exhibitor, extract:
- "company_name": The company/organization name
- "website": Their website URL if you can find it, otherwise empty string ""

Return ONLY a JSON array of objects. Example:
[
  {{"company_name": "Acme Corp", "website": "https://acme.com"}},
  {{"company_name": "Beta Inc", "website": ""}}
]

If you cannot access the page or find no exhibitors, return an empty array: []
Return ONLY the JSON array, no other text."""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
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
        exhibitors = json.loads(text)
        if isinstance(exhibitors, list):
            return [
                {
                    "company_name": str(e.get("company_name", "")).strip(),
                    "website": str(e.get("website", "")).strip(),
                }
                for e in exhibitors
                if str(e.get("company_name", "")).strip()
            ]
    except json.JSONDecodeError:
        pass

    return []


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
    """Build a company row dict from exhibitor info + Apollo enrichment data.

    Tracks the data source for transparency.
    """
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
            "website": exhibitor["website"],
            "location": "",
            "country": "",
            "industry_vertical": "Unknown",
            "sub_industry": "",
            "revenue_range": "Unknown",
            "data_source": "HTML only",
        }


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


def scrape_exhibitors(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple:
    """Scrape exhibitor links and return enriched company lists per exhibition.

    Pipeline per exhibition:
    1. Fetch exhibitor page HTML
    2. Parse HTML with Claude to extract company names + URLs
    3. If HTML parsing fails (JS-heavy), fall back to Claude Web Search
    4. Enrich each company via Apollo.io Organization API
    5. Apply shortlist criteria via Claude

    Returns:
        Tuple of (sheets_dict, logs_list).
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
            progress_callback(idx, total_exhibitions, f"Fetching page: {exhibition_name}")

        # Step 1: Fetch exhibitor page HTML
        html = _fetch_page_html(exhibitor_link)
        scrape_method = "HTML"
        exhibitors = []

        if html and not _is_js_heavy_page(html):
            logs.append(f"HTML fetched: {len(html)} chars")

            if progress_callback:
                progress_callback(idx, total_exhibitions, f"Parsing exhibitors from HTML: {exhibition_name}")

            # Step 2: Parse HTML with Claude
            exhibitors = _extract_exhibitors_from_html(client, html, exhibitor_link)
            logs.append(f"Exhibitors parsed from HTML: {len(exhibitors)}")
        else:
            if html:
                logs.append(f"⚠ Page appears JS-heavy ({len(html)} chars HTML but minimal text content)")
            else:
                logs.append("⚠ Could not fetch page HTML")

        # Step 2b: Fallback to Claude Web Search if HTML parsing yielded nothing
        if not exhibitors:
            logs.append("Falling back to Claude Web Search...")
            scrape_method = "Claude Web Search"

            if progress_callback:
                progress_callback(idx, total_exhibitions, f"Using Claude Web Search: {exhibition_name}")

            exhibitors = _extract_exhibitors_via_web_search(client, exhibitor_link)
            logs.append(f"Exhibitors found via Web Search: {len(exhibitors)}")

        if exhibitors:
            names_preview = [e["company_name"] for e in exhibitors[:20]]
            logs.append(f"Names: {', '.join(names_preview)}")
            if len(exhibitors) > 20:
                logs.append(f"  ... and {len(exhibitors) - 20} more")

        if not exhibitors:
            logs.append("⚠ No exhibitors found via any method — skipping")
            if progress_callback:
                progress_callback(idx, total_exhibitions, f"No exhibitors found for {exhibition_name}")
            continue

        logs.append(f"Scrape method: {scrape_method}")

        if progress_callback:
            progress_callback(
                idx, total_exhibitions,
                f"Enriching {len(exhibitors)} companies via Apollo: {exhibition_name}"
            )

        # Step 3: Enrich each company via Apollo.io
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
        logs.append(f"Companies enriched: {len(all_enriched)} (Apollo: {apollo_hits}, HTML only: {len(all_enriched) - apollo_hits})")

        # Step 4: Apply shortlist criteria
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
