"""Step 1: Scrape exhibition websites for exhibitor companies and enrich with company details.

For each exhibition row:
1. Fetches the exhibitor page HTML via requests
2. Uses BeautifulSoup to find the exhibitor section and extract company names + URLs
3. If BS4 parsing finds nothing, falls back to Claude HTML parsing (smaller section)
4. If that also fails, falls back to Claude Web Search
5. Uses Apollo.io Organization Enrichment to get company details
6. Applies shortlist criteria to produce a filtered subset

Data source tracking: each company row includes a "Data Source" field
indicating where the enrichment data came from (e.g. "Apollo", "HTML only").
"""

import json
import re
import time
from typing import Dict, List, Optional, Callable
from urllib.parse import urljoin

import requests
import pandas as pd
import anthropic
from bs4 import BeautifulSoup, Tag

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

# Words that indicate navigation/boilerplate rather than company names
NOISE_WORDS = {
    "home", "about", "contact", "register", "login", "sign up", "menu",
    "search", "privacy", "terms", "cookie", "back to top", "read more",
    "learn more", "view all", "see all", "click here", "download",
    "agenda", "schedule", "speakers", "venue", "hotel", "travel",
    "faq", "help", "support", "share", "tweet", "facebook", "linkedin",
    "instagram", "twitter", "youtube", "subscribe", "newsletter",
}


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
    """Fetch a web page and return raw HTML."""
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


# ── BeautifulSoup HTML Parsing ────────────────────────────────────────────────


def _find_exhibitor_section(soup: BeautifulSoup) -> Optional[Tag]:
    """Find the exhibitor/sponsor section in the page.

    Searches generically by:
    1. Element with id containing 'exhibitor' or 'sponsor'
    2. Element with class containing 'exhibitor' or 'sponsor'
    3. Heading (h1-h4) containing the word 'exhibitor' or 'sponsor', then takes parent section
    """
    # Strategy 1: Find by id
    for keyword in ["exhibitor", "sponsor", "partner"]:
        el = soup.find(id=re.compile(keyword, re.IGNORECASE))
        if el:
            return el

    # Strategy 2: Find by class
    for keyword in ["exhibitor", "sponsor", "partner"]:
        el = soup.find(class_=re.compile(keyword, re.IGNORECASE))
        if el:
            # Walk up to a section/div parent for broader context
            parent = el
            for _ in range(3):
                if parent.parent and parent.parent.name in ("section", "div", "main", "article"):
                    parent = parent.parent
                else:
                    break
            return parent

    # Strategy 3: Find by heading text
    for heading_tag in ["h1", "h2", "h3", "h4"]:
        for heading in soup.find_all(heading_tag):
            text = heading.get_text(strip=True).lower()
            if any(kw in text for kw in ["exhibitor", "sponsor", "partner"]):
                # Return the parent section/div
                parent = heading.parent
                for _ in range(3):
                    if parent.parent and parent.parent.name in ("section", "div", "main", "article"):
                        parent = parent.parent
                    else:
                        break
                return parent

    return None


def _is_likely_company_name(text: str) -> bool:
    """Check if a text string looks like a company name rather than navigation/boilerplate."""
    text = text.strip()
    if not text or len(text) < 2 or len(text) > 100:
        return False
    if text.lower() in NOISE_WORDS:
        return False
    # Skip if it's just a number or very short generic word
    if text.isdigit():
        return False
    # Skip if it looks like a sentence (too many words = probably a description)
    if len(text.split()) > 8:
        return False
    return True


def _extract_companies_from_section(section: Tag, base_url: str) -> List[Dict]:
    """Extract company names and URLs from an exhibitor section using structural patterns.

    Handles common patterns:
    - Logo images with alt text
    - Links to company websites
    - Cards/grid items with company info
    - List items with company names
    """
    companies = {}  # name_lower -> {company_name, website}

    def _add(name: str, website: str = ""):
        name = name.strip()
        if not _is_likely_company_name(name):
            return
        key = name.lower()
        if key not in companies:
            companies[key] = {"company_name": name, "website": website}
        elif website and not companies[key]["website"]:
            companies[key]["website"] = website

    # Pattern 1: Images with alt text (logo grids)
    for img in section.find_all("img"):
        alt = (img.get("alt") or "").strip()
        if alt and _is_likely_company_name(alt):
            # Check if the image is wrapped in a link
            parent_link = img.find_parent("a")
            href = ""
            if parent_link:
                href = parent_link.get("href", "")
                if href and not href.startswith(("http", "//")):
                    href = urljoin(base_url, href)
            _add(alt, href)

    # Pattern 2: Links with text that look like company names
    for a in section.find_all("a"):
        href = a.get("href", "")
        text = a.get_text(strip=True)

        # Skip internal/anchor links
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue

        if href and not href.startswith(("http", "//")):
            href = urljoin(base_url, href)

        # Skip links to the same domain (internal navigation)
        if href and base_url:
            from urllib.parse import urlparse
            link_domain = urlparse(href).netloc.lower().replace("www.", "")
            base_domain = urlparse(base_url).netloc.lower().replace("www.", "")
            is_external = link_domain and link_domain != base_domain

            if text and _is_likely_company_name(text):
                _add(text, href if is_external else "")

    # Pattern 3: Heading tags within repeating containers (cards)
    # Find repeating child divs/articles that might be exhibitor cards
    card_containers = section.find_all(["div", "article", "li"], recursive=False)
    if len(card_containers) < 3:
        # Try one level deeper
        for child in section.find_all(["div", "section"], recursive=False):
            deeper = child.find_all(["div", "article", "li"], recursive=False)
            if len(deeper) >= 3:
                card_containers = deeper
                break

    if len(card_containers) >= 3:
        for card in card_containers:
            # Look for a heading or strong tag as the company name
            name_el = card.find(["h2", "h3", "h4", "h5", "strong", "b"])
            if name_el:
                name = name_el.get_text(strip=True)
                # Find an external link in the card
                link = card.find("a", href=True)
                href = ""
                if link:
                    href = link.get("href", "")
                    if href and not href.startswith(("http", "//")):
                        href = urljoin(base_url, href)
                _add(name, href)

    return list(companies.values())


def _bs4_extract_exhibitors(html: str, url: str, logs: List[str]) -> List[Dict]:
    """Main BeautifulSoup extraction pipeline.

    1. Parse HTML
    2. Find exhibitor section
    3. Extract companies from that section
    """
    soup = BeautifulSoup(html, "html.parser")

    section = _find_exhibitor_section(soup)
    if not section:
        logs.append("BS4: No exhibitor section found in HTML")
        return []

    section_id = section.get("id", "")
    section_class = " ".join(section.get("class", []))[:50]
    logs.append(f"BS4: Found exhibitor section (id='{section_id}', class='{section_class}')")
    logs.append(f"BS4: Section size: {len(str(section))} chars")

    companies = _extract_companies_from_section(section, url)
    logs.append(f"BS4: Extracted {len(companies)} companies from section")

    return companies


# ── Claude Fallbacks ──────────────────────────────────────────────────────────


def _claude_parse_section_html(client, section_html: str, exhibitor_link: str) -> List[Dict]:
    """Use Claude to parse a smaller HTML section when BS4 structural extraction fails.

    This sends only the exhibitor section (not the whole page) to Claude.
    """
    # Clean the section HTML
    section_html = re.sub(r'<script[^>]*>.*?</script>', '', section_html, flags=re.DOTALL | re.IGNORECASE)
    section_html = re.sub(r'<style[^>]*>.*?</style>', '', section_html, flags=re.DOTALL | re.IGNORECASE)
    section_html = re.sub(r'\s+', ' ', section_html)

    # Truncate if still too large
    if len(section_html) > 60000:
        section_html = section_html[:60000] + "\n... [truncated]"

    prompt = f"""Parse this HTML section from an exhibition exhibitor listing page ({exhibitor_link}).
This is the exhibitor/sponsor section of the page. Extract ALL company names and their website URLs.

HTML SECTION:
{section_html}

Look for:
- Company names in headings, strong tags, alt text of images
- Website URLs in href attributes of links
- Any text that represents a company or organization name

For each exhibitor, extract:
- "company_name": The company/organization name
- "website": Their website URL if available, otherwise empty string ""

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


def _extract_exhibitors_via_web_search(client, exhibitor_link: str, exhibition_name: str = "") -> List[Dict]:
    """Last resort: Use Claude Web Search to find exhibitors when all HTML parsing fails."""
    search_hint = f'"{exhibition_name}" ' if exhibition_name else ""
    prompt = f"""I need to find the complete list of exhibiting companies for this conference/exhibition.

Exhibition page: {exhibitor_link}
{f'Exhibition name: {exhibition_name}' if exhibition_name else ''}

The exhibitor list on that page is loaded dynamically via JavaScript, so I need you to search the web to find this information.

Please search for:
1. {search_hint}exhibitors list
2. {search_hint}sponsors list
3. The conference name + "exhibitors" or "exhibitor directory"

Find as many exhibiting company names as possible. For each company, also find their website URL if available.

Return ONLY a JSON array of objects. Example:
[
  {{"company_name": "Acme Corp", "website": "https://acme.com"}},
  {{"company_name": "Beta Inc", "website": ""}}
]

If you truly cannot find any exhibitor information after searching, return an empty array: []
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
            "website": exhibitor["website"],
            "location": "",
            "country": "",
            "industry_vertical": "Unknown",
            "sub_industry": "",
            "revenue_range": "Unknown",
            "data_source": "HTML only",
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


def scrape_exhibitors(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple:
    """Scrape exhibitor links and return enriched company lists per exhibition.

    Extraction cascade per exhibition:
    1. Fetch HTML → BS4 structural parse (fastest, cheapest)
    2. If BS4 finds section but no companies → Claude parses the section HTML
    3. If no section found or all parsing fails → Claude Web Search

    Then:
    4. Enrich each company via Apollo.io
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

        # Step 1: Fetch HTML
        html = _fetch_page_html(exhibitor_link)
        scrape_method = ""
        exhibitors = []
        section_html = ""

        if not html:
            logs.append("⚠ Could not fetch page HTML")
        else:
            logs.append(f"HTML fetched: {len(html)} chars")

            if progress_callback:
                progress_callback(idx, total_exhibitions, f"Parsing exhibitors: {exhibition_name}")

            # Step 2: BeautifulSoup structural extraction
            exhibitors = _bs4_extract_exhibitors(html, exhibitor_link, logs)

            if exhibitors:
                scrape_method = "HTML (BS4)"
            else:
                # Step 2b: Claude parses the exhibitor section if BS4 found one
                soup = BeautifulSoup(html, "html.parser")
                section = None
                # Re-find the section for Claude parsing
                for keyword in ["exhibitor", "sponsor", "partner"]:
                    section = soup.find(id=re.compile(keyword, re.IGNORECASE))
                    if section:
                        break
                if not section:
                    for keyword in ["exhibitor", "sponsor", "partner"]:
                        section = soup.find(class_=re.compile(keyword, re.IGNORECASE))
                        if section:
                            break

                if section:
                    section_html = str(section)
                    logs.append(f"BS4 found section ({len(section_html)} chars) — sending to Claude for parsing")

                    if progress_callback:
                        progress_callback(idx, total_exhibitions, f"Claude parsing section: {exhibition_name}")

                    exhibitors = _claude_parse_section_html(client, section_html, exhibitor_link)
                    logs.append(f"Claude parsed from section: {len(exhibitors)} companies")

                    if exhibitors:
                        scrape_method = "HTML (Claude section parse)"

        # Step 3: Claude Web Search fallback
        if not exhibitors:
            logs.append("Falling back to Claude Web Search...")
            scrape_method = "Claude Web Search"

            if progress_callback:
                progress_callback(idx, total_exhibitions, f"Using Claude Web Search: {exhibition_name}")

            exhibitors = _extract_exhibitors_via_web_search(client, exhibitor_link, exhibition_name)
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

        # Step 4: Enrich each company via Apollo.io
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

        # Step 5: Apply shortlist criteria
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
