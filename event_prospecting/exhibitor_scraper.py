"""Step 1: Scrape exhibition websites for exhibitor companies and enrich with company details.

For each exhibition row:
1. Fetches the exhibitor page HTML via requests
2. Detects JS-heavy/SPA pages and skips with a clear warning
3. Uses BeautifulSoup to find the exhibitor section and extract company names + URLs
4. Uses Apollo.io Organization Enrichment to get company details
5. Applies shortlist criteria via Claude to produce a filtered subset

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


def _is_js_heavy_page(html: str, url: str) -> bool:
    """Detect if a page is likely JavaScript-rendered (SPA) with no server-side content.

    Indicators:
    - URL contains a hash fragment (e.g. /#exhibitors)
    - HTML body has very little visible text relative to script content
    - Presence of SPA framework markers (React root, Angular app, Vue app)
    """
    # Hash-based routing is a strong SPA signal
    if "#" in url:
        return True

    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")
    if not body:
        return True

    # Compare script content vs visible text
    scripts = body.find_all("script")
    script_chars = sum(len(s.get_text()) for s in scripts)

    # Remove scripts to measure visible text
    for s in scripts:
        s.decompose()
    visible_text = body.get_text(strip=True)

    # If visible text is tiny compared to script content, likely JS-rendered
    if len(visible_text) < 500 and script_chars > 5000:
        return True

    # SPA framework markers
    spa_markers = [
        body.find(id="root"),        # React
        body.find(id="app"),         # Vue
        body.find(id="__next"),      # Next.js
        body.find(attrs={"ng-app": True}),  # Angular
    ]
    if any(spa_markers) and len(visible_text) < 1000:
        return True

    return False


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

    Pipeline per exhibition:
    1. Fetch HTML → BS4 structural parse (Python only, no LLM)
    2. Enrich each company via Apollo.io
    3. Apply shortlist criteria via Claude

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
        exhibitors = []

        if not html:
            logs.append("⚠ Could not fetch page HTML — skipping")
            if progress_callback:
                progress_callback(idx, total_exhibitions, f"Could not fetch page for {exhibition_name}")
            continue

        logs.append(f"HTML fetched: {len(html)} chars")

        # Check for JS-heavy pages that won't have server-rendered content
        if _is_js_heavy_page(html, exhibitor_link):
            logs.append("⚠ Page appears to be JavaScript-rendered (SPA) — exhibitor data is loaded dynamically and cannot be extracted via HTTP fetch. Skipping.")
            if progress_callback:
                progress_callback(idx, total_exhibitions, f"JS-rendered page, skipping: {exhibition_name}")
            continue

        if progress_callback:
            progress_callback(idx, total_exhibitions, f"Parsing exhibitors: {exhibition_name}")

        # Step 2: BeautifulSoup structural extraction (Python only, no LLM)
        exhibitors = _bs4_extract_exhibitors(html, exhibitor_link, logs)

        if exhibitors:
            names_preview = [e["company_name"] for e in exhibitors[:20]]
            logs.append(f"Names: {', '.join(names_preview)}")
            if len(exhibitors) > 20:
                logs.append(f"  ... and {len(exhibitors) - 20} more")

        if not exhibitors:
            logs.append("⚠ No exhibitors found in HTML — skipping")
            if progress_callback:
                progress_callback(idx, total_exhibitions, f"No exhibitors found for {exhibition_name}")
            continue

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
