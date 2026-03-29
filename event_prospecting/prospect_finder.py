"""Step 2: Find and enrich prospects within scraped companies using Apollo.io.

Takes the multi-sheet workbook from Step 1 and adds prospect columns to the
right of existing company data in each worksheet.
"""

import re
import time
from typing import Dict, List, Optional, Callable

import requests
import pandas as pd

from event_prospecting import config

BATCH_DELAY = 2  # seconds between Apollo API calls

# Prospect columns added to the right of company data
PROSPECT_COLUMNS = [
    "First Name",
    "Last Name",
    "Job Title",
    "Department",
    "Phone",
    "Mobile",
    "Email",
    "LinkedIn Profile Link",
    "City",
    "State",
    "Prospect Country",
    "Industry",
    "Technology",
    "Lead Source",
]

# Role-based targeting by company size
LARGE_COMPANY_TITLES = [
    "Director of Engineering",
    "Director of Data Engineering",
    "Head of Applications",
    "VP of Engineering",
    "VP of Technology",
    "Director of IT",
    "Head of Engineering",
    "Head of Data",
]

SMALL_MID_COMPANY_TITLES = [
    "Chief Executive Officer",
    "CEO",
    "Chief Technology Officer",
    "CTO",
    "Chief Operating Officer",
    "COO",
    "Chief Information Officer",
    "CIO",
    "Chief Digital Officer",
    "CDO",
]

EXCLUDE_TITLES = ["Chief Financial Officer", "CFO"]


def _parse_revenue_to_millions(revenue_range: str) -> Optional[float]:
    """Parse a revenue range string into an approximate value in millions.

    Returns None if unparseable. Used to determine company size for targeting.
    """
    if not revenue_range or revenue_range.lower() in ("unknown", "n/a", ""):
        return None

    text = revenue_range.upper().replace(",", "").replace("$", "").strip()

    # Try to find a number with B/M suffix
    match = re.search(r"([\d.]+)\s*(B|M|K)?", text)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "B":
        return value * 1000
    elif unit == "K":
        return value / 1000
    else:
        # Default to millions
        return value


def _is_large_company(revenue_range: str) -> bool:
    """Determine if a company is 'large' based on revenue.

    Large = revenue > $100M. If unknown, default to small/mid targeting.
    """
    millions = _parse_revenue_to_millions(revenue_range)
    if millions is None:
        return False
    return millions > 100


def _extract_domain(website: str) -> str:
    """Extract domain from a website URL."""
    if not website:
        return ""
    website = website.strip().lower()
    if not website.startswith("http"):
        website = "https://" + website
    # Extract domain
    match = re.search(r"https?://(?:www\.)?([^/]+)", website)
    return match.group(1) if match else website


def _apollo_people_search(
    domain: str,
    titles: List[str],
    exclude_titles: List[str],
    per_page: int = 10,
) -> List[Dict]:
    """Search Apollo.io for people at a company matching role criteria.

    Args:
        domain: Company domain (e.g. 'acme.com').
        titles: Job titles to include.
        exclude_titles: Job titles to exclude.
        per_page: Max results per request.

    Returns:
        List of person dicts from Apollo API.
    """
    if not config.APOLLO_API_KEY:
        return []

    try:
        response = requests.post(
            "https://api.apollo.io/api/v1/mixed_people/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": config.APOLLO_API_KEY,
                "q_organization_domains": domain,
                "person_titles": titles,
                "person_not_titles": exclude_titles,
                "per_page": per_page,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("people", [])
    except requests.RequestException as e:
        print(f"Apollo API error for {domain}: {e}")
        return []


def _extract_prospect_row(person: Dict, company_name: str) -> Dict:
    """Extract prospect fields from an Apollo person record."""
    org = person.get("organization", {}) or {}

    return {
        "First Name": person.get("first_name", ""),
        "Last Name": person.get("last_name", ""),
        "Job Title": person.get("title", ""),
        "Department": person.get("departments", [""])[0] if person.get("departments") else "",
        "Phone": person.get("phone_number", "") or (person.get("phone_numbers", [{}]) or [{}])[0].get("sanitized_number", ""),
        "Mobile": person.get("mobile_phone", ""),
        "Email": person.get("email", ""),
        "LinkedIn Profile Link": person.get("linkedin_url", ""),
        "City": person.get("city", ""),
        "State": person.get("state", ""),
        "Prospect Country": person.get("country", ""),
        "Industry": org.get("industry", ""),
        "Technology": ", ".join(org.get("current_technologies", [])[:10]) if org.get("current_technologies") else "",
        "Lead Source": "Apollo.io",
    }


def find_prospects(
    sheets: Dict[str, pd.DataFrame],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Find prospects for each company and add as columns to the right.

    Args:
        sheets: Dict mapping sheet names to company DataFrames (from Step 1).
        progress_callback: Optional callback(current, total, status_text).

    Returns:
        Dict mapping the same sheet names to enriched DataFrames with prospect
        columns appended to the right of existing company columns.
    """
    total_companies = sum(len(df) for df in sheets.values())
    processed = 0
    result_sheets = {}

    for sheet_name, company_df in sheets.items():
        enriched_rows = []

        for _, company_row in company_df.iterrows():
            company_name = str(company_row.get("Company Name", "")).strip()
            website = str(company_row.get("Website", "")).strip()
            revenue_range = str(company_row.get("Revenue Range", "")).strip()

            if progress_callback:
                progress_callback(processed, total_companies, f"Finding prospects: {company_name}")

            domain = _extract_domain(website)
            if not domain:
                # No website — add company row with empty prospect columns
                row_data = company_row.to_dict()
                for col in PROSPECT_COLUMNS:
                    row_data[col] = ""
                enriched_rows.append(row_data)
                processed += 1
                continue

            # Determine targeting based on company size
            if _is_large_company(revenue_range):
                titles = LARGE_COMPANY_TITLES
            else:
                titles = SMALL_MID_COMPANY_TITLES

            # Search Apollo
            people = _apollo_people_search(domain, titles, EXCLUDE_TITLES)

            if people:
                for p_idx, person in enumerate(people):
                    prospect_data = _extract_prospect_row(person, company_name)
                    if p_idx == 0:
                        # First prospect row: include company data
                        row_data = company_row.to_dict()
                    else:
                        # Subsequent prospects: blank company columns for readability
                        row_data = {col: "" for col in company_df.columns}
                    row_data.update(prospect_data)
                    enriched_rows.append(row_data)
            else:
                # No prospects found — add company row with empty prospect columns
                row_data = company_row.to_dict()
                for col in PROSPECT_COLUMNS:
                    row_data[col] = ""
                enriched_rows.append(row_data)

            processed += 1
            if progress_callback:
                prospect_count = len(people) if people else 0
                progress_callback(processed, total_companies, f"{company_name}: {prospect_count} prospects found")

            time.sleep(BATCH_DELAY)

        # Build enriched DataFrame preserving column order
        all_columns = list(company_df.columns) + PROSPECT_COLUMNS
        result_sheets[sheet_name] = pd.DataFrame(enriched_rows, columns=all_columns)

    return result_sheets
