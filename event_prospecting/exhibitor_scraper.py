"""Step 1: Scrape exhibition websites for exhibitor companies and enrich with company details.

For each exhibition row, scrapes the exhibitor link to extract a company list,
enriches each company, and returns results as a dict of DataFrames keyed by
exhibition name (one worksheet per exhibition).
"""

from typing import Dict, Optional, Callable
import pandas as pd


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


def scrape_exhibitors(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Scrape exhibitor links and return enriched company lists per exhibition.

    Iterates through each row in the input DataFrame. For each exhibition,
    scrapes the exhibitor link to extract companies and enriches them with
    company details.

    Shortlist criteria (#INPUT-STEP1):
    - Not IT service providers
    - Revenue under $2 billion
    - Preferably financial services (not mandatory)
    - Using technologies in our expertise: Salesforce, Dell Boomi, Snowflake,
      .NET, Tableau, MS Fabric
    - Avoid casting too wide a net on data engineering tools

    Args:
        df: Input DataFrame with 'Exhibition' and 'Exhibitor Link' columns.
            Each row is a different exhibition.
        progress_callback: Optional callback(current, total, status_text).

    Returns:
        Dict mapping sheet names ('<ExhibitionName>-Exhibitors') to DataFrames.
        Each DataFrame has columns: Company Name, Website, Location of Company,
        Country, Industry Vertical, Sub-industry, Revenue Range.
    """
    raise NotImplementedError("Exhibitor scraping not yet implemented")
