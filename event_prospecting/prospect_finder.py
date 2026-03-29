"""Step 2: Find and enrich prospects within scraped companies using Apollo.io."""

import pandas as pd


def find_prospects(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """Find prospects for each company using role-based targeting + Apollo.io.

    Targeting rules:
    - Large companies: Director of Engineering, Director of Data Engineering, Head of Applications
    - Small/mid-size companies: C-level executives (CEO, CTO, etc.)
    - Exclude CFOs

    Args:
        df: Company DataFrame from Step 1 (output of scrape_exhibitors).
        progress_callback: Optional callback(current, total, status_text).

    Returns:
        DataFrame with columns: First Name, Last Name, Job Title, Department,
        Phone, Mobile, Email, LinkedIn Profile Link, City, State, Country,
        Company Name, Industry, Technology, Lead Source.
    """
    raise NotImplementedError("Prospect finding not yet implemented")
