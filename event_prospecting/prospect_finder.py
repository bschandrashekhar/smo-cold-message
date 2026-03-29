"""Step 2: Find and enrich prospects within scraped companies using Apollo.io.

Takes the multi-sheet workbook from Step 1 and adds prospect columns to the
right of existing company data in each worksheet.
"""

from typing import Dict, Optional, Callable
import pandas as pd


# Prospect columns added to the right of company data in Step 2
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


def find_prospects(
    sheets: Dict[str, pd.DataFrame],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """Find prospects for each company and add as columns to the right.

    For each company in each exhibition worksheet, finds relevant prospects
    using role-based targeting and enriches contact details via Apollo.io.
    Prospect rows are added alongside company data — multiple prospects per
    company result in the company fields repeated on each row.

    Targeting rules (ICP varies by company size):
    - Large companies: Director of Engineering, Director of Data Engineering,
      Head of Applications
    - Small/mid-size companies: C-level executives (CEO, CTO, etc.)
    - Exclude CFOs from targeting

    Args:
        sheets: Dict mapping sheet names to company DataFrames (from Step 1).
        progress_callback: Optional callback(current, total, status_text).

    Returns:
        Dict mapping the same sheet names to enriched DataFrames with prospect
        columns appended to the right of existing company columns.
    """
    raise NotImplementedError("Prospect finding not yet implemented")
