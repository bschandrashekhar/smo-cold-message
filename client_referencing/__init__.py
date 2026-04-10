"""Client Referencing — prospect-to-client matching and reference data management.

Use this module to:
  1. Match a prospect against the database of existing clients by industry,
     technology, and geography.
  2. Refresh/sync reference data (clients, industries, case studies) from
     Excel uploads via the Streamlit UI.

Install from Git in another project's requirements.txt:
    client-referencing @ git+https://github.com/<org>/smo-cold-message.git

Usage:
    from client_referencing import find_matches, invalidate_cache

    result = find_matches("Healthcare", "Salesforce, Mulesoft, Snowflake", "USA", max_matches=8)
    for m in result["matches"]:
        print(m.client_name, m.final_score)
"""

from client_referencing.matcher import ClientMatch, find_matches, invalidate_cache
from client_referencing.casestudy_matcher import CaseStudyMatch, find_casestudy_matches, invalidate_casestudy_cache

__all__ = [
    "find_matches", "invalidate_cache", "ClientMatch",
    "find_casestudy_matches", "invalidate_casestudy_cache", "CaseStudyMatch",
]
