"""Client Referencing — reusable VectorMatch pipeline.

Install from Git in another project's requirements.txt:
    client-referencing @ git+https://github.com/<org>/smo-cold-message.git

Usage:
    from client_referencing import find_matches, invalidate_cache

    result = find_matches("Banking", "Salesforce, Mulesoft", "USA")
    for m in result["matches"]:
        print(m.client_name, m.final_score)
"""

from client_referencing.matcher import ClientMatch, find_matches, invalidate_cache

__all__ = ["find_matches", "invalidate_cache", "ClientMatch"]
