"""Check semantic match details for all Tier 1 clients against 'boomi'."""

import sys
sys.path.insert(0, ".")

from client_referencing.matcher import (
    fetch_all_rows, filter_candidates, semantic_match,
    _fetch_industry_embeddings, _compute_industry_scores,
)

PROSPECT_IND = "financial services"
PROSPECT_CTRY = "australia"
PROSPECT_TECH = "boomi"

all_rows = fetch_all_rows()
industry_embeddings = _fetch_industry_embeddings()
industry_scores = _compute_industry_scores(PROSPECT_IND, all_rows, industry_embeddings)

candidate_rows, filter_level, tier1_clients, tier2_clients = filter_candidates(
    all_rows, PROSPECT_IND, PROSPECT_CTRY, industry_scores
)

print(f"Tier 1 clients: {tier1_clients}\n")

# Run semantic match on ALL candidate rows (tier 1) for "boomi"
sem_results = semantic_match(candidate_rows, [PROSPECT_TECH])

print(f"{'Client':<40} {'Semantic Matches'}")
print("-" * 90)
for c in tier1_clients:
    matches = sem_results.get(c, [])
    if matches:
        for tech, embed_text, sim in matches:
            print(f"  {c:<38} {tech} -> {embed_text} (sim={sim:.4f})")
    else:
        print(f"  {c:<38} (no semantic match)")
