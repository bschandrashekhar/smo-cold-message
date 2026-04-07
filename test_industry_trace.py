"""Trace industry shortlisting using live cosine similarity scores.

Tests the full pipeline: fetch industry embeddings, compute scores,
apply INDUSTRY_MATCH_THRESHOLD, and show tier decisions + backfill ordering.
"""

import sys
sys.path.insert(0, ".")

from client_referencing.config import INDUSTRY_MATCH_THRESHOLD
from client_referencing.matcher import (
    fetch_all_rows,
    _matches_geography,
    _fetch_industry_embeddings,
    _compute_industry_scores,
    _compute_industry_client_names,
    filter_candidates,
)

PROSPECT_IND = "financial services"
PROSPECT_CTRY = "australia"

print(f"Prospect Industry: \"{PROSPECT_IND}\"")
print(f"Prospect Country:  \"{PROSPECT_CTRY}\"")
print(f"INDUSTRY_MATCH_THRESHOLD: {INDUSTRY_MATCH_THRESHOLD}")
print()

# --- Step 1: Fetch data and compute scores ---
print("Fetching all rows...")
all_rows = fetch_all_rows()

# Unique clients with metadata
clients = {}
for r in all_rows:
    cname = r["client_name"]
    if cname not in clients:
        clients[cname] = {
            "industry_array": r.get("industry_array") or [],
            "industry_group": r.get("industry_group") or "",
            "client_industry": r.get("client_industry") or "",
            "client_geography": r.get("client_geography") or "",
        }

print(f"Total unique clients: {len(clients)}\n")

print("Fetching industry embeddings and computing scores...")
industry_embeddings = _fetch_industry_embeddings()
print(f"Industry terms in DB: {len(industry_embeddings)}")
industry_scores = _compute_industry_scores(PROSPECT_IND, all_rows, industry_embeddings)
print(f"Clients with scores > 0: {sum(1 for s in industry_scores.values() if s > 0)}\n")

# --- Step 2: Show all clients with their industry scores ---
print(f"=== All clients with industry_score >= {INDUSTRY_MATCH_THRESHOLD} (pass threshold) ===\n")

passing = []
failing_nonzero = []
for cname in sorted(clients.keys()):
    score = industry_scores.get(cname, 0.0)
    if score >= INDUSTRY_MATCH_THRESHOLD:
        passing.append((cname, score))
    elif score > 0:
        failing_nonzero.append((cname, score))

for cname, score in sorted(passing, key=lambda x: -x[1]):
    info = clients[cname]
    terms = [t for t in info["industry_array"]] + ([info["industry_group"]] if info["industry_group"] else [])
    geo = info["client_geography"]
    print(f"  {cname:<35} score={score:.4f}  geo={geo:<15}  terms={terms}")

print(f"\nTotal passing: {len(passing)}")

if failing_nonzero:
    print(f"\n--- Clients with 0 < score < {INDUSTRY_MATCH_THRESHOLD} (below threshold) ---")
    for cname, score in sorted(failing_nonzero, key=lambda x: -x[1]):
        info = clients[cname]
        terms = [t for t in info["industry_array"]] + ([info["industry_group"]] if info["industry_group"] else [])
        print(f"  {cname:<35} score={score:.4f}  terms={terms}")
    print(f"Total below threshold: {len(failing_nonzero)}")

# --- Step 3: Run 3-tier filter ---
print(f"\n{'='*60}")
print(f"=== 3-TIER PRE-FILTER ===\n")

candidate_rows, filter_level, tier1_clients, tier2_clients = filter_candidates(
    all_rows, PROSPECT_IND, PROSPECT_CTRY, industry_scores
)

print(f'TIER 1 (Industry >= {INDUSTRY_MATCH_THRESHOLD} + Country "{PROSPECT_CTRY}"):')
if tier1_clients:
    for c in tier1_clients:
        score = industry_scores.get(c, 0.0)
        print(f"  {c:<35} score={score:.4f}  geo={clients[c]['client_geography']}")
else:
    print("  (empty)")
print(f"Count: {len(tier1_clients)}")
print(f"Spec check: len > 4? {len(tier1_clients) > 4} --> {'USE TIER 1' if len(tier1_clients) > 4 else 'FALL TO TIER 2'}\n")

print(f'TIER 2 (Industry >= {INDUSTRY_MATCH_THRESHOLD} only):')
if tier2_clients:
    for c in tier2_clients:
        score = industry_scores.get(c, 0.0)
        print(f"  {c:<35} score={score:.4f}  geo={clients[c]['client_geography']}")
else:
    print("  (empty)")
print(f"Count: {len(tier2_clients)}")
print(f"Spec check: len > 2? {len(tier2_clients) > 2} --> {'USE TIER 2' if len(tier2_clients) > 2 else 'FALL TO TIER 3 (all clients)'}\n")

# --- Decision ---
print(f"=== DECISION ===")
print(f"filter_level = \"{filter_level}\"")
print(f"industry_applied = {filter_level != 'none'}")
print(f"Case = {'YES_I' if filter_level != 'none' else 'NO_I'}")
print(f"Candidate rows: {len(candidate_rows)} rows from {len(set(r['client_name'] for r in candidate_rows))} clients")

# --- Step 4: Backfill ordering simulation ---
print(f"\n{'='*60}")
print(f"=== BACKFILL ORDERING SIMULATION ===")
print(f"(Sort key: descending industry_score, then alphabetical)\n")

industry_client_names = _compute_industry_client_names(all_rows, PROSPECT_IND, industry_scores)

# Simulate backfill: all clients not in tier results, sorted by industry score
core_clients = set(tier1_clients or tier2_clients or [])
backfill = [(c, industry_scores.get(c, 0.0)) for c in sorted(clients.keys()) if c not in core_clients]
backfill.sort(key=lambda x: (-x[1], x[0]))

print(f"{'#':>3}  {'Client':<35} {'Score':>7}  {'Passes?':>7}  Geography")
print(f"{'-'*90}")
for i, (c, score) in enumerate(backfill[:20], 1):  # show top 20
    passes = "YES" if score >= INDUSTRY_MATCH_THRESHOLD else ("partial" if score > 0 else "no")
    print(f"  {i:>2}. {c:<35} {score:>6.4f}  {passes:>7}  {clients[c]['client_geography']}")

if len(backfill) > 20:
    print(f"  ... and {len(backfill) - 20} more")
