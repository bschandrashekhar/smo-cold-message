"""Trace industry shortlisting for a given prospect industry.

Supports hypothetical overrides to client industry data to simulate
what-if scenarios without touching the database.
"""

import sys
sys.path.insert(0, ".")

from client_referencing.matcher import fetch_all_rows, _matches_industry, _matches_geography

PROSPECT_IND = "financial services"
PROSPECT_CTRY = "australia"

# --- Hypothetical overrides ---
# Change industry_array / industry_group for specific clients to test ordering.
# Set to {} to use real data for all clients.
OVERRIDES = {
    "Cellulant Group": {
        "industry_array": ["banking", "lending"],
        "industry_group": "fintech",
    },
    "Gulf International Bank": {
        "industry_array": ["banking", "financial lending"],
        "industry_group": "fintech",
    },
}

all_rows = fetch_all_rows()

# Unique clients with their metadata (apply overrides)
clients = {}
for r in all_rows:
    cname = r["client_name"]
    if cname not in clients:
        override = OVERRIDES.get(cname, {})
        clients[cname] = {
            "industry_array": override.get("industry_array", r.get("industry_array") or []),
            "industry_group": override.get("industry_group", r.get("industry_group") or ""),
            "client_industry": r.get("client_industry") or "",
            "client_geography": r.get("client_geography") or "",
            "exact_key": r.get("exact_key") or "",
        }

if OVERRIDES:
    print("=== OVERRIDES APPLIED ===")
    for name, ov in OVERRIDES.items():
        print(f"  {name}: {ov}")
    print()

# --- All clients matching prospect industry ---
print(f'=== All clients matching industry "{PROSPECT_IND}" ===')
print(f'Match logic: "{PROSPECT_IND}" in item.lower() for industry_array, OR in industry_group.lower()\n')

industry_matches = []
for cname, info in sorted(clients.items()):
    arr = info["industry_array"]
    group = info["industry_group"]
    arr_hits = [item for item in arr if PROSPECT_IND in item.lower()]
    group_hit = PROSPECT_IND in group.lower()
    if arr_hits or group_hit:
        industry_matches.append(cname)
        reason = []
        if arr_hits:
            reason.append(f"arr={arr_hits}")
        if group_hit:
            reason.append(f'group="{group}"')
        print(f"  {cname:<35} geo={info['client_geography']:<15} matched via {', '.join(reason)}")

industry_client_names = set(industry_matches)
print(f"\nTotal: {len(industry_matches)} unique clients\n")

# --- Tier 1: Industry + Country ---
print(f'=== TIER 1: Industry "{PROSPECT_IND}" + Country "{PROSPECT_CTRY}" ===')
tier1 = sorted(c for c in industry_matches if PROSPECT_CTRY in clients[c]["client_geography"].lower())
for c in tier1:
    print(f"  {c:<35} geo={clients[c]['client_geography']}")
print(f"\nCount: {len(tier1)}")
print(f"Spec check: len <= 4? {len(tier1) <= 4} --> {'FALL TO TIER 2' if len(tier1) <= 4 else 'USE TIER 1 (industry_and_geography)'}\n")

# --- Tier 2: Industry only ---
print(f'=== TIER 2: Industry "{PROSPECT_IND}" only ===')
tier2 = sorted(industry_matches)
for c in tier2:
    print(f"  {c:<35} geo={clients[c]['client_geography']}")
print(f"\nCount: {len(tier2)}")
print(f"Spec check: len <= 2? {len(tier2) <= 2} --> {'FALL TO TIER 3 (Case NO_I — all clients)' if len(tier2) <= 2 else 'USE TIER 2 (Case YES_I — industry filtered)'}\n")

# --- Summary ---
if len(tier1) > 4:
    chosen = "TIER 1 (Industry + Country)"
    candidate_count = len(tier1)
    filter_level = "industry_and_geography"
elif len(tier2) > 2:
    chosen = "TIER 2 (Industry only)"
    candidate_count = len(tier2)
    filter_level = "industry"
else:
    chosen = "TIER 3 (All clients — no industry filter)"
    candidate_count = len(clients)
    filter_level = "none"

print(f"=== DECISION ===")
print(f"Selected: {chosen}")
print(f"Candidate rows: all rows belonging to {candidate_count} clients")
print(f"filter_level = \"{filter_level}\"")
print(f"industry_applied = {filter_level != 'none'}")
print(f"Case = {'YES_I' if filter_level != 'none' else 'NO_I'}")

# --- Backfill ordering simulation ---
# Simulates what happens when backfill clients are sorted by industry relevance
print(f"\n{'='*60}")
print(f"=== BACKFILL ORDERING SIMULATION ===")
print(f"(Assuming core shortlist has <=5 clients and backfill triggers)\n")

# Collect all clients with exact_key = "mobile application development"
# (the key that "ios" resolves to via alias)
ALIAS_KEY = "mobile application development"
backfill_candidates = []
for cname, info in sorted(clients.items()):
    if info["exact_key"].lower() == ALIAS_KEY:
        backfill_candidates.append(cname)

# Assume core shortlist from Tier 1 matching (SmartGroup, Moneyspot)
core_shortlist = {"SmartGroup Corporation", "Moneyspot"}
backfill_candidates = [c for c in backfill_candidates if c not in core_shortlist]

print(f"Core shortlist: {sorted(core_shortlist)}")
print(f"Backfill candidates with exact_key=\"{ALIAS_KEY}\": {len(backfill_candidates)}\n")

# Show before sort
print("--- Before industry sort ---")
for i, c in enumerate(backfill_candidates, 1):
    is_ind = c in industry_client_names
    print(f"  {i:>2}. {c:<35} industry_match={str(is_ind):<6} "
          f"arr={clients[c]['industry_array']}, group={clients[c]['industry_group']}, "
          f"geo={clients[c]['client_geography']}")

# Sort by industry relevance (same key as matcher.py line 499)
backfill_sorted = sorted(backfill_candidates, key=lambda x: (x not in industry_client_names, x))

print("\n--- After industry sort ---")
print(f"Sort key: (not in industry_client_names, client_name) — industry matches first, then alphabetical\n")
for i, c in enumerate(backfill_sorted, 1):
    is_ind = c in industry_client_names
    sort_key = (c not in industry_client_names, c)
    print(f"  {i:>2}. {c:<35} industry_match={str(is_ind):<6} sort_key={sort_key}  "
          f"geo={clients[c]['client_geography']}")

# Apply cap per new spec (DeficitNumber = 5 - len(core_shortlist))
generic_deficit = 5 - len(core_shortlist)
capped = backfill_sorted[:generic_deficit]

print(f"\n--- After cap (DeficitNumber = 5 - {len(core_shortlist)} = {generic_deficit}) ---")
for i, c in enumerate(capped, 1):
    is_ind = c in industry_client_names
    print(f"  {i:>2}. {c:<35} industry_match={str(is_ind):<6} geo={clients[c]['client_geography']}")

# Geo backfill check
combined = len(core_shortlist) + len(capped)
print(f"\n--- Geo backfill check ---")
print(f"Combined count: {len(core_shortlist)} core + {len(capped)} generic = {combined}")
print(f"Combined <= 5? {combined <= 5}")
if combined <= 5:
    geo_deficit = 6 - combined
    print(f"Geo DeficitNumber = 6 - {combined} = {geo_deficit}")
    exclude = core_shortlist | set(capped)
    geo_candidates = [c for c in sorted(clients.keys())
                      if c not in exclude and PROSPECT_CTRY in clients[c]["client_geography"].lower()]
    geo_picked = geo_candidates[:geo_deficit]
    print(f"Geo candidates (Australian, not already picked): {len(geo_candidates)}")
    if geo_picked:
        for c in geo_picked:
            print(f"  Geo pick: {c:<35} geo={clients[c]['client_geography']}")
    else:
        print("  (no geo candidates available)")

    # Final list
    print(f"\n=== FINAL SHORTLIST (core + capped generic + geo) ===")
    final = sorted(core_shortlist) + capped + geo_picked
    for i, c in enumerate(final, 1):
        source = "core" if c in core_shortlist else ("geo" if c in geo_picked else "generic_backfill")
        is_ind = c in industry_client_names
        print(f"  {i}. {c:<35} source={source:<18} industry_match={str(is_ind):<6} geo={clients[c]['client_geography']}")
else:
    print("Geo backfill skipped (combined > 5)")
    print(f"\n=== FINAL SHORTLIST (core + capped generic) ===")
    final = sorted(core_shortlist) + capped
    for i, c in enumerate(final, 1):
        source = "core" if c in core_shortlist else "generic_backfill"
        is_ind = c in industry_client_names
        print(f"  {i}. {c:<35} source={source:<18} industry_match={str(is_ind):<6} geo={clients[c]['client_geography']}")


# ── SEMANTIC INDUSTRY RELEVANCE SIMULATION ──────────────────────────────
print(f"\n{'='*60}")
print(f"=== WHAT-IF: SEMANTIC INDUSTRY SORT (instead of exact substring) ===")
print(f"Prospect industry: \"{PROSPECT_IND}\"\n")

# Semantic closeness: terms related to "financial services" with relevance weights.
# In a real implementation this could use embeddings; here we use a manual map.
INDUSTRY_ALIASES = {
    "financial services": 1.0,   # exact
    "banking": 0.85,
    "fintech": 0.80,
    "lending": 0.75,
    "financial": 0.70,           # partial word match
    "insurance": 0.65,
    "wealth management": 0.60,
}


def semantic_industry_score(info, prospect_ind):
    """Score how semantically close a client's industry columns are to the prospect industry.

    Checks each item in industry_array and industry_group against INDUSTRY_ALIASES.
    Returns the highest match score found (0.0 if no match).
    """
    best = 0.0
    all_terms = [t.lower().strip() for t in info["industry_array"]] + [info["industry_group"].lower().strip()]
    for term in all_terms:
        if not term:
            continue
        # Check exact alias match
        if term in INDUSTRY_ALIASES:
            best = max(best, INDUSTRY_ALIASES[term])
        # Check if any alias keyword appears as substring in the term
        for alias, weight in INDUSTRY_ALIASES.items():
            if alias in term and weight > best:
                best = weight
    return best


# Score all backfill candidates
scored = []
for c in backfill_candidates:
    score = semantic_industry_score(clients[c], PROSPECT_IND)
    scored.append((c, score))

# Sort: highest industry score first, then alphabetical
scored.sort(key=lambda x: (-x[1], x[0]))

print("--- Backfill candidates with semantic industry scores ---")
print(f"{'#':>3}  {'Client':<35} {'Score':>6}  Industry Terms")
print(f"{'':>3}  {'':35} {'':>6}  (arr + group)")
print(f"{'-'*95}")
for i, (c, score) in enumerate(scored, 1):
    terms = [t for t in clients[c]["industry_array"]] + [clients[c]["industry_group"]]
    terms = [t for t in terms if t]
    marker = " <-- industry relevant" if score > 0 else ""
    print(f"  {i:>2}. {c:<35} {score:>5.2f}  {terms}{marker}")

# Apply cap
semantic_capped = [c for c, _ in scored[:generic_deficit]]
print(f"\n--- After cap (DeficitNumber = {generic_deficit}) ---")
for i, c in enumerate(semantic_capped, 1):
    score = next(s for n, s in scored if n == c)
    print(f"  {i:>2}. {c:<35} score={score:.2f}  geo={clients[c]['client_geography']}")

# Geo backfill with semantic sort
combined_sem = len(core_shortlist) + len(semantic_capped)
print(f"\n--- Geo backfill check ---")
print(f"Combined: {len(core_shortlist)} core + {len(semantic_capped)} generic = {combined_sem}")
if combined_sem <= 5:
    geo_deficit_sem = 6 - combined_sem
    exclude_sem = core_shortlist | set(semantic_capped)
    geo_cand_sem = [c for c in sorted(clients.keys())
                    if c not in exclude_sem and PROSPECT_CTRY in clients[c]["client_geography"].lower()]
    geo_picked_sem = geo_cand_sem[:geo_deficit_sem]
    if geo_picked_sem:
        for c in geo_picked_sem:
            print(f"  Geo pick: {c:<35} geo={clients[c]['client_geography']}")

    print(f"\n=== FINAL SHORTLIST (semantic industry sort) ===")
    final_sem = sorted(core_shortlist) + semantic_capped + geo_picked_sem
else:
    geo_picked_sem = []
    final_sem = sorted(core_shortlist) + semantic_capped

for i, c in enumerate(final_sem, 1):
    source = "core" if c in core_shortlist else ("geo" if c in geo_picked_sem else "generic_backfill")
    score = next((s for n, s in scored if n == c), 0.0)
    print(f"  {i}. {c:<35} source={source:<18} ind_score={score:.2f}  geo={clients[c]['client_geography']}")

# Comparison
print(f"\n--- COMPARISON: Current vs Semantic ---")
print(f"{'Pos':<5} {'Current (exact substring)':<38} {'Semantic (alias-based)':<38}")
print(f"{'-'*80}")
current_final = final if 'final' in dir() else []
for pos in range(max(len(current_final), len(final_sem))):
    curr = current_final[pos] if pos < len(current_final) else "(empty)"
    sem = final_sem[pos] if pos < len(final_sem) else "(empty)"
    changed = " ***" if curr != sem else ""
    print(f"  {pos+1:<3} {curr:<38} {sem:<38}{changed}")
