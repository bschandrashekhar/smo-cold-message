"""Test find_matches end-to-end for two combinations."""

import sys
sys.path.insert(0, ".")

from client_referencing.matcher import find_matches

COMBOS = [
    ("financial services", "iOS", "Australia"),
    ("financial services", "boomi", "Australia"),
]

for industry, tech, country in COMBOS:
    print(f"\n{'='*80}")
    print(f"INPUT: industry={industry!r}, tech={tech!r}, country={country!r}")
    print(f"{'='*80}")

    result = find_matches(industry, tech, country, max_matches=8)

    print(f"\nFilter applied: {result['industry_filter_applied']}")
    print(f"Total matches: {len(result['matches'])}")
    print(f"Industry-filtered only: {result['industry_filtered_only']}")

    print(f"\n{'#':>3}  {'Client':<35} {'Score':>6}  {'Exact':>6}  {'Sem':>5}  {'IndScore':>8}  Source")
    print(f"{'-'*100}")
    for i, m in enumerate(result["matches"], 1):
        exact_str = ", ".join(m.exact_techs) if m.exact_techs else "-"
        sem_str = ", ".join(f"{t[0]}->{t[1][:20]}({t[2]:.2f})" for t in m.semantic_techs[:2]) if m.semantic_techs else "-"
        print(f"  {i:>2}. {m.client_name:<35} {m.final_score:>5.3f}  {exact_str:<6}  {sem_str:<5}  {m.industry_score:>7.4f}")
        if m.case_studies:
            print(f"       Case studies: {', '.join(cs['title'] for cs in m.case_studies)}")

    print(f"\nDebug log:")
    for label, value in result["debug_log"]:
        print(f"  {label}: {value}")
