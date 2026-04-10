"""Show all Australian clients and which ones are excluded."""
import sys
sys.path.insert(0, ".")
from client_referencing.matcher import fetch_all_rows

all_rows = fetch_all_rows()
exclude = {"CompareClub", "Remunerator Australia", "Australian Business Credit", "Moneyspot", "Ascentis"}

seen = set()
print("All Australian clients (excluding core + generic backfill):\n")
for r in all_rows:
    cname = r["client_name"]
    geo = (r.get("client_geography") or "").lower()
    if "australia" in geo and cname not in seen and cname not in exclude:
        seen.add(cname)
        print(f"  {cname:<40} geo={r.get('client_geography')}")

print(f"\nTotal: {len(seen)}")
