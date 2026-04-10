"""Update geo_priority column from sorted_by_country.xlsx."""
import sys
sys.path.insert(0, ".")

import openpyxl
from supabase import create_client
from prospect_outreach.config import SUPABASE_URL, SUPABASE_SERVICE_KEY

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

wb = openpyxl.load_workbook(r"C:\Users\sateesh\Desktop\sorted_by_country.xlsx")
ws = wb.active

# Build client -> priority mapping (skip header)
priorities = {}
for row in ws.iter_rows(min_row=2, values_only=True):
    country, company, priority = row
    if company and priority is not None:
        priorities[company] = int(priority)

print(f"Loaded {len(priorities)} client priorities from Excel\n")

# Update each unique client in the table
# Since table is denormalized (multiple rows per client), update by client_name
success = 0
errors = 0
for client_name, priority in sorted(priorities.items()):
    try:
        result = sb.table("client_referencing_data").update(
            {"geo_priority": priority}
        ).eq("client_name", client_name).execute()
        updated = len(result.data) if result.data else 0
        print(f"  {client_name:<45} priority={priority}  rows_updated={updated}")
        success += 1
    except Exception as e:
        print(f"  ERROR: {client_name} -> {e}")
        errors += 1

print(f"\nDone: {success} clients updated, {errors} errors")

# Verify a sample
print("\nVerification (sample):")
sample = sb.table("client_referencing_data").select(
    "client_name,client_geography,geo_priority"
).in_("client_name", ["Brickworks (Austral Bricks)", "SmartGroup Corporation", "Cellulant Group"]).execute()
seen = set()
for r in sample.data:
    if r["client_name"] not in seen:
        seen.add(r["client_name"])
        print(f"  {r['client_name']:<40} geo={r['client_geography']:<15} priority={r['geo_priority']}")
