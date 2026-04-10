"""Add geo_priority column to client_referencing_data table."""
import sys
sys.path.insert(0, ".")

from supabase import create_client
from prospect_outreach.config import SUPABASE_URL, SUPABASE_SERVICE_KEY

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Add column via RPC (raw SQL)
print("Adding geo_priority column...")
result = sb.rpc("exec_sql", {
    "query": "ALTER TABLE client_referencing_data ADD COLUMN IF NOT EXISTS geo_priority integer DEFAULT 999;"
}).execute()
print(f"Result: {result}")

# Verify
print("\nVerifying column exists...")
rows = sb.table("client_referencing_data").select("client_name,client_geography,geo_priority").limit(5).execute()
for r in rows.data:
    print(f"  {r['client_name']:<35} geo={r.get('client_geography'):<15} priority={r.get('geo_priority')}")
print("\nDone!")
