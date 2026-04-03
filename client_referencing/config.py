"""Client Referencing pipeline configuration.

Reuses shared keys from prospect_outreach.config.
"""

from prospect_outreach.config import VOYAGE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY

TABLE_NAME = "client_referencing_data"
VOYAGE_MODEL = "voyage-4-large"
VOYAGE_BATCH_SIZE = 128
EMBEDDING_DIM = 1024
