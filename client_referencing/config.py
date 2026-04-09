"""Client Referencing pipeline configuration.

Reuses shared keys from prospect_outreach.config.
"""

from prospect_outreach.config import VOYAGE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY

TABLE_NAME = "client_referencing_data"
TECH_TABLE_NAME = "client_tech_data"
INDUSTRY_TABLE_NAME = "industry_embeddings"
VOYAGE_MODEL = "voyage-4-large"
VOYAGE_BATCH_SIZE = 128
EMBEDDING_DIM = 1024
INDUSTRY_MATCH_THRESHOLD = 0.85
CASE_STUDIES_TABLE_NAME = "client_case_studies"
CACHE_TTL = 3600  # seconds — how long cached embeddings/rows stay fresh
