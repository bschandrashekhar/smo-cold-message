"""Client Referencing pipeline configuration."""

import os


def _get_secret(key: str) -> str:
    """Read from Streamlit secrets (cloud) or env vars (local)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, "")


VOYAGE_API_KEY = _get_secret("VOYAGE_API_KEY")
SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = _get_secret("SUPABASE_SERVICE_KEY")

TABLE_NAME = "client_referencing_data"
TECH_TABLE_NAME = "client_tech_data"
INDUSTRY_TABLE_NAME = "industry_embeddings"
VOYAGE_MODEL = "voyage-4-large"
VOYAGE_BATCH_SIZE = 128
EMBEDDING_DIM = 1024
INDUSTRY_MATCH_THRESHOLD = 0.85
CASE_STUDIES_TABLE_NAME = "client_case_studies"
CASESTUDY_TECH_MAPPING_TABLE = "client_case_studies_technology_mapping"
CACHE_TTL = 3600  # seconds — how long cached embeddings/rows stay fresh

# CaseStudyMatcher scoring weights
CS_EXACT_WEIGHT = 0.50
CS_SEMANTIC_WEIGHT = 0.20
CS_CONTEXT_WEIGHT = 0.30
