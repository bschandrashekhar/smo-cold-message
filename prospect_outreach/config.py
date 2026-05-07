import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present (local dev) — use explicit path to avoid CWD issues
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)


def _get_secret(key: str) -> str:
    """Read from Streamlit secrets (cloud) or env vars (local)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, "")


# Environment variables for APIs and paths
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = _get_secret("VOYAGE_API_KEY")
SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_SERVICE_KEY = _get_secret("SUPABASE_SERVICE_KEY")
SERPER_API_KEY = _get_secret("SERPER_API_KEY")

# Data directory where brand profiles are stored
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
BRAND_JSONS = {
    "LendingLogik": DATA_DIR / "brand_lendinglogik.json",
    "CloudChillies": DATA_DIR / "brand_cloudchillies.json",
}
