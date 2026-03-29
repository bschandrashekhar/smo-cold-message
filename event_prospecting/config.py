"""Event Prospecting pipeline configuration.

Apollo-specific key lives here. Shared keys (Anthropic) are
imported from prospect_outreach.config to avoid duplication.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str) -> str:
    """Read from Streamlit secrets (cloud) or env vars (local)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, "")


APOLLO_API_KEY = _get_secret("APOLLO_API_KEY")

# Shared keys — reuse from prospect_outreach config
from prospect_outreach.config import ANTHROPIC_API_KEY  # noqa: E402
