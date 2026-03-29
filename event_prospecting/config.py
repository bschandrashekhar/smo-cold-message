"""Event Prospecting pipeline configuration.

Apollo-specific key lives here. Shared keys (Anthropic, Serper) are
imported from prospect_outreach.config to avoid duplication.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str) -> str:
    """Read from environment variables."""
    return os.environ.get(key, "")


APOLLO_API_KEY = _get_secret("APOLLO_API_KEY")

# Shared keys — reuse from prospect_outreach config
from prospect_outreach.config import ANTHROPIC_API_KEY, SERPER_API_KEY  # noqa: E402
