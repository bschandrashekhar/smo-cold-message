"""Event Prospecting pipeline configuration."""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str) -> str:
    """Read from environment variables."""
    return os.environ.get(key, "")


APOLLO_API_KEY = _get_secret("APOLLO_API_KEY")
