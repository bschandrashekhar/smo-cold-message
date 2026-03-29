"""Legacy entry point — delegates to top-level app.py.

Streamlit Cloud may be configured to run this file. Instead of duplicating
the UI here, we import and run the top-level main() which has the sidebar
pipeline selector and both pipelines.
"""

import sys
import os

# Ensure the project root is on the path so top-level app imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import main  # noqa: E402

main()
