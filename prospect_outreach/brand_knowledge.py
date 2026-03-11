import json
import time
from pathlib import Path
from . import config
import anthropic


def are_brand_files_present():
    """Return a dict showing which brand profile files currently exist."""
    status = {}
    for name, path in config.BRAND_JSONS.items():
        status[name] = path.exists()
    return status


def _scrape_brand_with_claude(website_url: str, brand_name: str) -> dict:
    """Use Claude with web_search tool to research a brand website and extract structured profile."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    prompt = f"""Research the company website {website_url} ({brand_name}) thoroughly.

Extract the following information and return it as a JSON object:

{{
  "name": "{brand_name}",
  "website": "{website_url}",
  "services": ["list of services offered"],
  "verticals": ["industry verticals they serve"],
  "tech_stack": ["technologies and platforms they work with (e.g. Salesforce, Snowflake, AWS)"],
  "differentiators": ["what makes them unique / key value propositions"],
  "tone_and_positioning": "description of their brand voice, messaging style, and market positioning",
  "target_audience": "description of their ideal customer profile",
  "tagline": "their tagline or mission statement if available",
  "case_study_highlights": ["brief descriptions of notable client work or results mentioned on the site"]
}}

Search their website thoroughly — look at services pages, about pages, case studies, and any other relevant content. Return ONLY the JSON object, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text from response (may have multiple content blocks due to tool use)
    text_parts = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)

    full_text = "\n".join(text_parts).strip()

    # Parse JSON from the response — handle markdown code blocks
    if "```json" in full_text:
        full_text = full_text.split("```json")[1].split("```")[0].strip()
    elif "```" in full_text:
        full_text = full_text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(full_text)
    except json.JSONDecodeError:
        # If parsing fails, return a basic structure with the raw text
        return {
            "name": brand_name,
            "website": website_url,
            "raw_research": full_text,
            "services": [],
            "verticals": [],
            "tech_stack": [],
            "differentiators": [],
            "tone_and_positioning": "",
            "target_audience": "",
        }


BRAND_WEBSITES = {
    "LendingLogik": "https://www.lendinglogik.com",
    "CloudChillies": "https://www.cloudchillies.com",
}


def refresh_brand_profiles():
    """Scrape brand websites using Claude web search and persist JSON profiles."""
    config.DATA_DIR.mkdir(exist_ok=True)

    for name, website in BRAND_WEBSITES.items():
        profile = _scrape_brand_with_claude(website, name)
        path = config.BRAND_JSONS[name]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        time.sleep(2)  # Small delay between brand scrapes

    return are_brand_files_present()


def ensure_brand_profiles():
    """Load brand profiles, scraping if they don't exist yet."""
    status = are_brand_files_present()
    if not all(status.values()):
        refresh_brand_profiles()


def load_brand_profile(brand_name: str) -> dict:
    """Load a recorded brand profile from disk."""
    path = config.BRAND_JSONS.get(brand_name)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Brand profile for {brand_name} not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
