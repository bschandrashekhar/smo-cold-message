"""Prospect Outreach — Pass 1 Research Pipeline.

For each prospect:
  1. Brand match via find_brand_match (industry embeddings)
  2. EMEA country coding (early, before matching)
  3. Technology research via Serper (with Supabase cache, 3-month TTL)
  4. Case study matching via find_casestudy_matches
  5. Client matching via find_matches
  6. Intent scoring (optional)
"""

import json
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional

import anthropic
import pandas as pd
import requests
from supabase import create_client

from prospect_outreach import config
from prospect_outreach.brand_knowledge import load_brand_profile
from client_referencing.brand_matcher import find_brand_match
from client_referencing.matcher import find_matches
from client_referencing.casestudy_matcher import find_casestudy_matches

# ── Constants ─────────────────────────────────────────────────────────────

CACHE_TTL_DAYS = 90  # 3 months

EMEA_COUNTRIES = {
    # Middle East
    "bahrain", "cyprus", "egypt", "iran", "iraq", "israel", "jordan", "kuwait",
    "lebanon", "oman", "palestine", "qatar", "saudi arabia", "syria",
    "turkey", "united arab emirates", "uae", "yemen",
    # Africa
    "algeria", "angola", "benin", "botswana", "burkina faso", "burundi",
    "cabo verde", "cameroon", "central african republic", "chad", "comoros",
    "democratic republic of the congo", "djibouti", "equatorial guinea",
    "eritrea", "eswatini", "ethiopia", "gabon", "gambia", "ghana", "guinea",
    "guinea-bissau", "ivory coast", "kenya", "lesotho", "liberia", "libya",
    "madagascar", "malawi", "mali", "mauritania", "mauritius", "morocco",
    "mozambique", "namibia", "niger", "nigeria", "rwanda", "sao tome and principe",
    "senegal", "seychelles", "sierra leone", "somalia", "south africa",
    "south sudan", "sudan", "tanzania", "togo", "tunisia", "uganda",
    "zambia", "zimbabwe",
}

TECHNOLOGY_RESEARCH_SYSTEM_PROMPT = """You will be given search result snippets for a prospect company.
Based solely on these snippets, populate the TECHNOLOGY_RESEARCH Python dictionary.
Only include technologies with actual evidence.
Confidence levels: "high" = explicitly named in official source,
"medium" = indirect reliable signal, "low" = weak single mention.

Return ONLY a valid JSON object (no markdown, no code blocks, no variable assignment) with this structure:

{
    "company_name": "<company_name>",
    "website": "<url>",
    "research_date": "<YYYY-MM-DD>",
    "tech_stack": {
        "crm": [],
        "data_and_analytics": [],
        "backend_languages_frameworks": [],
        "frontend": [],
        "marketing_tech": [],
        "ecommerce_cms": [],
        "other": []
    }
}

Each array item should be a string like "Salesforce (high)" or "Snowflake (medium)".
Omit any category that has no findings — do not include empty arrays."""

TECHNOLOGY_RESEARCH_QUERIES = [
    '"{name}" technology software platform',
    '"{name}" technology partner announcement',
    '"{name}" site:zoominfo.com',
    '"{name}" site:linkedin.com jobs',
    '"{name}" Salesforce',
    '"{name}" Snowflake',
]

INTENT_SCORE_PROMPT = """Rate this prospect's likelihood of needing Salesforce/Snowflake/AI/data engineering services on a scale of 1-10.

Prospect: {prospect_name}, {designation} at {company_name}
Technology Research:
{technology_research}

Scoring guide:
- 8-10: Strong signals (active job postings for Salesforce/Snowflake/data roles, RFPs, digital transformation announcements)
- 5-7: Moderate signals (technology initiatives, some relevant hiring, growth indicators)
- 1-4: Weak signals (no tech signals, no relevant hiring, generic company info only)

Return ONLY a single integer from 1 to 10, nothing else."""

# ── Supabase client ────────────────────────────────────────────────────────

_supabase = None

def _get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    return _supabase


# ── Serper search ──────────────────────────────────────────────────────────

def _serper_search(query: str, num_results: int = 5) -> list[dict]:
    """Run a Serper search and return organic results."""
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": config.SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json().get("organic", [])
    except Exception:
        return []


def _results_to_text(results: list[dict]) -> str:
    parts = []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        parts.append(f"Title: {title}\nSnippet: {snippet}\nURL: {link}")
    return "\n\n".join(parts)


# ── Company research ───────────────────────────────────────────────────────

def _is_cache_stale(date_of_research) -> bool:
    """Return True if the cached research is older than CACHE_TTL_DAYS."""
    if date_of_research is None:
        return True
    if isinstance(date_of_research, str):
        dt = datetime.fromisoformat(date_of_research.replace("Z", "+00:00"))
    else:
        dt = date_of_research
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt) > timedelta(days=CACHE_TTL_DAYS)


def _run_technology_research(company_name: str, website: str) -> dict:
    """Run technology research using Serper + Claude synthesis.

    Runs 6 Serper queries, concatenates snippets, and asks Claude to
    populate a TECHNOLOGY_RESEARCH dict with tech_stack categories.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Run all 6 Serper queries and concatenate
    all_snippets = ""
    for query_template in TECHNOLOGY_RESEARCH_QUERIES:
        query = query_template.format(name=company_name)
        results = _serper_search(query, 5)
        all_snippets += f"\n\nQuery: {query}\n"
        all_snippets += _results_to_text(results)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=TECHNOLOGY_RESEARCH_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"company_name='{company_name}'\nwebsite='{website}'\n\n{all_snippets}",
        }],
    )
    text = response.content[0].text.strip() if response.content else ""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        result = json.loads(text)
        result["website"] = website
        return result
    except json.JSONDecodeError:
        return {"company_name": company_name, "website": website}


def _get_technology_research(company_name: str, website: str) -> dict:
    """Get technology research from cache or run fresh."""
    sb = _get_supabase()
    result = sb.table("Cache_Prospect_Company_Research").select("*").eq("Website", website).execute()
    if result.data:
        row = result.data[0]
        if not _is_cache_stale(row.get("Date_of_Research")):
            cr = row["Technology_research"]
            return cr if isinstance(cr, dict) else json.loads(cr)

    # Run fresh research
    research = _run_technology_research(company_name, website)
    sb.table("Cache_Prospect_Company_Research").upsert({
        "Website": website,
        "Technology_research": research,
        "Date_of_Research": datetime.now(timezone.utc).isoformat(),
    }).execute()
    return research




# ── Claude Web Search alternative ─────────────────────────────────────────

def _extract_websearch_text(response) -> str:
    """Extract the final text block from a Claude web search response."""
    for block in response.content:
        if hasattr(block, "type") and block.type == "text":
            return block.text.strip()
    return ""


def _parse_json_response(text: str, fallback: dict) -> dict:
    """Strip markdown fences and parse JSON, returning fallback on failure."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


WEBSEARCH_MODEL_HAIKU = "claude-haiku-4-5-20251001"
WEBSEARCH_MODEL_SONNET = "claude-sonnet-4-6"

# Pricing per million tokens (USD)
_MODEL_PRICING = {
    WEBSEARCH_MODEL_HAIKU: {"input": 1.0, "output": 5.0},
    WEBSEARCH_MODEL_SONNET: {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
}
_WEBSEARCH_COST_PER_SEARCH = 0.01  # $10 per 1000 searches


def _compute_cost(response, model: str) -> dict:
    """Compute cost from a Claude API response usage."""
    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    pricing = _MODEL_PRICING.get(model, {"input": 3.0, "output": 15.0})
    token_cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    # Count web searches from server_tool_use
    num_searches = 0
    for block in response.content:
        if hasattr(block, "type") and block.type == "server_tool_use":
            num_searches += 1
    search_cost = num_searches * _WEBSEARCH_COST_PER_SEARCH
    total = token_cost + search_cost
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_cost_usd": round(token_cost, 5),
        "web_searches": num_searches,
        "search_cost_usd": round(search_cost, 4),
        "total_cost_usd": round(total, 5),
        "model": model,
    }


def _run_company_research_websearch(company_name: str, website: str,
                                     model: str = WEBSEARCH_MODEL_HAIKU) -> dict:
    """Company research via Claude Web Search — test/comparison only."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    prompt = f"""You are a sales researcher. Search the web for "{company_name}" (website: {website}).

Find ONLY factual, verifiable information from the last 3 months about:
1. Technical initiatives (digital transformation, cloud migration, AI/automation, technology modernization)
2. Job openings (Salesforce, Snowflake, data engineering, custom development, other technical)
3. Additional context (company size, industry, awards, partnerships, recent news)

Rules:
- Only include findings that explicitly mention "{company_name}" by name
- Only include findings from the last 3 months — discard anything older or undated
- No generic industry articles, no speculation
- Only include keys where data was actually found — no empty lists, no null values

Return ONLY a valid JSON object (no markdown, no code blocks) with this structure (omit any key with no data):
{{
  "company_name": "{company_name}",
  "technical_initiatives": {{
    "digital_transformation": ["finding"],
    "technology_modernization": ["finding"],
    "cloud_migration": ["finding"],
    "ai_automation": ["finding"]
  }},
  "job_openings": {{
    "salesforce": [{{"title": "str", "date_posted": "str", "tools_mentioned": ["str"]}}],
    "snowflake": [...],
    "custom_development": [...],
    "data_engineering": [...],
    "other_technical": [...]
  }},
  "additional_context": ["finding"]
}}"""

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3,
        }],
    )
    text = _extract_websearch_text(response)
    cost = _compute_cost(response, model)
    return _parse_json_response(text, {"company_name": company_name}), cost


def _run_prospect_research_websearch(prospect_name: str, designation: str,
                                      company_name: str, city: str, country: str,
                                      model: str = WEBSEARCH_MODEL_HAIKU) -> dict:
    """Prospect research via Claude Web Search — test/comparison only."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    prompt = f"""You are a sales researcher. Search the web for "{prospect_name}" who is {designation} at {company_name}, based in {city}, {country}.

Search strategy:
1. First search LinkedIn for their profile
2. Then search for broader mentions (interviews, keynotes, podcasts, announcements)

Find ONLY factual, verifiable information. Rules:
- LinkedIn is the primary source; use broader results only to fill gaps
- Recent activity limited to last 3 months only
- previous_employers from LinkedIn career history only
- No speculation or unverified claims
- Only include keys where data was actually found — no empty lists, no null values

Return ONLY a valid JSON object (no markdown, no code blocks) with this structure (omit any key with no data):
{{
  "prospect_name": "{prospect_name}",
  "designation": "{designation}",
  "company_name": "{company_name}",
  "location": "{city}, {country}",
  "professional_background": {{
    "location_and_network": ["finding"],
    "educational_background": ["finding"],
    "role_scope": ["finding"]
  }},
  "strategic_focus_areas": {{
    "partnership_development": ["finding"],
    "sales_and_marketing": ["finding"],
    "domain_expertise": ["finding"]
  }},
  "recent_activity": {{
    "brand_and_initiatives": ["finding"],
    "thought_leadership": ["finding"]
  }},
  "previous_employers": ["Company A", "Company B"]
}}"""

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        tools=[{
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3,
        }],
    )
    text = _extract_websearch_text(response)
    cost = _compute_cost(response, model)
    return _parse_json_response(text, {
        "prospect_name": prospect_name,
        "designation": designation,
        "company_name": company_name,
        "location": f"{city}, {country}",
    }), cost


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_tech_names_from_dict(tech_research: dict) -> str:
    """Flatten TECHNOLOGY_RESEARCH tech_stack into comma-separated tech names.

    Strips confidence levels like "(high)" from each entry.
    Used to feed matchers which expect comma-separated input.
    """
    tech_stack = tech_research.get("tech_stack", {})
    names = []
    for category_techs in tech_stack.values():
        if isinstance(category_techs, list):
            for item in category_techs:
                # Strip confidence annotations like " (high)", " (medium)", " (low)"
                name = re.sub(r"\s*\((high|medium|low)\)\s*$", "", str(item), flags=re.IGNORECASE).strip()
                if name:
                    names.append(name)
    return ", ".join(names)


def _score_intent(prospect_name: str, designation: str,
                   company_name: str, technology_research: str) -> int:
    """Rate intent 1-10 via Claude."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    prompt = INTENT_SCORE_PROMPT.format(
        prospect_name=prospect_name,
        designation=designation,
        company_name=company_name,
        technology_research=technology_research,
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip() if response.content else ""
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 5


def _apply_emea_coding(country: str) -> str:
    """Return 'EMEA' if country is in Middle East or Africa, else return original."""
    if country and country.lower() in EMEA_COUNTRIES:
        return "EMEA"
    return country


# ── Main entry point ───────────────────────────────────────────────────────

def research_workbook(
    file_bytes: bytes,
    output_path: str,
    generate_intent_score: bool = True,
    max_case_studies: int = 5,
    max_client_matches: int = 5,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> None:
    """Run Pass 1 research pipeline on uploaded Excel workbook.

    Args:
        file_bytes: Raw bytes of the uploaded .xlsx file.
        output_path: Path to write data_output.xlsx.
        generate_intent_score: Whether to generate Intent_Score column.
        max_case_studies: Max matches for find_casestudy_matches.
        max_client_matches: Max matches for find_matches (must be >= 5).
        progress_callback: Optional fn(current, total, status_msg).
    """
    import io
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="prospects")

    # Load dates sheet if present
    try:
        dates_df = pd.read_excel(io.BytesIO(file_bytes), sheet_name="dates")
    except Exception:
        dates_df = pd.DataFrame()

    # Ensure generated columns exist
    for col in [
        "Case_Studies", "Industry_Client_References",
        "Suggested_Brand_Name_to_use", "Intent_Score",
        "Message_to_send", "Prospect_Technologies",
    ]:
        if col not in df.columns:
            df[col] = ""

    total = len(df)

    # Group by company for deduplication
    company_cache: dict[str, dict] = {}  # website -> technology_research

    for idx, row in df.iterrows():
        prospect_name = f"{row.get('First_Name', '')} {row.get('Last_Name', '')}".strip()
        designation = str(row.get("Designation", ""))
        company_name = str(row.get("Company_Name", ""))
        website = str(row.get("Website", ""))
        industry = str(row.get("Industry", ""))
        country = _apply_emea_coding(str(row.get("Country", "")))
        city = str(row.get("City", ""))
        state = str(row.get("State", ""))

        current = int(idx) + 1
        if progress_callback:
            progress_callback(current, total, f"Processing {prospect_name} ({company_name})...")

        # 1. Brand match
        brand_result = find_brand_match(prospect_industry=industry)
        df.at[idx, "Suggested_Brand_Name_to_use"] = brand_result["brand"]

        # 2. EMEA coding (already applied to local variable at row read; write to df)
        df.at[idx, "Country"] = country

        # 3. Technology research (cached per website)
        if website not in company_cache:
            if progress_callback:
                progress_callback(current, total, f"Researching {company_name}...")
            tech_research = _get_technology_research(company_name, website)
            company_cache[website] = tech_research
        else:
            tech_research = company_cache[website]

        # Store full TECHNOLOGY_RESEARCH dict as JSON in Prospect_Technologies
        df.at[idx, "Prospect_Technologies"] = json.dumps(tech_research)

        # Extract comma-separated tech names for matchers
        tech_names_csv = _extract_tech_names_from_dict(tech_research)

        # 4. Case study matching
        if progress_callback:
            progress_callback(current, total, f"Matching case studies for {prospect_name}...")
        try:
            cs_result = find_casestudy_matches(
                prospect_context="",
                prospect_industry=industry,
                prospect_technologies=tech_names_csv,
                prospect_country="",
                max_matches=max_case_studies,
            )
            df.at[idx, "Case_Studies"] = json.dumps([m.to_dict() if hasattr(m, "to_dict") else m for m in cs_result["matches"]])
        except Exception as e:
            df.at[idx, "Case_Studies"] = json.dumps({"error": str(e)})

        # 5. Client matching
        if progress_callback:
            progress_callback(current, total, f"Matching clients for {prospect_name}...")
        try:
            client_result = find_matches(
                prospect_industry=industry,
                prospect_technologies=tech_names_csv,
                prospect_country=country,
                max_matches=max(max_client_matches, 5),
            )
            df.at[idx, "Industry_Client_References"] = json.dumps([
                m.to_dict() if hasattr(m, "to_dict") else m for m in client_result["matches"]
            ])
        except Exception as e:
            df.at[idx, "Industry_Client_References"] = json.dumps({"error": str(e)})

        # 6. Intent scoring (optional)
        if generate_intent_score:
            if progress_callback:
                progress_callback(current, total, f"Scoring intent for {prospect_name}...")
            score = _score_intent(prospect_name, designation, company_name, json.dumps(tech_research))
            df.at[idx, "Intent_Score"] = score

    # Write output
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="prospects", index=False)
        if not dates_df.empty:
            dates_df.to_excel(writer, sheet_name="dates", index=False)

    if progress_callback:
        progress_callback(total, total, "Research complete.")
