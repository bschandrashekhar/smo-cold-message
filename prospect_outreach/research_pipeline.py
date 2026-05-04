"""Prospect Outreach — Pass 1 Research Pipeline.

For each prospect:
  1. Brand match via find_brand_match (industry embeddings)
  2. Company research via Serper (with Supabase cache, 3-month TTL)
  3. Prospect research via Serper (with Supabase cache, 3-month TTL)
  4. Research summary (5-6 bullet points)
  5. Case study matching via find_casestudy_matches
  6. Client matching via find_matches
  7. Intent scoring (optional)
  8. EMEA country coding
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

TECH_EXTRACTION_PROMPT = """From the following company research JSON, extract only the generic technology platform names mentioned.
Return a comma-separated list of only well-known, general-purpose platforms (e.g. Salesforce, Snowflake, Boomi, MuleSoft, AWS, Azure, GCP, SAP, ServiceNow, Workday, Tableau, Power BI, Agentforce, .NET, Python).
Do NOT include proprietary or company-specific product names.
Return ONLY the comma-separated list, nothing else. If none found, return empty string.

Research JSON:
{research_json}"""

RESEARCH_SUMMARY_PROMPT = """Based on the following company and prospect research, write a crisp research summary of exactly 5-6 bullet points.
Each bullet should be a key finding about the company's technology signals, initiatives, job openings, or the prospect's role and focus.
Keep each bullet concise (one sentence). Use plain text, no markdown formatting within bullets.

Company Research:
{company_research}

Prospect Research:
{prospect_research}

Return ONLY the bullet points, one per line, each starting with "- "."""

INTENT_SCORE_PROMPT = """Rate this prospect's likelihood of needing Salesforce/Snowflake/AI/data engineering services on a scale of 1-10.

Prospect: {prospect_name}, {designation} at {company_name}
Research Summary:
{research_summary}

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


def _run_company_research(company_name: str, website: str) -> dict:
    """Run company research using Serper per SKILL.md spec."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Step 1: Technical initiatives
    initiative_queries = [
        f'"{company_name}" digital transformation initiative',
        f'"{company_name}" technology modernization',
        f'"{company_name}" cloud migration AWS Azure GCP',
        f'"{company_name}" AI automation machine learning',
        f'"{company_name}" technology stack engineering',
    ]
    initiative_results = []
    for q in initiative_queries:
        initiative_results.extend(_serper_search(q, 3))

    # Step 2: Job openings (last 3 months)
    job_queries = [
        f'"{company_name}" jobs Salesforce',
        f'"{company_name}" jobs Snowflake',
        f'"{company_name}" jobs "data engineer" OR "data engineering"',
        f'"{company_name}" jobs "custom development" OR "software engineer" OR "full stack"',
        f'site:linkedin.com/jobs "{company_name}" Salesforce',
        f'site:linkedin.com/jobs "{company_name}" Snowflake',
    ]
    job_results = []
    for q in job_queries:
        job_results.extend(_serper_search(q, 3))

    # Step 3: Additional context
    context_results = _serper_search(f'"{company_name}" company overview size industry', 5)

    # Ask Claude to synthesize into COMPANY_RESEARCH dict
    synthesis_prompt = f"""You are a sales researcher. Based on the following search results about "{company_name}", build a COMPANY_RESEARCH Python dict.

Rules:
- Only include keys where data was actually found — no empty lists, no null values
- Only include findings that explicitly mention "{company_name}" by name
- For job_openings, only include postings from the last 3 months
- No generic industry articles, no speculation

INITIATIVE SEARCH RESULTS:
{_results_to_text(initiative_results)}

JOB SEARCH RESULTS:
{_results_to_text(job_results)}

CONTEXT SEARCH RESULTS:
{_results_to_text(context_results)}

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
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    text = response.content[0].text.strip()
    # Strip markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"company_name": company_name}


def _get_company_research(company_name: str, website: str) -> dict:
    """Get company research from cache or run fresh."""
    sb = _get_supabase()
    result = sb.table("Cache_Prospect_Company_Research").select("*").eq("Website", website).execute()
    if result.data:
        row = result.data[0]
        if not _is_cache_stale(row.get("Date_of_Research")):
            cr = row["Company_Research"]
            return cr if isinstance(cr, dict) else json.loads(cr)

    # Run fresh research
    research = _run_company_research(company_name, website)
    sb.table("Cache_Prospect_Company_Research").upsert({
        "Website": website,
        "Company_Research": research,
        "Date_of_Research": datetime.now(timezone.utc).isoformat(),
    }).execute()
    return research


# ── Prospect research ──────────────────────────────────────────────────────

def _run_prospect_research(prospect_name: str, designation: str,
                            company_name: str, city: str, country: str,
                            linkedin_url: str = "") -> dict:
    """Run prospect research using Serper per SKILL.md spec."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Step 1: LinkedIn-first search — use provided URL if available, else search by name
    if linkedin_url:
        linkedin_queries = [linkedin_url]
    else:
        linkedin_queries = [
            f'site:linkedin.com/in "{prospect_name}" "{company_name}"',
            f'site:linkedin.com "{prospect_name}" "{designation}" "{company_name}"',
        ]
    linkedin_results = []
    for q in linkedin_queries:
        linkedin_results.extend(_serper_search(q, 5))

    # Step 2: Broaden if insufficient
    broader_results = []
    if len(linkedin_results) < 3:
        broader_queries = [
            f'"{prospect_name}" "{designation}" "{company_name}"',
            f'"{prospect_name}" "{company_name}" interview OR keynote OR podcast',
            f'"{prospect_name}" "{company_name}" announcement OR partnership OR initiative',
        ]
        for q in broader_queries:
            broader_results.extend(_serper_search(q, 3))

    synthesis_prompt = f"""You are a sales researcher. Based on the following search results about "{prospect_name}" ({designation} at {company_name}), build a PROSPECT_RESEARCH Python dict.

Rules:
- Only include keys where data was actually found — no empty lists, no null values
- LinkedIn is the primary source; use broader results only to fill gaps
- Recent activity limited to last 6 months only
- previous_employers from LinkedIn career history only
- No speculation or unverified claims

LINKEDIN SEARCH RESULTS:
{_results_to_text(linkedin_results)}

BROADER SEARCH RESULTS:
{_results_to_text(broader_results)}

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
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
    text = response.content[0].text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "prospect_name": prospect_name,
            "designation": designation,
            "company_name": company_name,
            "location": f"{city}, {country}",
        }


def _get_prospect_research(prospect_name: str, designation: str,
                            company_name: str, city: str, country: str, email: str,
                            linkedin_url: str = "") -> dict:
    """Get prospect research from cache or run fresh."""
    sb = _get_supabase()
    result = sb.table("Cache_Prospect_Contact_Research").select("*").eq("Email", email).execute()
    if result.data:
        row = result.data[0]
        if not _is_cache_stale(row.get("Date_of_Research")):
            cr = row["Prospect_Research"]
            return cr if isinstance(cr, dict) else json.loads(cr)

    research = _run_prospect_research(prospect_name, designation, company_name, city, country, linkedin_url)
    sb.table("Cache_Prospect_Contact_Research").upsert({
        "Email": email,
        "Prospect_Research": research,
        "Date_of_Research": datetime.now(timezone.utc).isoformat(),
    }).execute()
    return research


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_technologies(company_research: dict) -> str:
    """Ask Claude to extract generic tech platform names from company research."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    prompt = TECH_EXTRACTION_PROMPT.format(research_json=json.dumps(company_research))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip() if response.content else ""


def _build_research_summary(company_research: dict, prospect_research: dict) -> str:
    """Generate 5-6 bullet point summary from research dicts."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    prompt = RESEARCH_SUMMARY_PROMPT.format(
        company_research=json.dumps(company_research),
        prospect_research=json.dumps(prospect_research),
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip() if response.content else ""


def _score_intent(prospect_name: str, designation: str,
                   company_name: str, research_summary: str) -> int:
    """Rate intent 1-10 via Claude."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    prompt = INTENT_SCORE_PROMPT.format(
        prospect_name=prospect_name,
        designation=designation,
        company_name=company_name,
        research_summary=research_summary,
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
        "Research_Summary", "Research_Summary_Compressed",
        "Case_Studies", "Industry_Client_References",
        "Suggested_Brand_Name_to_use", "Intent_Score",
        "Message_to_send", "Prospect_Technologies",
    ]:
        if col not in df.columns:
            df[col] = ""

    total = len(df)

    # Group by company for deduplication
    company_cache: dict[str, dict] = {}  # website -> company_research

    for idx, row in df.iterrows():
        prospect_name = f"{row.get('First_Name', '')} {row.get('Last_Name', '')}".strip()
        designation = str(row.get("Designation", ""))
        company_name = str(row.get("Company_Name", ""))
        email = str(row.get("Email", ""))
        website = str(row.get("Website", ""))
        _li = row.get("LinkedIn", "") if "LinkedIn" in df.columns else ""
        linkedin_url = "" if not _li or pd.isna(_li) else str(_li).strip()
        industry = str(row.get("Industry", ""))
        country = str(row.get("Country", ""))
        city = str(row.get("City", ""))
        state = str(row.get("State", ""))

        current = int(idx) + 1
        if progress_callback:
            progress_callback(current, total, f"Processing {prospect_name} ({company_name})...")

        # 1. Brand match
        brand_result = find_brand_match(prospect_industry=industry)
        df.at[idx, "Suggested_Brand_Name_to_use"] = brand_result["brand"]

        # 2. Company research (cached per website)
        if website not in company_cache:
            if progress_callback:
                progress_callback(current, total, f"Researching {company_name}...")
            company_research = _get_company_research(company_name, website)
            company_cache[website] = company_research
        else:
            company_research = company_cache[website]

        # 3. Extract technologies
        prospect_technologies = _extract_technologies(company_research)
        df.at[idx, "Prospect_Technologies"] = prospect_technologies

        # 4. Prospect research (cached per email)
        if progress_callback:
            progress_callback(current, total, f"Researching {prospect_name}...")
        prospect_research = _get_prospect_research(
            prospect_name, designation, company_name, city, country, email, linkedin_url
        )

        # 5. Research summary
        research_summary = _build_research_summary(company_research, prospect_research)
        df.at[idx, "Research_Summary"] = research_summary

        # 6. Case study matching
        if progress_callback:
            progress_callback(current, total, f"Matching case studies for {prospect_name}...")
        try:
            cs_result = find_casestudy_matches(
                prospect_context=research_summary,
                prospect_industry=industry,
                prospect_technologies=prospect_technologies,
                prospect_country="",
                max_matches=max_case_studies,
            )
            df.at[idx, "Case_Studies"] = json.dumps([m.to_dict() if hasattr(m, "to_dict") else m for m in cs_result["matches"]])
        except Exception as e:
            df.at[idx, "Case_Studies"] = json.dumps({"error": str(e)})

        # 7. Client matching
        if progress_callback:
            progress_callback(current, total, f"Matching clients for {prospect_name}...")
        try:
            client_result = find_matches(
                prospect_industry=industry,
                prospect_technologies=prospect_technologies,
                prospect_country=country,
                max_matches=max(max_client_matches, 5),
            )
            df.at[idx, "Industry_Client_References"] = json.dumps([
                m.to_dict() if hasattr(m, "to_dict") else m for m in client_result["matches"]
            ])
        except Exception as e:
            df.at[idx, "Industry_Client_References"] = json.dumps({"error": str(e)})

        # 8. Intent scoring (optional)
        if generate_intent_score:
            if progress_callback:
                progress_callback(current, total, f"Scoring intent for {prospect_name}...")
            score = _score_intent(prospect_name, designation, company_name, research_summary)
            df.at[idx, "Intent_Score"] = score

        # 9. EMEA coding
        df.at[idx, "Country"] = _apply_emea_coding(country)

    # Write output
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="prospects", index=False)
        if not dates_df.empty:
            dates_df.to_excel(writer, sheet_name="dates", index=False)

    if progress_callback:
        progress_callback(total, total, "Research complete.")
