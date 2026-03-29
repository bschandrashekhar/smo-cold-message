"""Prospect research pipeline: Claude web search per company/prospect,
case study matching, brand selection, and intent scoring.

Supports two search providers:
  - "claude" (default): Claude with web_search tool
  - "serper": Serper.dev Google Search API → Claude synthesis
"""

import json
import time
import requests
import pandas as pd
import anthropic
from pathlib import Path
from typing import Dict, List, Callable, Optional

from . import config, brand_knowledge, case_study_match

MAX_RETRIES = 3
RETRY_DELAY = 65


def _call_claude_with_retry(client, **kwargs):
    """Call Claude API with automatic retry on rate limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s before retry {attempt + 2}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise


def validate_prospects_sheet(df: pd.DataFrame) -> bool:
    required = [
        "Prospect Name",
        "Designation",
        "Company Name",
        "Website",
        "Location of Prospect",
    ]
    return all(col in df.columns for col in required)


# ---------------------------------------------------------------------------
# Step 2a: Company-level research (deduplicated)
# ---------------------------------------------------------------------------

def _research_company(company_name: str, website: str) -> str:
    """Use Claude with web search to research a company.

    Follows company-research-SKILL: targeted searches for technical initiatives and job openings.
    Returns JSON-serialized COMPANY_RESEARCH dict.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    prompt = f"""Research the company "{company_name}" (website: {website}) using web search.

Search for:
1. Technical initiatives: digital transformation, technology modernization, cloud migration (AWS/Azure/GCP), AI/automation projects, technology stack
2. Job openings (last 6 months only): Salesforce, Snowflake, data engineering, custom development/software engineering
3. Additional context: company size, industry, key partnerships, recent news

Then return a JSON object with this exact shape (omit any key where no data was found — no empty lists or null values):

{{
  "company_name": "{company_name}",
  "technical_initiatives": {{
    "digital_transformation": ["finding 1", "finding 2"],
    "technology_modernization": ["finding 1", "finding 2"],
    "cloud_migration": ["finding 1", "finding 2"],
    "ai_automation": ["finding 1", "finding 2"]
  }},
  "job_openings": {{
    "salesforce": [{{"title": "str", "date_posted": "str", "tools_mentioned": ["str"], "url": "str"}}],
    "snowflake": [],
    "custom_development": [],
    "data_engineering": [],
    "other_technical": []
  }},
  "additional_context": ["finding 1", "finding 2"]
}}

Rules:
- Omit any key with no findings. No empty lists, no null values.
- For job_openings, only include postings from the last 6 months.
- Prefer official company blog, press releases, and reputable news over speculation.
- Return ONLY the JSON object, no other text."""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_final_text(response)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return text


def _extract_final_text(response) -> str:
    """Extract all text blocks after the last web search result.

    Web search responses have many content blocks: tool calls, search results,
    and text. Text blocks before/between searches are narration ("Let me search...").
    Text blocks after the last search result are the actual summary — often split
    across many small blocks (one per bullet point).
    """
    # Find the index of the last web_search_tool_result block
    last_search_idx = -1
    for i, block in enumerate(response.content):
        if block.type == "web_search_tool_result":
            last_search_idx = i

    # Collect all text blocks after the last search result
    text_parts = []
    for i, block in enumerate(response.content):
        if block.type == "text" and i > last_search_idx:
            text_parts.append(block.text)

    return "".join(text_parts).strip()


# ---------------------------------------------------------------------------
# Serper.dev search provider
# ---------------------------------------------------------------------------

SERPER_CACHE_PATH = config.DATA_DIR / "serper_research_cache.json"


def _load_serper_cache() -> dict:
    """Load existing Serper research cache."""
    if SERPER_CACHE_PATH.exists():
        with open(SERPER_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_serper_cache(cache: dict) -> None:
    """Save Serper research cache."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SERPER_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _serper_search(query: str, num_results: int = 10) -> List[Dict]:
    """Call Serper.dev Google Search API and return organic results."""
    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": config.SERPER_API_KEY,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": num_results},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("organic", [])


def _serper_search_results_to_text(results: List[Dict]) -> str:
    """Format Serper search results into text for Claude synthesis."""
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        link = r.get("link", "")
        lines.append(f"{i}. {title}\n   {snippet}\n   URL: {link}")
    return "\n\n".join(lines)


def _research_company_serper(company_name: str, website: str) -> str:
    """Research a company using Serper Google Search + Claude synthesis.

    Follows company-research-SKILL: 5 technical initiative queries + 6 job queries.
    Returns JSON-serialized COMPANY_RESEARCH dict.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    tech_queries = [
        f'"{company_name}" digital transformation initiative',
        f'"{company_name}" technology modernization',
        f'"{company_name}" cloud migration AWS Azure GCP',
        f'"{company_name}" AI automation machine learning',
        f'"{company_name}" technology stack engineering',
        f'site:{website} technology innovation',
    ]

    job_queries = [
        f'"{company_name}" jobs Salesforce',
        f'"{company_name}" jobs Snowflake',
        f'"{company_name}" jobs "data engineer" OR "data engineering"',
        f'"{company_name}" jobs "custom development" OR "software engineer" OR "full stack"',
        f'site:linkedin.com/jobs "{company_name}" Salesforce',
        f'site:linkedin.com/jobs "{company_name}" Snowflake',
    ]

    all_results_text = []
    for query in tech_queries + job_queries:
        results = _serper_search(query, num_results=10)
        if results:
            all_results_text.append(f"Search: {query}\n{_serper_search_results_to_text(results)}")

    search_context = "\n\n---\n\n".join(all_results_text) if all_results_text else "No search results found."

    prompt = f"""Based on the following search results about "{company_name}", produce a structured JSON research summary.

SEARCH RESULTS:
{search_context}

Return a JSON object with this exact shape (omit any key where no data was found — no empty lists or null values):

{{
  "company_name": "{company_name}",
  "technical_initiatives": {{
    "digital_transformation": ["finding 1", "finding 2"],
    "technology_modernization": ["finding 1", "finding 2"],
    "cloud_migration": ["finding 1", "finding 2"],
    "ai_automation": ["finding 1", "finding 2"]
  }},
  "job_openings": {{
    "salesforce": [{{"title": "str", "date_posted": "str", "tools_mentioned": ["str"], "url": "str"}}],
    "snowflake": [],
    "custom_development": [],
    "data_engineering": [],
    "other_technical": []
  }},
  "additional_context": ["finding 1", "finding 2"]
}}

Rules:
- Omit any key with no findings. No empty lists, no null values.
- For job_openings, only include postings from the last 6 months.
- Prefer official company blog, press releases, and reputable news over speculation.
- Return ONLY the JSON object, no other text."""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        json.loads(text)
    except json.JSONDecodeError:
        pass  # return as-is if not valid JSON; downstream handles text fine

    return text


def run_company_research(company_name: str, website: str, provider: str = "serper") -> str:
    """Entry point for company-research skill."""
    if provider == "serper":
        return _research_company_serper(company_name, website)
    return _research_company(company_name, website)


def run_prospect_research(prospect_name: str, designation: str, company_name: str, company_research: str, provider: str = "serper") -> str:
    """Entry point for prospect-research skill."""
    if provider == "serper":
        return _research_prospect_serper(prospect_name, designation, company_name, company_research)
    return _research_prospect(prospect_name, designation, company_name, company_research)


def run_skill(skill_name: str, **kwargs) -> str:
    """Generic skill dispatcher for .claude/skills integration."""
    if skill_name == "company-research":
        return run_company_research(
            company_name=kwargs.get("company_name", ""),
            website=kwargs.get("url", ""),
            provider=kwargs.get("provider", "serper"),
        )
    if skill_name == "prospect-research":
        return run_prospect_research(
            prospect_name=kwargs.get("prospect_name", ""),
            designation=kwargs.get("designation", ""),
            company_name=kwargs.get("company_name", ""),
            company_research=kwargs.get("company_research", ""),
            provider=kwargs.get("provider", "serper"),
        )
    raise ValueError(f"Unknown skill: {skill_name}")


def _research_prospect_serper(prospect_name: str, designation: str, company_name: str, company_research: str) -> str:
    """Research an individual prospect using Serper Google Search + Claude synthesis.

    Follows prospect-research-SKILL: LinkedIn-first, broaden only if insufficient.
    Returns JSON-serialized PROSPECT_RESEARCH dict.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    # Step 1: LinkedIn-first queries
    linkedin_queries = [
        f'site:linkedin.com/in "{prospect_name}" "{company_name}"',
        f'site:linkedin.com "{prospect_name}" "{designation}" "{company_name}"',
        f'site:linkedin.com/posts "{prospect_name}" "{company_name}"',
    ]

    all_results_text = []
    for query in linkedin_queries:
        results = _serper_search(query, num_results=10)
        if results:
            all_results_text.append(f"Search: {query}\n{_serper_search_results_to_text(results)}")

    # Step 2: Broaden if LinkedIn yields insufficient data
    if len(all_results_text) < 2:
        broader_queries = [
            f'"{prospect_name}" "{designation}" "{company_name}"',
            f'"{prospect_name}" "{company_name}" interview OR keynote OR podcast',
            f'"{prospect_name}" "{company_name}" announcement OR partnership OR initiative',
        ]
        for query in broader_queries:
            results = _serper_search(query, num_results=10)
            if results:
                all_results_text.append(f"Search: {query}\n{_serper_search_results_to_text(results)}")

    search_context = "\n\n---\n\n".join(all_results_text) if all_results_text else "No search results found."

    prompt = f"""Based on the following search results, research "{prospect_name}", who is {designation} at {company_name}.

SEARCH RESULTS:
{search_context}

COMPANY CONTEXT:
{company_research[:2000]}

Return a JSON object with this exact shape (omit any key where no data was found — no empty lists or null values):

{{
  "prospect_name": "{prospect_name}",
  "designation": "{designation}",
  "company_name": "{company_name}",
  "location": "City, Country",
  "professional_background": {{
    "location_and_network": ["finding 1"],
    "educational_background": ["finding 1"],
    "role_scope": ["finding 1"]
  }},
  "strategic_focus_areas": {{
    "partnership_development": ["finding 1"],
    "sales_and_marketing": ["finding 1"],
    "domain_expertise": ["finding 1"]
  }},
  "recent_activity": {{
    "brand_and_initiatives": ["finding 1"],
    "thought_leadership": ["finding 1"]
  }},
  "previous_employers": ["Company A", "Company B"]
}}

Rules:
- Always include prospect_name, designation, company_name, location (use "Unknown" if not found).
- Omit any nested key with no findings. No empty lists, no null values.
- recent_activity: last 6 months only.
- previous_employers: flat list from LinkedIn career history, exclude current company.
- Return ONLY the JSON object, no other text."""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        json.loads(text)
    except json.JSONDecodeError:
        pass  # return as-is if not valid JSON; downstream handles text fine

    return text


# ---------------------------------------------------------------------------
# Step 2b: Prospect-level personalization
# ---------------------------------------------------------------------------

def _research_prospect(prospect_name: str, designation: str, company_name: str, company_research: str) -> str:
    """Use Claude with web search to research an individual prospect.

    Follows prospect-research-SKILL: LinkedIn-first approach, structured output.
    Returns JSON-serialized PROSPECT_RESEARCH dict.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    prompt = f"""Research "{prospect_name}", who is {designation} at {company_name}.

Search LinkedIn first (site:linkedin.com/in, site:linkedin.com/posts). Only broaden to other sources if LinkedIn yields insufficient data.

COMPANY CONTEXT:
{company_research[:2000]}

Return a JSON object with this exact shape (omit any key where no data was found — no empty lists or null values):

{{
  "prospect_name": "{prospect_name}",
  "designation": "{designation}",
  "company_name": "{company_name}",
  "location": "City, Country",
  "professional_background": {{
    "location_and_network": ["finding 1"],
    "educational_background": ["finding 1"],
    "role_scope": ["finding 1"]
  }},
  "strategic_focus_areas": {{
    "partnership_development": ["finding 1"],
    "sales_and_marketing": ["finding 1"],
    "domain_expertise": ["finding 1"]
  }},
  "recent_activity": {{
    "brand_and_initiatives": ["finding 1"],
    "thought_leadership": ["finding 1"]
  }},
  "previous_employers": ["Company A", "Company B"]
}}

Rules:
- Always include prospect_name, designation, company_name, location (use "Unknown" if not found).
- Omit any nested key with no findings. No empty lists, no null values.
- recent_activity: last 6 months only.
- previous_employers: flat list from LinkedIn career history, exclude current company.
- Return ONLY the JSON object, no other text."""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_final_text(response)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return text


# ---------------------------------------------------------------------------
# Step 3: Matching, brand selection, intent scoring
# ---------------------------------------------------------------------------

BFSI_KEYWORDS = [
    "bank", "banking", "credit union", "mutual bank", "lending", "loan",
    "mortgage", "finance", "financial", "fintech", "insurance", "insurer",
    "underwriting", "wealth management", "investment", "capital markets",
    "payments", "neobank",
]


INDUSTRY_VERTICALS = {
    "banking": ["bank", "banking", "credit union", "mutual bank", "neobank", "retail banking"],
    "lending": ["lending", "loan", "mortgage", "credit"],
    "financial services": ["finance", "financial", "fintech", "wealth management", "investment", "capital markets"],
    "insurance": ["insurance", "insurer", "underwriting"],
    "payments": ["payments", "payment processing"],
    "healthcare": ["healthcare", "health", "medical", "hospital", "clinical"],
    "nonprofit": ["nonprofit", "non-profit", "charity", "ngo", "foundation", "not-for-profit"],
    "real estate": ["real estate", "property", "construction"],
    "retail": ["retail", "ecommerce", "e-commerce"],
    "technology": ["saas", "software", "tech company", "platform"],
    "education": ["education", "university", "edtech"],
    "manufacturing": ["manufacturing", "industrial"],
}


def _detect_industry(company_research: str) -> str:
    """Detect the prospect company's industry from research text."""
    research_lower = company_research.lower()
    for industry, keywords in INDUSTRY_VERTICALS.items():
        for keyword in keywords:
            if keyword in research_lower:
                return industry
    return ""


def _select_brand(company_research: str) -> str:
    """Select brand based on industry: BFSI → LendingLogik, else → CloudChillies."""
    research_lower = company_research.lower()
    for keyword in BFSI_KEYWORDS:
        if keyword in research_lower:
            return "LendingLogik"
    return "CloudChillies"


def _score_intent(
    prospect_name: str,
    designation: str,
    company_name: str,
    company_research: str,
    prospect_research: str,
) -> int:
    """Use Claude to assign an intent score based on research signals."""
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    prompt = f"""Score this prospect's likelihood of needing Salesforce/Snowflake/AI/data engineering services.

PROSPECT: {prospect_name}, {designation} at {company_name}

COMPANY RESEARCH:
{company_research[:1500]}

PROSPECT RESEARCH:
{prospect_research[:500]}

Intent scoring criteria:
- 8-10: Strong signals (active job postings for relevant tech, recent RFPs, explicit digital transformation announcements)
- 5-7: Moderate signals (general growth indicators, industry trends, some tech adoption)
- 1-4: Weak signals (no specific indicators found, generic company)

Respond with ONLY a JSON object:
{{
  "intent_score": <number 1-10>,
  "intent_reasoning": "1-sentence explanation of the score"
}}"""

    response = _call_claude_with_retry(
        client,
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(text)
        score = int(result.get("intent_score", 5))
        return max(1, min(10, score))
    except (json.JSONDecodeError, ValueError):
        return 5


# ---------------------------------------------------------------------------
# Main pipeline: research_workbook
# ---------------------------------------------------------------------------

BATCH_SIZE = 5
BATCH_DELAY = 3  # seconds between batches
CLAUDE_WEB_SEARCH_DELAY = 60  # seconds after every Claude web search call (company + prospect) to stay under 30K TPM


def run_research(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    search_provider: str = "claude",
) -> pd.DataFrame:
    """Run the full research pipeline on a prospects DataFrame.

    Args:
        df: DataFrame with prospect data.
        progress_callback: Optional callback(current, total, status_text) for progress updates.
        search_provider: "claude" for Claude web search, "serper" for Serper Google Search.
    """
    use_serper = search_provider == "serper"
    research_company_fn = _research_company_serper if use_serper else _research_company
    research_prospect_fn = _research_prospect_serper if use_serper else _research_prospect

    # Load Serper cache if using Serper
    serper_cache = _load_serper_cache() if use_serper else {}

    df = df.copy()

    # Ensure output columns exist
    for col in ["Research Summary", "Case Studies", "Industry References", "Suggested Brand Name to use", "Intent Score"]:
        if col not in df.columns:
            df[col] = ""

    # Ensure brand profiles exist (needed later for message generation)
    brand_knowledge.ensure_brand_profiles()

    # Group prospects by company for deduplication
    company_groups: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        company = str(row.get("Company Name", "")).strip()
        if company not in company_groups:
            company_groups[company] = []
        company_groups[company].append(idx)

    total = len(df)
    processed = 0

    # Research company-by-company
    company_research_cache: Dict[str, str] = {}
    company_case_studies_cache: Dict[str, List[Dict]] = {}

    for company, indices in company_groups.items():
        # Company-level research (done once per company)
        website = str(df.at[indices[0], "Website"]).strip()
        if progress_callback:
            progress_callback(processed, total, f"Researching company: {company} [{'Serper' if use_serper else 'Claude'}]")

        company_research = research_company_fn(company, website)
        company_research_cache[company] = company_research

        # Proactive TPM throttle guard: Claude web search calls are token-heavy (~10-15K tokens each).
        # Wait 60s after every web search call (company + prospect) to stay under 30K TPM.
        if not use_serper:
            if progress_callback:
                progress_callback(processed, total, f"Throttle pause 60s after company research (TPM guard)...")
            time.sleep(CLAUDE_WEB_SEARCH_DELAY)

        # Case study matching for this company (done once, shared across all prospects)
        case_studies = case_study_match.match_case_studies(company_research)
        company_case_studies_cache[company] = case_studies
        case_studies_json = json.dumps(case_studies)

        # Industry reference clients (same vertical, done once per company)
        # Use Country column for geography-based sorting
        company_country = str(df.loc[indices[0]].get("Country", "")).strip()
        industry = _detect_industry(company_research)
        industry_refs = []
        if industry:
            industry_refs = case_study_match.find_industry_references(
                industry,
                prospect_geography=company_country,
                company_name=company,
            )
        industry_refs_json = json.dumps(industry_refs)

        # Brand selection (done once per company, same brand for all prospects)
        brand = _select_brand(company_research)

        # Per-prospect research within this company
        for idx in indices:
            row = df.loc[idx]
            prospect_name = str(row.get("Prospect Name", "")).strip()
            designation = str(row.get("Designation", "")).strip()

            if progress_callback:
                progress_callback(processed, total, f"Researching: {prospect_name} ({designation})")

            # Prospect-level research
            prospect_research_text = research_prospect_fn(
                prospect_name, designation, company, company_research
            )

            # Proactive TPM throttle guard after prospect research (Claude web search only)
            if not use_serper:
                if progress_callback:
                    progress_callback(processed, total, f"Throttle pause 60s after prospect research (TPM guard)...")
                time.sleep(CLAUDE_WEB_SEARCH_DELAY)

            # Build research summary (company + prospect findings)
            summary_parts = []
            summary_parts.append(f"Company: {company_research}")
            summary_parts.append(f"Prospect: {prospect_research_text}")
            if case_studies:
                cs_brief = "; ".join(
                    f"{cs['use_case']} ({cs['industry']})" for cs in case_studies[:3]
                )
                summary_parts.append(f"Relevant case studies: {cs_brief}")
            research_summary = "\n".join(summary_parts)

            # Intent scoring (Claude)
            intent_score = _score_intent(
                prospect_name, designation, company,
                company_research, prospect_research_text,
            )

            df.at[idx, "Research Summary"] = research_summary
            df.at[idx, "Case Studies"] = case_studies_json
            df.at[idx, "Industry References"] = industry_refs_json
            df.at[idx, "Suggested Brand Name to use"] = brand
            df.at[idx, "Intent Score"] = intent_score

            # Save to Serper cache
            if use_serper:
                cache_key = f"{prospect_name}|{company}"
                serper_cache[cache_key] = {
                    "prospect_name": prospect_name,
                    "designation": designation,
                    "company_name": company,
                    "website": website,
                    "location": str(row.get("Location of Prospect", "")).strip(),
                    "company_research": company_research,
                    "prospect_research": prospect_research_text,
                    "research_summary": research_summary,
                }
                _save_serper_cache(serper_cache)

            processed += 1
            if progress_callback:
                progress_callback(processed, total, f"Completed: {prospect_name}")

            # Rate limiting within batches
            if processed % BATCH_SIZE == 0:
                time.sleep(BATCH_DELAY)

    return df


def research_workbook(
    input_path: str,
    output_path: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    search_provider: str = "claude",
) -> None:
    """Load an Excel file, run research, and save results."""
    xls = pd.ExcelFile(input_path)
    if "prospects" not in xls.sheet_names:
        raise ValueError("Workbook must contain a 'prospects' sheet")

    df = pd.read_excel(xls, sheet_name="prospects")

    if not validate_prospects_sheet(df):
        raise ValueError("prospects sheet is missing required columns")

    result = run_research(df, progress_callback=progress_callback, search_provider=search_provider)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="prospects", index=False)
        if "dates" in xls.sheet_names:
            dates_df = pd.read_excel(xls, sheet_name="dates")
            dates_df.to_excel(writer, sheet_name="dates", index=False)
