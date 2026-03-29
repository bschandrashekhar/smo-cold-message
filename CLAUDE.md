# SMO Cold Message - Project Instructions

## Project Overview
Prospect Outreach Generator: Two-pass Streamlit app that researches prospects, matches case studies, and generates personalized cold outreach messages.

## Key References
- **Requirement Specs**: `requirements/requirement_specs.txt` — full pipeline spec, column definitions, matching logic, message style rules
- **Company Research Skill**: `.claude/skills/company-research/SKILL.md` — search queries, output schema, execution steps
- **Prospect Research Skill**: `.claude/skills/prospect-research/SKILL.md` — LinkedIn-first strategy, output schema, broadening logic
- **Brand Profiles**: `data/brand_lendinglogik.json`, `data/brand_cloudchillies.json`
- **Supabase Tables**: `case_studies` (vectorized), `client_references` (curated client list with industry/geography)

## Search Engine Default
This project uses **Serper** (Google Search API via serper.dev) as the default search provider for all research steps (company and prospect). Claude Web Search is available as a fallback provider.
