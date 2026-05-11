# SMO Cold Message - Project Instructions

## Project Overview
Prospect Outreach Generator: Two-pass Streamlit app that researches prospects, matches case studies, and generates personalized cold outreach messages.

## Key References
- **Requirement Specs**: `requirements/message_gen_requirement_specs.txt` — full pipeline spec, column definitions, matching logic, message style rules
- **Brand Profiles**: `data/brand_lendinglogik.json`, `data/brand_cloudchillies.json`
- **Supabase Tables**: `case_studies` (vectorized), `client_references` (curated client list with industry/geography)

## Search Engine Default
This project uses ONLY **Serper** (Google Search API via serper.dev) as the default search provider for all research steps (company and prospect). 

## Code Investigation Rule
Before answering any question about where a column, variable, or function is used, always search the **full codebase** (all .py files, all subdirectories) first. Never answer based on a single file lookup.

Always explicitly state at the start of every answer whether a full codebase search was performed or not, regardless of what was asked.
