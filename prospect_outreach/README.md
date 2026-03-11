# Prospect Outreach Generator

This repository contains a Streamlit application and supporting modules for
researching prospects, matching them to case studies, and generating
personalized outreach messages. The project was described in `requirements/requirements.txt`.

## Setup

1. **Create a Python environment** and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate      # or `venv\Scripts\activate` on Windows
   pip install -r requirements/requirements.txt
   ```

2. **Set environment variables** by creating a `.env` file in the project
   root (outside this package):

   ```ini
   ANTHROPIC_API_KEY=...
   VOYAGE_API_KEY=...
   SUPABASE_URL=...
   SUPABASE_SERVICE_KEY=...
   ```

3. **Run the Streamlit app**:

   ```bash
   streamlit run prospect_outreach/app.py
   ```

## Usage

- **Tab 1 (Research)**: Upload `data.xlsx` with `prospects` and `dates`
  sheets. Press **Run Research** to fill in research summaries and brand
  recommendations. Download `data_output.xlsx`.

- **Offline Review**: Open the downloaded workbook, adjust research results,
  clear rows you want to skip, then save it.

- **Tab 2 (Generate Messages)**: Upload the reviewed file and press
  **Generate Messages**. Download the resulting `data_final.xlsx` with
  personalized outreach copy.

- **Tab 3 (Settings)**: Check and refresh brand knowledge profiles.

## Modules

- `config.py`: environment variable and path configuration.
- `brand_knowledge.py`: handles loading/refreshing brand profile JSONs.
- `prospect_research.py`: stubbed research pipeline operating on Excel
  workbooks.
- `case_study_match.py`: placeholder for Supabase/Voyage AI integration.
- `message_generator.py`: constructs basic 3-paragraph messages.
- `app.py`: Streamlit front end.

## Notes

The current implementation uses placeholder logic for research, case study
matching, and message generation. Replace the stubs with real AI
integration as needed.
