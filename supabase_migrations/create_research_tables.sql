-- Run this in the Supabase SQL editor to create the research persistence tables.

-- Table 1: Company Research
-- Keyed by Website (base URL). Upsert on re-run overwrites with latest research.
CREATE TABLE IF NOT EXISTS "Prospect_Company_Research" (
    "Website"          text        PRIMARY KEY,
    "Company_Research" jsonb       NOT NULL,
    "Date_of_Research" timestamptz NOT NULL DEFAULT now()
);

-- Table 2: Prospect Contact Research
-- Keyed by Email. Upsert on re-run overwrites with latest research.
CREATE TABLE IF NOT EXISTS "Prospect_Contact_Research" (
    "Email"             text        PRIMARY KEY,
    "Prospect_Research" jsonb       NOT NULL,
    "Date_of_Research"  timestamptz NOT NULL DEFAULT now()
);
