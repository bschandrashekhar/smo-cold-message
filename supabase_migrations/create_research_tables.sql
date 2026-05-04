-- Run this in the Supabase SQL editor to create the research cache tables.
-- If you previously created Prospect_Company_Research / Prospect_Contact_Research, drop them first:
--   DROP TABLE IF EXISTS "Prospect_Company_Research";
--   DROP TABLE IF EXISTS "Prospect_Contact_Research";

-- Table 1: Company Research Cache
-- Keyed by Website (base URL). Upsert on re-run overwrites with latest research.
CREATE TABLE IF NOT EXISTS "Cache_Prospect_Company_Research" (
    "Website"          text        PRIMARY KEY,
    "Company_Research" jsonb       NOT NULL,
    "Date_of_Research" timestamptz NOT NULL DEFAULT now()
);

-- Table 2: Prospect Contact Research Cache
-- Keyed by Email. Upsert on re-run overwrites with latest research.
CREATE TABLE IF NOT EXISTS "Cache_Prospect_Contact_Research" (
    "Email"            text        PRIMARY KEY,
    "Contact_Research" jsonb       NOT NULL,
    "Date_of_Research" timestamptz NOT NULL DEFAULT now()
);
