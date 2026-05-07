-- Migration: Add Technology_Research column to Cache_Prospect_Company_Research
-- Company_Research column is kept (not renamed).
--
-- Already applied manually on 2026-05-07.

-- 1. Clear all existing rows (old schema)
TRUNCATE TABLE "Cache_Prospect_Company_Research";

-- 2. Add new column (Company_Research kept as-is)
ALTER TABLE "Cache_Prospect_Company_Research"
    ADD COLUMN IF NOT EXISTS "Technology_Research" jsonb;
