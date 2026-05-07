-- Migration: Rename Company_Research column to Technology_research
-- and clear all existing cache entries (schema change).
--
-- Run in Supabase SQL Editor.

-- 1. Clear all existing rows (old COMPANY_RESEARCH schema)
TRUNCATE TABLE "Cache_Prospect_Company_Research";

-- 2. Rename column
ALTER TABLE "Cache_Prospect_Company_Research"
    RENAME COLUMN "Company_Research" TO "Technology_research";
