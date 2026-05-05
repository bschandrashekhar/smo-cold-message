-- Supabase RPC function: match_casestudy_technologies
-- Performs vector similarity search on client_case_studies_technology_mapping.
-- casestudy_id is uuid (not bigint) — recreate if the column type was changed.
--
-- Run in Supabase SQL Editor if you get error:
--   "structure of query does not match function result type"
--   "Returned type uuid does not match expected type bigint in column 2"

DROP FUNCTION IF EXISTS match_casestudy_technologies(vector, int, float);

CREATE OR REPLACE FUNCTION match_casestudy_technologies(
    query_embedding vector(1024),
    match_count     int   DEFAULT 40,
    match_threshold float DEFAULT 0.3
)
RETURNS TABLE (
    casestudy_id        uuid,
    casestudy_technology text,
    similarity          float
)
LANGUAGE sql STABLE
AS $$
    SELECT
        m.casestudy_id,
        m.casestudy_technology,
        1 - (m.casestudy_technology_embedding <=> query_embedding) AS similarity
    FROM client_case_studies_technology_mapping m
    WHERE 1 - (m.casestudy_technology_embedding <=> query_embedding) >= match_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
$$;
