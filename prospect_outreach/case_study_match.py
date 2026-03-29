"""Case study matching via Supabase vector search + Voyage AI reranking.

Uses the same Supabase database and match_case_studies RPC as meeting_notes_prep.
"""

from typing import List, Dict
from . import config

# Lazy-initialized clients
_supabase = None
_voyage = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        _supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    return _supabase


def _get_voyage():
    global _voyage
    if _voyage is None:
        import voyageai
        _voyage = voyageai.Client(api_key=config.VOYAGE_API_KEY)
    return _voyage


def match_case_studies(query: str, match_count: int = 5) -> List[Dict]:
    """Search for case studies matching the query using vector similarity + reranking.

    Args:
        query: Search text built from prospect research (initiatives, industry, tech signals).
        match_count: Number of final results to return.

    Returns:
        List of dicts with keys: company_name, use_case, doc_type, summary, industry, tags, similarity_score
    """
    if not config.SUPABASE_URL or not config.VOYAGE_API_KEY:
        return []

    try:
        # Step 1: Generate query embedding
        voyage = _get_voyage()
        embed_result = voyage.embed(
            [query[:8000]],
            model="voyage-3-large",
            input_type="query",
        )
        query_embedding = embed_result.embeddings[0]

        # Step 2: Vector search via Supabase RPC
        supabase = _get_supabase()
        response = supabase.rpc(
            "match_case_studies",
            {
                "query_embedding": query_embedding,
                "match_count": match_count * 4,  # Over-fetch for reranking
                "match_threshold": 0.15,
            },
        ).execute()

        candidates = response.data or []
        if not candidates:
            return []

        # Step 3: Rerank with Voyage cross-encoder
        rerank_docs = []
        for c in candidates:
            parts = [f"{c.get('company_name', '')} — {c.get('use_case', '')}"]
            if c.get("industry"):
                parts.append(f"Industry: {c['industry']}")
            if c.get("tags"):
                parts.append(f"Tags: {c['tags']}")
            parts.append(c.get("summary", ""))
            rerank_docs.append(". ".join(parts))

        rerank_result = voyage.rerank(
            query=query[:8000],
            documents=rerank_docs,
            model="rerank-2",
            top_k=min(match_count, len(candidates)),
        )

        # Step 4: Build final results in reranked order
        results = []
        for item in rerank_result.results:
            c = candidates[item.index]
            results.append({
                "company_name": c.get("company_name", ""),
                "use_case": c.get("use_case", ""),
                "doc_type": c.get("doc_type", ""),
                "summary": c.get("summary", ""),
                "industry": c.get("industry", ""),
                "tags": c.get("tags", ""),
                "similarity_score": round(item.relevance_score, 3),
            })

        return results

    except Exception as e:
        # Graceful degradation — pipeline continues without case studies
        print(f"Case study search failed: {e}")
        return []



# Industry keyword groups for broadening search when exact matches are insufficient
RELATED_INDUSTRIES = {
    "banking": ["lending", "financial"],
    "lending": ["banking", "financial"],
    "financial services": ["banking", "lending"],
    "insurance": ["financial"],
    "payments": ["financial", "banking"],
    "healthcare": [],
    "nonprofit": [],
    "real estate": [],
    "retail": [],
    "technology": [],
    "education": [],
    "manufacturing": [],
}


def find_industry_references(
    industry: str,
    prospect_geography: str = "",
    company_name: str = "",
    top_k: int = 5,
) -> List[str]:
    """Find reference client names in the same industry/vertical.

    Uses the client_references table in Supabase (curated client list with
    industry and geography). Two-tier matching:
      Tier 1: Exact industry match, local geography first
      Tier 2: Related industries, local geography first (only if Tier 1 < top_k)

    Args:
        industry: Detected industry vertical (e.g. "banking").
        prospect_geography: Prospect's geography for local-first sorting.
        company_name: Prospect's company name to exclude from results.
        top_k: Number of unique reference clients to return.

    Returns:
        List of client company names.
    """
    try:
        supabase = _get_supabase()
        result = supabase.table("client_references").select(
            "client_name, industry, geography"
        ).execute()
        all_clients = result.data or []
    except Exception as e:
        print(f"Client references lookup failed: {e}")
        return []

    industry_lower = industry.lower()
    geo_lower = prospect_geography.lower()
    company_lower = company_name.lower()

    def _is_local(client_geo: str) -> bool:
        return geo_lower in client_geo.lower() if geo_lower else False

    # Tier 1: Exact industry match
    tier1 = []
    for c in all_clients:
        if industry_lower in c["industry"].lower() and c["client_name"].lower() != company_lower:
            tier1.append((c["client_name"], _is_local(c["geography"])))
    tier1.sort(key=lambda x: (not x[1], x[0]))

    picked = []
    seen = set()
    for name, _ in tier1:
        if name.lower() not in seen:
            picked.append(name)
            seen.add(name.lower())
        if len(picked) >= top_k:
            return picked

    # Tier 2: Related industries, local geography first
    related_keywords = RELATED_INDUSTRIES.get(industry_lower, [])
    if related_keywords and len(picked) < top_k:
        tier2 = []
        for c in all_clients:
            ind_lower = c["industry"].lower()
            if (
                any(k in ind_lower for k in related_keywords)
                and industry_lower not in ind_lower
                and c["client_name"].lower() not in seen
                and c["client_name"].lower() != company_lower
            ):
                tier2.append((c["client_name"], _is_local(c["geography"])))
        tier2.sort(key=lambda x: (not x[1], x[0]))

        for name, _ in tier2:
            if name.lower() not in seen:
                picked.append(name)
                seen.add(name.lower())
            if len(picked) >= top_k:
                break

    return picked
