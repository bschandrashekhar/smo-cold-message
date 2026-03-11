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
