"""VectorMatch: find best-matching existing clients for a prospect."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import voyageai
from supabase import create_client

from client_referencing.config import (
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    TABLE_NAME,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
)

# Scoring weights
EXACT_WEIGHT = 0.7
SEMANTIC_WEIGHT = 0.3

# Technology normalization: map keywords to canonical embed_text
TECH_ALIASES = {
    "ios": "Technology: Mobile Application Development",
    "android": "Technology: Mobile Application Development",
    "swift": "Technology: Mobile Application Development",
    "kotlin": "Technology: Mobile Application Development",
    "flutter": "Technology: Mobile Application Development",
    "react native": "Technology: Mobile Application Development",
    "xamarin": "Technology: Mobile Application Development",
    "javascript": "Technology: Open Source",
    "js": "Technology: Open Source",
    "node": "Technology: Open Source",
    "node.js": "Technology: Open Source",
    "nodejs": "Technology: Open Source",
    "angular": "Technology: Open Source",
    "typescript": "Technology: Open Source",
    "react": "Technology: Open Source",
    "vue": "Technology: Open Source",
    "vue.js": "Technology: Open Source",
}


@dataclass
class ClientMatch:
    client_name: str
    client_industry: str
    client_geography: str
    client_url: str
    industry_match: bool
    exact_techs: List[str]
    semantic_techs: List[Tuple[str, str, float]]  # (prospect_tech, matched_embed_text, similarity)
    match_ratio: float
    similarity_score: float
    final_score: float


# Lazy-initialized clients
_supabase = None
_voyage = None


def _get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


def _get_voyage():
    global _voyage
    if _voyage is None:
        _voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
    return _voyage


def fetch_all_rows() -> List[dict]:
    """Fetch all rows from client_referencing_data (excluding embedding)."""
    sb = _get_supabase()
    cols = ("id,client_name,client_industry,client_geography,client_url,"
            "industry_array,industry_primary,industry_group,embed_text,exact_key")
    result = sb.table(TABLE_NAME).select(cols).execute()
    return result.data or []


def filter_by_industry(rows: List[dict], prospect_industry: str) -> Tuple[List[dict], bool]:
    """Filter rows where prospect_industry appears in industry_array or industry_group.

    Returns (filtered_rows, was_filter_applied).
    If filter yields < 3 unique clients, return all rows with was_filter_applied=False.
    """
    prospect_ind = prospect_industry.lower().strip()
    if not prospect_ind:
        return rows, False

    filtered = []
    for r in rows:
        industry_arr = r.get("industry_array") or []
        arr_match = any(prospect_ind in item.lower() for item in industry_arr)
        group_match = prospect_ind in (r.get("industry_group") or "").lower()
        if arr_match or group_match:
            filtered.append(r)

    unique_clients = set(r["client_name"] for r in filtered)
    if len(unique_clients) < 3:
        return rows, False

    return filtered, True


def exact_match(rows: List[dict], prospect_techs: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Match each prospect technology against exact_key column.

    Returns:
        matched_by_client: {client_name: [matched tech strings]}
        unmatched_techs: prospect technologies with no exact match
    """
    key_to_rows = {}
    for r in rows:
        ek = r["exact_key"].lower().strip()
        key_to_rows.setdefault(ek, []).append(r)

    matched_by_client = {}
    matched_techs = set()

    for tech in prospect_techs:
        tech_lower = tech.lower().strip()
        if tech_lower in key_to_rows:
            matched_techs.add(tech)
            for r in key_to_rows[tech_lower]:
                matched_by_client.setdefault(r["client_name"], []).append(tech)

    unmatched = [t for t in prospect_techs if t not in matched_techs]
    return matched_by_client, unmatched


def normalize_tech_for_embedding(tech: str) -> str:
    """Apply alias rules for semantic matching."""
    tech_lower = tech.lower().strip()
    if tech_lower in TECH_ALIASES:
        return TECH_ALIASES[tech_lower]
    return f"Technology: {tech.strip()}"


def semantic_match(
    candidate_rows: List[dict],
    unmatched_techs: List[str],
) -> Dict[str, List[Tuple[str, str, float]]]:
    """Semantic matching for technologies that didn't match exactly.

    Returns: {client_name: [(prospect_tech, matched_embed_text, similarity), ...]}
    """
    if not unmatched_techs:
        return {}

    voyage = _get_voyage()
    sb = _get_supabase()

    # Normalize and deduplicate query texts
    tech_to_query = {}
    for tech in unmatched_techs:
        tech_to_query[tech] = normalize_tech_for_embedding(tech)

    unique_queries = list(set(tech_to_query.values()))

    # Generate embeddings for all unique query texts at once
    embed_result = voyage.embed(
        unique_queries,
        model=VOYAGE_MODEL,
        input_type="query",
    )
    query_embeddings = dict(zip(unique_queries, embed_result.embeddings))

    candidate_client_names = set(r["client_name"] for r in candidate_rows)
    semantic_by_client = {}

    for tech in unmatched_techs:
        query_text = tech_to_query[tech]
        embedding = query_embeddings[query_text]

        rpc_result = sb.rpc("match_client_technologies", {
            "query_embedding": embedding,
            "match_count": 40,
            "match_threshold": 0.3,
        }).execute()

        for row in (rpc_result.data or []):
            cname = row["client_name"]
            if cname in candidate_client_names:
                semantic_by_client.setdefault(cname, []).append(
                    (tech, row["embed_text"], row["similarity"])
                )

    # Rerank with Voyage rerank-2
    if semantic_by_client:
        all_clients = list(semantic_by_client.keys())
        rerank_docs = []
        for cname in all_clients:
            techs_text = ", ".join(
                f"{t[0]} -> {t[1]} ({t[2]:.2f})" for t in semantic_by_client[cname]
            )
            rerank_docs.append(f"Client: {cname}. Semantic matches: {techs_text}")

        rerank_query = "Technologies: " + ", ".join(unmatched_techs)

        try:
            rerank_result = voyage.rerank(
                query=rerank_query,
                documents=rerank_docs,
                model="rerank-2",
                top_k=min(len(all_clients), 10),
            )
            for item in rerank_result.results:
                cname = all_clients[item.index]
                boosted = []
                for (tech, embed_text, sim) in semantic_by_client[cname]:
                    boosted.append((tech, embed_text, sim * (1 + item.relevance_score) / 2))
                semantic_by_client[cname] = boosted
        except Exception:
            pass  # graceful degradation: keep original similarities

    return semantic_by_client


def find_matches(
    prospect_industry: str,
    prospect_technologies: str,
    top_k: int = 5,
) -> Dict:
    """Main orchestrator. Returns structured results for UI rendering.

    Returns dict with keys:
        - matches: List[ClientMatch] sorted by final_score desc
        - industry_filtered_only: List[str] clients that passed industry but not top_k
        - industry_filter_applied: bool
        - total_candidates: int
    """
    prospect_ind = prospect_industry.strip().lower()
    prospect_techs = [t.strip().lower() for t in prospect_technologies.split(",") if t.strip()]
    total_techs = len(prospect_techs)

    if total_techs == 0:
        return {"matches": [], "industry_filtered_only": [],
                "industry_filter_applied": False, "total_candidates": 0}

    # Step 1: Fetch all rows
    all_rows = fetch_all_rows()

    # Step 2: Industry filter
    candidate_rows, industry_applied = filter_by_industry(all_rows, prospect_ind)
    print(f"[DEBUG] Total rows: {len(all_rows)}, Candidate rows after industry filter: {len(candidate_rows)}, Filter applied: {industry_applied}")
    # Always track which clients match industry (for tiebreaker sorting)
    industry_client_names = set()
    if prospect_ind:
        for r in all_rows:
            industry_arr = r.get("industry_array") or []
            arr_match = any(prospect_ind in item.lower() for item in industry_arr)
            group_match = prospect_ind in (r.get("industry_group") or "").lower()
            if arr_match or group_match:
                industry_client_names.add(r["client_name"])

    # Step 3: Exact match
    exact_by_client, unmatched_techs = exact_match(candidate_rows, prospect_techs)
    print(f"[DEBUG] Exact matches by client: {dict((k, v) for k, v in exact_by_client.items())}")
    print(f"[DEBUG] Unmatched techs: {unmatched_techs}")

    # Step 4: Semantic match on unmatched technologies
    semantic_by_client = semantic_match(candidate_rows, unmatched_techs)
    print(f"[DEBUG] Semantic matches by client: {list(semantic_by_client.keys())}")

    # Step 5: Score and rank
    all_matched_clients = set(exact_by_client.keys()) | set(semantic_by_client.keys())
    print(f"[DEBUG] All matched clients: {all_matched_clients}")

    # Build client metadata lookup
    client_meta = {}
    for r in candidate_rows:
        if r["client_name"] not in client_meta:
            client_meta[r["client_name"]] = {
                "client_industry": r["client_industry"],
                "client_geography": r["client_geography"],
                "client_url": r["client_url"],
            }

    matches = []
    for cname in all_matched_clients:
        exact_techs = exact_by_client.get(cname, [])
        sem_techs = semantic_by_client.get(cname, [])

        match_ratio = len(set(exact_techs)) / total_techs if total_techs > 0 else 0

        # Similarity: average of best similarity per unmatched tech
        if sem_techs:
            best_per_tech = {}
            for (ptech, embed_text, sim) in sem_techs:
                if ptech not in best_per_tech or sim > best_per_tech[ptech][1]:
                    best_per_tech[ptech] = (embed_text, sim)
            similarity_score = sum(s for _, s in best_per_tech.values()) / len(best_per_tech)
        else:
            similarity_score = 0.0

        final_score = EXACT_WEIGHT * match_ratio + SEMANTIC_WEIGHT * similarity_score

        meta = client_meta.get(cname, {})
        matches.append(ClientMatch(
            client_name=cname,
            client_industry=meta.get("client_industry", ""),
            client_geography=meta.get("client_geography", ""),
            client_url=meta.get("client_url", ""),
            industry_match=cname in industry_client_names,
            exact_techs=list(set(exact_techs)),
            semantic_techs=sem_techs,
            match_ratio=round(match_ratio, 3),
            similarity_score=round(similarity_score, 3),
            final_score=round(final_score, 3),
        ))

    matches.sort(key=lambda m: (m.final_score, m.industry_match), reverse=True)
    top_matches = matches[:top_k]

    # Clients that passed industry filter but didn't make top_k
    top_names = set(m.client_name for m in top_matches)
    industry_only = sorted(industry_client_names - top_names) if industry_applied else []

    return {
        "matches": top_matches,
        "industry_filtered_only": industry_only,
        "industry_filter_applied": industry_applied,
        "total_candidates": len(all_matched_clients),
    }
