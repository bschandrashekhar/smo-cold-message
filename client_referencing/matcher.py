"""VectorMatch: find best-matching existing clients for a prospect."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

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

# Single source of truth: map input keywords to their canonical exact_key in the DB.
# Used for both exact matching and semantic search (with "Technology: " prefix).
EXACT_KEY_ALIASES = {
    "ios": "mobile application development",
    "android": "mobile application development",
    "swift": "mobile application development",
    "kotlin": "mobile application development",
    "flutter": "mobile application development",
    "react native": "mobile application development",
    "xamarin": "mobile application development",
    "javascript": "open source",
    "js": "open source",
    "node": "open source",
    "node.js": "open source",
    "nodejs": "open source",
    "angular": "open source",
    "typescript": "open source",
    "react": "open source",
    "vue": "open source",
    "vue.js": "open source",
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
    match_source: str  # "industry_exact", "industry_semantic", "backfill_exact", "backfill_semantic", "geography"


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
        # Check direct match and alias match (e.g. "ios" → "mobile application development")
        keys_to_check = [tech_lower]
        if tech_lower in EXACT_KEY_ALIASES:
            keys_to_check.append(EXACT_KEY_ALIASES[tech_lower])

        for key in keys_to_check:
            if key in key_to_rows:
                matched_techs.add(tech)
                for r in key_to_rows[key]:
                    matched_by_client.setdefault(r["client_name"], []).append(tech)

    unmatched = [t for t in prospect_techs if t not in matched_techs]
    return matched_by_client, unmatched


def normalize_tech_for_embedding(tech: str) -> str:
    """Apply alias rules for semantic matching. Derives from EXACT_KEY_ALIASES."""
    tech_lower = tech.lower().strip()
    if tech_lower in EXACT_KEY_ALIASES:
        return f"Technology: {EXACT_KEY_ALIASES[tech_lower].title()}"
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


def _compute_industry_client_names(all_rows: List[dict], prospect_ind: str) -> set:
    """Compute which clients match the prospect industry (for tiebreaking)."""
    if not prospect_ind:
        return set()
    names = set()
    for r in all_rows:
        industry_arr = r.get("industry_array") or []
        arr_match = any(prospect_ind in item.lower() for item in industry_arr)
        group_match = prospect_ind in (r.get("industry_group") or "").lower()
        if arr_match or group_match:
            names.add(r["client_name"])
    return names


def _build_client_meta(rows: List[dict]) -> Dict[str, dict]:
    """Build {client_name: {industry, geography, url}} lookup from rows."""
    meta = {}
    for r in rows:
        if r["client_name"] not in meta:
            meta[r["client_name"]] = {
                "client_industry": r["client_industry"],
                "client_geography": r["client_geography"],
                "client_url": r["client_url"],
            }
    return meta


def _score_client(
    cname: str,
    exact_by_client: Dict[str, List[str]],
    semantic_by_client: Dict[str, List[Tuple[str, str, float]]],
    total_techs: int,
) -> Tuple[float, float, float]:
    """Compute match_ratio, similarity_score, final_score for a client."""
    exact_techs = exact_by_client.get(cname, [])
    sem_techs = semantic_by_client.get(cname, [])

    match_ratio = len(set(exact_techs)) / total_techs if total_techs > 0 else 0

    if sem_techs:
        best_per_tech = {}
        for (ptech, embed_text, sim) in sem_techs:
            if ptech not in best_per_tech or sim > best_per_tech[ptech][1]:
                best_per_tech[ptech] = (embed_text, sim)
        similarity_score = sum(s for _, s in best_per_tech.values()) / len(best_per_tech)
    else:
        similarity_score = 0.0

    final_score = EXACT_WEIGHT * match_ratio + SEMANTIC_WEIGHT * similarity_score
    return round(match_ratio, 3), round(similarity_score, 3), round(final_score, 3)


def _build_shortlist(
    exact_by_client: Dict[str, List[str]],
    semantic_by_client: Dict[str, List[Tuple[str, str, float]]],
    industry_client_names: set,
    sort_by_industry: bool,
    source_exact: str,
    source_semantic: str,
    exclude_clients: set = None,
) -> List[Tuple[str, str]]:
    """Build ordered list of (client_name, match_source), exact-match clients first.

    Clients with exact matches come first, then semantic-only clients.
    Within each group, optionally sorted by industry relevance.
    """
    exclude = exclude_clients or set()

    exact_clients = [c for c in exact_by_client if c not in exclude]
    semantic_only = [c for c in semantic_by_client if c not in exclude and c not in exact_by_client]

    if sort_by_industry:
        exact_clients.sort(key=lambda c: (c not in industry_client_names, c))
        semantic_only.sort(key=lambda c: (c not in industry_client_names, c))

    result = [(c, source_exact) for c in exact_clients]
    result += [(c, source_semantic) for c in semantic_only]
    return result


def _geography_backfill(
    all_rows: List[dict],
    prospect_country: str,
    exclude_clients: set,
) -> List[str]:
    """Find unique clients matching prospect_country, excluding already-shortlisted."""
    if not prospect_country:
        return []
    seen = set()
    result = []
    for r in all_rows:
        cname = r["client_name"]
        geo = (r.get("client_geography") or "").lower()
        if cname not in exclude_clients and cname not in seen and prospect_country in geo:
            result.append(cname)
            seen.add(cname)
    return result


def find_matches(
    prospect_industry: str,
    prospect_technologies: str,
    prospect_country: str = "",
) -> Dict:
    """Main orchestrator with tiered matching and backfill.

    Returns dict with keys:
        - matches: List[ClientMatch] sorted by final_score desc
        - industry_filtered_only: List[str] clients that passed industry but not in results
        - industry_filter_applied: bool
        - total_candidates: int
    """
    prospect_ind = prospect_industry.strip().lower()
    prospect_techs = [t.strip().lower() for t in prospect_technologies.split(",") if t.strip()]
    prospect_ctry = prospect_country.strip().lower()
    total_techs = len(prospect_techs)

    if total_techs == 0:
        return {"matches": [], "industry_filtered_only": [],
                "industry_filter_applied": False, "total_candidates": 0}

    # Step 1: Fetch all rows and compute industry metadata
    all_rows = fetch_all_rows()
    industry_client_names = _compute_industry_client_names(all_rows, prospect_ind)
    client_meta = _build_client_meta(all_rows)

    # Step 2: Industry filter
    candidate_rows, industry_applied = filter_by_industry(all_rows, prospect_ind)

    # Step 3: Core matching (exact + semantic on candidate rows)
    exact_by_client, unmatched_techs = exact_match(candidate_rows, prospect_techs)
    semantic_by_client = semantic_match(candidate_rows, unmatched_techs)

    # Build shortlist: exact clients first, then semantic-only
    sort_by_ind = not industry_applied  # sort by industry only when filter was skipped
    shortlist = _build_shortlist(
        exact_by_client, semantic_by_client, industry_client_names,
        sort_by_industry=sort_by_ind,
        source_exact="industry_exact" if industry_applied else "exact",
        source_semantic="industry_semantic" if industry_applied else "semantic",
    )

    # Merge exact + semantic data for scoring (clients can have both)
    all_exact = dict(exact_by_client)
    all_semantic = dict(semantic_by_client)

    # Step 4: Tech backfill if shortlist has ≤5 unique clients
    shortlist_names = set(c for c, _ in shortlist)
    if len(shortlist_names) <= 5:
        remaining_rows = [r for r in all_rows if r["client_name"] not in shortlist_names]
        if remaining_rows:
            bf_exact, _ = exact_match(remaining_rows, prospect_techs)
            bf_semantic = semantic_match(remaining_rows, unmatched_techs)

            # Merge backfill data into scoring dicts
            for c, techs in bf_exact.items():
                all_exact.setdefault(c, []).extend(techs)
            for c, techs in bf_semantic.items():
                all_semantic.setdefault(c, []).extend(techs)

            bf_shortlist = _build_shortlist(
                bf_exact, bf_semantic, industry_client_names,
                sort_by_industry=True,
                source_exact="backfill_exact",
                source_semantic="backfill_semantic",
                exclude_clients=shortlist_names,
            )
            shortlist.extend(bf_shortlist)

    # Step 5: Geography backfill if still ≤5 unique clients
    shortlist_names = set(c for c, _ in shortlist)
    if len(shortlist_names) <= 5 and prospect_ctry:
        deficit = 6 - len(shortlist_names)
        if deficit > 0:
            geo_clients = _geography_backfill(all_rows, prospect_ctry, shortlist_names)
            for cname in geo_clients[:deficit]:
                shortlist.append((cname, "geography"))

    # Step 6: Score all shortlisted clients and build ClientMatch objects
    matches = []
    seen = set()
    for cname, source in shortlist:
        if cname in seen:
            continue
        seen.add(cname)

        match_ratio, similarity_score, final_score = _score_client(
            cname, all_exact, all_semantic, total_techs
        )
        meta = client_meta.get(cname, {})
        matches.append(ClientMatch(
            client_name=cname,
            client_industry=meta.get("client_industry", ""),
            client_geography=meta.get("client_geography", ""),
            client_url=meta.get("client_url", ""),
            industry_match=cname in industry_client_names,
            exact_techs=list(set(all_exact.get(cname, []))),
            semantic_techs=all_semantic.get(cname, []),
            match_ratio=match_ratio,
            similarity_score=similarity_score,
            final_score=final_score,
            match_source=source,
        ))

    # Final sort: score desc, then exact-match preference, then industry
    matches.sort(
        key=lambda m: (m.final_score, len(m.exact_techs) > 0, m.industry_match),
        reverse=True,
    )

    # Cap at 6 results (5 tech + 1 geography max per spec)
    matches = matches[:6]

    # Clients that passed industry filter but didn't make the results
    result_names = set(m.client_name for m in matches)
    industry_only = sorted(industry_client_names - result_names) if industry_applied else []

    return {
        "matches": matches,
        "industry_filtered_only": industry_only,
        "industry_filter_applied": industry_applied,
        "total_candidates": len(result_names),
    }
