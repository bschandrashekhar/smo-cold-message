"""VectorMatch: find best-matching existing clients for a prospect."""

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import voyageai
from supabase import create_client

from client_referencing.config import (
    CACHE_TTL,
    INDUSTRY_MATCH_THRESHOLD,
    INDUSTRY_TABLE_NAME,
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
    industry_score: float
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


class _DataCache:
    """In-memory cache for industry embeddings, client rows, and prospect vectors."""

    def __init__(self, ttl: int = CACHE_TTL):
        self.ttl = ttl
        self._industry_embeddings = None       # {term: raw_embedding}
        self._industry_term_vecs = None         # {term: normalized_numpy_array}
        self._all_rows = None                   # List[dict]
        self._prospect_embeddings = {}          # {prospect_string: normalized_numpy_array}
        self._last_refresh = 0.0

    def _is_stale(self) -> bool:
        return time.time() - self._last_refresh > self.ttl

    def get_industry_embeddings(self) -> Dict[str, list]:
        if self._industry_embeddings is None or self._is_stale():
            self._refresh()
        return self._industry_embeddings

    def get_industry_term_vecs(self) -> Dict[str, np.ndarray]:
        if self._industry_term_vecs is None or self._is_stale():
            self._refresh()
        return self._industry_term_vecs

    def get_all_rows(self) -> List[dict]:
        if self._all_rows is None or self._is_stale():
            self._refresh()
        return self._all_rows

    def get_prospect_embedding(self, prospect_ind: str) -> np.ndarray:
        if prospect_ind not in self._prospect_embeddings:
            voyage = _get_voyage()
            result = voyage.embed([prospect_ind], model=VOYAGE_MODEL, input_type="document")
            vec = np.array(result.embeddings[0])
            norm = np.linalg.norm(vec)
            self._prospect_embeddings[prospect_ind] = vec / norm if norm > 0 else vec
        return self._prospect_embeddings[prospect_ind]

    def _refresh(self):
        """Reload industry embeddings and client rows from Supabase."""
        sb = _get_supabase()
        # Industry embeddings
        result = sb.table(INDUSTRY_TABLE_NAME).select("term,embedding").execute()
        self._industry_embeddings = {row["term"]: row["embedding"] for row in (result.data or [])}
        # Pre-normalize vectors
        self._industry_term_vecs = {}
        for term, emb in self._industry_embeddings.items():
            v = np.array(json.loads(emb) if isinstance(emb, str) else emb)
            n = np.linalg.norm(v)
            if n > 0:
                self._industry_term_vecs[term] = v / n
        # Client rows
        cols = ("id,client_name,client_industry,client_geography,client_url,"
                "industry_array,industry_primary,industry_group,embed_text,exact_key,geo_priority")
        result = sb.table(TABLE_NAME).select(cols).execute()
        self._all_rows = result.data or []
        self._last_refresh = time.time()

    def invalidate(self):
        """Force cache refresh on next access."""
        self._last_refresh = 0.0
        self._prospect_embeddings.clear()


_cache = _DataCache()


def invalidate_cache():
    """Public API: call after sync operations to force fresh data on next query."""
    _cache.invalidate()


def fetch_all_rows() -> List[dict]:
    """Fetch all rows from client_referencing_data (cached with TTL)."""
    return _cache.get_all_rows()


def _matches_industry(row: dict, prospect_ind: str, industry_scores: Dict[str, float]) -> bool:
    """Check if a row's client passes the industry threshold via cosine similarity."""
    cname = row.get("client_name", "")
    return industry_scores.get(cname, 0.0) >= INDUSTRY_MATCH_THRESHOLD


def _matches_geography(row: dict, prospect_ctry: str) -> bool:
    """Check if a row matches the prospect country."""
    geo = (row.get("client_geography") or "").lower()
    return prospect_ctry in geo


def filter_candidates(
    rows: List[dict],
    prospect_industry: str,
    prospect_country: str,
    industry_scores: Dict[str, float] = None,
) -> Tuple[List[dict], str, List[str], List[str]]:
    """3-tier pre-filter: Industry+Country → Industry → All.

    Returns (filtered_rows, filter_level, tier1_clients, tier2_clients).
    filter_level is "industry_and_geography", "industry", or "none".
    tier1_clients/tier2_clients are unique client name lists for debug logging.
    """
    prospect_ind = prospect_industry.lower().strip()
    prospect_ctry = prospect_country.lower().strip()
    scores = industry_scores or {}

    tier1_clients = []
    tier2_clients = []

    if not prospect_ind:
        return rows, "none", tier1_clients, tier2_clients

    # Tier 1: Industry + Country
    if prospect_ctry:
        tier1_rows = [r for r in rows if _matches_industry(r, prospect_ind, scores) and _matches_geography(r, prospect_ctry)]
        tier1_clients = sorted(set(r["client_name"] for r in tier1_rows))

        if len(tier1_clients) > 4:
            return tier1_rows, "industry_and_geography", tier1_clients, tier2_clients

    # Tier 2: Industry only
    tier2_rows = [r for r in rows if _matches_industry(r, prospect_ind, scores)]
    tier2_clients = sorted(set(r["client_name"] for r in tier2_rows))

    if len(tier2_clients) > 2:
        return tier2_rows, "industry", tier1_clients, tier2_clients

    # Tier 3: No filter (Case A)
    return rows, "none", tier1_clients, tier2_clients


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


def _compute_industry_client_names(all_rows: List[dict], prospect_ind: str, industry_scores: Dict[str, float]) -> set:
    """Compute which clients pass the industry threshold via cosine similarity."""
    if not prospect_ind:
        return set()
    return {r["client_name"] for r in all_rows
            if industry_scores.get(r["client_name"], 0.0) >= INDUSTRY_MATCH_THRESHOLD}


def _fetch_industry_embeddings() -> Dict[str, list]:
    """Fetch all industry term embeddings from Supabase (cached with TTL)."""
    return _cache.get_industry_embeddings()


def _compute_industry_scores(
    prospect_ind: str,
    all_rows: List[dict],
) -> Dict[str, float]:
    """Compute per-client industry relevance via cosine similarity.

    For each client, finds the max cosine similarity between the prospect
    industry embedding and the embeddings of each term in the client's
    industry_array + industry_group.

    Uses cached industry term vectors and prospect embeddings to avoid
    redundant Supabase queries and Voyage API calls.

    Returns {client_name: score} where score is 0.0 to 1.0.
    """
    if not prospect_ind:
        return {}

    term_vecs = _cache.get_industry_term_vecs()
    if not term_vecs:
        return {}

    prospect_unit = _cache.get_prospect_embedding(prospect_ind)
    if np.linalg.norm(prospect_unit) == 0:
        return {}

    # Build client -> industry terms mapping (deduplicated by client)
    client_terms: Dict[str, set] = {}
    for r in all_rows:
        cname = r["client_name"]
        if cname in client_terms:
            continue
        terms = set()
        for item in (r.get("industry_array") or []):
            terms.add(item.lower().strip())
        grp = (r.get("industry_group") or "").lower().strip()
        if grp:
            terms.add(grp)
        client_terms[cname] = terms

    # Score each client: max cosine similarity across their industry terms
    scores = {}
    for cname, terms in client_terms.items():
        best = 0.0
        for term in terms:
            if term in term_vecs:
                sim = float(np.dot(prospect_unit, term_vecs[term]))
                if sim > best:
                    best = sim
        scores[cname] = round(best, 4)

    return scores


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
    industry_scores: Dict[str, float],
    sort_by_industry: bool,
    source_exact: str,
    source_semantic: str,
    exclude_clients: set = None,
) -> List[Tuple[str, str]]:
    """Build ordered list of (client_name, match_source), exact-match clients first.

    Clients with exact matches come first, then semantic-only clients.
    Within each group, optionally sorted by industry relevance score (descending).
    """
    exclude = exclude_clients or set()

    exact_clients = [c for c in exact_by_client if c not in exclude]
    semantic_only = [c for c in semantic_by_client if c not in exclude and c not in exact_by_client]

    if sort_by_industry:
        exact_clients.sort(key=lambda c: (-industry_scores.get(c, 0.0), c))
        semantic_only.sort(key=lambda c: (-industry_scores.get(c, 0.0), c))

    result = [(c, source_exact) for c in exact_clients]
    result += [(c, source_semantic) for c in semantic_only]
    return result


def _geography_backfill(
    all_rows: List[dict],
    prospect_country: str,
    exclude_clients: set,
) -> List[str]:
    """Find unique clients matching prospect_country, sorted by geo_priority ascending."""
    if not prospect_country:
        return []
    seen = set()
    candidates = []
    for r in all_rows:
        cname = r["client_name"]
        geo = (r.get("client_geography") or "").lower()
        if cname not in exclude_clients and cname not in seen and prospect_country in geo:
            candidates.append((cname, r.get("geo_priority") or 999))
            seen.add(cname)
    candidates.sort(key=lambda x: x[1])
    return [c for c, _ in candidates]


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
        - debug_log: List[Tuple[str, str]] named log entries for UI display
    """
    prospect_ind = prospect_industry.strip().lower()
    prospect_techs = [t.strip().lower() for t in prospect_technologies.split(",") if t.strip()]
    prospect_ctry = prospect_country.strip().lower()
    total_techs = len(prospect_techs)
    debug_log = []

    if total_techs == 0:
        return {"matches": [], "industry_filtered_only": [],
                "industry_filter_applied": False, "total_candidates": 0,
                "debug_log": []}

    # Step 1: Fetch all rows and compute industry metadata
    all_rows = fetch_all_rows()
    client_meta = _build_client_meta(all_rows)

    # Compute continuous industry relevance scores via cached embeddings
    industry_scores = _compute_industry_scores(prospect_ind, all_rows)

    # Industry client names (threshold-based) for Case YES_I vs NO_I determination
    industry_client_names = _compute_industry_client_names(all_rows, prospect_ind, industry_scores)

    # Step 2: 3-tier pre-filter (Industry+Country → Industry → All)
    candidate_rows, filter_level, tier1_clients, tier2_clients = filter_candidates(
        all_rows, prospect_ind, prospect_ctry, industry_scores
    )
    industry_applied = filter_level != "none"

    # Debug: Always log tier 1 (Industry + Geography)
    debug_log.append((
        "Prep shortlistExistingClients (I+G)",
        ", ".join(tier1_clients) if tier1_clients else "(empty)",
    ))
    # Debug: Log tier 2 only if tier 1 was insufficient (≤4)
    if len(tier1_clients) <= 4:
        debug_log.append((
            "Prep shortlistExistingClients (I Only)",
            ", ".join(tier2_clients) if tier2_clients else "(empty)",
        ))

    # Step 3: Core matching (exact + semantic on candidate rows)
    exact_by_client, unmatched_techs = exact_match(candidate_rows, prospect_techs)
    semantic_by_client = semantic_match(candidate_rows, unmatched_techs)

    # Build shortlist: exact clients first, then semantic-only
    sort_by_ind = not industry_applied  # sort by industry only when filter was skipped
    shortlist = _build_shortlist(
        exact_by_client, semantic_by_client, industry_scores,
        sort_by_industry=sort_by_ind,
        source_exact="industry_exact" if industry_applied else "exact",
        source_semantic="industry_semantic" if industry_applied else "semantic",
    )

    # Debug: Case-specific logging after core matching
    _shortlist_names = [c for c, _ in shortlist]
    if not industry_applied:
        # Case A (NO_I): log after exact, then after semantic separately
        exact_clients_str = ", ".join(exact_by_client.keys()) if exact_by_client else "(none)"
        debug_log.append((
            "Case NO_I: shortlistExistingClients (After Exact Match)",
            exact_clients_str,
        ))
        debug_log.append((
            "Case NO_I: shortlistExistingClients (After Semantic Search)",
            ", ".join(_shortlist_names) if _shortlist_names else "(empty)",
        ))
    else:
        # Case B (YES_I): log after both matches together
        debug_log.append((
            "Case YES_I: shortlistExistingClients (After Both Matches)",
            ", ".join(_shortlist_names) if _shortlist_names else "(empty)",
        ))

    # Merge exact + semantic data for scoring (clients can have both)
    all_exact = dict(exact_by_client)
    all_semantic = dict(semantic_by_client)

    # Step 4+5: Combined backfill (generic + geo) collected, sorted, appended once
    shortlist_names = set(c for c, _ in shortlist)
    backfill_entries = []  # List[(client_name, match_source)]

    if len(shortlist_names) <= 5:
        # Generic backfill: exact + semantic from all clients excluding shortlist
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
                bf_exact, bf_semantic, industry_scores,
                sort_by_industry=True,
                source_exact="backfill_exact",
                source_semantic="backfill_semantic",
                exclude_clients=shortlist_names,
            )
            backfill_entries.extend(bf_shortlist)

        # Debug: generic backfill log (spec line 86) — shows ALL candidates before cap
        if backfill_entries:
            generic_names = list(dict.fromkeys(c for c, _ in backfill_entries))
            debug_log.append((
                "Case BACKFILL for Generic: shortlistAllBackFillClients",
                ", ".join(generic_names),
            ))

        # Cap generic backfill at deficit (spec lines 88-90)
        generic_deficit = 5 - len(shortlist_names)
        if generic_deficit > 0 and backfill_entries:
            capped = []
            capped_seen = set()
            for entry in backfill_entries:
                if entry[0] not in capped_seen:
                    if len(capped_seen) >= generic_deficit:
                        break
                    capped_seen.add(entry[0])
                capped.append(entry)
            backfill_entries = capped
        elif generic_deficit <= 0:
            backfill_entries = []

        # Geo backfill if combined count still ≤5 (spec lines 96-107)
        backfill_names = set(c for c, _ in backfill_entries)
        combined_count = len(shortlist_names) + len(backfill_names)
        if combined_count <= 5 and prospect_ctry:
            deficit = 6 - combined_count
            if deficit > 0:
                exclude_all = shortlist_names | backfill_names
                geo_clients = _geography_backfill(all_rows, prospect_ctry, exclude_all)
                geo_added = geo_clients[:deficit]
                for cname in geo_added:
                    backfill_entries.append((cname, "geography"))

                # Debug: geo backfill log — only geo clients (spec line 105)
                if geo_added:
                    debug_log.append((
                        "Case BACKFILL for Geo: shortlistGeoBackFillClients",
                        ", ".join(geo_added),
                    ))

        # Append backfill (generic first, then geo) without re-sorting
        if backfill_entries:
            shortlist.extend(backfill_entries)

            # Debug: single combined backfill log
            all_names = list(dict.fromkeys(c for c, _ in shortlist))
            debug_log.append((
                "Case BACKFILL for Generic + Geo: shortlistExistingClients",
                ", ".join(all_names) if all_names else "(empty)",
            ))

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
            industry_score=industry_scores.get(cname, 0.0),
            exact_techs=list(set(all_exact.get(cname, []))),
            semantic_techs=all_semantic.get(cname, []),
            match_ratio=match_ratio,
            similarity_score=similarity_score,
            final_score=final_score,
            match_source=source,
        ))

    # Preserve shortlist order (core first, backfill after) per spec.
    # _build_shortlist already orders: exact clients first, then semantic-only,
    # with industry_score sorting applied where spec requires it.
    # Only cap at 6 results (spec line 118).
    matches = matches[:6]

    # Clients that passed industry filter but didn't make the results
    result_names = set(m.client_name for m in matches)
    industry_only = sorted(industry_client_names - result_names) if industry_applied else []

    return {
        "matches": matches,
        "industry_filtered_only": industry_only,
        "industry_filter_applied": industry_applied,
        "total_candidates": len(result_names),
        "debug_log": debug_log,
    }
