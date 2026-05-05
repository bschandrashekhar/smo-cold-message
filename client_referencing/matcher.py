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
    TECH_TABLE_NAME,
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
    client_id: str = ""

    def to_dict(self) -> dict:
        return {
            "client_name": self.client_name,
            "client_id": self.client_id,
            "client_industry": self.client_industry,
            "client_geography": self.client_geography,
            "client_url": self.client_url,
            "industry_match": self.industry_match,
            "industry_score": self.industry_score,
            "exact_techs": self.exact_techs,
            "semantic_techs": [
                {"prospect_tech": t[0], "matched_text": t[1], "similarity": t[2]}
                for t in self.semantic_techs
            ],
            "match_ratio": self.match_ratio,
            "similarity_score": self.similarity_score,
            "final_score": self.final_score,
            "match_source": self.match_source,
        }


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
        self._tech_embeddings = {}              # {query_text: raw_embedding_list}
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

    def get_tech_embeddings(self, query_texts: List[str]) -> Dict[str, list]:
        """Return Voyage embeddings for technology query strings, cached per string.

        Only calls Voyage for strings not already in cache. Returns raw
        embedding lists (not normalized) since they go straight into the
        Supabase RPC for vector similarity search.
        """
        missing = [q for q in query_texts if q not in self._tech_embeddings]
        if missing:
            voyage = _get_voyage()
            result = voyage.embed(missing, model=VOYAGE_MODEL, input_type="query")
            for text, emb in zip(missing, result.embeddings):
                self._tech_embeddings[text] = emb
        return {q: self._tech_embeddings[q] for q in query_texts}

    def _refresh(self):
        """Reload industry embeddings and client+tech rows from Supabase."""
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
        # Client rows (metadata only)
        client_cols = ("id,client_name,client_industry,client_geography,client_url,"
                       "industry_array,industry_primary,industry_group,geo_priority")
        client_result = sb.table(TABLE_NAME).select(client_cols).execute()
        client_lookup = {r["id"]: r for r in (client_result.data or [])}
        # Tech rows
        tech_cols = "id,client_id,exact_key,embed_text"
        tech_result = sb.table(TECH_TABLE_NAME).select(tech_cols).execute()
        # Join: reconstruct flat rows (same shape as before the split)
        merged = []
        for t in (tech_result.data or []):
            c = client_lookup.get(t["client_id"])
            if c is None:
                continue
            merged.append({
                "id": c["id"],
                "client_name": c["client_name"],
                "client_industry": c["client_industry"],
                "client_geography": c["client_geography"],
                "client_url": c["client_url"],
                "industry_array": c["industry_array"],
                "industry_primary": c["industry_primary"],
                "industry_group": c["industry_group"],
                "geo_priority": c["geo_priority"],
                "exact_key": t["exact_key"],
                "embed_text": t["embed_text"],
            })
        self._all_rows = merged
        self._last_refresh = time.time()

    def invalidate(self):
        """Force cache refresh on next access.

        Preserves prospect industry and technology embeddings since those
        are deterministic (same input string always produces the same
        Voyage output).
        """
        self._last_refresh = 0.0


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

    # Tier 2: Industry only (when Tier 1 has <= 4 clients)
    tier2_rows = [r for r in rows if _matches_industry(r, prospect_ind, scores)]
    tier2_clients = sorted(set(r["client_name"] for r in tier2_rows))

    if len(tier2_clients) > 2:
        return tier2_rows, "industry", tier1_clients, tier2_clients

    # Tier 3: No filter — tech-only shortlist
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

    sb = _get_supabase()

    # Normalize and deduplicate query texts
    tech_to_query = {}
    for tech in unmatched_techs:
        tech_to_query[tech] = normalize_tech_for_embedding(tech)

    unique_queries = list(set(tech_to_query.values()))

    # Get embeddings from cache (only calls Voyage for uncached strings)
    query_embeddings = _cache.get_tech_embeddings(unique_queries)

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
            rerank_result = _get_voyage().rerank(
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
    """Build {client_name: {client_id, industry, geography, url}} lookup from rows."""
    meta = {}
    for r in rows:
        if r["client_name"] not in meta:
            meta[r["client_name"]] = {
                "client_id": r["id"],
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
    prospect_country: str,
    max_matches: int,
) -> Dict:
    """Main orchestrator with tiered matching and backfill.

    Returns dict with keys:
        - matches: List[ClientMatch] sorted by final_score desc
        - industry_filtered_only: List[str] clients that passed industry but not in results
        - industry_filter_applied: bool
        - total_candidates: int
        - debug_log: List[Tuple[str, str]] named log entries for UI display
    """
    if max_matches < 5 or max_matches > 10:
        raise ValueError("max_matches must be between 5 and 10")

    prospect_ind = prospect_industry.strip().lower()
    prospect_techs = [t.strip().lower() for t in prospect_technologies.split(",") if t.strip()]
    prospect_ctry = prospect_country.strip().lower()
    total_techs = len(prospect_techs)
    debug_log = []

    explanation = {}  # structured explanation data for UI

    if total_techs == 0:
        return {"matches": [], "industry_filtered_only": [],
                "industry_filter_applied": False, "total_candidates": 0,
                "debug_log": [], "explanation": {}}

    # Print tier reference for log clarity (exact formatting per spec)
    debug_log.append((
        "Tier Reference",
        "\n\t\tTier 1: Industry + Geography Only Shortlist"
        "\n\t\tTier 2: Industry Only Shortlist"
        "\n\t\tTier 3: Tech Only Shortlist\n",
    ))

    # Step 1: Fetch all rows and compute industry metadata
    all_rows = fetch_all_rows()
    client_meta = _build_client_meta(all_rows)

    # Compute continuous industry relevance scores via cached embeddings
    industry_scores = _compute_industry_scores(prospect_ind, all_rows)

    # Industry client names (threshold-based) for tier determination
    industry_client_names = _compute_industry_client_names(all_rows, prospect_ind, industry_scores)

    # Step 2: 3-tier pre-filter (Industry+Country → Industry → All)
    candidate_rows, filter_level, tier1_clients, tier2_clients = filter_candidates(
        all_rows, prospect_ind, prospect_ctry, industry_scores
    )
    industry_applied = filter_level != "none"
    flag_tier_3 = filter_level == "none"

    # Explanation: tier selection
    if filter_level == "industry_and_geography":
        explanation["tier_used"] = "Tier 1 (Industry + Geography)"
        explanation["tier_reason"] = f"Tier 1 has {len(tier1_clients)} clients (> 4 threshold)"
    elif filter_level == "industry":
        explanation["tier_used"] = "Tier 2 (Industry Only)"
        explanation["tier_reason"] = (
            f"Tier 1 has {len(tier1_clients)} clients (≤ 4), "
            f"Tier 2 has {len(tier2_clients)} clients (> 2 threshold)"
        )
    else:
        explanation["tier_used"] = "Tier 3 (Tech Only)"
        explanation["tier_reason"] = (
            f"Tier 1 has {len(tier1_clients)} clients (≤ 4), "
            f"Tier 2 has {len(tier2_clients)} clients (≤ 2) — not enough industry matches"
        )
    explanation["tier1_clients"] = tier1_clients
    explanation["tier2_clients"] = tier2_clients
    explanation["flag_tier_3"] = flag_tier_3

    # Debug: Always log Tier 1 (Industry + Geography)
    debug_log.append((
        "Tier 1 shortlistExistingClients (Shortlist Only)",
        ", ".join(tier1_clients) if tier1_clients else "(empty)",
    ))
    # Debug: Log Tier 2 only if Tier 1 was insufficient (≤4)
    if len(tier1_clients) <= 4:
        debug_log.append((
            "Tier 2 shortlistExistingClients (Shortlist Only)",
            ", ".join(tier2_clients) if tier2_clients else "(empty)",
        ))

    # Step 3: Core matching (exact + semantic on candidate rows)
    exact_by_client, unmatched_techs = exact_match(candidate_rows, prospect_techs)
    semantic_by_client = semantic_match(candidate_rows, unmatched_techs)

    # Build shortlist: exact clients first, then semantic-only
    sort_by_ind = flag_tier_3  # sort by industry only when filter was skipped (Tier 3)
    shortlist = _build_shortlist(
        exact_by_client, semantic_by_client, industry_scores,
        sort_by_industry=sort_by_ind,
        source_exact="industry_exact" if industry_applied else "exact",
        source_semantic="industry_semantic" if industry_applied else "semantic",
    )

    # Debug: Case-specific logging after core matching
    _shortlist_names = [c for c, _ in shortlist]
    if flag_tier_3:
        # Tier 3: log after exact, then after semantic separately
        exact_clients_str = ", ".join(exact_by_client.keys()) if exact_by_client else "(none)"
        debug_log.append((
            "Tier 3: shortlistExistingClients (After Exact Match)",
            exact_clients_str,
        ))
        debug_log.append((
            "Tier 3: shortlistExistingClients (Order after matches)",
            ", ".join(_shortlist_names) if _shortlist_names else "(empty)",
        ))
    else:
        # Tier 1 or Tier 1+2: log after both matches together
        debug_log.append((
            "(Tier 1) or (Tier 1+Tier 2) : shortlistExistingClients (Order after matches)",
            ", ".join(_shortlist_names) if _shortlist_names else "(empty)",
        ))

    # Explanation: core matching results
    explanation["prospect_techs"] = prospect_techs
    explanation["unmatched_techs"] = unmatched_techs
    explanation["core_match_clients"] = [c for c, _ in shortlist]
    explanation["core_exact_by_client"] = {c: list(set(ts)) for c, ts in exact_by_client.items()}
    explanation["core_semantic_by_client"] = {
        c: [(pt, et, sim) for pt, et, sim in ts]
        for c, ts in semantic_by_client.items()
    }

    # Merge exact + semantic data for scoring (clients can have both)
    all_exact = dict(exact_by_client)
    all_semantic = dict(semantic_by_client)

    # Truncate shortlist to top 5 before backfill (spec line 90)
    seen_top5 = set()
    shortlist_top5 = []
    for entry in shortlist:
        if entry[0] not in seen_top5:
            if len(seen_top5) >= 5:
                break
            seen_top5.add(entry[0])
        shortlist_top5.append(entry)
    shortlist = shortlist_top5

    # Explanation: top-5 truncation
    explanation["top5_clients"] = list(dict.fromkeys(c for c, _ in shortlist))
    explanation["top5_count"] = len(seen_top5)

    # Step 4+5: Backfill logic — depends on FLAG_TIER_3
    shortlist_names = set(c for c, _ in shortlist)
    backfill_entries = []  # List[(client_name, match_source)]

    if len(shortlist_names) <= 5 and not flag_tier_3:
        # Branch A: We had industry matches (Tier 1/2), backfill with tech first, then geo
        bf_exact, bf_semantic = {}, {}
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

        # Debug: generic backfill log — shows ALL candidates before cap
        generic_all_names = []
        if backfill_entries:
            generic_all_names = list(dict.fromkeys(c for c, _ in backfill_entries))
            debug_log.append((
                "Case BACKFILL for Generic: shortlistAllBackFillClients",
                ", ".join(generic_all_names),
            ))

        explanation["generic_backfill_all"] = generic_all_names
        explanation["bf_exact_by_client"] = {c: list(set(ts)) for c, ts in bf_exact.items()} if remaining_rows else {}
        explanation["bf_semantic_by_client"] = {
            c: [(pt, et, sim) for pt, et, sim in ts]
            for c, ts in bf_semantic.items()
        } if remaining_rows else {}

        # Cap generic backfill at deficit to reach 5
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

        # Geo backfill if combined count still ≤5
        backfill_names = set(c for c, _ in backfill_entries)
        combined_count = len(shortlist_names) + len(backfill_names)
        if combined_count <= 5 and prospect_ctry:
            deficit = max_matches - combined_count
            if deficit > 0:
                exclude_all = shortlist_names | backfill_names
                geo_clients = _geography_backfill(all_rows, prospect_ctry, exclude_all)
                geo_added = geo_clients[:deficit]
                for cname in geo_added:
                    backfill_entries.append((cname, "geography"))

                if geo_added:
                    debug_log.append((
                        "Case BACKFILL for Geo: shortlistGeoBackFillClients",
                        ", ".join(geo_added),
                    ))

        explanation["generic_deficit"] = generic_deficit
        # Separate the picked generic and geo from final backfill_entries
        explanation["generic_backfill_picked"] = list(dict.fromkeys(
            c for c, s in backfill_entries if s != "geography"
        ))
        explanation["geo_backfill_clients"] = list(dict.fromkeys(
            c for c, s in backfill_entries if s == "geography"
        ))
        explanation["geo_deficit"] = (max_matches - combined_count) if (combined_count <= 5 and prospect_ctry) else 0

        # Append backfill (generic first, then geo) without re-sorting
        if backfill_entries:
            shortlist.extend(backfill_entries)

            all_names = list(dict.fromkeys(c for c, _ in shortlist))
            debug_log.append((
                "Case BACKFILL for Generic + Geo: shortlistExistingClients",
                ", ".join(all_names) if all_names else "(empty)",
            ))

    elif len(shortlist_names) <= 5 and flag_tier_3:
        # Branch B: We already used tech for Tier 3 shortlisting, skip tech backfill,
        # go straight to geo backfill
        explanation["generic_backfill_all"] = []
        explanation["generic_backfill_picked"] = []
        explanation["generic_deficit"] = 0
        explanation["bf_exact_by_client"] = {}
        explanation["bf_semantic_by_client"] = {}
        geo_added = []
        if prospect_ctry:
            deficit = max_matches - len(shortlist_names)
            if deficit > 0:
                geo_clients = _geography_backfill(all_rows, prospect_ctry, shortlist_names)
                geo_added = geo_clients[:deficit]
                for cname in geo_added:
                    backfill_entries.append((cname, "geography"))

                if geo_added:
                    debug_log.append((
                        "Case BACKFILL for Geo: shortlistGeoBackFillClients",
                        ", ".join(geo_added),
                    ))

        explanation["geo_backfill_clients"] = geo_added
        explanation["geo_deficit"] = deficit if prospect_ctry else 0

        if backfill_entries:
            shortlist.extend(backfill_entries)

            all_names = list(dict.fromkeys(c for c, _ in shortlist))
            debug_log.append((
                "Case BACKFILL for Geo: shortlistExistingClients",
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
            client_id=meta.get("client_id", ""),
        ))

    # Preserve shortlist order (core first, backfill after) per spec.
    # _build_shortlist already orders: exact clients first, then semantic-only,
    # with industry_score sorting applied where spec requires it.
    # Cap at max_matches results.
    matches = matches[:max_matches]

    # Clients that passed industry filter but didn't make the results
    result_names = set(m.client_name for m in matches)
    industry_only = sorted(industry_client_names - result_names) if industry_applied else []

    # Explanation: final composition
    explanation["final_clients"] = [
        {"name": m.client_name, "source": m.match_source, "score": m.final_score,
         "industry_score": m.industry_score}
        for m in matches
    ]

    return {
        "matches": matches,
        "industry_filtered_only": industry_only,
        "industry_filter_applied": industry_applied,
        "total_candidates": len(result_names),
        "debug_log": debug_log,
        "explanation": explanation,
    }
