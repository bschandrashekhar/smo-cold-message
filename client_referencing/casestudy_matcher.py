"""CaseStudyMatcher: find best-matching case studies for a prospect.

Matches prospect against case study database using three signals:
  1. Exact technology match (via casestudy_tech_exact_match)
  2. Semantic technology match (via casestudy_technology_embedding)
  3. Context similarity (prospect_context embedding vs summary_embedding)

Tiered ranking:
  Tier 1: Industry + Technology (case studies whose client passes industry filter)
  Tier 2: Technology Only (remaining matches)
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from client_referencing.config import (
    CACHE_TTL,
    CASE_STUDIES_TABLE_NAME,
    CASESTUDY_TECH_MAPPING_TABLE,
    CS_CONTEXT_WEIGHT,
    CS_EXACT_WEIGHT,
    CS_SEMANTIC_WEIGHT,
    INDUSTRY_MATCH_THRESHOLD,
    INDUSTRY_TABLE_NAME,
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    TABLE_NAME,
    VOYAGE_MODEL,
)
from client_referencing.matcher import (
    EXACT_KEY_ALIASES,
    _get_supabase,
    _get_voyage,
    normalize_tech_for_embedding,
)


@dataclass
class CaseStudyMatch:
    casestudy_id: int
    casestudy_name: str
    client_id: str
    client_name: str
    client_industry: str
    url: str
    summary_problem: str
    summary_solution: str
    summary_outcomes: str
    exact_techs: List[str]
    semantic_techs: List[Tuple[str, str, float]]
    match_ratio: float
    similarity_score: float
    context_score: float
    final_score: float
    tier: str
    industry_score: float

    def to_dict(self) -> dict:
        return {
            "casestudy_id": self.casestudy_id,
            "casestudy_name": self.casestudy_name,
            "client_id": self.client_id,
            "client_name": self.client_name,
            "client_industry": self.client_industry,
            "url": self.url,
            "summary_problem": self.summary_problem,
            "summary_solution": self.summary_solution,
            "summary_outcomes": self.summary_outcomes,
            "exact_techs": self.exact_techs,
            "semantic_techs": [
                {"prospect_tech": t[0], "matched_tech": t[1], "similarity": t[2]}
                for t in self.semantic_techs
            ],
            "match_ratio": self.match_ratio,
            "similarity_score": self.similarity_score,
            "context_score": self.context_score,
            "final_score": self.final_score,
            "tier": self.tier,
            "industry_score": self.industry_score,
        }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _CaseStudyCache:
    """In-memory cache for case study data, industry embeddings, and prospect vectors."""

    def __init__(self, ttl: int = CACHE_TTL):
        self.ttl = ttl
        self._case_studies = None          # List[dict] from client_case_studies
        self._tech_mappings = None         # List[dict] from tech mapping table
        self._client_data = None           # {client_id: dict} from client_referencing_data
        self._industry_term_vecs = None    # {term: normalized numpy array}
        self._prospect_ctx_embeddings = {} # {context_str: normalized numpy array}
        self._prospect_ind_embeddings = {} # {industry_str: normalized numpy array}
        self._tech_query_embeddings = {}   # {query_text: raw embedding list}
        self._last_refresh = 0.0

    def _is_stale(self) -> bool:
        return time.time() - self._last_refresh > self.ttl

    def get_case_studies(self) -> List[dict]:
        if self._case_studies is None or self._is_stale():
            self._refresh()
        return self._case_studies

    def get_tech_mappings(self) -> List[dict]:
        if self._tech_mappings is None or self._is_stale():
            self._refresh()
        return self._tech_mappings

    def get_client_data(self) -> Dict[str, dict]:
        if self._client_data is None or self._is_stale():
            self._refresh()
        return self._client_data

    def get_industry_term_vecs(self) -> Dict[str, np.ndarray]:
        if self._industry_term_vecs is None or self._is_stale():
            self._refresh()
        return self._industry_term_vecs

    def get_prospect_context_embedding(self, context: str) -> np.ndarray:
        if context not in self._prospect_ctx_embeddings:
            voyage = _get_voyage()
            result = voyage.embed([context[:8000]], model=VOYAGE_MODEL, input_type="query")
            vec = np.array(result.embeddings[0])
            norm = np.linalg.norm(vec)
            self._prospect_ctx_embeddings[context] = vec / norm if norm > 0 else vec
        return self._prospect_ctx_embeddings[context]

    def get_prospect_industry_embedding(self, industry: str) -> np.ndarray:
        if industry not in self._prospect_ind_embeddings:
            voyage = _get_voyage()
            result = voyage.embed([industry], model=VOYAGE_MODEL, input_type="document")
            vec = np.array(result.embeddings[0])
            norm = np.linalg.norm(vec)
            self._prospect_ind_embeddings[industry] = vec / norm if norm > 0 else vec
        return self._prospect_ind_embeddings[industry]

    def get_tech_embeddings(self, query_texts: List[str]) -> Dict[str, list]:
        missing = [q for q in query_texts if q not in self._tech_query_embeddings]
        if missing:
            voyage = _get_voyage()
            result = voyage.embed(missing, model=VOYAGE_MODEL, input_type="query")
            for text, emb in zip(missing, result.embeddings):
                self._tech_query_embeddings[text] = emb
        return {q: self._tech_query_embeddings[q] for q in query_texts}

    def _refresh(self):
        sb = _get_supabase()

        # Case studies
        cs_result = sb.table(CASE_STUDIES_TABLE_NAME).select(
            "casestudy_id,client_id,casestudy_name,url,"
            "summary_problem,summary_solution,summary_outcomes,summary_embedding"
        ).execute()
        self._case_studies = cs_result.data or []

        # Tech mappings
        tech_result = sb.table(CASESTUDY_TECH_MAPPING_TABLE).select(
            "id,casestudy_id,casestudy_technology,casestudy_tech_exact_match"
        ).execute()
        self._tech_mappings = tech_result.data or []

        # Client data (for industry info)
        client_result = sb.table(TABLE_NAME).select(
            "id,client_name,client_industry,client_geography,client_url,"
            "industry_array,industry_primary,industry_group"
        ).execute()
        self._client_data = {str(r["id"]): r for r in (client_result.data or [])}

        # Industry term embeddings
        ind_result = sb.table(INDUSTRY_TABLE_NAME).select("term,embedding").execute()
        self._industry_term_vecs = {}
        for row in (ind_result.data or []):
            emb = row["embedding"]
            v = np.array(json.loads(emb) if isinstance(emb, str) else emb)
            n = np.linalg.norm(v)
            if n > 0:
                self._industry_term_vecs[row["term"]] = v / n

        self._last_refresh = time.time()

    def invalidate(self):
        self._last_refresh = 0.0


_cache = _CaseStudyCache()


def invalidate_casestudy_cache():
    """Public API: force fresh data on next query."""
    _cache.invalidate()


# ---------------------------------------------------------------------------
# Technology input massaging
# ---------------------------------------------------------------------------

def _expand_prospect_techs(prospect_technologies: str) -> List[str]:
    """Parse comma-separated tech string, lowercase, and expand aliases.

    Returns deduplicated list of lowercase tech keywords.
    """
    raw = [t.strip().lower() for t in prospect_technologies.split(",") if t.strip()]
    expanded = set(raw)
    for tech in raw:
        if tech in EXACT_KEY_ALIASES:
            expanded.add(EXACT_KEY_ALIASES[tech])
    return list(expanded)


# ---------------------------------------------------------------------------
# Exact matching
# ---------------------------------------------------------------------------

def _exact_match_by_casestudy(
    tech_mappings: List[dict],
    prospect_techs: List[str],
) -> Tuple[Dict[int, List[str]], List[str]]:
    """Match prospect techs against casestudy_tech_exact_match.

    Returns:
        matched_by_cs: {casestudy_id: [matched prospect tech strings]}
        unmatched_techs: prospect techs with no exact match in any case study
    """
    # Build index: exact_key -> list of (casestudy_id, tech_name)
    key_index: Dict[str, List[Tuple[int, str]]] = {}
    for t in tech_mappings:
        ek = (t.get("casestudy_tech_exact_match") or "").lower().strip()
        if ek:
            key_index.setdefault(ek, []).append((t["casestudy_id"], t["casestudy_technology"]))

    matched_by_cs: Dict[int, List[str]] = {}
    matched_techs = set()

    for tech in prospect_techs:
        keys_to_check = [tech]
        if tech in EXACT_KEY_ALIASES:
            keys_to_check.append(EXACT_KEY_ALIASES[tech])

        for key in keys_to_check:
            if key in key_index:
                matched_techs.add(tech)
                for cs_id, _cs_tech in key_index[key]:
                    matched_by_cs.setdefault(cs_id, []).append(tech)

    unmatched = [t for t in prospect_techs if t not in matched_techs]
    return matched_by_cs, unmatched


# ---------------------------------------------------------------------------
# Semantic matching
# ---------------------------------------------------------------------------

def _semantic_match_by_casestudy(
    unmatched_techs: List[str],
) -> Dict[int, List[Tuple[str, str, float]]]:
    """Semantic match for techs that didn't match exactly.

    Returns: {casestudy_id: [(prospect_tech, matched_cs_tech, similarity), ...]}
    """
    if not unmatched_techs:
        return {}

    sb = _get_supabase()

    # Normalize query texts
    tech_to_query = {}
    for tech in unmatched_techs:
        tech_to_query[tech] = normalize_tech_for_embedding(tech)

    unique_queries = list(set(tech_to_query.values()))
    query_embeddings = _cache.get_tech_embeddings(unique_queries)

    semantic_by_cs: Dict[int, List[Tuple[str, str, float]]] = {}

    for tech in unmatched_techs:
        query_text = tech_to_query[tech]
        embedding = query_embeddings[query_text]

        rpc_result = sb.rpc("match_casestudy_technologies", {
            "query_embedding": embedding,
            "match_count": 40,
            "match_threshold": 0.3,
        }).execute()

        for row in (rpc_result.data or []):
            cs_id = row["casestudy_id"]
            semantic_by_cs.setdefault(cs_id, []).append(
                (tech, row["casestudy_technology"], row["similarity"])
            )

    return semantic_by_cs


# ---------------------------------------------------------------------------
# Context similarity
# ---------------------------------------------------------------------------

def _compute_context_scores(
    prospect_context: str,
    case_studies: List[dict],
) -> Dict[int, float]:
    """Compute cosine similarity between prospect_context and each case study's summary_embedding.

    Returns: {casestudy_id: similarity_score}
    """
    if not prospect_context.strip():
        return {}

    ctx_vec = _cache.get_prospect_context_embedding(prospect_context)
    scores = {}

    for cs in case_studies:
        cs_id = cs["casestudy_id"]
        emb_raw = cs.get("summary_embedding")
        if not emb_raw:
            continue
        emb = np.array(json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw)
        norm = np.linalg.norm(emb)
        if norm == 0:
            continue
        emb_unit = emb / norm
        scores[cs_id] = float(np.dot(ctx_vec, emb_unit))

    return scores


# ---------------------------------------------------------------------------
# Industry scoring
# ---------------------------------------------------------------------------

def _compute_industry_scores(
    prospect_industry: str,
    case_studies: List[dict],
    client_data: Dict[str, dict],
) -> Dict[str, float]:
    """Compute per-client industry relevance via cosine similarity.

    Returns {client_id: score} where score is 0.0 to 1.0.
    """
    if not prospect_industry.strip():
        return {}

    term_vecs = _cache.get_industry_term_vecs()
    if not term_vecs:
        return {}

    prospect_unit = _cache.get_prospect_industry_embedding(prospect_industry)
    if np.linalg.norm(prospect_unit) == 0:
        return {}

    # Build client -> industry terms (deduplicated)
    client_ids_in_cs = set(str(cs["client_id"]) for cs in case_studies)
    client_terms: Dict[str, set] = {}
    for cid in client_ids_in_cs:
        c = client_data.get(cid)
        if not c or cid in client_terms:
            continue
        terms = set()
        for item in (c.get("industry_array") or []):
            terms.add(item.lower().strip())
        grp = (c.get("industry_group") or "").lower().strip()
        if grp:
            terms.add(grp)
        client_terms[cid] = terms

    scores = {}
    for cid, terms in client_terms.items():
        best = 0.0
        for term in terms:
            if term in term_vecs:
                sim = float(np.dot(prospect_unit, term_vecs[term]))
                if sim > best:
                    best = sim
        scores[cid] = round(best, 4)

    return scores


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_casestudy(
    cs_id: int,
    exact_by_cs: Dict[int, List[str]],
    semantic_by_cs: Dict[int, List[Tuple[str, str, float]]],
    context_scores: Dict[int, float],
    total_techs: int,
) -> Tuple[float, float, float, float]:
    """Compute match_ratio, similarity_score, context_score, final_score for a case study."""
    exact_techs = exact_by_cs.get(cs_id, [])
    sem_techs = semantic_by_cs.get(cs_id, [])

    match_ratio = len(set(exact_techs)) / total_techs if total_techs > 0 else 0

    if sem_techs:
        best_per_tech = {}
        for (ptech, cs_tech, sim) in sem_techs:
            if ptech not in best_per_tech or sim > best_per_tech[ptech][1]:
                best_per_tech[ptech] = (cs_tech, sim)
        similarity_score = sum(s for _, s in best_per_tech.values()) / len(best_per_tech)
    else:
        similarity_score = 0.0

    ctx_score = context_scores.get(cs_id, 0.0)

    final_score = (
        CS_EXACT_WEIGHT * match_ratio
        + CS_SEMANTIC_WEIGHT * similarity_score
        + CS_CONTEXT_WEIGHT * ctx_score
    )
    return (
        round(match_ratio, 3),
        round(similarity_score, 3),
        round(ctx_score, 3),
        round(final_score, 3),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_casestudy_matches(
    prospect_context: str,
    prospect_industry: str,
    prospect_technologies: str,
    prospect_country: str,
    max_matches: int = 8,
) -> dict:
    """Match a prospect against case study database.

    Returns dict with keys:
        - matches: List[CaseStudyMatch]
        - debug_log: List[Tuple[str, str]]
    """
    debug_log: List[Tuple[str, str]] = []

    # Header
    debug_log.append((
        "Tier Reference",
        "\n\t\tTier 1: Industry + Technology Shortlist"
        "\n\t\tTier 2: Technology Only Shortlist\n",
    ))

    prospect_techs = _expand_prospect_techs(prospect_technologies)
    total_techs = len(prospect_techs)

    if total_techs == 0 and not prospect_context.strip():
        return {"matches": [], "debug_log": debug_log}

    debug_log.append(("Prospect Techs (expanded)", ", ".join(prospect_techs)))

    # Load cached data
    case_studies = _cache.get_case_studies()
    tech_mappings = _cache.get_tech_mappings()
    client_data = _cache.get_client_data()

    # Build case study lookup
    cs_by_id = {cs["casestudy_id"]: cs for cs in case_studies}

    # --- TIER 2: Technology-Only Shortlist ---

    # Exact match
    exact_by_cs, unmatched_techs = _exact_match_by_casestudy(tech_mappings, prospect_techs)
    debug_log.append(("Exact matched case studies", str(list(exact_by_cs.keys())) if exact_by_cs else "(none)"))
    debug_log.append(("Unmatched techs for semantic", ", ".join(unmatched_techs) if unmatched_techs else "(none)"))

    # Semantic match
    semantic_by_cs = _semantic_match_by_casestudy(unmatched_techs)

    # Context similarity
    context_scores = _compute_context_scores(prospect_context, case_studies)

    # Score all case studies that have at least some signal
    candidate_cs_ids = set(exact_by_cs.keys()) | set(semantic_by_cs.keys()) | set(
        cs_id for cs_id, score in context_scores.items() if score > 0.1
    )

    scored: List[Tuple[int, float, float, float, float]] = []
    for cs_id in candidate_cs_ids:
        match_ratio, sim_score, ctx_score, final_score = _score_casestudy(
            cs_id, exact_by_cs, semantic_by_cs, context_scores, total_techs
        )
        if final_score > 0:
            scored.append((cs_id, match_ratio, sim_score, ctx_score, final_score))

    scored.sort(key=lambda x: -x[4])  # descending by final_score

    tier2_shortlist = scored  # all scored case studies

    if not tier2_shortlist:
        debug_log.append(("Result", "No matching case studies found"))
        return {"matches": [], "debug_log": debug_log}

    debug_log.append((
        "TIER 2 shortlist (all tech matches)",
        ", ".join(
            f"{cs_by_id.get(s[0], {}).get('casestudy_name', f'ID:{s[0]}')} ({s[4]:.3f})"
            for s in tier2_shortlist
        ),
    ))

    # --- TIER 1: Industry + Technology Shortlist ---

    industry_scores = _compute_industry_scores(
        prospect_industry, case_studies, client_data
    )

    # Determine which case studies belong to industry-matching clients
    tier1_shortlist = []
    tier2_remaining = []

    for entry in tier2_shortlist:
        cs_id = entry[0]
        cs = cs_by_id.get(cs_id)
        if not cs:
            continue
        cid = str(cs["client_id"])
        ind_score = industry_scores.get(cid, 0.0)
        if ind_score >= INDUSTRY_MATCH_THRESHOLD:
            tier1_shortlist.append(entry)
        else:
            tier2_remaining.append(entry)

    debug_log.append((
        "TIER 1 shortlist (industry + tech)",
        ", ".join(
            f"{cs_by_id.get(s[0], {}).get('casestudy_name', f'ID:{s[0]}')} ({s[4]:.3f})"
            for s in tier1_shortlist
        ) if tier1_shortlist else "(empty — no industry matches)",
    ))
    debug_log.append((
        "TIER 2 remaining (tech only)",
        ", ".join(
            f"{cs_by_id.get(s[0], {}).get('casestudy_name', f'ID:{s[0]}')} ({s[4]:.3f})"
            for s in tier2_remaining
        ) if tier2_remaining else "(empty)",
    ))

    # --- FINAL SHORTLIST ---
    final_shortlist = tier1_shortlist + tier2_remaining

    debug_log.append((
        "ALL FINAL_SHORTLIST",
        ", ".join(
            f"{cs_by_id.get(s[0], {}).get('casestudy_name', f'ID:{s[0]}')} ({s[4]:.3f})"
            for s in final_shortlist
        ),
    ))

    top_results = final_shortlist[:max_matches]

    debug_log.append((
        f"Top {max_matches} from FINAL_SHORTLIST",
        ", ".join(
            f"{cs_by_id.get(s[0], {}).get('casestudy_name', f'ID:{s[0]}')} ({s[4]:.3f})"
            for s in top_results
        ),
    ))

    # Build CaseStudyMatch objects
    matches = []
    for cs_id, match_ratio, sim_score, ctx_score, final_score in top_results:
        cs = cs_by_id.get(cs_id)
        if not cs:
            continue
        cid = str(cs["client_id"])
        client = client_data.get(cid, {})
        ind_score = industry_scores.get(cid, 0.0)
        tier = "tier_1" if ind_score >= INDUSTRY_MATCH_THRESHOLD else "tier_2"

        matches.append(CaseStudyMatch(
            casestudy_id=cs_id,
            casestudy_name=cs.get("casestudy_name", ""),
            client_id=cid,
            client_name=client.get("client_name", ""),
            client_industry=client.get("client_industry", ""),
            url=cs.get("url", ""),
            summary_problem=cs.get("summary_problem", "") or "",
            summary_solution=cs.get("summary_solution", "") or "",
            summary_outcomes=cs.get("summary_outcomes", "") or "",
            exact_techs=list(set(exact_by_cs.get(cs_id, []))),
            semantic_techs=semantic_by_cs.get(cs_id, []),
            match_ratio=match_ratio,
            similarity_score=sim_score,
            context_score=ctx_score,
            final_score=final_score,
            tier=tier,
            industry_score=ind_score,
        ))

    return {
        "matches": matches,
        "debug_log": debug_log,
    }
