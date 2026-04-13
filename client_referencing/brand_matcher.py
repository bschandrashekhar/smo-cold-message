"""Brand Matcher — match a prospect to LendingLogik or CloudChillies via industry embeddings."""

import numpy as np

from client_referencing.matcher import _cache

LENDINGLOGIK_INDUSTRIES = {"banking", "financial services", "fintech", "insurance", "lending"}
BRAND_MATCH_THRESHOLD = 0.850


def find_brand_match(
    prospect_context: str,
    prospect_industry: str,
) -> dict:
    """Determine which brand best fits a prospect based on industry similarity.

    Returns dict with brand name, matched term, similarity score, all matches,
    and a debug log.
    """
    debug_log = []
    industry_lower = prospect_industry.strip().lower()

    debug_log.append(("Input", f"prospect_industry: {prospect_industry!r}\nNormalized: {industry_lower!r}"))

    if not industry_lower:
        debug_log.append(("Result", "No industry provided -> defaulting to CloudChillies"))
        return {
            "brand": "CloudChillies",
            "matched_industry_term": None,
            "similarity_score": 0.0,
            "all_matches": [],
            "debug_log": debug_log,
        }

    # Get pre-normalized industry term vectors from cache
    term_vecs = _cache.get_industry_term_vecs()
    prospect_unit = _cache.get_prospect_embedding(industry_lower)

    debug_log.append(("Industry Embeddings", f"Loaded {len(term_vecs)} industry terms from cache"))

    # Compute cosine similarity against all industry terms
    scored = []
    for term, vec in term_vecs.items():
        sim = float(np.dot(prospect_unit, vec))
        scored.append((term, round(sim, 4)))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Filter to threshold
    above_threshold = [(t, s) for t, s in scored if s >= BRAND_MATCH_THRESHOLD]

    debug_log.append((
        "Similarity Scores",
        f"Total terms: {len(scored)}\n"
        f"Above threshold ({BRAND_MATCH_THRESHOLD}): {len(above_threshold)}\n"
        f"Top 5: {scored[:5]}"
    ))

    if not above_threshold:
        debug_log.append(("Result", f"No industry term above {BRAND_MATCH_THRESHOLD} threshold -> defaulting to CloudChillies"))
        return {
            "brand": "CloudChillies",
            "matched_industry_term": None,
            "similarity_score": 0.0,
            "all_matches": [],
            "debug_log": debug_log,
        }

    best_term, best_score = above_threshold[0]
    brand = "LendingLogik" if best_term.lower() in LENDINGLOGIK_INDUSTRIES else "CloudChillies"

    debug_log.append((
        "Result",
        f"Best match: {best_term!r} (similarity: {best_score})\n"
        f"In LendingLogik set: {best_term.lower() in LENDINGLOGIK_INDUSTRIES}\n"
        f"Brand: {brand}"
    ))

    return {
        "brand": brand,
        "matched_industry_term": best_term,
        "similarity_score": best_score,
        "all_matches": above_threshold,
        "debug_log": debug_log,
    }
