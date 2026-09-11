"""Preserve source-bound rejection reasons without rehabilitating invalid votes.

The v1 builder and frozen experiments remain unchanged. Re-normalize the raw
reviews here so diagnostics cannot be supplied for another seat or candidate.
"""
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.review_feedback import build_review_feedback
from surgical_agent.research.verification.review_normalization import normalize_review


def build_repair_feedback(pool, raw_reviews, reviews, issues, *, image_count=3):
    if not isinstance(raw_reviews, dict) or set(raw_reviews) != set(SEATS):
        raise ValueError("five source-bound raw reviewer seats required")
    original = {}
    for seat in SEATS:
        clean, diagnostic = normalize_review(raw_reviews[seat], pool, seat=seat, image_count=image_count)
        if clean != reviews[seat]:
            raise ValueError("raw and normalized reviewer bindings disagree")
        original[seat] = diagnostic
    result = build_review_feedback(pool, reviews, issues, image_count=image_count)
    result["schema_version"] = "candidate_review_feedback_v2"
    for candidate in result["candidates"]:
        for item in candidate["invalid_reviewers"]:
            codes = original[item["reviewer_seat"]]["errors"].get(candidate["candidate_id"])
            if not codes:
                raise ValueError("missing original rejection diagnostic")
            item["normalization_reason"] = item.pop("reason")
            item["reason_codes"] = list(codes)
            item["use"] = "Invalid judgment: diagnose what needs checking; this casts no vote and supplies no verified visual evidence."
    return result
