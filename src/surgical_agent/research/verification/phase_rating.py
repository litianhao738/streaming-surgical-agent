"""Isolated Phase ratings; reuse the existing unique-best mean-4 selector."""
from copy import deepcopy

from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.phase_extension import (
    phase_apply,
    phase_choice_error,
)


def rating_schema():
    return {"type": "object", "properties": {
        "ratings": {"type": "object", "properties": {str(p): {"type": "integer", "minimum": 1, "maximum": 5}
            for p in range(7)}, "required": [str(p) for p in range(7)], "additionalProperties": False},
        "image_indices": {"type": "array", "items": {"type": "integer", "enum": [0, 1, 2]}},
        "observation": {"type": "string"}},
        "required": ["ratings", "image_indices", "observation"], "additionalProperties": False}


def rating_error(raw, image_count):
    if not isinstance(raw, dict) or set(raw) != {"ratings", "image_indices", "observation"}:
        return "INVALID_FIELDS"
    ratings = raw["ratings"]
    if not isinstance(ratings, dict) or set(ratings) != {str(p) for p in range(7)}:
        return "INCOMPLETE_PHASE_RATINGS"
    if any(type(v) is not int or not 1 <= v <= 5 for v in ratings.values()):
        return "INVALID_RATING"
    # Cite the current image even when all seven scores express uncertainty.
    return phase_choice_error({"phase_id": 0, "image_indices": raw["image_indices"],
        "observation": raw["observation"]}, image_count)


def apply_ratings(current, raw_reviews, image_count=3):
    means = {f"phase_{p}": None for p in range(7)}
    if not isinstance(raw_reviews, dict) or set(raw_reviews) != set(SEATS):
        errors = {"panel": "INVALID_SEATS"}
    else:
        errors = {s: error for s in SEATS if (error := rating_error(raw_reviews[s], image_count))}
    if not errors:
        means = {f"phase_{p}": sum(raw_reviews[s]["ratings"][str(p)] for s in SEATS) / 5
            for p in range(7)}
    result, decision = phase_apply(deepcopy(current), means, threshold=4.0)
    decision.update(means=means, errors=errors, valid_panel=not errors)
    return result, decision
