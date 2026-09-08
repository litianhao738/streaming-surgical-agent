"""Versioned evidence-bound review, independent of GT and network transport.

The schema binds judgments to propositions; it cannot certify visual truth.
An invalid item or explicit positive/negative conflict blocks that item only.
"""
from jsonschema import Draft202012Validator

from surgical_agent.research.verification.candidate_coordinator import SEATS

FINDINGS = ("MATCH", "REFUTED", "UNCLEAR", "VISIBLE_ONLY", "INDIRECT_EFFECT")


def item_schema(image_count):
    properties = {
        "rating": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "finding": {"type": "string", "enum": list(FINDINGS)},
        "scope": {"type": "string", "enum": ["WHOLE_FRAME", "LOCAL_REGION", "UNCERTAIN"]},
        "image_indices": {"type": "array", "items": {"type": "integer", "minimum": 0,
                            "maximum": image_count - 1}, "maxItems": image_count},
        "observation": {"type": "string", "minLength": 1, "maxLength": 1000},
    }
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def review_schema(pool, image_count):
    props = {p["id"]: item_schema(image_count) for p in pool["propositions"]}
    return {"type": "object", "properties": {"judgments": {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}},
            "required": ["judgments"], "additionalProperties": False}


def item_error(value, task, image_count):
    if not isinstance(value, dict) or not Draft202012Validator(item_schema(image_count)).is_valid(value):
        return "SCHEMA_INVALID"
    if type(value["rating"]) is not int or any(type(i) is not int for i in value["image_indices"]):
        return "NON_INTEGER"
    if len(set(value["image_indices"])) != len(value["image_indices"]):
        return "DUPLICATE_IMAGE"
    rating, finding = value["rating"], value["finding"]
    if ((finding == "MATCH" and rating < 4) or (finding == "UNCLEAR" and rating != 3)
            or (finding in ("REFUTED", "VISIBLE_ONLY", "INDIRECT_EFFECT") and rating > 2)):
        return "RATING_FINDING_CONFLICT"
    if finding in ("VISIBLE_ONLY", "INDIRECT_EFFECT") and task != "target":
        return "TARGET_ONLY_FINDING"
    if finding != "UNCLEAR" and image_count - 1 not in value["image_indices"]:
        return "NO_CURRENT_FRAME_EVIDENCE"
    if finding == "MATCH" and value["scope"] == "UNCERTAIN":
        return "UNCERTAIN_SUPPORT"
    if rating <= 2 and value["scope"] != "WHOLE_FRAME":
        return "LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL"
    return None


def aggregate(reviews, pool, *, image_count=3):
    if not isinstance(reviews, dict) or set(reviews) != set(SEATS):
        raise ValueError("five named reviewer seats required")
    if not 1 <= image_count <= 3:
        raise ValueError("only real causal short histories accepted")
    ids = {p["id"] for p in pool["propositions"]}
    clean = {}
    for seat in SEATS:
        raw = reviews[seat]
        outer_valid = (isinstance(raw, dict) and set(raw) == {"judgments"}
                       and isinstance(raw["judgments"], dict)
                       and not set(raw["judgments"]) - ids)
        judgments = raw["judgments"] if outer_valid else {}
        errors = {p["id"]: item_error(judgments.get(p["id"]), p["task"], image_count)
                  for p in pool["propositions"]}
        clean[seat] = {"scores": {pid: judgments[pid]["rating"] if errors[pid] is None else 3 for pid in ids},
                       "errors": {pid: error for pid, error in errors.items() if error},
                       "judgments": judgments, "blocked": {}}
    means = {}
    for pid in ids:
        scores = [clean[s]["scores"][pid] for s in SEATS]
        reason = ("INVALID_EVIDENCE" if any(pid in clean[s]["errors"] for s in SEATS)
                  else "EXPLICIT_CONFLICT" if min(scores) <= 2 and max(scores) >= 4 else None)
        mean = sum(scores) / len(SEATS)
        means[pid] = 3 if reason else mean
        if reason:
            for seat in SEATS:
                clean[seat]["blocked"][pid] = {"reason": reason, "effective_score": 3,
                                               "valid_items_mean": None if reason == "INVALID_EVIDENCE" else mean}
    return means, clean
