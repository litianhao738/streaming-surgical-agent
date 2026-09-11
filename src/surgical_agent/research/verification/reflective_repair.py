"""Candidate-bounded joint patch validation; no GT or model-confidence oracle."""
from copy import deepcopy

from surgical_agent.research.verification import candidate_coordinator as candidates
from surgical_agent.research.verification.semantic_coordinator import (
    item_error,
    item_schema,
)


def reflection_schema(pool, image_count):
    properties = {q: {"type": "array", "items": {"type": "integer", "enum": sorted(
        p["label_id"] for p in pool["propositions"] if p["task"] == q)}, "maxItems": 8} for q in candidates.TASKS}
    edit = item_schema(image_count)
    edit["properties"].update(candidate_id={"type": "string", "enum": [p["id"] for p in pool["propositions"]]},
                              operation={"type": "string", "enum": ["ADD", "REMOVE"]})
    edit["required"].extend(["candidate_id", "operation"])
    return {"type": "object", "properties": {
        "prediction": {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False},
        "changes": {"type": "array", "items": edit, "maxItems": 64}},
        "required": ["prediction", "changes"], "additionalProperties": False}


def accept_reflection(h0, pool, raw, image_count=3):
    h0 = candidates.labels(h0)
    if not isinstance(raw, dict) or set(raw) != {"prediction", "changes"}:
        raise ValueError("invalid reflection envelope")
    if not isinstance(raw["prediction"], dict) or set(raw["prediction"]) != set(candidates.TASKS):
        raise ValueError("four-head prediction required; Phase stays frozen")
    proposed = candidates.labels({**raw["prediction"], "phase": h0["phase"]})
    lookup = {p["id"]: (p["task"], p["label_id"]) for p in pool["propositions"]}
    allowed = {q: {c for task, c in lookup.values() if task == q} for q in candidates.TASKS}
    if any(not set(proposed[q]) <= allowed[q] for q in candidates.TASKS):
        raise ValueError("unreviewed label outside candidate pool")
    expected = {f"{q}_{c}": "ADD" if c in proposed[q] else "REMOVE"
                for q in candidates.TASKS for c in set(h0[q]) ^ set(proposed[q])}
    if not isinstance(raw["changes"], list):
        raise TypeError("change evidence must be an array")
    seen = set()
    for change in raw["changes"]:
        if not isinstance(change, dict):
            raise TypeError("change must be an object")
        pid = change.get("candidate_id")
        if not isinstance(pid, str) or pid in seen or pid not in expected:
            raise ValueError("unknown, duplicate or non-change evidence")
        seen.add(pid)
        if change.get("operation") != expected[pid]:
            raise ValueError("operation does not match the actual label diff")
        item = {k: v for k, v in change.items() if k not in ("candidate_id", "operation")}
        if item_error(item, lookup[pid][0], image_count):
            raise ValueError("invalid evidence for actual change")
        if (expected[pid] == "ADD" and item["rating"] < 4) or (expected[pid] == "REMOVE" and item["rating"] > 2):
            raise ValueError("evidence direction contradicts edit")
    if seen != set(expected):
        raise ValueError("every actual change requires evidence")
    # An accepted new IVT needs its components; do not reconstruct independent heads.
    for c in set(proposed["ivt"]) - set(h0["ivt"]):
        if any(v not in proposed[q] for q, v in candidates.COMPONENTS[c].items()):
            raise ValueError("added IVT lacks required components")
    for q in candidates.TASKS[:3]:
        if any(any(candidates.COMPONENTS[c][q] == removed for c in proposed["ivt"])
               for removed in set(h0[q]) - set(proposed[q])):
            raise ValueError("removed component still required by a retained IVT")
    return deepcopy(proposed)
