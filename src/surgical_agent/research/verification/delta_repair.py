"""Sparse LLM edits with full-ontology IDs, followed by independent visual review.

The model returns only actual changes and their evidence. Python reconstructs
the tentative answer; no omitted head or label is interpreted as a deletion.
"""
from copy import deepcopy

from jsonschema import Draft202012Validator

from surgical_agent.research.verification import candidate_coordinator as candidates
from surgical_agent.research.verification.prior_panel import BOUNDS
from surgical_agent.research.verification.reflective_repair import accept_reflection
from surgical_agent.research.verification.semantic_coordinator import item_schema

MAX_CHANGES = 24


def delta_schema(image_count=3):
    edit = item_schema(image_count)
    edit["properties"].update(
        candidate_id={"type": "string", "enum": [f"{q}_{c}" for q in candidates.TASKS for c in range(BOUNDS[q])]},
        operation={"type": "string", "enum": ["ADD", "REMOVE"]})
    edit["required"].extend(["candidate_id", "operation"])
    return {"type": "object", "properties": {"changes": {"type": "array", "items": edit, "maxItems": MAX_CHANGES}},
            "required": ["changes"], "additionalProperties": False}


def compile_delta(current, raw, image_count=3):
    """Validate all explicit edits before any patch or second-round API call."""
    if not Draft202012Validator(delta_schema(image_count)).is_valid(raw):
        raise ValueError("invalid sparse repair schema")
    current = candidates.labels(current)
    proposed, seen, pairs = deepcopy(current), set(), set()
    for edit in raw["changes"]:
        pid = edit["candidate_id"]
        q, value = pid.rsplit("_", 1)
        c = int(value)
        if pid in seen:
            raise ValueError("duplicate or contradictory edit")
        seen.add(pid)
        pairs.add((q, c))
        if edit["operation"] == "ADD":
            if c in current[q]:
                raise ValueError("ADD must change current state")
            proposed[q].append(c)
        else:
            if c not in current[q]:
                raise ValueError("REMOVE must change current state")
            proposed[q].remove(c)
    extra = {"propositions": [{"task": q, "label_id": c} for q, c in sorted(pairs)]}
    full_pool = candidates.make_pool(current, previous=extra)
    validated = accept_reflection(current, full_pool,
        {"prediction": {q: proposed[q] for q in candidates.TASKS}, "changes": raw["changes"]}, image_count)
    needed = set(seen)
    for edit in raw["changes"]:
        if edit["operation"] == "ADD" and edit["candidate_id"].startswith("ivt_"):
            c = int(edit["candidate_id"].split("_")[1])
            needed.update(f"{q}_{v}" for q, v in candidates.COMPONENTS[c].items())
    review_pool = {"propositions": [p for p in full_pool["propositions"] if p["id"] in needed]}
    return {"current": current, "tentative": validated, "changes": deepcopy(raw["changes"]),
            "full_pool": full_pool, "review_pool": review_pool}


def apply_reviewed_delta(compiled, means):
    """Only requested edits can take effect; omitted candidates remain unchanged."""
    pool = compiled["review_pool"]
    if set(means) != {p["id"] for p in pool["propositions"]}:
        raise ValueError("review does not cover exactly the needed propositions")
    changes = {e["candidate_id"]: e for e in compiled["changes"]}
    effective = {p["id"]: means[p["id"]] if p["id"] in changes else 3
                 for p in compiled["full_pool"]["propositions"]}
    for pid, edit in changes.items():
        if edit["operation"] == "ADD" and pid.startswith("ivt_"):
            c = int(pid.split("_")[1])
            if any(means[f"{q}_{v}"] < 4 for q, v in candidates.COMPONENTS[c].items()):
                effective[pid] = 3
    final = candidates.select(compiled["current"], compiled["full_pool"], effective)
    actual = {f"{q}_{c}": "ADD" if c in final[q] else "REMOVE" for q in candidates.TASKS
              for c in set(final[q]) ^ set(compiled["current"][q])}
    if any(pid not in changes or changes[pid]["operation"] != op for pid, op in actual.items()):
        raise ValueError("review caused an unrequested edit")
    return final


def issue_evidence(state, history):
    """Compact saved feedback, excluding unrelated passed claims and all GT."""
    issues = candidates.feedback(state, history["pool"], history["means"], history["normalized"])
    result = []
    for issue in issues:
        pid = issue["candidate_id"]
        opinions = []
        for seat in candidates.SEATS:
            normalized = history["normalized"][seat]
            opinions.append({"seat": seat, "judgment": normalized["judgments"].get(pid),
                             "validation_error": normalized["errors"].get(pid)})
        result.append({"candidate_id": pid, "currently_selected": issue["currently_selected"],
                       "issue": issue["issue"], "effective_score": issue["mean"], "opinions": opinions})
    return result
