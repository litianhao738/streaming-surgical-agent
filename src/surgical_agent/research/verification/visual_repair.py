"""Unscored visual edit hypotheses followed by the original five-seat gate.

No network/GT. An ADD or REMOVE is a hypothesis, never a self-approved vote.
RECHECK can revisit existing candidates without candidate-pool growth.
"""
import re
from copy import deepcopy

from jsonschema import Draft202012Validator

from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import make_pool
from surgical_agent.research.verification.prior_panel import BOUNDS, COMPONENTS, labels

MAX_CHANGES = 16
MAX_RECHECK = 16


def split_id(pid):
    match = re.fullmatch(r"(instrument|verb|target|ivt)_(0|[1-9][0-9]*)", pid) if isinstance(pid, str) else None
    if not match or int(match[2]) >= BOUNDS[match[1]]:
        raise ValueError("candidate ID outside four-head ontology")
    return match[1], int(match[2])


def repair_schema(image_count):
    if type(image_count) is not int or not 1 <= image_count <= 3:
        raise ValueError("real causal history of one to three images required")
    ids = [f"{task}_{value}" for task, bound in BOUNDS.items() for value in range(bound)]
    change = {"type": "object", "properties": {
        "candidate_id": {"type": "string", "enum": ids},
        "operation": {"type": "string", "enum": ["ADD", "REMOVE"]},
        "image_indices": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": image_count - 1},
                          "minItems": 1, "maxItems": image_count, "uniqueItems": True},
        "observation": {"type": "string", "minLength": 1, "maxLength": 1000}},
        "required": ["candidate_id", "operation", "image_indices", "observation"], "additionalProperties": False}
    return {"type": "object", "properties": {
        "changes": {"type": "array", "items": change, "maxItems": MAX_CHANGES},
        "recheck": {"type": "array", "items": {"type": "string", "enum": ids}, "maxItems": MAX_RECHECK, "uniqueItems": True}},
        "required": ["changes", "recheck"], "additionalProperties": False}


def compile_repair(current, previous_pool, raw, issues, *, image_count=3):
    """Build a temporary hypothesis and a review queue, with no self-rating test.

Component additions needed by a new IVT are explicitly recorded as derived
hypotheses and require their own review. Removing a relation never erases its
components. A requested component removal conflicting with a retained relation
is blocked and recorded, rather than silently destroying that relation.
"""
    current = labels(current)
    if not Draft202012Validator(repair_schema(image_count)).is_valid(raw):
        raise ValueError("invalid visual repair schema")
    base_pool = make_pool(current, previous=previous_pool)
    known = {p["id"] for p in base_pool["propositions"]}
    queued = set()
    for issue in issues:
        pid = issue["candidate_id"]
        if pid not in known:
            raise ValueError("issue is not bound to the previous pool")
        queued.add(pid)
    directions, observations = {}, {}
    for change in raw["changes"]:
        pid = change["candidate_id"]
        task, value = split_id(pid)
        if pid in directions:
            raise ValueError("duplicate or conflicting edit")
        if image_count - 1 not in change["image_indices"]:
            raise ValueError("edit must refer to the target image")
        if any(type(i) is not int for i in change["image_indices"]):
            raise ValueError("image indices must be integers")
        if (change["operation"] == "ADD") == (value in current[task]):
            raise ValueError("requested edit does not change current labels")
        directions[pid] = change["operation"]
        observations[pid] = deepcopy(change)
    queued.update(raw["recheck"])
    pairs = [split_id(pid) for pid in queued | directions.keys()]
    expanded = deepcopy(base_pool)
    expanded["propositions"].extend({"task": task, "label_id": value} for task, value in pairs)
    pool = deepcopy(make_pool(current, previous=expanded))
    temporary = deepcopy(current)
    for pid, operation in directions.items():
        task, value = split_id(pid)
        if operation == "ADD":
            temporary[task].append(value)
        else:
            temporary[task].remove(value)
    derived, blocked = {}, {}
    for ivt in temporary["ivt"]:
        for task, value in COMPONENTS[ivt].items():
            pid = f"{task}_{value}"
            if pid in directions and directions[pid] == "REMOVE":
                blocked[pid] = "COMPONENT_NEEDED_BY_RETAINED_OR_PROPOSED_IVT"
                if value not in temporary[task]:
                    temporary[task].append(value)
            elif ivt not in current["ivt"] and value not in temporary[task]:
                temporary[task].append(value)
                derived[pid] = "ADD"
    temporary = labels(temporary)
    active = queued | directions.keys() | derived.keys()
    # New or rechecked IVT admissions need component judgments, even if the
    # component was already in the initial label set.
    for pid in tuple(active):
        task, value = split_id(pid)
        if task == "ivt":
            active.update(f"{t}_{v}" for t, v in COMPONENTS[value].items())
    review_pool = {"propositions": [p for p in pool["propositions"] if p["id"] in active]}
    return {"before": current, "temporary": temporary, "pool": pool, "review_pool": review_pool,
            "requested_changes": observations, "directions": directions, "derived_changes": derived,
            "blocked_temporary_changes": blocked, "recheck_ids": sorted(queued)}


def apply_reviewed_repair(compiled, means):
    """Use five-seat means only, allowing explicit edits or actual rechecks.

The repairer's explanation or desired edit never enters the reviewer score.
Unreviewed labels keep their current state. Component dependencies use the
existing selector; no automatic IVT projection deletes independent heads.
"""
    expected = {p["id"] for p in compiled["review_pool"]["propositions"]}
    if set(means) != expected:
        raise ValueError("review scores must cover exactly the review pool")
    if any(v is not None and (type(v) not in (int, float) or not 1 <= v <= 5) for v in means.values()):
        raise ValueError("invalid reviewer means")
    allowed = set(compiled["recheck_ids"]) | compiled["directions"].keys() | compiled["derived_changes"].keys()
    # Components of a requested/rechecked new relation must be addable if the
    # relation passes. They are hypotheses subject to the same score threshold.
    for pid in tuple(allowed):
        task, value = split_id(pid)
        if task == "ivt" and value not in compiled["before"]["ivt"]:
            allowed.update(f"{t}_{v}" for t, v in COMPONENTS[value].items())
    effective = {p["id"]: means.get(p["id"]) if p["id"] in allowed else None
                 for p in compiled["pool"]["propositions"]}
    # Existing component scores must remain available for checking admission,
    # but a dependency-only review is not permission to delete that component.
    for pid in means.keys() - allowed:
        if means[pid] is not None and means[pid] >= 4:
            effective[pid] = means[pid]
    return panel.select(compiled["before"], compiled["pool"], effective, threshold=4)
