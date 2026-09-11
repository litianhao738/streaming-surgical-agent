"""One evidence recheck for eligible novel IVTs; no GT, APIs or score rewriting."""
from copy import deepcopy

from jsonschema import Draft202012Validator

from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import COMPONENTS, labels
from surgical_agent.research.verification.semantic_coordinator import item_error

VERSION = "unclear_relation_single_recheck_v1"


def eligible_groups(h0, graph, guard, *, image_count=3):
    """Only old-admitted new IVTs vetoed by valid unclear novel-Verb votes.

    All four related propositions must have five valid ratings, mean >=4,
    and no rating <=2. Invalid votes never become abstentions or evidence.
    """
    h0, original, strict = labels(h0), labels(graph["prediction"]), labels(guard["prediction"])
    by_id = {p["id"]: p for p in graph["pool"]["propositions"]}
    decisions = {d["candidate_id"]: d for d in guard["new_verb_decisions"]}
    groups, excluded = [], []
    for ivt in sorted(set(original["ivt"]) - set(strict["ivt"]) - set(h0["ivt"])):
        comp = COMPONENTS[ivt]
        verb_id = f"verb_{comp['verb']}"
        related = [f"ivt_{ivt}", *(f"{task}_{value}" for task, value in comp.items())]
        decision = decisions.get(verb_id)
        reason = None
        if comp["verb"] in h0["verb"] or decision is None or decision["reason"] != "NOT_ALL_FIVE_SUPPORT":
            reason = "NOT_AN_UNCLEAR_NOVEL_VERB_VETO"
        else:
            for pid in related:
                task = by_id[pid]["task"]
                items = [graph["reviews"][s]["judgments"].get(pid) for s in SEATS]
                if any(item_error(item, task, image_count) for item in items):
                    reason = "INVALID_RELATED_EVIDENCE"
                    break
                scores = [item["rating"] for item in items]
                if graph["means"][pid] != sum(scores) / 5 or graph["means"][pid] < 4:
                    reason = "RELATED_MEAN_NOT_SUPPORTED"
                    break
                if min(scores) <= 2:
                    reason = "EXPLICIT_RELATED_REFUTATION"
                    break
            if reason is None:
                verb_items = [graph["reviews"][s]["judgments"][verb_id] for s in SEATS]
                if not any(x["finding"] == "UNCLEAR" and x["rating"] == 3 for x in verb_items):
                    reason = "NO_VALID_UNCLEAR_VERB_VOTE"
        if reason:
            excluded.append({"group_id": f"ivt_{ivt}", "reason": reason})
            continue
        groups.append({"group_id": f"ivt_{ivt}", "ivt_id": ivt,
                       "components": deepcopy(comp),
                       "names": {task: _TASK_NAMES[task][value] for task, value in comp.items()}})
    if len(groups) > 4:
        raise ValueError("original novel-IVT cap exceeded; never silently truncate")
    return {"groups": groups, "excluded": excluded}


def response_schema(groups, image_count):
    item = {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["SUPPORT", "REFUTE", "UNCLEAR"]},
        "image_indices": {"type": "array", "items": {"type": "integer", "minimum": 0,
            "maximum": image_count - 1}, "uniqueItems": True, "maxItems": image_count},
        "observation": {"type": "string", "minLength": 1, "maxLength": 1000}},
        "required": ["verdict", "image_indices", "observation"], "additionalProperties": False}
    props = {g["group_id"]: deepcopy(item) for g in groups}
    return {"type": "object", "properties": {"judgments": {"type": "object", "properties": props,
        "required": list(props), "additionalProperties": False}}, "required": ["judgments"],
        "additionalProperties": False}


def apply_groups(strict, original, groups, raw, *, image_count=3):
    """Explicit second admission path, preserving true five-review scores.

    Refuting one relation never deletes a frame-wide component. No new pool
    expansion, whole-frame projection rebuild or Phase change is permitted.
    """
    initial, original = labels(strict), labels(original)
    result = {"prediction": deepcopy(initial), "status": "NO_ELIGIBLE_GROUPS", "accepted_groups": [],
              "decisions": {}, "validation_error": None}
    if not groups:
        return result
    if not Draft202012Validator(response_schema(groups, image_count)).is_valid(raw):
        result.update(status="INVALID_RESPONSE", validation_error="SCHEMA_INVALID")
        return result
    for value in raw["judgments"].values():
        if (any(type(i) is not int for i in value["image_indices"])
                or (value["verdict"] != "UNCLEAR" and image_count - 1 not in value["image_indices"])):
            result.update(status="INVALID_RESPONSE", validation_error="MISSING_CURRENT_IMAGE_EVIDENCE")
            return result
    out = deepcopy(initial)
    for group in groups:
        gid, ivt = group["group_id"], group["ivt_id"]
        if group["components"] != COMPONENTS[ivt] or ivt not in original["ivt"]:
            raise ValueError("group must be an original-admitted ontology relation")
        decision = raw["judgments"][gid]
        result["decisions"][gid] = deepcopy(decision)
        if decision["verdict"] != "SUPPORT":
            continue
        out["ivt"] = sorted(set(out["ivt"]) | {ivt})
        for task, value in group["components"].items():
            if value not in original[task]:
                raise ValueError("component lacks original admission support")
            out[task] = sorted(set(out[task]) | {value})
        result["accepted_groups"].append(gid)
    out = labels(out)
    if out["phase"] != initial["phase"]:
        raise AssertionError("Phase must remain fixed")
    for task in ("instrument", "verb", "target", "ivt"):
        if not set(initial[task]) <= set(out[task]) <= set(original[task]):
            raise AssertionError("only restore eligible original-admitted label groups")
    result.update(prediction=out, status="RECHECKED")
    return result
