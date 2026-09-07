"""Five independent ordinal predictions; arithmetic means, at most 3 reviews.

No GT, network, prior frequencies or accumulated cross-round votes here.
"""
from copy import deepcopy

from jsonschema import Draft202012Validator, ValidationError

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.prior_panel import (
    BOUNDS,
    COMPONENTS,
    TASKS,
    apply_patch_response,
    digest,
    labels,
)


def response_schema():
    props = {"schema_version": {"type": "string", "const": "five_mean_scores_v1"}}
    for q, n in BOUNDS.items():
        props[q] = {"type": "array", "minItems": n, "maxItems": n,
                    "items": {"type": "integer", "enum": [1, 2, 3, 4, 5]}}
    props["observation"] = {"type": "string", "maxLength": 1000}
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def full_universe():
    props, eligible = [], {}
    for q, n in BOUNDS.items():
        for c in range(n):
            pid = f"p{len(props) + 1:03}"
            comp = COMPONENTS[c] if q == "ivt" else None
            name = "/".join(_TASK_NAMES[k][v] for k, v in comp.items()) if comp else _TASK_NAMES[q][c]
            props.append({"proposition_id": pid, "task": q, "label_id": c,
                          "components": comp, "name": name})
            eligible[pid] = {"independent": q != "ivt"}
    return {"propositions": props, "eligibility": eligible}


def mean_scores(responses):
    if responses is None or len(responses) != 5 or any(r is None for r in responses):
        raise ValueError("exactly five complete model responses required")
    for r in responses:
        Draft202012Validator(response_schema()).validate(r)
        if any(type(v) is not int for q in TASKS for v in r[q]):
            raise ValueError("scores must be non-boolean integers")
    return {q: [sum(r[q][c] for r in responses) / 5 for c in range(n)] for q, n in BOUNDS.items()}


def issues_for(current, means, responses):
    result = []
    identifiers = {(p["task"], p["label_id"]): p["proposition_id"] for p in full_universe()["propositions"]}
    for q, n in BOUNDS.items():
        for c in range(n):
            present = c in current[q]
            if not ((present and means[q][c] <= 2) or (not present and means[q][c] >= 4)):
                continue
            op, pid = ("REMOVE" if present else "ADD"), identifiers[q, c]
            result.append({"issue_id": f"mean:{pid}:{op}", "proposition_id": pid,
                           "operation": op, "mean_score": means[q][c],
                           "individual_scores": [r[q][c] for r in responses],
                           "observations": [r["observation"] for r in responses]})
    return result


def admit(h0, draft, means):
    """Review net patch against immutable H0; components need their own support."""
    h0, draft = labels(h0), labels(draft)
    if h0["phase"] != draft["phase"]:
        raise ValueError("Phase is frozen")
    out = deepcopy(h0)
    for c in set(h0["ivt"]) - set(draft["ivt"]):
        if means["ivt"][c] <= 2:
            out["ivt"].remove(c)
    for c in sorted(set(draft["ivt"]) - set(h0["ivt"])):
        if means["ivt"][c] >= 4 and all(v in draft[q] and (v in h0[q] or means[q][v] >= 4)
                                       for q, v in COMPONENTS[c].items()):
            out["ivt"].append(c)
    for q in TASKS[:3]:
        for c in sorted(set(draft[q]) - set(h0[q])):
            if means[q][c] >= 4:
                out[q].append(c)
        for c in set(h0[q]) - set(draft[q]):
            if means[q][c] <= 2 and not any(COMPONENTS[p][q] == c for p in out["ivt"]):
                out[q].remove(c)
    return labels(out)


def run_arm(h0, panel_call, patch_call, patch_schema, image_refs, save):
    h0 = labels(h0)
    state, draft = deepcopy(h0), deepcopy(h0)
    universe = full_universe()
    history, snapshots, own_previous = [], [], None
    stop = "MAX_ROUNDS"
    for round_no in range(1, 4):
        raw = panel_call(round_no, draft, own_previous)
        try:
            means = mean_scores(raw)
            accepted = state if round_no == 1 else admit(h0, draft, means)
        except (ValueError, TypeError, KeyError, ApiSchemaError, ValidationError) as exc:
            stop = "ROUND_INVALID:" + type(exc).__name__
            record = {"round": round_no, "draft": draft, "responses": raw, "status": stop}
            history.append(record)
            save(f"round_{round_no}", record)
            break
        record = {"round": round_no, "draft": draft, "responses": raw, "means": means,
                  "last_accepted": accepted, "status": "VALID"}
        save(f"round_{round_no}", record)
        state = deepcopy(accepted)
        snapshots.append(deepcopy(state))
        history.append(record)
        own_previous = raw
        if round_no == 3:
            break
        issues = issues_for(state, means, raw)
        if not issues:
            stop = "NO_ACTIONABLE_SCORE_DISAGREEMENT"
            break
        patch = patch_call(round_no, state, issues)
        record.update(issues=issues, patch=patch)
        save(f"round_{round_no}", record)
        try:
            new_draft = apply_patch_response(state, universe, issues, patch, image_refs, patch_schema)
        except (ValueError, TypeError, KeyError, ApiSchemaError, ValidationError) as exc:
            stop = "PATCH_INVALID:" + type(exc).__name__
            break
        if digest(new_draft) == digest(state):
            stop = "REPAIR_NO_CHANGE"
            break
        # Repeated suggestions may be reassessed; no cumulative voting and no >R3.
        draft = new_draft
    snapshots.extend(deepcopy(state) for _ in range(3 - len(snapshots)))
    result = {"h0": h0, "final": state, "snapshots": snapshots, "history": history, "stop_reason": stop}
    save("result", result)
    return result
