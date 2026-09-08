"""Bounded multi-label candidate review; no network, GT, or learned confidence.

The base model proposes candidates. Only the coordinator can change labels,
after all five distinct reviewers have scored the same ID-keyed pool.
"""
from copy import deepcopy

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.prior_panel import (
    BOUNDS,
    COMPONENTS,
    TASKS,
    labels,
)

SEATS = ("grok", "qwen", "gpt", "gemini", "deepseek")
MAX_POOL = 64
MAX_NEW = {"instrument": 2, "verb": 2, "target": 2, "ivt": 4}


def make_pool(h0, proposals=None, previous=None):
    """Only legal novel candidates; adding a component to U is NOT accepting it."""
    h0 = labels(h0)
    pairs = {(q, c) for q in TASKS for c in h0[q]}
    if previous is not None:
        pairs.update((p["task"], p["label_id"]) for p in previous["propositions"])
    if proposals is not None:
        if not isinstance(proposals, dict) or set(proposals) != set(TASKS):
            raise ValueError("proposal must contain four candidate-ID lists, not edits")
        for q, limit in MAX_NEW.items():
            values = proposals[q]
            if (not isinstance(values, list) or len(values) > limit
                    or any(type(c) is not int or not 0 <= c < BOUNDS[q] for c in values)
                    or len(set(values)) != len(values)):
                raise ValueError("invalid or excessive candidate IDs")
            pairs.update((q, c) for c in values)
    # Necessary component hypotheses are reviewed too; never force projection.
    for q, c in tuple(pairs):
        if q == "ivt":
            pairs.update(COMPONENTS[c].items())
    if len(pairs) > MAX_POOL:
        raise ValueError("candidate pool cap exceeded")
    props = []
    for q, c in sorted(pairs):
        comp = COMPONENTS[c] if q == "ivt" else None
        name = "/".join(_TASK_NAMES[k][v] for k, v in comp.items()) if comp else _TASK_NAMES[q][c]
        props.append({"id": f"{q}_{c}", "task": q, "label_id": c, "name": name,
                      "components": comp})
    return {"propositions": props}


def score_schema(pool):
    scores = {p["id"]: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for p in pool["propositions"]}
    return {"type": "object", "properties": {"scores": {"type": "object", "properties": scores,
            "required": list(scores), "additionalProperties": False}},
            "required": ["scores"], "additionalProperties": False}


def validate_scores(raw, pool):
    """Auxiliary prose cannot destroy correct numeric fields; raw stays archived."""
    if not isinstance(raw, dict) or not isinstance(raw.get("scores"), dict):
        raise TypeError("missing score mapping")
    ids = {p["id"] for p in pool["propositions"]}
    if set(raw["scores"]) != ids:
        raise ValueError("missing or unknown candidate ID")
    if any(type(v) is not int or not 1 <= v <= 5 for v in raw["scores"].values()):
        raise ValueError("ratings must be integers 1..5")
    # We never feed arbitrary auxiliary prose to the coordinator or proposer.
    # Report that the numeric contract passed despite extra top-level output.
    return {"scores": dict(raw["scores"]), "ignored_auxiliary_fields": sorted(set(raw) - {"scores"})}


def aggregate(reviews, pool):
    if not isinstance(reviews, dict) or set(reviews) != set(SEATS):
        raise ValueError("five named reviewer seats required")
    clean = {seat: validate_scores(reviews[seat], pool) for seat in SEATS}
    means = {p["id"]: sum(clean[seat]["scores"][p["id"]] for seat in SEATS) / 5
             for p in pool["propositions"]}
    return means, clean


def select(h0, pool, means):
    """Multi-label selection, not one winning triplet. Uncertain edits keep H0."""
    h0 = labels(h0)
    ids = {(p["task"], p["label_id"]): p["id"] for p in pool["propositions"]}
    if set(means) != set(ids.values()):
        raise ValueError("pool and means differ")

    def score(q, c):
        return means[ids[q, c]]

    out = deepcopy(h0)
    for c in h0["ivt"]:
        if score("ivt", c) <= 2:
            out["ivt"].remove(c)
    for q, c in sorted(ids):
        if q != "ivt" or c in h0[q] or score(q, c) < 4:
            continue
        if all(v in h0[t] or score(t, v) >= 4 for t, v in COMPONENTS[c].items()):
            out["ivt"].append(c)
    for q in TASKS[:3]:
        for task, c in sorted(ids):
            if task == q and c not in h0[q] and score(q, c) >= 4:
                out[q].append(c)
        for c in h0[q]:
            if score(q, c) <= 2 and not any(COMPONENTS[ivt][q] == c for ivt in out["ivt"]):
                out[q].remove(c)
    # Existing caps still apply. Do not arbitrarily rank/truncate a crowded set.
    return labels(out)


def feedback(state, pool, means, reviews):
    """Machine-generated low-support/disagreement issues; no invented GT oracle."""
    result = []
    for p in pool["propositions"]:
        pid, current = p["id"], p["label_id"] in state[p["task"]]
        scores = [reviews[seat]["scores"][pid] for seat in SEATS]
        if (current and means[pid] < 4) or (2 < means[pid] < 4 and max(scores) >= 4):
            result.append({"candidate_id": pid, "currently_selected": current,
                           "mean": means[pid], "scores": scores,
                           "issue": "LOW_SUPPORT" if current else "DISAGREEMENT"})
    return result


def proposal_schema():
    props = {q: {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": BOUNDS[q] - 1},
                 "maxItems": n, "uniqueItems": True} for q, n in MAX_NEW.items()}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def run(h0, propose, review, save, *, aggregate_review=aggregate, max_rounds=3):
    """One initial pool expansion; at most three panel rounds and two refills.

    A proposal is never a patch and has no access to mutable accepted state.
    If proposing or reviewing fails, preserve the last reviewed state/H0.
    """
    if type(max_rounds) is not int or not 1 <= max_rounds <= 3:
        raise ValueError("review round cap must be 1..3")
    h0 = labels(h0)
    state = deepcopy(h0)
    pool = make_pool(h0)
    histories, snapshots = [], []
    stop = "MAX_ROUNDS"
    proposed = propose(0, deepcopy(h0), deepcopy(pool), [])
    try:
        if proposed is None:
            raise ValueError("initial candidate generation failed")
        pool = make_pool(h0, proposed, pool)
    except (ApiSchemaError, ValueError, TypeError, KeyError):
        stop = "PROPOSAL_FAILED"
    else:
        for round_no in range(1, max_rounds + 1):
            raw = review(round_no, deepcopy(state), deepcopy(pool))
            record = {"round": round_no, "pool": deepcopy(pool), "proposal": proposed, "raw": raw}
            try:
                means, normalized = aggregate_review(raw, pool)
                accepted = select(h0, pool, means)
            except (ApiSchemaError, ValueError, TypeError, KeyError):
                stop = "REVIEW_FAILED"
                record["status"] = stop
                histories.append(record)
                save(f"round_{round_no}", record)
                break
            record.update(status="VALID", means=means, normalized=normalized, accepted=accepted)
            save(f"round_{round_no}", record)
            histories.append(record)
            state = accepted
            snapshots.append(deepcopy(state))
            issues = feedback(state, pool, means, normalized)
            if not issues:
                stop = "NO_UNRESOLVED_ISSUES"
                break
            if round_no == max_rounds:
                break
            proposed = propose(round_no, deepcopy(state), deepcopy(pool), issues)
            record.update(refill=proposed, feedback=issues)
            save(f"round_{round_no}", record)
            try:
                if proposed is None:
                    raise ValueError("refill failed")
                expanded = make_pool(h0, proposed, pool)
            except (ApiSchemaError, ValueError, TypeError, KeyError):
                stop = "REFILL_FAILED"
                break
            if expanded == pool:
                stop = "NO_NEW_CANDIDATES"
                break
            pool = expanded
    snapshots.extend(deepcopy(state) for _ in range(3 - len(snapshots)))
    result = {"h0": h0, "final": state, "history": histories, "snapshots": snapshots, "stop_reason": stop}
    save("result", result)
    return result
