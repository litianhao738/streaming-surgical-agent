"""Five valid evidence ratings, arithmetic mean admission; no GT or transport.

Unlike semantic_v1, disagreement is recorded but is not itself a veto.
Incomplete/invalid evidence never becomes a fabricated average or a pass.
"""
from copy import deepcopy

from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import COMPONENTS, TASKS, labels
from surgical_agent.research.verification.semantic_coordinator import (
    aggregate as strict_aggregate,
)


def aggregate(reviews, pool, *, image_count=3):
    if not pool["propositions"]:
        raise ValueError("an empty candidate pool cannot establish a model pass")
    _, clean = strict_aggregate(reviews, pool, image_count=image_count)
    means, diagnostics = {}, {}
    for p in pool["propositions"]:
        pid = p["id"]
        invalid = {s: clean[s]["errors"][pid] for s in SEATS if pid in clean[s]["errors"]}
        scores = [clean[s]["scores"][pid] for s in SEATS]
        means[pid] = None if invalid else sum(scores) / 5
        diagnostics[pid] = {"scores": [None if s in invalid else clean[s]["scores"][pid] for s in SEATS],
                            "invalid": invalid, "explicit_conflict": min(scores) <= 2 and max(scores) >= 4}
    return means, diagnostics


def select(current, pool, means, *, threshold=4.0):
    """Keep uncertain labels; independent heads are not rebuilt from IVTs.

    Support >= threshold admits; refutation support (6 - mean) >= threshold
    removes. New IVTs require all three component scores to pass too.
    """
    if threshold not in (3.5, 4.0):
        raise ValueError("predeclared thresholds are 3.5 and 4")
    current = labels(current)
    ids = {(p["task"], p["label_id"]): p["id"] for p in pool["propositions"]}
    if set(means) != set(ids.values()) or any(
            x is not None and (type(x) not in (int, float) or not 1 <= x <= 5) for x in means.values()):
        raise ValueError("invalid or incomplete pool means")
    if any((q, c) not in ids for q in TASKS for c in current[q]):
        raise ValueError("current labels must be in reviewed pool")

    def supported(q, c):
        value = means[ids[q, c]]
        return value is not None and value >= threshold

    def refuted(q, c):
        value = means[ids[q, c]]
        return value is not None and value <= 6 - threshold

    out = deepcopy(current)
    out["ivt"] = [c for c in current["ivt"] if not refuted("ivt", c)]
    for q, c in sorted(ids):
        if (q == "ivt" and c not in current[q] and supported(q, c)
                and all(supported(t, v) for t, v in COMPONENTS[c].items())):
            out[q].append(c)
    for q in TASKS[:3]:
        for task, c in sorted(ids):
            if task == q and c not in current[q] and supported(q, c):
                out[q].append(c)
        for c in current[q]:
            if refuted(q, c) and not any(COMPONENTS[ivt][q] == c for ivt in out["ivt"]):
                out[q].remove(c)
    return labels(out)


def unresolved(state, pool, means, diagnostics, *, threshold=4.0):
    """No holistic average that lets easy labels hide unsupported relations."""
    result = []
    for p in pool["propositions"]:
        pid, mean = p["id"], means[p["id"]]
        present = p["label_id"] in state[p["task"]]
        # Every reviewed alternative must be either supported-and-selected or
        # refuted-and-absent; unclear/missing evidence keeps the loop unresolved.
        passed = mean is not None and (mean >= threshold if present else mean <= 6 - threshold)
        if not passed:
            result.append({"candidate_id": pid, "currently_selected": present, "mean": mean,
                           "scores": diagnostics[pid]["scores"], "invalid": diagnostics[pid]["invalid"]})
    return result
