"""Isolated admission guard for novel Verb labels; no API or GT dependencies."""
from copy import deepcopy

from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.prior_panel import labels

VERSION = "new_verb_all_five_support_v1"


def select_with_verb_guard(h0, pool, means, diagnostics):
    """Require every valid reviewer to rate a new Verb >=4 before selection.

Run the original local selector from H0 using a copied mean map. This makes
the existing IVT component requirement honor a blocked new Verb too. Existing
Verb labels, their deletion rule and every other candidate score are unchanged.
"""
    initial = labels(h0)
    ids = {p["id"] for p in pool["propositions"]}
    if len(ids) != len(pool["propositions"]) or set(means) != ids or set(diagnostics) != ids:
        raise ValueError("complete unique candidate means and diagnostics required")
    guarded = deepcopy(means)
    decisions = []
    for proposition in pool["propositions"]:
        if proposition["task"] != "verb" or proposition["label_id"] in initial["verb"]:
            continue
        pid = proposition["id"]
        item = diagnostics[pid]
        scores = item.get("scores") if isinstance(item, dict) else None
        valid = (isinstance(item, dict) and item.get("invalid") == {}
                 and isinstance(scores, list) and len(scores) == 5
                 and all(type(score) is int and 1 <= score <= 5 for score in scores)
                 and means[pid] is not None and means[pid] == sum(scores) / 5)
        supported = valid and all(score >= 4 for score in scores)
        if not supported:
            guarded[pid] = None
        decisions.append({"candidate_id": pid, "original_mean": means[pid], "scores": deepcopy(scores),
                          "all_five_valid": valid, "admissible": bool(supported),
                          "reason": "ALL_FIVE_SUPPORT" if supported else
                          "NOT_ALL_FIVE_SUPPORT" if valid else "INVALID_OR_INCOMPLETE_EVIDENCE"})
    prediction = panel.select(initial, pool, guarded, threshold=4)
    if prediction["phase"] != initial["phase"]:
        raise AssertionError("Verb admission guard cannot change Phase")
    return {"version": VERSION, "prediction": prediction, "guarded_means": guarded,
            "new_verb_decisions": decisions}
