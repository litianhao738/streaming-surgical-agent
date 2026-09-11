"""Gate features that do not require a ranked or scored perception head.

The published extractor in `features.py` is built on `candidate.confidence` and
`rankings`: top-1 confidence, top-1 minus top-2 margin, and the min/mean
confidence of the selected classes. The adopted H0 contract
`joint_perception_final_only_v1` returns selected label IDs only, so every one
of those features is a constant and no Gate can be fitted on them.

These features are derived from evidence the final-only contract does produce:
the shape and internal consistency of the five label sets, and the retrieval
hints, which cost no provider call. `structural` can be computed before any
repair call is spent; `with_proposal` additionally uses the single candidate
proposal, so it only saves the five review calls rather than all six.

Nothing here predicts correctness. These are inputs to a decision about whether
spending review calls on a frame is worthwhile.
"""
from surgical_agent.research.verification.prior_panel import BOUNDS, COMPONENTS

HEADS = ("instrument", "verb", "target", "ivt")
NULL_IVT_FROM = 94
NULL_TARGET = 14
NULL_VERB = 9


def _labels(payload, head):
    value = payload.get(head)
    if isinstance(value, dict):
        value = value.get("selected_ids", [value.get("selected_id")])
    return {v for v in (value or []) if type(v) is int}


def structural(h0, hints=None):
    """Shape and self-consistency of one final-only H0, plus retrieval hints."""
    sets = {head: _labels(h0, head) for head in HEADS}
    phase = _labels(h0, "phase")
    used = {"instrument": set(), "verb": set(), "target": set()}
    for ivt in sets["ivt"]:
        if 0 <= ivt < BOUNDS["ivt"]:
            for task, value in COMPONENTS[ivt].items():
                used[task].add(value)
    # A relation whose own components are absent from the component heads is the
    # model disagreeing with itself, which no confidence score is needed to see.
    missing = sum(len(used[t] - sets[t]) for t in used)
    orphan = sum(len(sets[t] - used[t]) for t in used)
    selected = hints["audit"]["selected"] if hints and hints.get("audit") else []
    hinted = {row["ivt"] for row in selected if isinstance(row.get("ivt"), int)}
    return {
        "n_instrument": len(sets["instrument"]), "n_verb": len(sets["verb"]),
        "n_target": len(sets["target"]), "n_ivt": len(sets["ivt"]),
        "n_labels_total": sum(len(v) for v in sets.values()),
        "phase_id": next(iter(phase), -1),
        "has_null_ivt": int(any(v >= NULL_IVT_FROM for v in sets["ivt"])),
        "has_null_target": int(NULL_TARGET in sets["target"]),
        "has_null_verb": int(NULL_VERB in sets["verb"]),
        "components_missing_from_heads": missing,
        "orphan_components": orphan,
        "self_inconsistent": int(missing > 0 or orphan > 0),
        "ivt_per_instrument": round(len(sets["ivt"]) / max(len(sets["instrument"]), 1), 3),
        "empty_ivt": int(not sets["ivt"]),
        "n_graph_hints": len(hinted),
        "graph_hints_new": len(hinted - sets["ivt"]),
    }


def with_proposal(h0, proposal, hints=None):
    """Structural features plus what the single candidate proposal returned."""
    base = structural(h0, hints)
    counts = {head: len(_labels(proposal or {}, head)) for head in HEADS}
    sets = {head: _labels(h0, head) for head in HEADS}
    novel = sum(len(_labels(proposal or {}, head) - sets[head]) for head in HEADS)
    base.update({
        "proposed_total": sum(counts.values()), "proposed_ivt": counts["ivt"],
        "proposed_verb": counts["verb"], "proposed_target": counts["target"],
        "proposal_empty": int(sum(counts.values()) == 0),
        "proposal_novel": novel,
        "pool_growth": novel,
    })
    return base


def feature_names(with_proposal_stage=False):
    empty = {head: [] for head in (*HEADS, "phase")}
    empty["phase"] = [0]
    sample = with_proposal(empty, None) if with_proposal_stage else structural(empty)
    return sorted(sample)
