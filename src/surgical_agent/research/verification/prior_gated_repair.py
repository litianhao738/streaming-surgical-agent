"""Prior-gated IVT admission on top of the frozen mean-panel selector.

Motivation, measured on archived answers (no new calls): five-seat rating
means separate correct from wrong IVT candidates with AUC ~0.44-0.46, while
the leave-query-video-out Training rate of the same IVT class, conditioned on
the H0 Phase, separates them with AUC ~0.75. Ground-truth I/V/T are exactly the
component projection of ground-truth IVT in every archived frame, so the four
interaction heads are determined by the IVT set.

The rule keeps the panel selector byte-for-byte and adds three Python steps:

  veto   drop a selected non-null IVT whose phase-conditioned prior rate is
         below `veto_rate` (a relation the Training annotators essentially
         never used in that workflow phase);
  add    admit a pool non-null IVT whose phase-conditioned prior rate is at
         least `add_rate`, together with its components;
  prune  optionally remove verb/target labels no surviving IVT anchors; off by
         default because it lowered Verb/Target F1 on the development replay.

Defaults (veto 0.01, add 0.7, no prune) were chosen on 136 pooled Training
targets by `tools/audit/prior_gated_replay.py` before Validation was scored.

No ground truth, transport or model text is read here. The prior table must be
fitted with the query video excluded; that is asserted by the caller.
"""
from copy import deepcopy

from surgical_agent.research.verification.prior_panel import (
    BOUNDS,
    COMPONENTS,
    TASKS,
    labels,
)
from surgical_agent.research.verification.recent_mean_panel import (
    select as panel_select,
)

NULL_IVT_FROM = 94
VERSION = "prior_gated_ivt_v1"


def phase_rate(prior, task, label_id, phase):
    """Phase-bucket rate when that bucket has any eligible class, else global rate."""
    table = prior["tasks"][task]
    bucket = table["phase"].get(str(phase)) or []
    rows = bucket if any(r["eligible"] for r in bucket) else table["global"]
    row = rows[label_id]
    if row["id"] != label_id:
        raise ValueError("prior table rows must be indexed by class id")
    return row["rate"]


def is_null(ivt):
    return ivt >= NULL_IVT_FROM


def select_prior_gated(current, pool, means, prior, *, phase, threshold=4.0, veto_rate=0.01,
                       add_rate=0.7, prune=(), universe_add=False):
    """Return (four heads, decision log). `current` may carry a phase; it is not changed."""
    if prior is None or type(phase) is not int or not 0 <= phase < 7:
        raise ValueError("leave-video-out prior and a valid phase are required")
    if veto_rate is not None and not 0 <= veto_rate < 1:
        raise ValueError("veto rate must be a fraction")
    if add_rate is not None and not 0 < add_rate <= 1:
        raise ValueError("add rate must be a fraction")
    if any(p not in TASKS[:3] for p in prune):
        raise ValueError("prune must name interaction component heads")
    base = panel_select(current, pool, means, threshold=threshold) if means is not None else labels(current)
    out = deepcopy(base)
    log = {"version": VERSION, "phase_used": phase, "vetoed": [], "prior_added": [], "pruned": {}}

    if veto_rate is not None:
        for c in list(out["ivt"]):
            if not is_null(c) and phase_rate(prior, "ivt", c, phase) < veto_rate:
                out["ivt"].remove(c)
                log["vetoed"].append(c)

    if add_rate is not None:
        pool_ivts = {p["label_id"] for p in pool["propositions"] if p["task"] == "ivt"}
        if universe_add:
            pool_ivts |= {c for c in range(BOUNDS["ivt"]) if COMPONENTS[c]["instrument"] in out["instrument"]}
        for c in sorted(pool_ivts):
            if c in out["ivt"] or is_null(c) or c in log["vetoed"]:
                continue
            if phase_rate(prior, "ivt", c, phase) >= add_rate:
                out["ivt"].append(c)
                log["prior_added"].append(c)
                for task, value in COMPONENTS[c].items():
                    if value not in out[task]:
                        out[task].append(value)

    for task in prune:
        anchored = {COMPONENTS[c][task] for c in out["ivt"]}
        removed = [v for v in out[task] if v not in anchored]
        if removed:
            out[task] = [v for v in out[task] if v in anchored]
            log["pruned"][task] = removed

    result = labels({**out, "phase": current["phase"]}) if "phase" in current else labels(out)
    return result, log
