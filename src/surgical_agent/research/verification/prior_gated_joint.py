"""Unified five-head admission: prior-gated four heads plus a jointly verified Phase.

Every head now follows the same three steps, propose -> five-seat verify ->
Python admit:

  four heads  graph proposal -> five compact reviews -> frozen mean selector ->
              prior gate (`prior_gated_repair`, thresholds frozen);
  Phase       one visual Phase recommendation -> the same five seats rate all
              seven Phase alternatives together with the four-head pool ->
              `phase_extension.phase_apply` (unique best >= 4 beats the current).

Measured on archived answers before this module existed (zero new calls): the
joint panel's Phase decision fixed 1 and broke 0 Phase on 40 Training targets
and again on 32 VID110 targets; the blind five-seat majority vote broke more
than it fixed on four batches. Feeding a better Phase into the gate changed the
four heads on 8/136 Training targets, 5 better and 3 worse, and not at all with
the ground-truth Phase, so the gate keeps the H0 Phase as its bucket.

Leakage rules enforced here, not just documented:

  * reviewers and the Phase recommender see H0 and the raw candidate pool only;
    the gate runs after every answer is saved, so no request can carry a gated
    label (`assert_ungated_request`);
  * the Phase recommender receives no prior-derived relation hints; the gate
    and the Phase decision must not share the same Training table
    (`assert_no_prior_hints`);
  * the prior table is leave-query-video-out, asserted by `select_prior_gated`;
  * no ground truth, transport or model text is read here.
"""
from copy import deepcopy

from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.five_head_repair import (
    aggregate_five_heads,
    normalize_five_heads,
)
from surgical_agent.research.verification.phase_extension import phase_apply
from surgical_agent.research.verification.prior_gated_repair import (
    select_prior_gated,
)
from surgical_agent.research.verification.prior_panel import TASKS, labels
from surgical_agent.research.verification.recent_mean_panel import (
    aggregate as compact_aggregate,
)
from surgical_agent.research.verification.recent_mean_panel import (
    select as compact_select,
)

VERSION = "prior_gated_joint_phase_v1"
PHASE_THRESHOLD = 4.0
ARMS = ("h0", "control", "gated_control", "gated_control_jointphase", "gated_joint_jointphase")
PRIMARY = "gated_control_jointphase"
FORBIDDEN_REQUEST_KEYS = frozenset({"gt", "ground_truth", "labels", "gated_prediction", "gate_log",
                                    "prior_added", "vetoed", "candidate_relation_hints", "phase_compatibility_hints"})


def joint_pool(pool):
    """Four-head candidates plus all seven Phase alternatives; ids bind task_label."""
    if any(p["task"] == "phase" for p in pool["propositions"]):
        raise ValueError("pool already carries phase candidates")
    result = {"propositions": [deepcopy(p) for p in pool["propositions"]]}
    result["propositions"] += [{"id": f"phase_{i}", "task": "phase", "label_id": i,
                                "name": _TASK_NAMES["phase"][i], "components": None} for i in range(7)]
    return result


def four_pool(pool):
    return {"propositions": [deepcopy(p) for p in pool["propositions"] if p["task"] != "phase"]}


def phase_means(means):
    """All seven Phase scores, or all None when any seat left one Phase unrated."""
    values = {f"phase_{i}": means.get(f"phase_{i}") for i in range(7)}
    if any(v is None for v in values.values()):
        return {k: None for k in values}, False
    return values, True


def decide_phase(current, joint_means):
    """Frozen Phase admission over the joint panel; four heads of `current` untouched."""
    values, complete = phase_means(joint_means)
    out, decision = phase_apply(labels(current), values, threshold=PHASE_THRESHOLD)
    decision["valid_panel"] = complete
    return out["phase"], decision


def assert_ungated_request(packet, h0):
    """A reviewer or recommender packet may describe H0 only, never a gated result."""
    if not isinstance(packet, dict):
        raise TypeError("request packet must be a JSON object")
    if FORBIDDEN_REQUEST_KEYS & packet.keys():
        raise ValueError("request carries ground truth, gate output or prior hints")
    for key in ("current_prediction", "current_prediction_hypothesis"):
        if key in packet and labels(packet[key]) != labels(h0):
            raise ValueError("request describes a prediction other than H0")


def decide(h0, pool, prior, *, compact_raw, joint_raw, phase_raw=None, gate, apply_phase_choices,
           normalize_compact, image_count=3):
    """All arm predictions from saved answers. No transport, no GT.

    `apply_phase_choices` and `normalize_compact` are the frozen v1.3.0 helpers
    passed in by the caller so this module does not import script code.
    """
    h0 = labels(h0)
    kw = {"phase": h0["phase"][0], "veto_rate": gate["veto_rate"], "add_rate": gate["add_rate"],
          "prune": tuple(gate["prune"])}
    reviews, formats = normalize_compact(compact_raw, pool, image_count)
    means, diagnostics = compact_aggregate(reviews, pool, image_count=image_count)
    four = compact_select(h0, pool, means, threshold=4)
    if phase_raw is not None:
        control, vote = apply_phase_choices(four, phase_raw, image_count)
    else:
        control, vote = labels({**four, "phase": h0["phase"]}), None
    gated_control, log_control = select_prior_gated(h0, pool, means, prior, **kw)

    jpool = joint_pool(pool)
    jreviews, jformats = normalize_five_heads(joint_raw, jpool, image_count=image_count)
    jmeans, jdiagnostics = aggregate_five_heads(jreviews, jpool, image_count=image_count)
    jphase, phase_decision = decide_phase(h0, jmeans)
    sub = four_pool(jpool)
    gated_joint, log_joint = select_prior_gated(h0, sub, {p["id"]: jmeans[p["id"]] for p in sub["propositions"]},
                                                prior, **kw)
    primary = labels({**{t: gated_control[t] for t in TASKS}, "phase": jphase})
    if any(primary[t] != gated_control[t] for t in TASKS):
        raise AssertionError("Phase admission must not change the gated four heads")
    predictions = {"h0": deepcopy(h0), "control": control, "gated_control": gated_control,
                   "gated_control_jointphase": primary,
                   "gated_joint_jointphase": labels({**{t: gated_joint[t] for t in TASKS}, "phase": jphase})}
    detail = {"version": VERSION, "means": means, "diagnostics": diagnostics, "format_diagnostics": formats,
              "phase_vote": vote, "gate_log_control": log_control, "joint_pool": jpool, "joint_means": jmeans,
              "joint_diagnostics": jdiagnostics, "joint_format_diagnostics": jformats,
              "phase_decision": phase_decision, "gate_log_joint": log_joint}
    return predictions, detail
