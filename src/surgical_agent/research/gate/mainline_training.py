"""Five-head, pre-verification Gate contracts for the frozen joint mainline.

Features have exactly the same 31 inference-time inputs as the final-only
experiment. Their version is intentionally new: the supervised policy now
repairs Phase as well as the four interaction heads. No GT, final prediction,
review response, or future frame is accepted by the feature extractor.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.research.gate.final_only_training import (
    BASE_FEATURES,
    FEATURE_ORDER,
    TARGETS,
    TASKS,
    TRACKER_FEATURES,
    canonical_labels,
    extract_features,
    mask_tracker,
    validate_features,
)

FEATURE_VERSION = "mainline_five_head_gate_features_v1"
LABEL_VERSION = "mainline_five_head_episode_utility_v1"
UTILITY_DEFINITION = "mean_per_frame_set_f1_five_heads_empty_empty_one"
REQUIRED_GT_TASKS = tuple(TASKS)
EPSILON = 1e-12


def _truth(gt: Mapping, mask: Mapping) -> dict[str, set[int] | None]:
    if not isinstance(mask, Mapping) or set(mask) != set(TASKS):
        raise ValueError("explicit masks for all five heads are required")
    if any(type(mask[task]) is not bool for task in TASKS):
        raise ValueError("task masks must be booleans")
    if not isinstance(gt, Mapping):
        raise TypeError("GT must be a mapping; masked labels may be absent")
    result = {}
    for task in TASKS:
        if not mask[task]:
            result[task] = None
            continue
        value = gt.get(task)
        low, high = TASK_ID_BOUNDS[task]
        if (not isinstance(value, (list, tuple))
                or any(type(label) is not int or not low <= label <= high for label in value)
                or len(set(value)) != len(value)
                or (task == "phase" and len(value) != 1)):
            raise ValueError(f"invalid observed GT labels for {task}")
        result[task] = set(value)
    return result


def _counts(predicted: set[int], expected: set[int]) -> dict:
    tp, fp, fn = len(predicted & expected), len(predicted - expected), len(expected - predicted)
    denominator = 2 * tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
            "f1": 2 * tp / denominator if denominator else 1.0,
            "set_errors": fp + fn}


def label_outcome(h0, final, *, gt, mask, repair_observed: bool) -> dict:
    """Join GT only after freezing inference features and all API outcomes.

    Utility is the mean of five *per-frame* set F1 changes (empty/empty = 1),
    not the pooled micro-F1 used to evaluate a cohort. Phase is a singleton
    set, so its per-frame F1 equals correctness. Every head must be annotated
    for any aggregate target. Missing GT is never treated as a negative.

    ``repair_observed`` means the complete prescribed repair episode returned
    usable responses. An API failure or rejected response makes its benefit
    unknown, even if the pipeline emitted a fallback final prediction.
    ``safe_benefit`` is a training label, not a deployment safety guarantee:
    it requires positive utility and no added incorrect or removed correct
    label in *any* head, including harms hidden by net within-head gains.
    """
    if type(repair_observed) is not bool:
        raise ValueError("repair_observed must be an explicit boolean")
    expected = _truth(gt, mask)
    initial = canonical_labels(h0) if h0 is not None else None
    if repair_observed and (initial is None or final is None):
        raise ValueError("an observed repair requires both H0 and final predictions")
    # A partial/fallback result does not become a counterfactual repair label.
    repaired = canonical_labels(final) if repair_observed else None
    before_counts = dict.fromkeys(TASKS)
    after_counts = dict.fromkeys(TASKS)
    errors = dict.fromkeys(TASKS)
    gains = dict.fromkeys(TASKS)
    loss_delta = dict.fromkeys(TASKS)
    fp_fn_delta = dict.fromkeys(TASKS)
    edits = dict.fromkeys(TASKS)
    for task in TASKS:
        truth = expected[task]
        if truth is None or initial is None:
            continue
        before = set(initial[task])
        before_counts[task] = _counts(before, truth)
        errors[task] = int(before != truth)
        if repaired is None:
            continue
        after = set(repaired[task])
        after_counts[task] = _counts(after, truth)
        gains[task] = after_counts[task]["f1"] - before_counts[task]["f1"]
        loss_delta[task] = after_counts[task]["set_errors"] - before_counts[task]["set_errors"]
        fp_fn_delta[task] = {
            "fp_delta": after_counts[task]["fp"] - before_counts[task]["fp"],
            "fn_delta": after_counts[task]["fn"] - before_counts[task]["fn"],
            "error_count_delta": loss_delta[task],
        }
        added, removed = after - before, before - after
        edits[task] = {
            "added_correct": sorted(added & truth),
            "removed_incorrect": sorted(removed - truth),
            "added_incorrect": sorted(added - truth),
            "removed_correct": sorted(removed & truth),
        }
        edits[task]["beneficial_count"] = len(edits[task]["added_correct"]) + len(edits[task]["removed_incorrect"])
        edits[task]["harmful_count"] = len(edits[task]["added_incorrect"]) + len(edits[task]["removed_correct"])

    complete = all(mask.values())
    observed = complete and repair_observed and initial is not None
    delta = sum(gains[t] for t in TASKS) / len(TASKS) if observed else None
    label_harm = any(edits[t]["harmful_count"] for t in TASKS) if observed else None
    phase_observed = expected["phase"] is not None and repaired is not None
    phase_before = errors["phase"] == 0 if errors["phase"] is not None else None
    phase_after = after_counts["phase"]["fn"] == 0 if phase_observed else None
    phase_changed = initial["phase"] != repaired["phase"] if repaired is not None else None
    if not phase_observed:
        phase_outcome = "unknown"
    elif phase_before is False and phase_after:
        phase_outcome = "improved"
    elif phase_before and phase_after is False:
        phase_outcome = "harmed"
    elif phase_changed:
        phase_outcome = "wrong_to_wrong"
    else:
        phase_outcome = "unchanged"

    return {
        "label_version": LABEL_VERSION,
        "utility_definition": UTILITY_DEFINITION,
        "utility_tasks": list(TASKS),
        "h0_observed": initial is not None,
        "repair_observed": repair_observed,
        "complete_five_head_gt": complete,
        "mask_by_task": dict(mask),
        "h0_error_by_task": errors,
        "h0_error": int(any(errors.values())) if complete and initial is not None else None,
        "benefit": int(delta > EPSILON) if observed else None,
        "safe_benefit": int(delta > EPSILON and not label_harm) if observed else None,
        "utility_delta": delta,
        "h0_mean_per_frame_f1": sum(before_counts[t]["f1"] for t in TASKS) / len(TASKS)
        if complete and initial is not None else None,
        "final_mean_per_frame_f1": sum(after_counts[t]["f1"] for t in TASKS) / len(TASKS)
        if observed else None,
        "h0_counts_by_task": before_counts,
        "final_counts_by_task": after_counts,
        "delta_f1_by_task": gains,
        "set_loss_delta_by_task": loss_delta,
        "fp_fn_delta_by_task": fp_fn_delta,
        "label_edits_by_task": edits,
        "beneficial_label_edits": sum(edits[t]["beneficial_count"] for t in TASKS) if observed else None,
        "harmful_label_edits": sum(edits[t]["harmful_count"] for t in TASKS) if observed else None,
        "any_label_harm": bool(label_harm) if observed else None,
        "any_head_harm": any(loss_delta[t] > 0 for t in TASKS) if observed else None,
        "phase": {"changed": phase_changed, "h0_correct": phase_before,
                  "final_correct": phase_after, "outcome": phase_outcome},
    }


def readiness(rows: Sequence[Mapping], *, target: str = "benefit", min_class_per_fit: int = 2) -> dict:
    """Check the five-head contract and class support for video-held-out fits.

    This is only an engineering prerequisite for pilot fitting; it does not
    establish power, select a Validation threshold, or approve deployment.
    """
    if target not in TARGETS or type(min_class_per_fit) is not int or min_class_per_fit < 1:
        raise ValueError("invalid training target or minimum class support")
    identities = [(row["video_id"], row["frame_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate target frames cannot be independent examples")
    if any(row["source_split"] != "Training" for row in rows):
        raise ValueError("Gate development accepts Training only")
    if any(not isinstance(row.get("policy_id"), str) or not row["policy_id"].strip() for row in rows):
        raise ValueError("each row must bind a nonempty frozen policy_id")
    if len({row["policy_id"] for row in rows}) > 1:
        raise ValueError("do not mix H0/repair policy versions in a Gate training set")
    for row in rows:
        if row.get("feature_version") != FEATURE_VERSION:
            raise ValueError("feature version differs from mainline five-head contract")
        labels = row["labels"]
        if (labels.get("label_version") != LABEL_VERSION
                or row.get("label_version", LABEL_VERSION) != LABEL_VERSION
                or labels.get("utility_definition") != UTILITY_DEFINITION
                or labels.get("utility_tasks") != list(TASKS)):
            raise ValueError("label version/utility differs from mainline five-head contract")
        for variant in ("features_no_tracker", "features_with_tracker"):
            validate_features(row[variant])
        if row["features_no_tracker"] != mask_tracker(row["features_with_tracker"]):
            raise ValueError("paired views must differ only by masking Tracker features")
        for name in TARGETS:
            value = labels[name]
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise ValueError("binary labels must be 0, 1 or null")
            if value is not None:
                if labels.get("complete_five_head_gt") is not True or labels.get("h0_observed") is not True:
                    raise ValueError("known targets require observed H0 and all five GT heads")
                if name != "h0_error" and labels.get("repair_observed") is not True:
                    raise ValueError("known repair targets require a completed repair episode")
    usable = [row for row in rows if row["labels"][target] is not None]
    videos = sorted({row["video_id"] for row in usable})
    reasons = []
    if len(videos) < 3:
        reasons.append("FEWER_THAN_THREE_LABELED_VIDEOS")
    folds = []
    for video in videos:
        fit = [row for row in usable if row["video_id"] != video]
        held = [row for row in usable if row["video_id"] == video]
        counts = Counter(row["labels"][target] for row in fit)
        if min(counts[0], counts[1]) < min_class_per_fit:
            reasons.append(f"INSUFFICIENT_FIT_CLASSES_WHEN_HOLDING_OUT_{video}")
        folds.append({"held_out_video_id": video,
                      "fit_video_ids": sorted({row["video_id"] for row in fit}),
                      "fit_negative": counts[0], "fit_positive": counts[1],
                      "held_out_examples": len(held),
                      "held_out_positive": sum(row["labels"][target] for row in held)})
    return {
        "feature_version": FEATURE_VERSION, "label_version": LABEL_VERSION,
        "policy_id": rows[0]["policy_id"] if rows else None,
        "target": target, "can_fit_pilot": not reasons,
        "observations": len(rows), "labeled": len(usable), "unknown": len(rows) - len(usable),
        "positive": sum(row["labels"][target] for row in usable),
        "negative": sum(row["labels"][target] == 0 for row in usable),
        "positive_videos": sorted({row["video_id"] for row in usable if row["labels"][target] == 1}),
        "negative_videos": sorted({row["video_id"] for row in usable if row["labels"][target] == 0}),
        "min_class_per_fit": min_class_per_fit, "folds": folds, "blocking_reasons": reasons,
        "statistical_sufficiency_proven": False, "deployable": False,
    }
