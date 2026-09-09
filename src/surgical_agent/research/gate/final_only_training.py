"""Preparation contracts for a final-only, whole-episode Benefit Gate.

This is independent of the legacy top-k / three-scope Gate. Feature extraction
has no supervision or post-verification arguments. Labels are joined offline.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.perception.final_only import (
    FINAL_ONLY_SCHEMA_VERSION,
    TASKS,
    validate_final_only,
)
from surgical_agent.research.signals.frame_evidence import load_ivt_components

FEATURE_VERSION = "final_only_gate_features_v1"
LABEL_VERSION = "final_only_episode_utility_v1"
INTERACTION_TASKS = TASKS[:-1]
BASE_FEATURES = (
    tuple(f"h0_{task}_count" for task in INTERACTION_TASKS)
    + tuple(f"h0_phase_{label}" for label in range(7))
    + ("h0_interaction_empty", "causal_image_count", "h0_null_ivt_count")
    + tuple(
        f"h0_{task}_{suffix}"
        for task in ("instrument", "verb", "target")
        for suffix in ("missing_ivt_component_count", "independent_extra_count")
    )
)
TRACKER_FEATURES = (
    "tracker_available",
    "tracker_tool_count",
    "tracker_class_count",
    "tracker_detector_score_mean",
    "tracker_detector_score_min",
    "tracker_h0_missing_class_count",
    "tracker_h0_extra_class_count",
    "tracker_history_available",
    "tracker_new_track_count",
    "tracker_lost_track_count",
    "tracker_retained_track_fraction",
)
FEATURE_ORDER = BASE_FEATURES + TRACKER_FEATURES
TARGETS = ("benefit", "safe_benefit", "h0_error")


def canonical_labels(value: Mapping) -> dict[str, list[int]]:
    """Validate either the canonical wire payload or a compact five-head record."""
    if not isinstance(value, Mapping):
        raise TypeError("prediction must be a mapping")
    if "schema_version" in value:
        validate_final_only(value)
        return {
            task: [value[task]["selected_id"]]
            if task == "phase" else sorted(value[task]["selected_ids"])
            for task in TASKS
        }
    if set(value) != set(TASKS):
        raise ValueError("compact prediction requires exactly five heads")
    if any(not isinstance(value[t], (list, tuple)) for t in TASKS):
        raise ValueError("compact labels must be lists or tuples")
    if len(value["phase"]) != 1:
        raise ValueError("phase requires exactly one label")
    payload = {"schema_version": FINAL_ONLY_SCHEMA_VERSION}
    payload.update({
        task: {"selected_id": value[task][0]}
        if task == "phase" else {"selected_ids": list(value[task])}
        for task in TASKS
    })
    validate_final_only(payload)
    return {task: sorted(value[task]) for task in TASKS}


def _tracks(frame: Mapping) -> tuple[Mapping, ...]:
    tracks = frame["tracks"]
    if not isinstance(tracks, (list, tuple)):
        raise TypeError("Tracker tracks must be a sequence")
    identities = []
    for track in tracks:
        if not isinstance(track, Mapping):
            raise TypeError("invalid Tracker item")
        identity = track.get("track_id")
        instrument = track.get("instrument_id")
        score = track.get("score")
        if (not isinstance(identity, str) or not identity
                or type(instrument) is not int
                or not TASK_ID_BOUNDS["instrument"][0] <= instrument <= TASK_ID_BOUNDS["instrument"][1]
                or type(score) not in (float, int)
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("invalid Tracker identity, class or detector score")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate Tracker identity in one frame")
    return tuple(tracks)


def extract_features(
    h0: Mapping,
    *,
    target_frame_id: int,
    causal_frame_ids: Sequence[int],
    tracker_snapshot: Mapping | None = None,
) -> dict[str, float]:
    """Only H0 and pre-verification causal predicted tracks may enter features."""
    labels = canonical_labels(h0)
    frames = tuple(causal_frame_ids)
    if (type(target_frame_id) is not int or target_frame_id < 0
            or not 1 <= len(frames) <= 3
            or any(type(frame) is not int or frame < 0 for frame in frames)
            or frames != tuple(target_frame_id + 25 * i for i in range(1 - len(frames), 1))):
        raise ValueError("expected a real contiguous short causal H0 window")
    values = dict.fromkeys(FEATURE_ORDER, 0.0)
    for task in INTERACTION_TASKS:
        values[f"h0_{task}_count"] = float(len(labels[task]))
    values[f"h0_phase_{labels['phase'][0]}"] = 1.0
    values["h0_interaction_empty"] = float(not any(labels[t] for t in INTERACTION_TASKS))
    values["causal_image_count"] = float(len(frames))
    components = load_ivt_components()
    values["h0_null_ivt_count"] = float(sum(i >= 94 for i in labels["ivt"]))
    for index, task in enumerate(("instrument", "verb", "target")):
        projected = {components[i][index] for i in labels["ivt"]}
        selected = set(labels[task])
        # Soft features only: the independent heads need not equal IVT projection.
        values[f"h0_{task}_missing_ivt_component_count"] = float(len(projected - selected))
        values[f"h0_{task}_independent_extra_count"] = float(len(selected - projected))
    if tracker_snapshot is None:
        return values
    if tracker_snapshot.get("status") != "AVAILABLE":
        raise ValueError("explicit unavailable Tracker must be passed as None")
    snapshots = tuple(tracker_snapshot.get("frames", ()))
    if (tuple(frame["frame_id"] for frame in snapshots) != frames
            or tracker_snapshot.get("source_max_frame_id") != target_frame_id):
        raise ValueError("Tracker must match the exact causal H0 window, without future frames")
    current = _tracks(snapshots[-1])
    present = {track["instrument_id"] for track in current}
    scores = [float(track["score"]) for track in current]
    values.update({
        "tracker_available": 1.0,
        "tracker_tool_count": float(len(current)),
        "tracker_class_count": float(len(present)),
        "tracker_detector_score_mean": sum(scores) / len(scores) if scores else 0.0,
        "tracker_detector_score_min": min(scores) if scores else 0.0,
        "tracker_h0_missing_class_count": float(len(present - set(labels["instrument"]))),
        "tracker_h0_extra_class_count": float(len(set(labels["instrument"]) - present)),
    })
    if len(snapshots) > 1:
        previous = _tracks(snapshots[-2])
        before = {track["track_id"] for track in previous}
        after = {track["track_id"] for track in current}
        values.update({
            "tracker_history_available": 1.0,
            "tracker_new_track_count": float(len(after - before)),
            "tracker_lost_track_count": float(len(before - after)),
            "tracker_retained_track_fraction": len(before & after) / len(before) if before else 0.0,
        })
    return values


def mask_tracker(features: Mapping[str, float]) -> dict[str, float]:
    validate_features(features)
    return {key: 0.0 if key in TRACKER_FEATURES else float(features[key]) for key in FEATURE_ORDER}


def validate_features(features: Mapping[str, float]) -> None:
    if set(features) != set(FEATURE_ORDER):
        raise ValueError("feature names differ from final-only schema")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in features.values()):
        raise ValueError("features must be finite numbers")


def label_outcome(h0, final, *, gt, mask, repair_observed: bool) -> dict:
    """Join masks/GT only after freezing features; missing labels stay unknown.

    Benefit is an improvement in mean per-frame set F1 over I/V/T/IVT. This
    utility is a training target, distinct from pooled micro-F1 evaluation.
    Phase is audited but not a repair scope of the frozen five-seat policy.
    """
    initial = canonical_labels(h0)
    repaired = canonical_labels(final)
    if type(repair_observed) is not bool or set(mask) != set(TASKS):
        raise ValueError("explicit observation and all task masks are required")
    if any(type(mask[task]) is not bool for task in TASKS):
        raise ValueError("task masks must be booleans")
    if repaired["phase"] != initial["phase"]:
        raise ValueError("this whole-episode Gate contract cannot repair Phase")
    errors = {}
    gains = {}
    loss_delta = {}
    for task in TASKS:
        if not mask[task]:
            errors[task] = None
            continue
        expected = gt[task]
        low, high = TASK_ID_BOUNDS[task]
        if (not isinstance(expected, (list, tuple)) or len(set(expected)) != len(expected)
                or any(type(x) is not int or not low <= x <= high for x in expected)
                or (task == "phase" and len(expected) != 1)):
            raise ValueError("invalid observed GT labels")
        truth = set(expected)
        before, after = set(initial[task]), set(repaired[task])
        errors[task] = int(before != truth)
        loss_delta[task] = len(after ^ truth) - len(before ^ truth)

        def score(prediction, truth=truth):
            denominator = len(prediction) + len(truth)
            return 2 * len(prediction & truth) / denominator if denominator else 1.0

        gains[task] = score(after) - score(before)
    complete = all(mask[task] for task in INTERACTION_TASKS)
    observed = complete and repair_observed
    delta = sum(gains[t] for t in INTERACTION_TASKS) / 4 if observed else None
    any_harm = any(loss_delta[t] > 0 for t in INTERACTION_TASKS) if observed else None
    return {
        "h0_error_by_task": errors,
        "h0_error": int(any(errors[t] for t in INTERACTION_TASKS)) if complete else None,
        "benefit": int(delta > 1e-12) if observed else None,
        "safe_benefit": int(delta > 1e-12 and not any_harm) if observed else None,
        "utility_delta": delta,
        "any_head_harm": any_harm,
        "set_loss_delta_by_task": loss_delta if repair_observed else {},
        "repair_observed": repair_observed,
        "complete_interaction_gt": complete,
    }


def readiness(rows: Sequence[Mapping], *, target: str = "benefit", min_class_per_fit: int = 2) -> dict:
    """Check class support in every leave-one-video-out fitting partition.

    This is an engineering minimum, not a statistical power guarantee or a
    final Validation operating-point selection.
    """
    if target not in TARGETS or type(min_class_per_fit) is not int or min_class_per_fit < 1:
        raise ValueError("invalid training target or minimum class support")
    identities = [(row["video_id"], row["frame_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate target frames cannot be treated as independent examples")
    if any(row["source_split"] != "Training" for row in rows):
        raise ValueError("Gate development accepts Training only")
    if len({row["policy_id"] for row in rows}) > 1:
        raise ValueError("do not mix H0/repair policy versions in a Gate training set")
    for row in rows:
        for variant in ("features_no_tracker", "features_with_tracker"):
            validate_features(row[variant])
        if row["features_no_tracker"] != mask_tracker(row["features_with_tracker"]):
            raise ValueError("paired views must differ only by masking Tracker features")
        value = row["labels"][target]
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError("binary labels must be 0, 1 or null")
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
        folds.append({
            "held_out_video_id": video,
            "fit_video_ids": sorted({row["video_id"] for row in fit}),
            "fit_negative": counts[0], "fit_positive": counts[1],
            "held_out_examples": len(held),
            "held_out_positive": sum(row["labels"][target] for row in held),
        })
    return {
        "target": target, "can_fit_pilot": not reasons,
        "observations": len(rows), "labeled": len(usable), "unknown": len(rows) - len(usable),
        "positive": sum(row["labels"][target] for row in usable),
        "negative": sum(row["labels"][target] == 0 for row in usable),
        "positive_videos": sorted({row["video_id"] for row in usable if row["labels"][target] == 1}),
        "min_class_per_fit": min_class_per_fit, "folds": folds, "blocking_reasons": reasons,
        "statistical_sufficiency_proven": False, "deployable": False,
    }
