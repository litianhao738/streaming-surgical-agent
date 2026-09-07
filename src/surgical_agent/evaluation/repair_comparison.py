"""Offline, task-masked comparisons of frozen H0, candidate H1 and final labels.

No GT enters a proposal or admission decision here. ``None`` represents an
unavailable/failed prediction, not a successful empty prediction. H1 availability
is reported separately; missing H1 falls back to H0 for the overall policy arm.
Metrics are pooled label counts, not confidence/ranking metrics or video means.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from surgical_agent.data.constants import TASK_ID_BOUNDS

TASKS = tuple(TASK_ID_BOUNDS)
ARMS = ("h0", "h1_policy", "final")
PAIRS = (("h0", "h1_policy"), ("h0", "final"), ("h1_policy", "final"))
CHANGE_CATEGORIES = ("improved", "worsened", "equal_loss_changed", "unchanged")


def _labels(value: object, tasks: Iterable[str], owner: str) -> dict[str, set[int]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{owner} must be a task-to-label mapping")
    result = {}
    for task in tasks:
        ids = value.get(task)
        if not isinstance(ids, (list, tuple)):
            raise TypeError(f"{owner}.{task} must be a list or tuple of integers")
        lower, upper = TASK_ID_BOUNDS[task]
        if any(type(label) is not int for label in ids):
            raise TypeError(f"{owner}.{task} labels must be non-boolean integers")
        if any(not lower <= label <= upper for label in ids):
            raise ValueError(f"{owner}.{task} labels must lie in {lower}..{upper}")
        result[task] = set(ids)
    return result


def _prepare(rows: Iterable[Mapping]) -> list[dict]:
    prepared, identities = [], set()
    for row in rows:
        video_id, frame_id = row["video_id"], row["frame_id"]
        if not isinstance(video_id, str) or not video_id:
            raise TypeError("video_id must be a nonempty string")
        if type(frame_id) is not int or frame_id < 0:
            raise TypeError("frame_id must be a non-negative integer")
        identity = (video_id, frame_id)
        if identity in identities:
            raise ValueError(f"duplicate target identity: {identity}")
        identities.add(identity)
        mask = row["mask"]
        if not isinstance(mask, Mapping) or any(
            type(mask.get(task)) is not bool for task in TASKS
        ):
            raise TypeError("mask must explicitly provide a boolean for every task")
        valid_tasks = tuple(task for task in TASKS if mask[task])
        predictions = {
            arm: None if row[arm] is None else _labels(row[arm], TASKS, arm)
            for arm in ("h0", "h1", "final")
        }
        prepared.append(
            {
                "video_id": video_id,
                "frame_id": frame_id,
                "valid_tasks": valid_tasks,
                "gt": _labels(row["gt"], valid_tasks, "gt"),
                "h0": predictions["h0"],
                "h1_policy": predictions["h1"]
                if predictions["h1"] is not None
                else predictions["h0"],
                "final": predictions["final"],
                "candidate_available": predictions["h1"] is not None,
            }
        )
    return prepared


def _predicted(row: dict, arm: str, task: str) -> set[int]:
    return row[arm][task] if row[arm] is not None else set()


def _exact(row: dict, arm: str, task: str) -> bool:
    return row[arm] is not None and row[arm][task] == row["gt"][task]


def _all_exact(row: dict, arm: str) -> bool:
    return bool(row["valid_tasks"]) and all(
        _exact(row, arm, task) for task in row["valid_tasks"]
    )


def _arm_metrics(rows: list[dict], arm: str) -> dict:
    tasks = {}
    for task in TASKS:
        valid = [row for row in rows if task in row["valid_tasks"]]
        tp = fp = fn = exact = failed = 0
        for row in valid:
            predicted, expected = _predicted(row, arm, task), row["gt"][task]
            tp += len(predicted & expected)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
            exact += _exact(row, arm, task)
            failed += row[arm] is None
        tasks[task] = {
            "valid_targets": len(valid),
            "failed_predictions": failed,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "exact_matches": exact,
            "micro_precision": (tp / (tp + fp) if tp + fp else 0.0) if valid else None,
            "micro_recall": (tp / (tp + fn) if tp + fn else 0.0) if valid else None,
            "micro_f1": (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
            if valid
            else None,
            "exact_set_accuracy": exact / len(valid) if valid else None,
        }
    valid_rows = [row for row in rows if row["valid_tasks"]]
    exact = sum(_all_exact(row, arm) for row in valid_rows)
    return {
        "tasks": tasks,
        "all_valid_heads_exact": {
            "valid_targets": len(valid_rows),
            "exact_matches": exact,
            "accuracy": exact / len(valid_rows) if valid_rows else None,
        },
    }


def _pair_detail(row: dict, before: str, after: str) -> dict:
    tasks = {}
    for task in row["valid_tasks"]:
        left, right = _predicted(row, before, task), _predicted(row, after, task)
        expected = row["gt"][task]
        left_loss, right_loss = len(left ^ expected), len(right ^ expected)
        changed = left != right or (row[before] is None) != (row[after] is None)
        category = (
            "improved"
            if right_loss < left_loss
            else "worsened"
            if right_loss > left_loss
            else "equal_loss_changed"
            if changed
            else "unchanged"
        )
        left_exact, right_exact = _exact(row, before, task), _exact(row, after, task)
        tasks[task] = {
            "category": category,
            "loss_before": left_loss,
            "loss_after": right_loss,
            "wrong_to_exact": not left_exact and right_exact,
            "exact_to_wrong": left_exact and not right_exact,
        }
    categories = {detail["category"] for detail in tasks.values()}
    category = (
        "unscored"
        if not tasks
        else "mixed"
        if {"improved", "worsened"} <= categories
        else "improved"
        if "improved" in categories
        else "worsened"
        if "worsened" in categories
        else "equal_loss_changed"
        if "equal_loss_changed" in categories
        else "unchanged"
    )
    left_exact, right_exact = _all_exact(row, before), _all_exact(row, after)
    return {
        "tasks": tasks,
        "category": category,
        "wrong_to_exact": bool(tasks) and not left_exact and right_exact,
        "exact_to_wrong": bool(tasks) and left_exact and not right_exact,
        "loss_before": sum(detail["loss_before"] for detail in tasks.values()),
        "loss_after": sum(detail["loss_after"] for detail in tasks.values()),
    }


def _change_counts(details: list[dict], *, frames: bool = False) -> dict:
    categories = (
        (*CHANGE_CATEGORIES, "mixed", "unscored") if frames else CHANGE_CATEGORIES
    )
    return {
        "valid_targets": sum(detail["category"] != "unscored" for detail in details),
        **{
            category: sum(detail["category"] == category for detail in details)
            for category in categories
        },
        "wrong_to_exact": sum(detail["wrong_to_exact"] for detail in details),
        "exact_to_wrong": sum(detail["exact_to_wrong"] for detail in details),
    }


def _cohort_report(rows: list[dict]) -> dict:
    paired = {}
    for before, after in PAIRS:
        details = [_pair_detail(row, before, after) for row in rows]
        paired[f"{before}_to_{after}"] = {
            "frames": _change_counts(details, frames=True),
            "tasks": {
                task: _change_counts(
                    [
                        detail["tasks"][task]
                        for detail in details
                        if task in detail["tasks"]
                    ]
                )
                for task in TASKS
            },
        }
    return {
        "targets": len(rows),
        "arms": {arm: _arm_metrics(rows, arm) for arm in ARMS},
        "paired": paired,
    }


def compute_repair_comparison(rows: Iterable[Mapping]) -> dict:
    """Compare saved predictions without modifying them or consulting any model.

    Each row requires video_id, frame_id, h0, h1, final, gt and mask. Predictions
    are five-head mappings of integer lists, or None. GT needs labels only for
    true task masks; every mask must be explicit. Duplicate identities fail.

    F1 has zero_division=0; a head with no valid GT returns null metrics. A failed
    prediction counts as an empty set for TP/FP/FN, but cannot earn exact credit,
    including on valid empty GT. All-valid-head exact uses frames with at least
    one valid head and compares every valid head, not only complete-GT frames.

    Changes use symmetric-difference loss per head: lower=improved,
    higher=worsened, changed with equal loss=equal_loss_changed. A failure-status
    change also counts as a change. Mixed frames have both improved and worsened
    heads, irrespective of net loss. Partial improvement is not wrong_to_exact.
    Masked-only changes are unscored for that head. No costs are inferred here.
    """
    prepared = _prepare(rows)
    available = [row for row in prepared if row["candidate_available"]]
    return {
        "schema_version": "repair_comparison_v1",
        "protocol": "task_masked_pooled_micro; zero_division=0; missing_h1_falls_back_to_h0",
        **_cohort_report(prepared),
        "candidate_availability": {
            "available": len(available),
            "total": len(prepared),
            "rate": len(available) / len(prepared) if prepared else None,
        },
        "candidate_available": _cohort_report(available),
        "details": [
            {
                "video_id": row["video_id"],
                "frame_id": row["frame_id"],
                "valid_tasks": list(row["valid_tasks"]),
                "candidate_available": row["candidate_available"],
                "pairs": {
                    f"{before}_to_{after}": _pair_detail(row, before, after)
                    for before, after in PAIRS
                },
            }
            for row in prepared
        ],
    }
