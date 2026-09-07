"""Audit saved presence-review experiments; never call a model or choose a policy.

GT is consumed only from the completed run's scored_predictions.json. The
candidate oracle is a retrospective bound on edits exposed by H1, not a usable
decision rule. Admission means an edit actually applied to the output: model
approval and application can differ because of dependency checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from surgical_agent.evaluation.repair_comparison import (
    TASKS,
    compute_repair_comparison,
)

FINAL_ARMS = ("old_final", "final_a", "final_b")
EDIT_TASKS = tuple(task for task in TASKS if task != "phase")


def _edits(before: Mapping, after: Mapping | None, task: str) -> set[tuple[str, int]]:
    if after is None:
        return set()
    left, right = set(before[task]), set(after[task])
    return {("ADD", label) for label in right - left} | {
        ("REMOVE", label) for label in left - right
    }


def _empty_admission() -> dict:
    return {
        "valid_targets_with_h0": 0,
        "masked_targets": 0,
        "missing_h0_targets": 0,
        "unavailable_final_targets": 0,
        "beneficial_proposed": 0,
        "beneficial_applied": 0,
        "harmful_proposed": 0,
        "harmful_applied": 0,
        "unexpected_applied": 0,
    }


def _finish_admission(counts: dict) -> dict:
    return {
        **counts,
        "beneficial_application_rate": (
            counts["beneficial_applied"] / counts["beneficial_proposed"]
            if counts["beneficial_proposed"]
            else None
        ),
        "harmful_application_rate": (
            counts["harmful_applied"] / counts["harmful_proposed"]
            if counts["harmful_proposed"]
            else None
        ),
    }


def _edit_admission(rows: Sequence[Mapping], arm: str) -> dict:
    by_task = {task: _empty_admission() for task in EDIT_TASKS}
    unexpected = []
    details = []
    phase_changes = []
    for row in rows:
        identity = {"video_id": row["video_id"], "frame_id": row["frame_id"]}
        h0, h1, final = row["h0"], row["h1"], row[arm]
        if h0 is not None and final is not None and set(h0["phase"]) != set(final["phase"]):
            phase_changes.append(identity)
        for task in EDIT_TASKS:
            counts = by_task[task]
            proposed = set() if h0 is None else _edits(h0, h1, task)
            applied = set() if h0 is None else _edits(h0, final, task)
            for operation, label in sorted(applied - proposed):
                unexpected.append({
                    **identity, "task": task, "operation": operation, "label_id": label,
                    "task_gt_valid": row["mask"][task],
                })
            if not row["mask"][task]:
                counts["masked_targets"] += 1
                continue
            if h0 is None:
                counts["missing_h0_targets"] += 1
                continue
            counts["valid_targets_with_h0"] += 1
            counts["unavailable_final_targets"] += final is None
            counts["unexpected_applied"] += len(applied - proposed)
            gt = set(row["gt"][task])
            for operation, label in sorted(proposed):
                beneficial = (label in gt) if operation == "ADD" else (label not in gt)
                benefit = "beneficial" if beneficial else "harmful"
                counts[f"{benefit}_proposed"] += 1
                counts[f"{benefit}_applied"] += (operation, label) in applied
                details.append({
                    **identity, "task": task, "operation": operation, "label_id": label,
                    "beneficial": beneficial, "applied": (operation, label) in applied,
                })
    pooled = {
        key: sum(counts[key] for counts in by_task.values())
        for key in _empty_admission()
    }
    return {
        "meaning": "actual output edits, not model approval; pooled target counts are task-target pairs",
        "approval_metadata_evaluated": False,
        "tasks": {task: _finish_admission(counts) for task, counts in by_task.items()},
        "pooled": _finish_admission(pooled),
        "applied_subset_of_proposed": not unexpected,
        "unexpected_edits": unexpected,
        "phase_changed_targets": phase_changes,
        "details": details,
    }


def _independent_counts(rows: Sequence[Mapping], arm: str) -> dict:
    """Separate counting implementation cross-checks scorer TP/FP/FN and exact."""
    by_task = {}
    for task in TASKS:
        counts = dict.fromkeys(
            ("valid_targets", "failed_predictions", "tp", "fp", "fn", "exact_matches"), 0
        )
        for row in rows:
            if not row["mask"][task]:
                continue
            pred = row["h1"] if arm == "h1_policy" else row[arm]
            if arm == "h1_policy" and pred is None:
                pred = row["h0"]
            expected = set(row["gt"][task])
            actual = set() if pred is None else set(pred[task])
            counts["valid_targets"] += 1
            counts["failed_predictions"] += pred is None
            counts["tp"] += sum(label in expected for label in actual)
            counts["fp"] += sum(label not in expected for label in actual)
            counts["fn"] += sum(label not in actual for label in expected)
            counts["exact_matches"] += pred is not None and actual == expected
        by_task[task] = counts
    return by_task


def _assert_counts(rows: Sequence[Mapping], arms: Mapping) -> None:
    for arm, report in arms.items():
        recomputed = _independent_counts(rows, arm)
        for task in TASKS:
            for key, value in recomputed[task].items():
                if report["tasks"][task][key] != value:
                    raise AssertionError(f"independent count mismatch: {arm}.{task}.{key}")


def _candidate_diagnostic(rows: Sequence[Mapping]) -> dict:
    details = []
    oracle_rows = []
    for row in rows:
        oracle = None if row["h0"] is None else {
            task: list(row["h0"][task]) for task in TASKS
        }
        if row["mask"]["ivt"] and row["h0"] is not None:
            h0 = set(row["h0"]["ivt"])
            h1 = h0 if row["h1"] is None else set(row["h1"]["ivt"])
            gt = set(row["gt"]["ivt"])
            correct_new = (h1 - h0) & gt
            removable_false = (h0 - h1) - gt
            optimal = (h0 | correct_new) - removable_false
            oracle["ivt"] = sorted(optimal)
            details.append({
                "video_id": row["video_id"], "frame_id": row["frame_id"],
                "candidate_available": row["h1"] is not None,
                "h0_missing_gt": sorted(gt - h0),
                "correct_ivt_additions_exposed": sorted(correct_new),
                "missed_by_both": sorted(gt - (h0 | h1)),
                "h0_correct_removed_by_h1": sorted((h0 - h1) & gt),
                "false_ivt_additions": sorted((h1 - h0) - gt),
                "false_h0_ivt_removals_exposed": sorted(removable_false),
                "labelwise_oracle_loss_reduction": len(h0 ^ gt) - len(optimal ^ gt),
                "whole_h1_loss_reduction": len(h0 ^ gt) - len(h1 ^ gt),
                "h0_exact": h0 == gt,
                "h1_exact": row["h1"] is not None and h1 == gt,
                "either_whole_answer_exact": h0 == gt or (row["h1"] is not None and h1 == gt),
                "labelwise_oracle_exact": optimal == gt,
            })
        oracle_rows.append({**row, "final": oracle})
    comparison = compute_repair_comparison(oracle_rows)
    keys = (
        "h0_missing_gt", "correct_ivt_additions_exposed", "missed_by_both",
        "h0_correct_removed_by_h1", "false_ivt_additions", "false_h0_ivt_removals_exposed",
    )
    return {
        "scope": "IVT only; retrospective GT oracle, never used for model or policy selection",
        "oracle_definition": "Apply only beneficial H0-to-H1 IVT edits, ignore component dependencies; optimistic bound",
        "valid_targets_with_h0": len(details),
        "valid_targets_missing_h0": sum(row["mask"]["ivt"] and row["h0"] is None for row in rows),
        "candidate_available_targets": sum(item["candidate_available"] for item in details),
        "counts": {key: sum(len(item[key]) for item in details) for key in keys},
        "targets_with_potential_labelwise_improvement": sum(
            item["labelwise_oracle_loss_reduction"] > 0 for item in details
        ),
        "targets_with_whole_candidate_improvement": sum(
            item["whole_h1_loss_reduction"] > 0 for item in details
        ),
        "either_whole_answer_exact_targets": sum(item["either_whole_answer_exact"] for item in details),
        "labelwise_oracle_exact_targets": sum(item["labelwise_oracle_exact"] for item in details),
        "labelwise_oracle_ivt_metrics": comparison["arms"]["final"]["tasks"]["ivt"],
        "details": details,
    }


def compute_presence_audit(rows: Sequence[Mapping]) -> dict:
    comparisons = {}
    for arm in FINAL_ARMS:
        comparisons[arm] = compute_repair_comparison([
            {**row, "final": row[arm]} for row in rows
        ])
    first = comparisons[FINAL_ARMS[0]]
    arms = {
        "h0": first["arms"]["h0"],
        "h1_policy": first["arms"]["h1_policy"],
        **{arm: comparisons[arm]["arms"]["final"] for arm in FINAL_ARMS},
    }
    available = [row for row in rows if row["h1"] is not None]
    available_arms = {
        "h0": first["candidate_available"]["arms"]["h0"],
        "h1_policy": first["candidate_available"]["arms"]["h1_policy"],
        **{arm: comparisons[arm]["candidate_available"]["arms"]["final"] for arm in FINAL_ARMS},
    }
    _assert_counts(rows, arms)
    _assert_counts(available, available_arms)
    coverage = {
        "targets": len(rows),
        "videos": dict(sorted(Counter(row["video_id"] for row in rows).items())),
        "candidate_availability": first["candidate_availability"],
        "predictions_available": {
            arm: sum(row[arm] is not None for row in rows)
            for arm in ("h0", "h1", *FINAL_ARMS)
        },
        "decision_metadata": {
            key: dict(Counter(str(row.get(key, "UNRECORDED")) for row in rows))
            for key in ("decision_a", "decision_b", "old_decision", "status")
        },
    }
    return {
        "schema_version": "presence_review_offline_audit_v1",
        "offline_only": True,
        "protocol": first["protocol"],
        "coverage": coverage,
        "arms": arms,
        "paired_vs_h0": {
            arm: comparisons[arm]["paired"]["h0_to_final"] for arm in FINAL_ARMS
        },
        "candidate_available": {
            "targets": len(available), "arms": available_arms,
            "paired_vs_h0": {
                arm: comparisons[arm]["candidate_available"]["paired"]["h0_to_final"]
                for arm in FINAL_ARMS
            },
        },
        "edit_application": {arm: _edit_admission(rows, arm) for arm in FINAL_ARMS},
        "candidate_ivt_diagnostic": _candidate_diagnostic(rows),
        "independent_metric_counts_match": True,
        "comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    source = args.run_dir / "scored_predictions.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError("scored_predictions.json must contain a list of rows")
    report = compute_presence_audit(rows)
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    report["sources"] = {
        "scored_predictions_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    calls_path = args.run_dir / "calls_summary.json"
    if calls_path.exists():
        report["calls_summary_as_recorded"] = json.loads(calls_path.read_text(encoding="utf-8"))
        report["sources"]["calls_summary_sha256"] = hashlib.sha256(calls_path.read_bytes()).hexdigest()
    else:
        report["calls_summary_as_recorded"] = None
    output = args.output or args.run_dir / "presence_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(output), "targets": report["coverage"]["targets"],
        "independent_metric_counts_match": True,
    }))


if __name__ == "__main__":
    main()
