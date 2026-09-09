"""Independently score a closed, frozen GraphRAG reviewer comparison.

This module never calls a model. All selected targets, including failed H0 and
fallback repairs, remain in the denominator for every available GT task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_grounded_api_pipeline import score_saved
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison

TASKS = tuple(TASK_ID_BOUNDS)
ARMS = ("none", "flat", "graph")
EDIT_FIELDS = ("correct_additions", "wrong_additions", "correct_deletions", "wrong_deletions")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_snapshots(output):
    """Run every isolation/coverage guard before the first dataset GT read."""
    output = Path(output)
    plan = read(output / "plan.json")
    completion = read(output / "completion.json")
    budget = read(output / "budget.json")
    snapshots = {}
    for name in ("plan", "predictions", "budget"):
        snapshots[name] = sha(output / f"{name}.json")
        if completion.get(f"{name}_sha256") != snapshots[name]:
            raise ValueError(f"{name} snapshot differs from closed completion record")
    if not completion.get("closed_utc") or budget.get("stopped") is not True:
        raise ValueError("GT scoring requires a closed inference ledger")
    if any(row.get("status") == "DISPATCHED" for row in budget["calls"]):
        raise ValueError("an inference call is still outstanding")
    if tuple(plan.get("arms", ())) != ARMS:
        raise ValueError("expected the frozen none/flat/graph comparison")
    if not plan.get("source_sha256"):
        raise ValueError("the preflight source snapshot is missing")
    for name, digest in plan["source_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != digest:
            raise ValueError(f"frozen source changed or is missing: {name}")
    expected = [(r["key"], r["video_id"], r["frame_id"]) for r in plan["selection"]]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("empty or duplicate planned target identities")
    if len({(v, f) for _, v, f in expected}) != len(expected):
        raise ValueError("a video/frame appears more than once in the plan")
    if len({k for k, _, _ in expected}) != len(expected):
        raise ValueError("target keys must be unique")
    targets = read(output / "predictions.json")["targets"]
    actual = [(r["key"], r["video_id"], r["frame_id"]) for r in targets]
    if actual != expected:
        raise ValueError("predictions must retain every planned target in frozen order")
    for row in targets:
        if "h0" not in row or set(row.get("arms", {})) != set(ARMS):
            raise ValueError("every target requires H0 and all three arms, including failures")
        for arm in ARMS:
            if not {"prediction", "status"} <= row["arms"][arm].keys():
                raise ValueError("every arm must explicitly record prediction and status")
        for prediction in (row["h0"], *(row["arms"][a]["prediction"] for a in ARMS)):
            if prediction is None:
                continue
            if not isinstance(prediction, dict) or set(prediction) != set(TASKS):
                raise ValueError("predictions require the five canonical label-list heads")
            for task, (lower, upper) in TASK_ID_BOUNDS.items():
                labels = prediction[task]
                if (not isinstance(labels, list)
                        or any(type(v) is not int or not lower <= v <= upper for v in labels)
                        or len(labels) != len(set(labels))):
                    raise ValueError(f"invalid canonical prediction labels for {task}")
    return plan, targets, budget, snapshots


def frame_delta(before, after, gt, mask):
    """Attribute edits only to heads with GT, with failures kept explicit."""
    tasks = {}
    for task in TASKS:
        if not mask[task]:
            continue
        left = set(before[task]) if before is not None else set()
        right = set(after[task]) if after is not None else set()
        expected = set(gt[task])
        added, deleted = right - left, left - right
        tasks[task] = {
            "before": sorted(left), "after": sorted(right), "gt": sorted(expected),
            "error_count_before": len(left ^ expected),
            "error_count_after": len(right ^ expected),
            "exact_before": before is not None and left == expected,
            "exact_after": after is not None and right == expected,
            "correct_additions": sorted(added & expected),
            "wrong_additions": sorted(added - expected),
            "correct_deletions": sorted(deleted - expected),
            "wrong_deletions": sorted(deleted & expected),
        }
    loss_before = sum(t["error_count_before"] for t in tasks.values())
    loss_after = sum(t["error_count_after"] for t in tasks.values())
    improved = any(t["error_count_after"] < t["error_count_before"] for t in tasks.values())
    worsened = any(t["error_count_after"] > t["error_count_before"] for t in tasks.values())
    changed = any(t["before"] != t["after"] for t in tasks.values())
    exact_before = bool(tasks) and all(t["exact_before"] for t in tasks.values())
    exact_after = bool(tasks) and all(t["exact_after"] for t in tasks.values())
    if not tasks:
        category = "unscored"
    elif before is None or after is None:
        category = ("unchanged_failure" if before is after else
                    "recovered_prediction" if before is None else "prediction_failed")
    elif not exact_before and exact_after:
        category = "fully_corrected"
    elif improved and worsened:
        category = "mixed"
    elif loss_after < loss_before:
        category = "partial_improvement"
    elif loss_after > loss_before:
        category = "harm"
    elif changed:
        category = "no_benefit_replacement"
    else:
        category = "unchanged"
    edits = {key: sum(len(t[key]) for t in tasks.values()) for key in EDIT_FIELDS}
    fixed = edits["correct_additions"] + edits["correct_deletions"]
    introduced = edits["wrong_additions"] + edits["wrong_deletions"]
    return {
        "category": category, "tasks": tasks,
        "error_count_before": loss_before, "error_count_after": loss_after,
        "net_errors_removed": loss_before - loss_after,
        "fixed_label_errors": fixed, "introduced_label_errors": introduced,
        "both_beneficial_and_harmful_edits": bool(fixed and introduced),
        "before_prediction_available": before is not None,
        "after_prediction_available": after is not None,
        "wrong_to_exact": before is not None and after is not None and not exact_before and exact_after,
        "exact_to_wrong": before is not None and after is not None and exact_before and not exact_after,
        **edits,
    }


def summarize_deltas(rows):
    return {
        "categories": dict(Counter(r["category"] for r in rows)),
        **{key: sum(r[key] for r in rows) for key in (
            "error_count_before", "error_count_after", "net_errors_removed",
            "fixed_label_errors", "introduced_label_errors", "wrong_to_exact",
            "exact_to_wrong", *EDIT_FIELDS)},
        "tasks": {
            task: {
                "valid_targets": sum(task in r["tasks"] for r in rows),
                **{key: sum(len(r["tasks"][task][key]) for r in rows if task in r["tasks"])
                   for key in EDIT_FIELDS},
            } for task in TASKS
        },
    }


def _distribution(values):
    values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0]
    return {"n": len(values), "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def runtime_summary(targets, budget):
    """Report measured elapsed time; never assign independent time to cached calls."""
    timing = {}
    for arm in ARMS:
        observed = [r["arms"][arm] for r in targets]
        independent = [r for r in observed if not r.get("shared_from")]
        timing[arm] = {
            "panel_seconds": _distribution(r.get("panel_seconds") for r in independent),
            "local_retrieval_seconds": _distribution(r.get("retrieval_seconds") for r in observed),
            "reference_chars": _distribution(r.get("reference_chars") for r in observed),
            "shared_panels_excluded_from_independent_latency": len(observed) - len(independent),
        }
    for arm in ("flat", "graph"):
        pairs = []
        for row in targets:
            base, compared = row["arms"]["none"], row["arms"][arm]
            left, right = base.get("panel_seconds"), compared.get("panel_seconds")
            if (not base.get("shared_from") and not compared.get("shared_from")
                    and isinstance(left, (int, float)) and left > 0
                    and isinstance(right, (int, float)) and right >= 0):
                pairs.append({"key": row["key"], "seconds_added": right - left,
                              "percent_added": (right / left - 1) * 100})
        timing[arm]["paired_to_none"] = {
            "pairs": pairs,
            "median_seconds_added": statistics.median(p["seconds_added"] for p in pairs) if pairs else None,
            "median_percent_added": statistics.median(p["percent_added"] for p in pairs) if pairs else None,
        }
    groups = defaultdict(lambda: {
        "calls": 0, "charges_by_account": defaultdict(Decimal),
        "unknown_reserved_calls": 0, "charge_kinds": Counter(),
        "prompt_tokens": 0, "completion_tokens": 0, "calls_with_token_usage": 0,
        "elapsed_seconds": [], "statuses": Counter(),
    })
    for row in budget["calls"]:
        stage = row["stage"]
        arm = next((a for a in ARMS if stage.startswith(a + "_")), "shared_h0_and_proposal")
        for name in (arm, "all"):
            group = groups[name]
            group["calls"] += 1
            group["charges_by_account"][row["account"]] += Decimal(str(row["charge"]))
            group["charge_kinds"][row.get("charge_kind", "unknown")] += 1
            group["unknown_reserved_calls"] += row.get("charge_kind") == "unknown_reserved"
            group["statuses"][row["status"]] += 1
            usage = row.get("usage", {})
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                group["calls_with_token_usage"] += 1
                group["prompt_tokens"] += usage["prompt_tokens"]
                group["completion_tokens"] += usage["completion_tokens"]
            group["elapsed_seconds"].append(row.get("elapsed_seconds"))
    costs = {}
    for name, group in groups.items():
        costs[name] = {**group,
                       "charges_by_account": {k: str(v) for k, v in group["charges_by_account"].items()},
                       "charge_kinds": dict(group["charge_kinds"]),
                       "statuses": dict(group["statuses"]),
                       "elapsed_seconds": _distribution(group["elapsed_seconds"])}
    return {"timing": timing, "dispatched_calls": costs,
            "notes": ["Native USD and estimated CNY are reported separately; no exchange rate is assumed.",
                      "Shared panels carry no new API charge and are not independent latency observations.",
                      "Small-sample timings include provider variation; no stable p95 or causal latency claim."]}


def score(output, adapter):
    output = Path(output)
    plan, targets, budget, snapshots = validate_snapshots(output)
    data = [{"video_id": r["video_id"], "frame_id": r["frame_id"],
             "h0": r["h0"], "h1": None, "final": r["arms"]["none"]["prediction"]}
            for r in targets]
    # The first GT read is deliberately below every inference/snapshot guard.
    baseline_report, truth = score_saved(adapter, data)
    truth_by_id = {(r["video_id"], r["frame_id"]): r for r in truth}
    reports = {"h0": baseline_report["arms"]["h0"]}
    details = []
    comparisons = {}
    pairs = [("h0", a) for a in ARMS] + [("none", a) for a in ("flat", "graph")]
    for arm in ARMS:
        arm_data = []
        for row in targets:
            gt = truth_by_id[(row["video_id"], row["frame_id"])]
            arm_data.append({**gt, "final": row["arms"][arm]["prediction"]})
        reports[arm] = compute_repair_comparison(arm_data)["arms"]["final"]
    for row in targets:
        gt = truth_by_id[(row["video_id"], row["frame_id"])]
        predictions = {"h0": row["h0"], **{a: row["arms"][a]["prediction"] for a in ARMS}}
        details.append({"key": row["key"], "video_id": row["video_id"], "frame_id": row["frame_id"],
                        "mask": gt["mask"],
                        "comparisons": {f"{left}_to_{right}": frame_delta(
                            predictions[left], predictions[right], gt["gt"], gt["mask"])
                            for left, right in pairs}})
    for left, right in pairs:
        name = f"{left}_to_{right}"
        comparisons[name] = summarize_deltas([r["comparisons"][name] for r in details])
    coverage = {
        "selected_targets": len(targets), "scored_targets": len(truth),
        "h0_available": sum(r["h0"] is not None for r in targets),
        "arm_predictions_available": {a: sum(r["arms"][a]["prediction"] is not None for r in targets) for a in ARMS},
        "arm_statuses": {a: dict(Counter(r["arms"][a]["status"] for r in targets)) for a in ARMS},
        "all_planned_targets_retained": True, "missing_gt_excluded_per_task_mask": True,
        "failed_predictions_never_receive_exact_credit": True,
        "fallback_prediction_is_not_evidence_of_successful_review": True,
    }
    metrics = {
        "schema_version": "graph_review_comparison_v1", "snapshot_sha256": snapshots,
        "source_sha256": plan["source_sha256"], "coverage": coverage,
        "metrics": reports, "comparisons": comparisons,
        "runtime": runtime_summary(targets, budget),
        "protocol": "task-masked pooled label micro metrics; zero_division=0; all frozen targets retained",
        "delta_definition": "Mixed means some valid heads improve and others worsen; same-loss changes are no-benefit replacements. Beneficial and harmful edits are also counted individually.",
    }
    # Detect modification or resumed inference while the independent score ran.
    validate_snapshots(output)
    if any(sha(output / f"{name}.json") != digest for name, digest in snapshots.items()):
        raise ValueError("inference snapshots changed during scoring")
    atomic_write_json(output / "metrics.json", metrics)
    atomic_write_json(output / "frame_deltas.json", {"snapshot_sha256": snapshots, "targets": details})
    atomic_write_json(output / "scored_truth.json", truth)
    print(json.dumps({"coverage": coverage,
                      "f1": {a: {t: m["micro_f1"] for t, m in report["tasks"].items()}
                             for a, report in reports.items()},
                      "changes": {name: value["categories"] for name, value in comparisons.items()}}), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    score(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
