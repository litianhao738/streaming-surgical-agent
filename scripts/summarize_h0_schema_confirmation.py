"""Read saved scores and GT sets; no provider calls or direct dataset reads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TASKS = ("instrument", "verb", "target", "ivt", "phase")
ARMS = ("baseline", "schema_only")
ROOT = Path(__file__).resolve().parents[1]


def aggregate(frames):
    keys = [(r["video_id"], r["frame_id"]) for r in frames]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate targets in pooled evidence")
    if any(not all(r["mask"].get(t, False) for t in TASKS) for r in frames):
        raise ValueError("comparison requires all five valid masks")
    metrics = {}
    for task in TASKS:
        tp = fp = fn = correct = 0
        for row in frames:
            gt = set(row["gt"][task])
            predicted = set(row["h0"][task]) if row["status"] == "OK" else set()
            tp += len(gt & predicted)
            fp += len(predicted - gt)
            fn += len(gt - predicted)
            correct += row["status"] == "OK" and predicted == gt
        metrics[task] = {
            "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": tp / (tp + fp) if tp + fp else 0,
            "micro_recall": tp / (tp + fn) if tp + fn else 0,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "correct": correct, "valid_gt": len(frames),
            "exact_accuracy": correct / len(frames) if frames else None,
        }
    joint = sum(row["status"] == "OK" and all(
        set(row["h0"][task]) == set(row["gt"][task]) for task in TASKS
    ) for row in frames)
    return {
        "tasks": metrics, "frames": len(frames), "all_five_exact": joint,
        "five_head_equal_weight_mean_micro_f1": sum(v["micro_f1"] for v in metrics.values()) / len(TASKS),
        "per_video": {video: len([r for r in frames if r["video_id"] == video])
                      for video in sorted({r["video_id"] for r in frames})},
    }


def compare(baseline, candidate):
    before = {(r["video_id"], r["frame_id"]): r for r in baseline}
    after = {(r["video_id"], r["frame_id"]): r for r in candidate}
    if set(before) != set(after):
        raise ValueError("unpaired target sets")
    for key, row in before.items():
        if row["gt"] != after[key]["gt"] or row["mask"] != after[key]["mask"]:
            raise ValueError("GT differs between paired arms")
    comparisons = {}
    for task in TASKS:
        same = gain = loss = 0
        for key, row in before.items():
            changed = after[key]
            ok_b, ok_c = row["status"] == "OK", changed["status"] == "OK"
            pred_b = set(row["h0"][task]) if ok_b else set()
            pred_c = set(changed["h0"][task]) if ok_c else set()
            truth = set(row["gt"][task])
            same += ok_b and ok_c and pred_b == pred_c
            exact_b, exact_c = ok_b and pred_b == truth, ok_c and pred_c == truth
            gain += exact_c and not exact_b
            loss += exact_b and not exact_c
        comparisons[task] = {"same_prediction_set": same, "exact_gain": gain, "exact_loss": loss}
    return comparisons


def summarize(current, previous):
    result = {
        "primary": "new24", "secondary": "pooled40",
        "caveat": "Pooled results include 16 previously examined development targets; not an independent 40-target confirmation. Equal-head mean is descriptive, not the adoption rule.",
        "cohorts": {},
    }
    old_keys = {(r["video_id"], r["frame_id"]) for r in previous["groups"]["baseline"]["evaluation"]["frames"]}
    new_keys = {(r["video_id"], r["frame_id"]) for r in current["groups"]["baseline"]["evaluation"]["frames"]}
    if old_keys & new_keys:
        raise ValueError("new confirmation overlaps the previous experiment")
    for name, sources in (("new24", [current]), ("previous16", [previous]), ("pooled40", [previous, current])):
        arm_frames, groups = {}, {}
        for arm in ARMS:
            arm_frames[arm] = [frame for source in sources for frame in source["groups"][arm]["evaluation"]["frames"]]
            groups[arm] = aggregate(arm_frames[arm])
            cost = sum(source["groups"][arm]["usage"]["reported_cost_usd"] for source in sources)
            calls = sum(source["groups"][arm]["usage"]["provider_calls"] for source in sources)
            native_n = sum(source["groups"][arm]["native_usage_count"] for source in sources)
            equivalent = sum(source["groups"][arm]["native_means"]["uncached_price_equivalent_not_actual_cost"] * source["groups"][arm]["native_usage_count"] for source in sources)
            groups[arm]["usage"] = {
                "actual_cost_usd": cost, "provider_calls": calls,
                "mean_actual_cost_usd": cost / calls if calls else None,
                "mean_uncached_price_equivalent_usd": equivalent / native_n if native_n else None,
            }
            groups[arm]["per_video_metrics"] = {video: aggregate([r for r in arm_frames[arm] if r["video_id"] == video])
                                              for video in groups[arm]["per_video"]}
        base_cost = groups["baseline"]["usage"]["actual_cost_usd"]
        candidate_cost = groups["schema_only"]["usage"]["actual_cost_usd"]
        result["cohorts"][name] = {
            "groups": groups,
            "paired_exact_changes": compare(arm_frames["baseline"], arm_frames["schema_only"]),
            "cost_saving_percent": 100 * (1 - candidate_cost / base_cost) if base_cost else None,
            "f1_delta_percentage_points": {t: 100 * (groups["schema_only"]["tasks"][t]["micro_f1"] - groups["baseline"]["tasks"][t]["micro_f1"]) for t in TASKS},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, default=ROOT / "artifacts/preflight/h0_schema_confirmation_qwen0902_20260906/summary.json")
    parser.add_argument("--previous", type=Path, default=ROOT / "artifacts/preflight/h0_prompt_refinement_qwen0902_20260906/summary.json")
    args = parser.parse_args()
    paths = (args.current, args.previous)
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result = summarize(*(json.loads(path.read_text(encoding="utf-8")) for path in paths))
    result["source_summary_sha256"] = before
    if before != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}:
        raise ValueError("source summaries changed during analysis")
    output = args.current.parent / "comparison_new24_and_pooled40.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, cohort in result["cohorts"].items():
        print(json.dumps({"cohort": name, "cost_saving_percent": cohort["cost_saving_percent"],
                          "f1_delta_percentage_points": cohort["f1_delta_percentage_points"],
                          "equal_head_means": {arm: group["five_head_equal_weight_mean_micro_f1"] for arm, group in cohort["groups"].items()}}))


if __name__ == "__main__":
    main()
