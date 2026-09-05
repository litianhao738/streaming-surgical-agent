"""Offline paired smoke audit. No model calls; GT is read only here.

This reports masked set-exact counts, not full-dataset mAP. Failed inference
timestamps remain in the execution counts and are excluded from paired
semantic comparisons, never scored as empty correct predictions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.frame_ground_truth import aggregate_evaluation_target

TASKS = ("instrument", "verb", "target", "ivt", "phase")
ATTRIBUTES = ("instrument_ids", "verb_ids", "target_ids", "triplet_ids", "phase_id")


def _jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _labels(hypothesis):
    return {
        task: ([hypothesis[attr]] if task == "phase" else sorted(hypothesis[attr]))
        for task, attr in zip(TASKS, ATTRIBUTES, strict=True)
    }


def _usage(run):
    rows = _jsonl(run / "api_usage.jsonl")
    return rows, {
        "logical_calls": len(rows),
        "cache_hits": sum(row["cache_hit"] for row in rows),
        "provider_calls": sum(row["provider_call_count"] for row in rows),
        "reported_provider_cost_usd": round(
            sum(row.get("provider_cost") or 0 for row in rows), 6
        ),
        "unpriced_provider_calls": sum(
            row["provider_call_count"]
            for row in rows
            if row.get("provider_cost") is None
        ),
        "errors": dict(
            Counter(row["error"]["code"] for row in rows if row.get("error"))
        ),
        "returned_models": sorted(
            {
                row["returned_model_identifier"]
                for row in rows
                if row.get("returned_model_identifier")
            }
        ),
    }


def audit(*, before, after, cache, dataset, video):
    records = {
        name: {
            row["observation"]["frame_id"]: row
            for row in _jsonl(run / "final_records" / f"{video}.jsonl")
        }
        for name, run in (("before", before), ("after", after))
    }
    initial_usage, before_usage = _usage(before)
    _, after_usage = _usage(after)
    paired = sorted(set(records["before"]) & set(records["after"]))
    paired = [
        frame
        for frame in paired
        if all(
            records[name][frame]["outcome"]["kind"] == "SEMANTIC" for name in records
        )
    ]
    h0 = {}
    for row in initial_usage:
        if row.get("error") or not row["request"]["prompt_version"].startswith(
            "joint_"
        ):
            continue
        frame = int(row["request"]["images"][-1]["identifier"].rsplit(":", 1)[-1])
        h0[frame] = json.loads(
            (cache / f"{row['request_hash']}.json").read_text(encoding="utf-8")
        )["response"]["parsed_payload"]

    adapter = CholecTrack20DatasetAdapter(dataset, causal_window_size=3)
    targets = {}
    for resolved in adapter.iter_video(video, frame_ids=paired):
        if resolved.evaluation is None:
            raise ValueError(
                "This smoke audit requires instance-supervised validation frames"
            )
        target = aggregate_evaluation_target(
            resolved.evaluation, source="offline_smoke_audit"
        )
        targets[target.frame_id] = target
    if set(targets) != set(paired):
        raise ValueError("Missing paired evaluation targets")
    rows = []
    for frame in paired:
        truth = _labels(asdict(targets[frame]))
        mask = asdict(targets[frame].mask)
        initial = {
            task: (
                [h0[frame][task]["selected_id"]]
                if task == "phase"
                else sorted(h0[frame][task]["selected_ids"])
            )
            for task in TASKS
        }
        row = {"frame": frame, "gt": truth, "mask": mask, "h0": initial}
        row["h0_topk_contains_all_gt"] = {
            task: set(truth[task]) <= {item["id"] for item in h0[frame][task]["topk"]}
            for task in TASKS
        }
        for name in records:
            record = records[name][frame]
            row[name] = _labels(record["outcome"]["hypothesis"])
            pool = record["audit"]["candidate_set"]["allowed_ids"]
            row[f"{name}_pool_contains_all_gt"] = {
                task: set(truth[task]) <= set(pool[task]) for task in TASKS
            }
            row[f"{name}_attempts"] = record["audit"]["proposal"]["attempts"]
            row[f"{name}_verified_tasks"] = record["outcome"]["verified_tasks"]
        rows.append(row)

    exact = {
        name: {
            task: sum(
                row["mask"][task] and row[name][task] == row["gt"][task] for row in rows
            )
            for task in TASKS
        }
        for name in ("h0", "before", "after")
    }
    deltas = {
        name: {
            task: {
                "rescued_vs_h0": sum(
                    row["mask"][task]
                    and row["h0"][task] != row["gt"][task]
                    and row[name][task] == row["gt"][task]
                    for row in rows
                ),
                "harmed_vs_h0": sum(
                    row["mask"][task]
                    and row["h0"][task] == row["gt"][task]
                    and row[name][task] != row["gt"][task]
                    for row in rows
                ),
                "wrong_but_verified": sum(
                    row["mask"][task]
                    and task in row[f"{name}_verified_tasks"]
                    and row[name][task] != row["gt"][task]
                    for row in rows
                ),
            }
            for task in TASKS
        }
        for name in records
    }
    return {
        "metric": "masked_set_exact_count_NOT_mAP",
        "paired_frames": paired,
        "processed_timestamps": {name: len(value) for name, value in records.items()},
        "task_denominators": {
            task: sum(row["mask"][task] for row in rows) for task in TASKS
        },
        "exact_counts": exact,
        "deltas": deltas,
        "all_five_exact": {
            name: sum(
                all(row["mask"].values()) and row[name] == row["gt"] for row in rows
            )
            for name in exact
        },
        "candidate_recall_counts": {
            key: {
                task: sum(row["mask"][task] and row[key][task] for row in rows)
                for task in TASKS
            }
            for key in (
                "h0_topk_contains_all_gt",
                "before_pool_contains_all_gt",
                "after_pool_contains_all_gt",
            )
        },
        "usage": {"before": before_usage, "after": after_usage},
        "frames": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "cache", "dataset"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--video", default="VID110")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    result = audit(
        before=args.before,
        after=args.after,
        cache=args.cache,
        dataset=args.dataset,
        video=args.video,
    )
    if args.summary_only:
        result.pop("frames")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
