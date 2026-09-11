"""Post-score per-video, coverage, whole-frame and billed-stage diagnostics."""
import argparse
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(output):
    rows, truths = read(output / "predictions.json"), read(output / "scored_truth.json")
    report = read(output / "metrics.json")
    truth = {(r["video_id"], r["frame_id"]): r for r in truths}
    tasks = ("instrument", "verb", "target", "ivt", "phase")
    groups, coverage, exact, phase_dist = {}, {}, {}, Counter()
    for r in rows:
        gt = truth[r["video_id"], r["frame_id"]]
        if gt["mask"]["phase"]:
            phase_dist.update(gt["gt"]["phase"])
    for arm in report["metrics"]:
        exact[arm] = sum(all(set(r[arm][t]) == set(truth[r["video_id"], r["frame_id"]]["gt"][t])
            for t in tasks) for r in rows if all(truth[r["video_id"], r["frame_id"]]["mask"].values()))
        groups[arm] = {}
        for video in sorted({r["video_id"] for r in rows}):
            counts = {t: Counter() for t in tasks}
            for r in rows:
                if r["video_id"] != video:
                    continue
                gt = truth[r["video_id"], r["frame_id"]]
                for t in tasks:
                    if gt["mask"][t]:
                        p, g = set(r[arm][t]), set(gt["gt"][t])
                        counts[t].update(tp=len(p & g), fp=len(p-g), fn=len(g-p))
            groups[arm][video] = {t: {**c, "micro_f1": 2*c["tp"]/(2*c["tp"]+c["fp"]+c["fn"])
                if 2*c["tp"]+c["fp"]+c["fn"] else 0} for t, c in counts.items()}
    for stage in ("joint_r1", "joint_r2"):
        coverage[stage] = {t: Counter() for t in tasks}
        for r in rows:
            raw = read(output / "targets" / r["key"] / "result.json")
            pool = raw.get(stage, raw["joint_r1"])["pool"]["propositions"]
            gt = truth[r["video_id"], r["frame_id"]]
            for t in tasks:
                if gt["mask"][t]:
                    g = set(gt["gt"][t])
                    p = {x["label_id"] for x in pool if x["task"] == t}
                    coverage[stage][t].update(gt_labels=len(g), covered=len(g&p), missing=len(g-p))
    calls = read(output / "budget.json")["calls"]
    effective = {}
    archive, chain = output, []
    while archive is not None:
        chain.append(read(archive / "budget.json")["calls"])
        parent = read(archive / "plan.json").get("parent_archive")
        archive = Path(parent) if parent else None
    for saved in reversed(chain):
        for call in saved:
            effective[call["target"], call["stage"], call["seat"]] = call
    sensitivity = {}
    for scope in ("all_calls_parsed", "control_calls_parsed"):
        relevant = [c for c in effective.values() if scope == "all_calls_parsed" or c["stage"].startswith("control_")]
        bad = {c["target"] for c in relevant if not c["status"].startswith("JSON_PARSED")}
        subset = [r for r in rows if r["key"] not in bad]
        sensitivity[scope] = {"target_count": len(subset), "excluded_keys": sorted(bad), "f1": {}}
        for arm in report["metrics"]:
            counts = {t: Counter() for t in tasks}
            for r in subset:
                gt = truth[r["video_id"], r["frame_id"]]
                for t in tasks:
                    if gt["mask"][t]:
                        p, g = set(r[arm][t]), set(gt["gt"][t])
                        counts[t].update(tp=len(p & g), fp=len(p-g), fn=len(g-p))
            sensitivity[scope]["f1"][arm] = {t: 2*c["tp"]/(2*c["tp"]+c["fp"]+c["fn"])
                if 2*c["tp"]+c["fp"]+c["fn"] else 0 for t, c in counts.items()}
    arm_stages = {"control": ("control_graph", "control_phase"),
        "joint_r1": ("phase_recommendation", "joint_r1"),
        "joint_r2": ("phase_recommendation", "joint_r1", "repair_r2", "joint_r2")}
    costs, seconds = {}, {}
    for arm, stages in arm_stages.items():
        cost = defaultdict(lambda: Decimal(0))
        critical = defaultdict(lambda: defaultdict(float))
        for c in calls:
            if c["stage"] not in stages:
                continue
            cost[c["account"] + ":" + c["charge_kind"]] += Decimal(c["charge"])
            stage = "parallel_control" if arm == "control" else c["stage"]
            if "elapsed_seconds" in c:
                critical[c["target"]][stage] = max(critical[c["target"]][stage], c["elapsed_seconds"])
        costs[arm] = {k: str(v) for k, v in cost.items()}
        seconds[arm] = {"api_critical_path_sum_seconds": sum(sum(s.values()) for s in critical.values()),
            "per_target_seconds": {k: sum(s.values()) for k, s in critical.items()}}
    result = {"per_video": groups, "all_five_heads_exact": exact, "phase_gt_counts": dict(phase_dist),
        "calls_without_elapsed_seconds": [c["index"] for c in calls if "elapsed_seconds" not in c],
        "transport_sensitivity_diagnostic_only": sensitivity,
        "candidate_coverage": coverage, "incremental_branch_costs": costs,
        "branch_api_critical_path": seconds,
        "notes": ["Costs exclude shared cached H0 and original four-head graph proposal.",
            "joint_r2_targeted shares all joint_r2 calls: no additional cost.",
            "Branch latency is sum of stage maxima; actual wall time includes dispatch and local overhead."]}
    if "parent_archive" in read(output / "plan.json"):
        result["reused_parent_archive"] = read(output / "plan.json")["parent_archive"]
        result["notes"].append("Continuation ledger contains only NEW round-two calls; parent control and round-one costs/time are excluded, not free.")
    (output / "detailed_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "per_video"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
