"""Development-only zero-call evaluation of the targeted Repair admission rule."""
import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_joint_phase_feedback_trial as trial


def replay(source, output):
    if output.exists():
        raise ValueError("new output required")
    rows = trial.read(source / "predictions.json")
    truths = trial.read(source / "scored_truth.json")
    truth = {(r["video_id"], r["frame_id"]): r for r in truths}
    rejected = {}
    for row in rows:
        raw = trial.read(source / "targets" / row["key"] / "result.json")
        if "compiled_repair" in raw:
            row["joint_r2_targeted"], rejected[row["key"]] = trial.apply_targeted_round(
                row["joint_r1"], raw["compiled_repair"]["paper_style"], row["joint_r2"])
        else:
            row["joint_r2_targeted"] = deepcopy(row["joint_r1"])
    metrics = {a: trial.common.compute_repair_comparison([{**truth[r['video_id'], r['frame_id']],
        "h0": r["h0"], "h1": None, "final": r[a]} for r in rows])["arms"]["final"] for a in trial.ARMS}
    deltas = {a: [{"key": r["key"], **trial.common.frame_delta(r[a], r["joint_r2_targeted"],
        truth[r['video_id'], r['frame_id']]["gt"], truth[r['video_id'], r['frame_id']]["mask"])}
        for r in rows] for a in trial.ARMS[:-1]}
    report = {"created_utc": trial.now(), "source": str(source.resolve()),
        "source_metrics_sha256": trial.sha(source / "metrics.json"), "api_calls": 0,
        "rule": trial.RULE, "rule_sha256": trial.sha(ROOT / "src/surgical_agent/research/verification/targeted_joint_repair.py"),
        "development_only": True, "metrics": metrics,
        "deltas": deltas, "changes": {a: trial.common.summarize_deltas(d) for a, d in deltas.items()},
        "rejected_changes": rejected}
    trial.save(output / "predictions.json", rows)
    trial.save(output / "metrics.json", report)
    print({"f1": {a: {t: round(v["micro_f1"] * 100, 3) for t, v in m["tasks"].items()} for a, m in metrics.items()},
        "changes": report["changes"], "rejected": rejected})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    replay(args.source, args.output)
