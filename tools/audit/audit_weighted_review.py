"""Independent counts plus exploratory transfer to the archived compact prompt."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import sha
from scripts.run_prior_panel_trial import read, save
from scripts.run_repair_revision_trial import normalize_five
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import weighted_review as weighted

OUTPUT = ROOT / "artifacts/preflight/weighted_review_confirmation_20260909_v1"
QUERY = ROOT / "artifacts/preflight/default_improvement_confirm_20260909_v1"
COMPACT = ROOT / "artifacts/preflight/compact_verifier_paired_20260909_v2_continued"
TASKS = (*weighted.TASKS, "phase")


def main():
    report = read(OUTPUT / "metrics.json")
    truth = {(r["video_id"], r["frame_id"]): r for r in read(QUERY / "scored_truth.json")}
    rows = read(OUTPUT / "predictions.json")
    for version in ("h0", "control", "weighted"):
        for task in TASKS:
            tp = fp = fn = exact = valid = 0
            for row in rows:
                gold = truth[row["video_id"], row["frame_id"]]
                if not gold["mask"][task]:
                    continue
                valid += 1
                pred, gt = set(row[version][task]), set(gold["gt"][task])
                tp += len(pred & gt)
                fp += len(pred - gt)
                fn += len(gt - pred)
                exact += pred == gt
            head = report["metrics"][version]["tasks"][task]
            assert (tp, fp, fn, exact, valid) == tuple(head[k] for k in ("tp", "fp", "fn", "exact_matches", "valid_targets"))
            assert head["micro_precision"] == (tp / (tp + fp) if tp + fp else 0)
            assert head["micro_f1"] == (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
            assert head["exact_set_accuracy"] == exact / valid
    save(OUTPUT / "independent_audit.json", {"verified_head_tables": 15,
        "metrics_sha256": sha(OUTPUT / "metrics.json"), "predictions_sha256": sha(OUTPUT / "predictions.json")})

    # No refitting or threshold changes: this old cohort is exploratory only.
    assert read(COMPACT / "metrics.json")["raw_replay_passed"]
    models, initial = read(OUTPUT / "models.json"), read(COMPACT / "initial_state.json")
    truths = {(r["video_id"], r["frame_id"]): r for r in read(COMPACT / "scored_truth.json")}
    rows, details, hashes = read(COMPACT / "predictions.json"), [], {}
    for row in rows:
        path = COMPACT / "targets" / row["key"] / "compact.json"
        record = read(path)
        hashes[str(path)] = sha(path)
        pool = initial[row["key"]]["pool"]
        reviews, _ = normalize_five(record["raw"]["graph"], pool, 3)
        result = weighted.select(row["h0"], pool, reviews, models[row["video_id"]], row["video_id"])
        row["weighted"] = deepcopy(result["prediction"])
        row["weighted"]["phase"] = deepcopy(row["compact"]["phase"])
        details.append({"key": row["key"], "result": result})
    metrics = {v: compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]],
        "h0": r["h0"], "h1": None, "final": r[v]} for r in rows])["arms"]["final"]
        for v in ("h0", "compact", "weighted")}
    deltas = [{"key": r["key"], **frame_delta(r["compact"], r["weighted"],
        truths[r["video_id"], r["frame_id"]]["gt"], truths[r["video_id"], r["frame_id"]]["mask"])} for r in rows]
    result = {"exploratory_only": True, "api_calls": 0, "refitted": False,
        "models_sha256": sha(OUTPUT / "models.json"), "source_records_sha256": hashes,
        "metrics": metrics, "changes": summarize_deltas(deltas), "predictions": rows, "details": details}
    save(OUTPUT / "compact_transfer.json", result)
    print(json.dumps({"independent_tables": 15, "compact_transfer_changes": result["changes"],
        "compact_transfer_f1": {v: {t: m["tasks"][t]["micro_f1"] for t in TASKS} for v, m in metrics.items()}}))


if __name__ == "__main__":
    main()
