"""Independent all-head counts, Phase switches and paired-complete sensitivity."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_phase_mechanism_comparison as trial


def main(output):
    trial.verify(output)
    metrics = trial.old.read(output / "metrics.json")
    assert metrics["raw_replayed"]
    rows = trial.old.read(output / "predictions.json")
    truths = {(r["video_id"], r["frame_id"]): r for r in trial.old.read(output / "scored_truth.json")}
    initial = trial.old.read(output / "initial_state.json")
    calls = trial.old.read(output / "budget.json")["calls"]
    assert len(rows) == 32
    for row in rows:
        before = initial[row["key"]]["graph_r1"]["prediction"]
        for arm in trial.PHASE_ARMS:
            assert all(row[arm][task] == before[task] for task in trial.source_runner.TASKS if task != "phase")
    for arm in trial.VERSIONS:
        for task in trial.source_runner.TASKS:
            tp = fp = fn = exact = valid = 0
            for r in rows:
                truth = truths[r["video_id"], r["frame_id"]]
                if not truth["mask"][task]:
                    continue
                valid += 1
                pred, gold = set(r[arm][task]), set(truth["gt"][task])
                tp += len(pred & gold)
                fp += len(pred - gold)
                fn += len(gold - pred)
                exact += pred == gold
            saved = metrics["metrics"][arm]["tasks"][task]
            assert (tp, fp, fn, exact, valid) == tuple(saved[k] for k in ("tp", "fp", "fn", "exact_matches", "valid_targets"))
            assert saved["micro_f1"] == (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
            assert saved["micro_precision"] == (tp/(tp+fp) if tp+fp else 0)
    for c in calls:
        folder = output / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
        request = trial.old.read(folder / "request.json")
        assert request == trial.old.read(output / "request_intents" / f"{c['target']}_{c['stage']}_{c['seat']}.json")
        assert request["model"] == (trial.legacy.PROPOSER if c["seat"] == "base" else trial.old.MODELS[c["seat"]])
    switches, per_video, sensitivity = {}, {}, {}
    for arm in trial.VERSIONS:
        count = Counter()
        for row in rows:
            truth = truths[row["video_id"], row["frame_id"]]
            if not truth["mask"]["phase"]:
                continue
            gold = truth["gt"]["phase"]
            before, after = row["h0"]["phase"], row[arm]["phase"]
            label = "unchanged" if before == after else "corrected" if after == gold else "harmed" if before == gold else "wrong_to_wrong"
            count[label] += 1
        switches[arm] = dict(count)
    def evaluate(selected, arm):
        return trial.old.compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]], "h0": r["h0"],
            "h1": None, "final": r[arm]} for r in selected])["arms"]["final"]
    for video in sorted({r["video_id"] for r in rows}):
        per_video[video] = {arm: evaluate([r for r in rows if r["video_id"] == video], arm) for arm in trial.VERSIONS}
    for arm in trial.ARMS[1:]:
        stages = {"compact_repeat", arm} if arm in trial.PHASE_ARMS else {"compact_repeat", trial.legacy.REPAIR_STAGE, trial.legacy.REVIEW_STAGE}
        excluded = {c["target"] for c in calls if c["stage"] in stages and not c["status"].startswith("JSON_PARSED")}
        selected = [r for r in rows if r["key"] not in excluded]
        sensitivity[arm] = {"targets": len(selected), "excluded": sorted(excluded),
            "metrics": {a: evaluate(selected, a) for a in ("h0", "compact_repeat", arm)} if selected else {}}
    report = {"independent_head_tables_verified": len(trial.VERSIONS)*5, "phase_switches_vs_h0": switches,
        "per_video": per_video, "paired_transport_complete_sensitivity": sensitivity,
        "request_models_and_intents_verified": True, "metrics_sha256": trial.old.sha(output / "metrics.json")}
    trial.old.save(output / "independent_audit.json", report)
    print(json.dumps({"independent_tables": report["independent_head_tables_verified"], "phase_switches": switches,
        "sensitivity_counts": {a: s["targets"] for a, s in sensitivity.items()}}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
