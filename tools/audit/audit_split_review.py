"""Independent metric counts and review failure/admission attribution."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_split_review_trial as trial

TASKS = ("instrument", "verb", "target", "ivt", "phase")


def audit(output):
    plan = trial.verify(output)
    report, done = trial.read(output / "metrics.json"), trial.read(output / "completion.json")
    assert report["raw_replayed"] and not done["fatal_error"]
    for name, digest in done["hashes"].items():
        assert trial.sha(output / name) == digest
    rows, truth = trial.read(output / "predictions.json"), trial.read(output / "scored_truth.json")
    truths = {(r["video_id"], r["frame_id"]): r for r in truth}
    counts = {}
    for arm in ("h0", "control", "split"):
        counts[arm] = {}
        for task in TASKS:
            tp = fp = fn = exact = valid = 0
            for row in rows:
                gold = truths[row["video_id"], row["frame_id"]]
                if not gold["mask"][task]:
                    continue
                valid += 1
                pred, gt = set(row[arm][task]), set(gold["gt"][task])
                tp += len(pred & gt)
                fp += len(pred - gt)
                fn += len(gt - pred)
                exact += pred == gt
            expected = report["metrics"][arm]["tasks"][task]
            assert (tp, fp, fn, exact, valid) == tuple(expected[k] for k in ("tp", "fp", "fn", "exact_matches", "valid_targets"))
            assert expected["micro_f1"] == (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
            assert expected["micro_precision"] == (tp / (tp + fp) if tp + fp else 0)
            assert expected["exact_set_accuracy"] == exact / valid
            counts[arm][task] = {k: expected[k] for k in ("tp", "fp", "fn", "exact_matches")}
    initials = trial.read(output / "initial_state.json")
    evidence, invalid = [], {a: Counter() for a in trial.ARMS}
    relation_audit = {a: Counter() for a in trial.ARMS}
    for row in rows:
        records = {a: trial.read(output / "targets" / row["key"] / f"{a}.json") for a in trial.ARMS}
        pool = initials[row["key"]]["pool"]
        gold = truths[row["video_id"], row["frame_id"]]
        for arm, record in records.items():
            fmt = record["formatting"]
            for group in ((fmt,) if arm == "control" else fmt.values()):
                for seat, diagnostics in group.items():
                    for errors in diagnostics["errors"].values():
                        invalid[arm].update((seat, error) for error in errors)
        for p in pool["propositions"]:
            task, pid, label = p["task"], p["id"], p["label_id"]
            selected = {a: label in row[a][task] for a in trial.ARMS}
            means = {a: records[a]["means"][pid] for a in trial.ARMS}
            if task == "ivt" and gold["mask"]["ivt"]:
                for arm in trial.ARMS:
                    present = label in gold["gt"]["ivt"]
                    tally = relation_audit[arm]
                    tally["true_candidates" if present else "false_candidates"] += 1
                    if means[arm] is None:
                        tally["true_without_five_valid_votes" if present else "false_without_five_valid_votes"] += 1
                    if present and means[arm] is not None and means[arm] >= 4:
                        tally["true_relation_supported"] += 1
                        if label not in row["h0"]["ivt"] and not selected[arm]:
                            tally["true_relation_supported_but_component_gate_blocks"] += 1
                    if selected[arm]:
                        tally["true_selected" if present else "false_selected"] += 1
            if means["control"] != means["split"] or selected["control"] != selected["split"]:
                evidence.append({"key": row["key"], "candidate": p, "gt_valid": gold["mask"][task],
                    "gt_present": label in gold["gt"][task] if gold["mask"][task] else None,
                    "means": means, "selected": selected,
                    "scores": {a: records[a]["diagnostics"][pid]["scores"] for a in trial.ARMS}})
    calls = trial.read(output / "budget.json")["calls"]
    for call in calls:
        assert call["model"] == plan["models"][call["seat"]]
        folder = output / "calls" / f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
        if call["status"].startswith("JSON_PARSED"):
            response = trial.read(folder / "response.json")["body"]
            assert response["model"] == call["model"]
            if call["seat"] in trial.PROVIDERS:
                assert response["provider"] == trial.PROVIDERS[call["seat"]]
    failed_targets = {c["target"] for c in calls if not c["status"].startswith("JSON_PARSED")}
    paired = [r for r in rows if r["key"] not in failed_targets]
    sensitivity = {a: trial.compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]],
        "h0": r["h0"], "h1": None, "final": r[a]} for r in paired])["arms"]["final"]
        for a in ("h0", *trial.ARMS)} if paired else {}
    result = {"independent_head_tables_verified": 15, "counts": counts,
        "models_and_providers_verified": True, "score_changes": evidence,
        "relation_audit": {a: dict(v) for a, v in relation_audit.items()},
        "transport_complete_sensitivity": {"target_count": len(paired), "excluded_keys": sorted(failed_targets),
            "metrics": sensitivity, "does_not_replace_primary_eight": True},
        "invalid_reasons": {a: [{"seat": seat, "reason": reason, "count": count}
            for (seat, reason), count in values.items()] for a, values in invalid.items()},
        "metrics_sha256": trial.sha(output / "metrics.json")}
    trial.save(output / "independent_audit.json", result)
    print(json.dumps({"verified_head_tables": 15, "changed_means_or_selections": len(evidence),
        "invalid_reasons": result["invalid_reasons"]}))


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
