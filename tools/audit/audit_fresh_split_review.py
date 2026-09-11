"""Independent counts, stage totals, transport sensitivity and shared-input checks."""
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_fresh_split_review as fresh


def audit(output):
    fresh.verify(output)
    done, report, rows, truth = [fresh.old.read(output / f"{name}.json") for name in ("completion", "metrics", "predictions", "scored_truth")]
    assert report["raw_replayed"] and not done["fatal_error"]
    for name, digest in done["hashes"].items():
        assert fresh.old.sha(output / name) == digest
    truths = {(r["video_id"], r["frame_id"]): r for r in truth}
    aggregates = {}
    for arm in ("h0", "control", "split"):
        for task in fresh.TASKS:
            tp = fp = fn = exact = valid = failed = 0
            for row in rows:
                g = truths[row["video_id"], row["frame_id"]]
                if not g["mask"][task]:
                    continue
                valid += 1
                failed += row[arm] is None
                pred = set(row[arm][task]) if row[arm] is not None else set()
                gt = set(g["gt"][task])
                tp += len(pred & gt)
                fp += len(pred - gt)
                fn += len(gt - pred)
                exact += row[arm] is not None and pred == gt
            h = report["metrics"][arm]["tasks"][task]
            assert (tp, fp, fn, exact, valid, failed) == tuple(h[k] for k in ("tp", "fp", "fn", "exact_matches", "valid_targets", "failed_predictions"))
            assert h["micro_f1"] == (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
            assert h["micro_precision"] == (tp/(tp+fp) if tp+fp else 0)
            assert h["exact_set_accuracy"] == exact/valid
        heads = report["metrics"][arm]["tasks"]
        aggregates[arm] = {"mean_f1": sum(h["micro_f1"] for h in heads.values()) / 5,
            "mean_precision": sum(h["micro_precision"] for h in heads.values()) / 5,
            "fp_fn": sum(h["fp"]+h["fn"] for h in heads.values())}
    calls = fresh.old.read(output / "budget.json")["calls"]
    for c in calls:
        folder = output / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
        request = fresh.old.read(folder / "request.json")
        assert request == fresh.old.read(output / "request_intents" / f"{c['target']}_{c['stage']}_{c['seat']}.json")
        assert request["model"] == (fresh.PROPOSER if c["seat"] == "base" else fresh.old.MODELS[c["seat"]])
        if c["status"].startswith("JSON_PARSED"):
            response = fresh.old.read(folder / "response.json")["body"]
            assert response["model"] == request["model"]
            if c["seat"] in fresh.old.PROVIDERS:
                assert response["provider"] == fresh.old.PROVIDERS[c["seat"]]
    invalid, coverage, relations = {a: Counter() for a in fresh.old.ARMS}, Counter(), {a: Counter() for a in fresh.old.ARMS}
    stage_seconds = Counter()
    for c in calls:
        stage_seconds[c["stage"]] += c["elapsed_seconds"]
    for row in rows:
        path = output / "targets" / row["key"] / "result.json"
        if not path.exists():
            continue
        r = fresh.old.read(path)
        if r["h0"] is None:
            continue
        assert row["control"]["phase"] == row["split"]["phase"]
        gt = truths[row["video_id"], row["frame_id"]]
        if gt["mask"]["ivt"]:
            pool_ivt = {p["label_id"] for p in r["pool"]["propositions"] if p["task"] == "ivt"}
            gold = set(gt["gt"]["ivt"])
            coverage.update({"gt": len(gold), "covered": len(gold & pool_ivt), "missing": len(gold-pool_ivt), "false_candidates": len(pool_ivt-gold)})
        for arm, result in r["arms"].items():
            formats = [result["formatting"]] if arm == "control" else result["formatting"].values()
            for fmt in formats:
                for seat, diagnostic in fmt.items():
                    for errors in diagnostic["errors"].values():
                        invalid[arm].update((seat, error) for error in errors)
            for p in r["pool"]["propositions"]:
                if p["task"] != "ivt" or not gt["mask"]["ivt"]:
                    continue
                mean = result["means"][p["id"]]
                positive = p["label_id"] in gt["gt"]["ivt"]
                if positive and mean is not None and mean >= 4:
                    relations[arm]["true_relation_supported"] += 1
                    if p["label_id"] not in r["h0"]["ivt"] and p["label_id"] not in result["final"]["ivt"]:
                        relations[arm]["true_relation_blocked_by_components"] += 1
                if p["label_id"] in result["final"]["ivt"]:
                    relations[arm]["true_selected" if positive else "false_selected"] += 1
    excluded = {c["target"] for c in calls if not c["status"].startswith("JSON_PARSED")}
    paired = [r for r in rows if r["key"] not in excluded and r["h0"] is not None]
    sensitivity = {a: fresh.old.compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]],
        "h0": r["h0"], "h1": None, "final": r[a]} for r in paired])["arms"]["final"] for a in ("h0", *fresh.old.ARMS)} if paired else {}
    projected_costs = {}
    for arm in fresh.old.ARMS:
        stages = ("h0", "proposal", "phase", *( ("control",) if arm == "control" else ("components", "relations")))
        projected_costs[arm] = {a: {kind: str(sum(Decimal(c["charge"]) for c in calls if c["stage"] in stages
            and c["account"] == a and c["charge_kind"] == kind)) for kind in ("native", "conservative_estimate", "unknown_reserved")} for a in fresh.LIMITS}
    audit = {"independent_head_tables_verified": 15, "aggregate_metrics": aggregates,
        "request_intents_models_and_providers_verified": True, "same_phase_verified": True,
        "candidate_coverage": dict(coverage), "relation_audit": {a: dict(v) for a, v in relations.items()},
        "invalid_reasons": {a: [{"seat": seat, "reason": reason, "count": n} for (seat, reason), n in v.items()] for a, v in invalid.items()},
        "transport_complete_sensitivity": {"targets": len(paired), "excluded_keys": sorted(excluded), "metrics": sensitivity},
        "standalone_equivalent_costs": projected_costs, "sum_request_seconds_not_wall_time": dict(stage_seconds),
        "metrics_sha256": fresh.old.sha(output / "metrics.json")}
    fresh.old.save(output / "independent_audit.json", audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in ("transport_complete_sensitivity", "invalid_reasons")}, ensure_ascii=False))


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
