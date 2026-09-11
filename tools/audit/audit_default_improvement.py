"""Independently count closed trial predictions and summarize actual stage timing."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import sha
from scripts.run_prior_panel_trial import read, save

TASKS = ("instrument", "verb", "target", "ivt", "phase")


def audit(output):
    plan, done, report = [read(output / f"{n}.json") for n in ("plan", "completion", "metrics")]
    if done["fatal_error"] or not report["raw_replayed"]:
        raise ValueError("successful closure and raw replay required before this offline audit")
    for n, h in done["hashes"].items():
        assert sha(output / n) == h, n
    rows, truth = read(output / "predictions.json"), read(output / "scored_truth.json")
    truths = {(r["video_id"], r["frame_id"]): r for r in truth}
    checked, by_version = 0, {}
    for version in plan["versions"]:
        computed = {}
        all_exact = 0
        for task in TASKS:
            tp = fp = fn = exact = valid = failed = 0
            for row in rows:
                gt = truths[row["video_id"], row["frame_id"]]
                if not gt["mask"][task]:
                    continue
                valid += 1
                failed += row[version] is None
                pred = set(row[version][task]) if row[version] is not None else set()
                gold = set(gt["gt"][task])
                tp += len(pred & gold)
                fp += len(pred - gold)
                fn += len(gold - pred)
                exact += row[version] is not None and pred == gold
            expected = {"tp": tp, "fp": fp, "fn": fn, "exact_matches": exact, "valid_targets": valid,
                "failed_predictions": failed, "micro_precision": (tp / (tp + fp) if tp + fp else 0) if valid else None,
                "micro_recall": (tp / (tp + fn) if tp + fn else 0) if valid else None,
                "micro_f1": (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0) if valid else None,
                "exact_set_accuracy": exact / valid if valid else None}
            assert expected == report["metrics"][version]["tasks"][task], (version, task)
            computed[task] = expected
            checked += 1
        for row in rows:
            gt = truths[row["video_id"], row["frame_id"]]
            valid_tasks = [t for t in TASKS if gt["mask"][t]]
            all_exact += bool(valid_tasks) and row[version] is not None and all(set(row[version][t]) == set(gt["gt"][t]) for t in valid_tasks)
        assert all_exact == report["metrics"][version]["all_valid_heads_exact"]["exact_matches"]
        by_version[version] = {"mean_f1": sum(computed[t]["micro_f1"] for t in TASKS) / 5,
            "mean_precision": sum(computed[t]["micro_precision"] for t in TASKS) / 5,
            "summed_fp_fn": sum(computed[t]["fp"] + computed[t]["fn"] for t in TASKS),
            "all_heads_exact": all_exact}
    records = [read(output / "targets" / r["key"] / "result.json") for r in rows]
    for record in records:
        if record["h0"] is not None:
            assert len({tuple(record["predictions"][a]["phase"]) for a in plan["versions"] if a != "h0"}) == 1
            if "verifier_current" in record["arms"]:
                assert record["arms"]["verifier_current"]["pool"] == record["arms"]["control"]["pool"]
    ledger = read(output / "budget.json")
    ids = [(c["target"], c["stage"], c["seat"]) for c in ledger["calls"]]
    assert len(set(ids)) == len(ids)
    for call in ledger["calls"]:
        folder = output / "calls" / f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
        request = read(folder / "request.json")
        intent = read(output / "request_intents" / f"{call['target']}_{call['stage']}_{call['seat']}.json")
        assert request == intent["request"]
        assert request["model"] == (plan["proposer"] if call["seat"] == "base" else plan["models"][call["seat"]])
        if call["status"].startswith("JSON_PARSED"):
            response = read(folder / "response.json")["body"]
            assert response["model"] == request["model"]
            if call["seat"] in plan["providers"]:
                assert response["provider"] == plan["providers"][call["seat"]]
        if call["stage"] != "h0":
            first = request["messages"][0]["content"][0]["text"]
            assert next(iter(json.loads(first))) == "academic_context"
    for account, amount in ledger["occupied"].items():
        actual = sum(Decimal(c["charge"]) for c in ledger["calls"] if c["account"] == account)
        assert actual == Decimal(amount) <= Decimal(plan["limits"][account])
    def equivalent(record, arm, field, shared_field):
        visited = set()
        while arm not in visited:
            visited.add(arm)
            value = record["arms"].get(arm, {})
            other = value.get(shared_field)
            if other is None:
                return value.get(field, 0)
            arm = other
        raise ValueError("reuse cycle")

    timing = {arm: {"proposal_seconds": sum(r["arms"].get(arm, {}).get("proposal_seconds", 0) for r in records),
                   "panel_seconds": sum(r["arms"].get(arm, {}).get("panel_seconds", 0) for r in records),
                   "standalone_equivalent_proposal_seconds": sum(equivalent(r, arm, "proposal_seconds", "proposal_shared_from") for r in records),
                   "standalone_equivalent_panel_seconds": sum(equivalent(r, arm, "panel_seconds", "shared_from") for r in records),
                   "shared_panels": sum(bool(r["arms"].get(arm, {}).get("shared_from")) for r in records)} for arm in plan["arms"]}
    invalid = {arm: dict(Counter(error for r in records for item in r["arms"].get(arm, {}).get("diagnostics", {}).values()
                                for error in item["invalid"].values())) for arm in plan["arms"]}
    result = {"independent_head_tables_verified": checked, "models_and_providers_verified": True,
        "shared_phase_verified": True, "same_pool_verifier_control_verified": True,
        "pre_dispatch_requests_verified": True, "no_duplicate_post_calls": True,
        "versions": by_version, "stage_wall_seconds": timing, "invalid_review_items": invalid,
        "phase_wall_seconds": sum(r.get("phase", {}).get("phase_seconds", 0) for r in records),
        "h0_request_seconds": sum(c["elapsed_seconds"] for c in ledger["calls"] if c["stage"] == "h0"),
        "elapsed_seconds": done["elapsed_seconds"], "source_sha256": sha(__file__)}
    save(output / "independent_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
