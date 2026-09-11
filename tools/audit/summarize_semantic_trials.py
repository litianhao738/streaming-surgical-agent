"""Offline supplement: paired successful-transport arms, causes and total bills."""
import json
import sys
from collections import Counter
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_prior_panel_trial import read, save
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison

RUNS = ["semantic_candidate_20260908_v2", "semantic_review_contract_20260908_v1",
        "semantic_gemini_strict_probe_20260908", "semantic_gemini_rows_probe_20260908",
        "semantic_gemini_object_rows_probe_20260908", "semantic_complete_gt_20260908_v1",
        "semantic_blind_candidate_20260908_v1"]


def main():
    base = ROOT / "artifacts/preflight"
    full = base / "semantic_complete_gt_20260908_v1"
    new, old = read(full / "scored_predictions.json"), read(full / "old_scored_predictions.json")
    keys = []
    for row in new:
        key = f"{row['video_id']}_{row['frame_id']}"
        numeric = read(full / "targets" / key / "old_review.json")
        semantic = read(full / "targets" / key / "round_1.json")
        if numeric["status"] == "VALID" and all(isinstance(v, dict) for v in semantic["raw"].values()):
            keys.append((row["video_id"], row["frame_id"]))
    paired = {"selection": keys, "criterion": "old numeric contract valid and all new reviews JSON parsed; item failures remain treatment outcomes",
              "new": compute_repair_comparison(r for r in new if (r["video_id"], r["frame_id"]) in keys),
              "old": compute_repair_comparison(r for r in old if (r["video_id"], r["frame_id"]) in keys)}
    save(full / "successful_transport_paired_comparison.json", paired)
    ledgers, totals, status, by_run = [], {}, Counter(), {}
    for name in RUNS:
        budget = read(base / name / "budget.json")
        if not budget["stopped"]:
            raise ValueError("all inference must finish before combined report")
        ledgers.append(budget)
        by_run[name] = {"post_calls": len(budget["calls"]), "charges": {}}
        for row in budget["calls"]:
            key = f"{row['account']}:{row['charge_kind']}"
            totals[key] = totals.get(key, Decimal(0)) + Decimal(row["charge"])
            by_run[name]["charges"][key] = str(Decimal(by_run[name]["charges"].get(key, "0")) + Decimal(row["charge"]))
            status[str(row.get("http_status", "no_response"))] += 1
    for previous, following in pairwise(ledgers):
        assert previous["occupied"] == following["carried_occupied"]
    result = {"post_calls": sum(len(b["calls"]) for b in ledgers), "http_status_counts": dict(status),
              "incremental_totals": {k: str(v) for k, v in totals.items()}, "by_run": by_run,
              "successful_paired_selection": keys, "continuous_budget_carry_verified": True,
              "unknown_reserved_is_not_native_charge": True}
    save(ROOT / "artifacts/research/semantic_verifier_repair_20260908/combined_audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
