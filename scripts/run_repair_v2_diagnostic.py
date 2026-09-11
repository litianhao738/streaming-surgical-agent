"""Shared-image Training replay isolating pool completion and quorum policy.

Twenty paid reviews at most. H0 and Gemini seed hypotheses are reused exactly.
Four arms: old pool/strict, old pool/quorum (offline), completed pool/strict,
completed pool/quorum. GT is used only after all new predictions are frozen.
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import Calls, sha
from scripts.run_complete_gt_semantic_trial import normalize_review_wire, review_body_v3
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import build_base, now, read, save
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.verification import candidate_coordinator as old
from surgical_agent.research.verification import semantic_coordinator as strict
from surgical_agent.research.verification.repair_v2 import (
    aggregate_quorum,
    complete_target_hypotheses,
)

SOURCE = ROOT / "artifacts/preflight/semantic_complete_gt_20260908_v1"
PREVIOUS = ROOT / "artifacts/preflight/semantic_blind_candidate_20260908_v1"
INCREMENT = {"openrouter_usd": Decimal("0.60"), "xai_usd": Decimal("0.25"), "aliyun_cny": Decimal("0.50")}


def main(output):
    if output.exists():
        raise ValueError("new output required; no paid replay")
    source_plan, previous = read(SOURCE / "plan.json"), read(PREVIOUS / "budget.json")
    if not previous["stopped"]:
        raise ValueError("previous inference still active")
    selection, seeds = source_plan["selection"], read(SOURCE / "predictions.json")
    prepared = []
    for s, row in zip(selection, seeds, strict=True):
        if not all(s["gt_availability_only"].values()):
            raise ValueError("complete GT masks required; do not inspect label answers")
        record = read(SOURCE / "targets" / s["key"] / "round_1.json")
        pool, expansion = complete_target_hypotheses(row["h0"], record["pool"])
        means, normalized = aggregate_quorum(record["raw"], record["pool"])
        prepared.append({"key": s["key"], "h0": old.labels(row["h0"]), "seed_pool": record["pool"],
                         "old_strict": old.labels(row["final"]), "old_quorum": old.select(row["h0"], record["pool"], means),
                         "old_quorum_judgments": normalized, "pool": pool, "expansion": expansion})
    limits = {k: Decimal(previous["occupied"][k]) + value for k, value in INCREMENT.items()}
    paths = [Path(__file__), ROOT / "scripts/run_complete_gt_semantic_trial.py",
             ROOT / "scripts/run_candidate_panel_trial.py", ROOT / "scripts/check_candidate_panel_providers.py",
             ROOT / "scripts/replay_semantic_review.py", ROOT / "scripts/run_semantic_candidate_trial.py",
             *sorted((ROOT / "src/surgical_agent/research/verification").glob("*.py")),
             ROOT / "scripts/run_grounded_api_pipeline.py", ROOT / "scripts/run_prior_panel_trial.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    plan = {"created_utc": now(), "selection": selection, "prepared": prepared, "max_calls": 20,
            "limits": {k: str(v) for k, v in limits.items()}, "carried_occupied": previous["occupied"],
            "source": str(SOURCE), "source_prediction_sha256": sha(SOURCE / "predictions.json"),
            "previous_ledger": str(PREVIOUS), "source_sha256": hashes,
            "scope": "four previously scored Training targets; development ablation, not held-out validation",
            "no_gt_in_pool_completion_or_inference": True,
            "candidate_policy": "all legal targets for existing instrument/verb hypotheses, cap64 or keep original",
            "quorum": "at least4 valid seats, at least3 directional votes, mean>=4/add <=2/delete, explicit conflict veto"}
    save(output / "plan.json", plan)
    for path in paths:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    print(json.dumps({"samples": [s["key"] for s in selection], "pools": [p["expansion"] for p in prepared]}), flush=True)
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    calls = Calls(output, previous["occupied"], limits=limits, max_calls=20)
    results = []
    for s, p in zip(selection, prepared, strict=True):
        base, key = build_base(adapter, s), s["key"]
        def one(seat, s=s, base=base, key=key, pool=p["pool"]):
            raw = calls.call(key, "completed_review", seat, review_body_v3(seat, base, s, pool))
            return normalize_review_wire(seat, raw, pool)
        with ThreadPoolExecutor(max_workers=5) as workers:
            raw = dict(zip(old.SEATS, workers.map(one, old.SEATS), strict=True))
        strict_means, strict_clean = strict.aggregate(raw, p["pool"])
        quorum_means, quorum_clean = aggregate_quorum(raw, p["pool"])
        row = {**p, "video_id": s["video_id"], "frame_id": s["frame_id"], "raw": raw,
               "expanded_strict": old.select(p["h0"], p["pool"], strict_means),
               "expanded_quorum": old.select(p["h0"], p["pool"], quorum_means),
               "strict_means": strict_means, "strict_reviews": strict_clean,
               "quorum_means": quorum_means, "quorum_reviews": quorum_clean}
        results.append(row)
        save(output / "targets" / f"{key}.json", row)
        save(output / "predictions.json", results)
        print(json.dumps({"target": key, "strict_changed": row["expanded_strict"] != p["h0"],
                          "quorum_changed": row["expanded_quorum"] != p["h0"], "calls": len(calls.rows)}), flush=True)
    calls.stopped = True
    calls.persist()
    frozen = sha(output / "predictions.json")
    if any(sha(ROOT / f) != v for f, v in hashes.items()):
        raise ValueError("source changed during inference")
    reports, truths = {}, {}
    for arm in ("old_strict", "old_quorum", "expanded_strict", "expanded_quorum"):
        rows = [{"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": r["h0"], "h1": None,
                 "final": r[arm]} for r in results]
        reports[arm], truths[arm] = score_saved(adapter, rows)
    coverage = []
    for r, truth in zip(results, truths["expanded_quorum"], strict=True):
        assert r["key"] == f"{truth['video_id']}_{truth['frame_id']}"
        coverage.append({"key": r["key"], "tasks": {q: {
            "gt": len(truth["gt"][q]),
            "old_pool_hits": len(set(truth["gt"][q]) & {p["label_id"] for p in r["seed_pool"]["propositions"] if p["task"] == q}),
            "expanded_pool_hits": len(set(truth["gt"][q]) & {p["label_id"] for p in r["pool"]["propositions"] if p["task"] == q}),
        } for q in old.TASKS if truth["mask"][q]}})
    assert sha(output / "predictions.json") == frozen
    save(output / "comparison.json", reports)
    save(output / "scored_predictions.json", truths)
    save(output / "coverage.json", coverage)
    summary = {"post_calls": len(calls.rows), "prediction_sha256": frozen, "coverage": coverage,
               "metrics": {k: v["arms"]["final"] for k, v in reports.items()},
               "changes": {k: v["paired"]["h0_to_final"] for k, v in reports.items()},
               "native_costs": {k: str(sum(Decimal(r["charge"]) for r in calls.rows
                                           if r["account"] == k and r["charge_kind"] == "native")) for k in limits}}
    save(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("post_calls", "native_costs", "coverage")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
