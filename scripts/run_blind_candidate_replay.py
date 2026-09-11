"""Replace H0-conditioned proposing with blind proposing; same frozen H0/judges.

Four already scored Training targets are a development ablation, never a new
confirmation set. One proposal and one five-seat panel per target, no retries.
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import PROVIDERS, Calls, score, sha
from scripts.run_complete_gt_semantic_trial import normalize_review_wire, review_body_v3
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import build_base, now, read, save
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.verification import candidate_coordinator as old
from surgical_agent.research.verification import semantic_coordinator as semantic


def blind_proposal_body(base, selected):
    body = gemini_proposal(base, selected, {}, {"propositions": []}, [])
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    for field in ("current_prediction", "candidate_pool", "issues"):
        del packet[field]
    packet["task"] = "Independently identify plausible CURRENT-frame instrument, verb, target and IVT label IDs from the images."
    packet["instructions"] = (
        "Inspect all tools and their current interactions, including partially visible tools. "
        "Use the earlier images only as causal motion context. Return only labels with visual support. "
        "Different tools may contribute different IVTs. The output supplies hypotheses to a later verifier, "
        "not an accepted answer. Do not fill a quota or infer labels from the surgical phase. "
        "Return exactly the four bounded ID lists in the supplied schema."
    )
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return body


def execute(source, output):
    if output.exists():
        raise ValueError("new output required")
    original, previous = read(source / "plan.json"), read(source / "budget.json")
    if not previous["stopped"]:
        raise ValueError("previous run still active")
    selected, cached = original["selection"], read(source / "predictions.json")
    dependencies = [Path(__file__), ROOT / "scripts/run_complete_gt_semantic_trial.py",
                    ROOT / "scripts/replay_semantic_review.py", ROOT / "scripts/run_semantic_candidate_trial.py",
                    ROOT / "scripts/run_candidate_panel_trial.py", ROOT / "scripts/check_candidate_panel_providers.py",
                    ROOT / "src/surgical_agent/research/verification/candidate_coordinator.py",
                    ROOT / "src/surgical_agent/research/verification/semantic_coordinator.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    save(output / "plan.json", {"created_utc": now(), "selection": selected, "max_calls": 24, "round_cap": 1,
        "source": str(source), "source_predictions_sha256": sha(source / "predictions.json"),
        "source_sha256": hashes, "carried_occupied": previous["occupied"], "limits": previous["limits"],
        "ablation": "H0-conditioned Gemini proposing vs blind Gemini proposing; same semantic verifier",
        "development_replay_after_prior_scoring": True, "no_gt_in_requests": True,
        "old_arm": "prior saved conditioned-pool semantic result, not the scores-only verifier"})
    for path in dependencies:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    calls = Calls(output, previous["occupied"], limits={k: Decimal(v) for k, v in previous["limits"].items()},
                  providers={**PROVIDERS, PROPOSER: "Google AI Studio"}, max_calls=24)
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    rows, results = [], []
    for s, cache in zip(selected, cached, strict=True):
        key, base, h0 = s["key"], build_base(adapter, s), old.labels(cache["h0"])
        proposed = calls.call(key, "blind_proposal", "base", blind_proposal_body(base, s))
        pool, final, history = old.make_pool(h0), deepcopy(h0), []
        status = "PROPOSAL_FAILED"
        try:
            if proposed is None:
                raise ValueError("missing proposal")
            pool = old.make_pool(h0, proposed)
        except (ApiSchemaError, ValueError, TypeError, KeyError):
            pass
        else:
            def one(seat, key=key, base=base, s=s, pool=pool):
                raw = calls.call(key, "semantic_1", seat, review_body_v3(seat, base, s, pool))
                return normalize_review_wire(seat, raw, pool)
            with ThreadPoolExecutor(max_workers=5) as workers:
                raw = dict(zip(old.SEATS, workers.map(one, old.SEATS), strict=True))
            means, clean = semantic.aggregate(raw, pool, image_count=len(base.images))
            final = old.select(h0, pool, means)
            status = "FROZEN_ONE_ROUND_ABLATION"
            history = [{"round": 1, "pool": pool, "proposal": proposed, "raw": raw, "means": means,
                        "normalized": clean, "accepted": final, "status": "VALID"}]
            save(output / "targets" / key / "round_1.json", history[0])
        result = {"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": h0, "final": final,
                  "history": history, "snapshots": [deepcopy(final) for _ in range(3)], "stop_reason": status}
        save(output / "targets" / key / "result.json", result)
        results.append(result)
        rows.append({"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": h0,
                     "h1": None, "final": final, "status": status})
        save(output / "predictions.json", rows)
        print(json.dumps({"target": key, "changed": final != h0, "candidates": len(pool["propositions"]),
                          "calls": len(calls.rows)}), flush=True)
    calls.stopped = True
    calls.persist()
    if any(sha(ROOT / p) != v for p, v in hashes.items()):
        raise ValueError("inference source changed")
    frozen = sha(output / "predictions.json")
    report, details = score(output, adapter, rows, results)
    prior_report, prior_truth = score_saved(adapter, cached)
    save(output / "old_scored_predictions.json", prior_truth)
    save(output / "old_comparison.json", prior_report)
    assert sha(output / "predictions.json") == frozen
    summary = {"targets": len(rows), "post_calls": len(calls.rows), "prediction_sha256": frozen,
               "comparison": report, "old_comparison": prior_report, "details": details,
               "native_costs": {k: str(sum(Decimal(r["charge"]) for r in calls.rows
                                           if r["account"] == k and r["charge_kind"] == "native")) for k in previous["limits"]}}
    save(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("targets", "post_calls", "native_costs")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.source, args.output)
