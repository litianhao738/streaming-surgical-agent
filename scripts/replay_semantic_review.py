"""One-round wire-contract revision on identical saved first-round candidates.

No new H0, proposal, GT-dependent sampling, threshold changes or retries. The
first Gemini contract response is checked before dispatching the other seats.
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

from scripts.run_candidate_panel_trial import Calls, score, sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import build_base, now, read, save
from scripts.run_semantic_candidate_trial import semantic_body
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.verification import candidate_coordinator as old
from surgical_agent.research.verification import semantic_coordinator as semantic


def review_body_v2(seat, base, selected, pool):
    body = semantic_body(seat, base, selected, pool)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    target_index = len(base.images) - 1
    packet["output_contract_clarification"] = {
        "finding": "Use exactly MATCH, REFUTED, UNCLEAR, VISIBLE_ONLY or INDIRECT_EFFECT. "
                   "Never put rating-scale prose such as 'likely supported' in finding.",
        "current_image_index": target_index,
        "image_order": "Images are supplied oldest first, current LAST. Indices are zero-based. "
                       f"Every MATCH or conclusive negative must include index {target_index}. "
                       "If evidence exists only in a past image, return UNCLEAR/rating 3.",
        "scope": "Scope describes your CLAIM, not the size of the tool or where you looked. "
                 "A whole-frame absence claim requires checking every visible tool. "
                 "Return REFUTED + WHOLE_FRAME only if this check supports absence. "
                 "If you can refute only one local instance, use UNCLEAR + rating 3; "
                 "never claim no frame-level label from one local negative.",
        "single_score": "rating is the only numeric judgment; use the exact finding enum and do not add auxiliary scores.",
    }
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return body


def execute(source, output):
    if output.exists():
        raise ValueError("new output directory required")
    original, prior = read(source / "plan.json"), read(source / "budget.json")
    if not prior["stopped"]:
        raise ValueError("previous inference must finish before replay")
    predictions = read(source / "predictions.json")
    selected = original["selection"]
    if len(predictions) != len(selected):
        raise ValueError("cannot silently exclude failed H0 targets")
    dependencies = [Path(__file__), ROOT / "scripts/run_semantic_candidate_trial.py",
                    ROOT / "scripts/run_candidate_panel_trial.py", ROOT / "scripts/check_candidate_panel_providers.py",
                    ROOT / "src/surgical_agent/research/verification/semantic_coordinator.py",
                    ROOT / "src/surgical_agent/research/verification/candidate_coordinator.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in dependencies}
    allowance = 96 - len(prior["calls"])
    if allowance < 5 * len(selected):
        raise ValueError("combined original 96-call allowance would be exceeded")
    plan = {"created_utc": now(), "source": str(source), "selection": selected,
            "source_predictions_sha256": sha(source / "predictions.json"), "source_sha256": hashes,
            "max_calls": 5 * len(selected), "combined_call_cap": 96,
            "limits": prior["limits"], "carried_occupied": prior["occupied"],
            "changed": "Only explicit enum/image-index/scope clarification; same schema and acceptance rules",
            "scope": "same-four-target development replay; no independent generalization claim"}
    save(output / "plan.json", plan)
    for p in dependencies:
        dest = output / "frozen_source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    calls = Calls(output, prior["occupied"], limits={k: Decimal(v) for k, v in prior["limits"].items()},
                  max_calls=plan["max_calls"])
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    rows, results, old_rows = [], [], []
    for index, (s, original_row) in enumerate(zip(selected, predictions, strict=True)):
        key, base = s["key"], build_base(adapter, s)
        first = read(source / "targets" / key / "round_1.json")
        pool, h0 = first["pool"], original_row["h0"]
        raw = {}
        def one(seat, key=key, base=base, s=s, pool=pool):
            return calls.call(key, "semantic_wire_v2", seat, review_body_v2(seat, base, s, pool))
        if index == 0:
            raw["gemini"] = one("gemini")
            # Configuration/schema failure is diagnosed before paying four more seats.
            probe = {seat: raw["gemini"] for seat in old.SEATS}
            _, clean = semantic.aggregate(probe, pool, image_count=len(base.images))
            schema_errors = {pid: e for pid, e in clean["gemini"]["errors"].items() if e == "SCHEMA_INVALID"}
            save(output / "contract_probe.json", {"schema_errors": schema_errors, "raw": raw["gemini"]})
            if schema_errors:
                calls.stopped = True
                calls.persist()
                print("Contract probe failed; remaining calls withheld.", flush=True)
                return
        seats = [seat for seat in old.SEATS if seat not in raw]
        with ThreadPoolExecutor(max_workers=5) as workers:
            raw.update(zip(seats, workers.map(one, seats), strict=True))
        means, clean = semantic.aggregate(raw, pool, image_count=len(base.images))
        final = old.select(h0, pool, means)
        history = {"round": 1, "pool": pool, "raw": raw, "means": means,
                   "normalized": clean, "accepted": final, "status": "VALID"}
        result = {"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": h0, "final": final,
                  "history": [history], "snapshots": [deepcopy(final) for _ in range(3)],
                  "stop_reason": "FROZEN_FIRST_ROUND_REPLAY"}
        save(output / "targets" / key / "round_1.json", history)
        save(output / "targets" / key / "result.json", result)
        previous_old = read(source / "targets" / key / "old_review.json")
        save(output / "targets" / key / "old_review.json", previous_old)
        rows.append({**original_row, "final": final, "status": result["stop_reason"]})
        old_rows.append({**rows[-1], "final": previous_old["final"]})
        results.append(result)
        save(output / "predictions.json", rows)
        print(json.dumps({"target": key, "changed": old.labels(final) != old.labels(h0),
                          "invalid_items": sum(len(r["errors"]) for r in clean.values()),
                          "calls": len(calls.rows)}), flush=True)
    calls.stopped = True
    calls.persist()
    frozen_hash = sha(output / "predictions.json")
    if any(sha(ROOT / p) != v for p, v in hashes.items()):
        raise ValueError("inference source changed")
    report, details = score(output, adapter, rows, results)
    old_report, old_truth = score_saved(adapter, old_rows)
    save(output / "old_scored_predictions.json", old_truth)
    save(output / "old_predictions.json", old_rows)
    save(output / "old_comparison.json", old_report)
    assert sha(output / "predictions.json") == frozen_hash
    summary = {"targets": len(rows), "post_calls": len(calls.rows), "prediction_sha256": frozen_hash,
               "comparison": report, "old_comparison": old_report, "details": details,
               "native_costs": {k: str(sum(Decimal(r["charge"]) for r in calls.rows
                                           if r["account"] == k and r["charge_kind"] == "native")) for k in prior["limits"]}}
    save(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("targets", "post_calls", "native_costs")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.source, args.output)
