"""Four one-call joint reflection repairs on a frozen expanded candidate pool."""
import argparse
import json
import shutil
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import PROVIDERS, Calls, sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import build_base, now, read, save
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.verification.reflective_repair import (
    accept_reflection,
    reflection_schema,
)


def reflection_body(base, selected, h0, pool):
    body = gemini_proposal(base, selected, h0, pool, [])
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    packet.pop("issues")
    packet["task"] = "Visually recheck the current-frame prediction jointly against candidate alternative interactions."
    packet["instructions"] = (
        "This is one bounded reflection, not permission to assume the initial answer is right or wrong. "
        "Independently inspect the supplied images, including frame-edge tools. Identify the action-object "
        "relationship for each observed tool; compare plausible alternative targets instead of independently "
        "endorsing every plausible anatomy label. Multiple tools may have the same class and different actions. "
        "All candidate memberships are hypotheses, not evidence. Do not equate visible anatomy with operated anatomy. "
        "Return the four final label sets under prediction, restricted to the candidate IDs. "
        "Return changes as an array, exactly one entry for every label you actually add or remove relative to "
        "current_prediction. Each change has candidate_id, operation, rating, finding, scope, image_indices, observation. "
        "ADD needs MATCH/rating4 or5 and current-image visual support. REMOVE needs REFUTED/rating1 or2, "
        "WHOLE_FRAME scope, and evidence excluding the label across all visible tools. "
        "Uncertainty or one local negative is insufficient to change a label: keep that initial label. "
        f"Cite current image index {len(base.images) - 1} for every change; use earlier images only for causal motion context. "
        "Give a concise observed fact for each change (at most1000 characters); no second score or unbound explanation. "
        "New IVTs require their components in the final sets; do not drop a component still needed by a retained IVT."
    )
    schema = reflection_schema(pool, len(base.images))
    packet["response_schema"] = schema
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    body.update(reasoning={"effort": "medium"}, max_tokens=8192, response_format={"type": "json_object"})
    return body


def main(source, output):
    if output.exists():
        raise ValueError("new output required")
    plan, budget, cached = read(source / "plan.json"), read(source / "budget.json"), read(source / "predictions.json")
    if not budget["stopped"]:
        raise ValueError("previous inference still running")
    limits = {k: Decimal(v) for k, v in budget["occupied"].items()}
    limits["openrouter_usd"] += Decimal("0.25")
    paths = [Path(__file__), ROOT / "src/surgical_agent/research/verification/reflective_repair.py",
             ROOT / "src/surgical_agent/research/verification/semantic_coordinator.py",
             ROOT / "src/surgical_agent/research/verification/candidate_coordinator.py",
             ROOT / "scripts/run_candidate_panel_trial.py", ROOT / "scripts/run_semantic_candidate_trial.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    save(output / "plan.json", {"created_utc": now(), "selection": plan["selection"], "max_calls": 4,
        "source": str(source), "source_prediction_sha256": sha(source / "predictions.json"), "source_sha256": hashes,
        "model": PROPOSER, "reasoning": "medium", "limits": {k: str(v) for k, v in limits.items()},
        "policy": "one joint repair; every actual diff needs bound evidence; no GT in requests",
        "scope": "same four Training development targets; not independent validation"})
    for path in paths:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    calls = Calls(output, budget["occupied"], limits=limits, providers={**PROVIDERS, PROPOSER: "Google AI Studio"}, max_calls=4)
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    rows = []
    for s, cached_row in zip(plan["selection"], cached, strict=True):
        h0, pool = cached_row["h0"], cached_row["pool"]
        raw = calls.call(s["key"], "joint_reflection", "base", reflection_body(build_base(adapter, s), s, h0, pool))
        final, status = h0, "FAILED_KEEP_H0"
        try:
            final = accept_reflection(h0, pool, raw)
            status = "VALID_PATCH" if final != h0 else "VALID_KEEP"
        except (ApiSchemaError, TypeError, ValueError, KeyError) as exc:
            status = f"{status}:{type(exc).__name__}"
        save(output / "targets" / f"{s['key']}.json", {"h0": h0, "pool": pool, "raw": raw, "final": final, "status": status})
        rows.append({"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": h0, "h1": None, "final": final, "status": status})
        save(output / "predictions.json", rows)
        print(json.dumps({"target": s["key"], "status": status, "calls": len(calls.rows)}), flush=True)
    calls.stopped = True
    calls.persist()
    frozen = sha(output / "predictions.json")
    if any(sha(ROOT / p) != v for p, v in hashes.items()):
        raise ValueError("source changed during inference")
    report, truth = score_saved(adapter, rows)
    assert sha(output / "predictions.json") == frozen
    save(output / "comparison.json", report)
    save(output / "scored_predictions.json", truth)
    summary = {"post_calls": len(calls.rows), "prediction_sha256": frozen, "comparison": report,
               "native_usd": str(sum(Decimal(r["charge"]) for r in calls.rows if r["charge_kind"] == "native"))}
    save(output / "summary.json", summary)
    print(json.dumps({"post_calls": len(calls.rows), "native_usd": summary["native_usd"], "metrics": report["arms"]["final"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
