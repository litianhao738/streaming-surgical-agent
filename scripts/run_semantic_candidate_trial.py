"""Training-only frozen-H0 / Gemini proposal / old-vs-evidence panel experiment.

Prepare freezes timeline samples and source hashes. Execute never overwrites a
paid run. The old score arm shares the exact first pool/images with the new arm.
Only the new arm may refill, at most three rounds; report round one separately.
"""
import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import MODELS, body_for
from scripts.run_candidate_panel_trial import (
    PROVIDERS,
    RATES,
    Calls,
    proposal_body,
    score,
    sha,
)
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_presence_review_trial import wire_body
from scripts.run_prior_panel_trial import build_base, now, read, save
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.verification import candidate_coordinator as old
from surgical_agent.research.verification import semantic_coordinator as semantic

PROPOSER = "google/gemini-3.8-flash"
PREVIOUS = ROOT / "artifacts/preflight/target_interaction_single_field_20260908_v2"
ALLOWANCE = {"openrouter_usd": Decimal("1.20"), "xai_usd": Decimal("0.50"), "aliyun_cny": Decimal(1)}
BOUNDARIES = (
    "Each candidate is a CURRENT-frame dataset proposition, not an anatomy visibility question. "
    "Instrument: some visible tool belongs to this class. Verb: some tool performs this action on an object. "
    "Target: this anatomy is the object directly acted on by at least one instrument in the current frame. "
    "Background visibility, proximity, downstream organ deformation or transmitted force alone do not establish that Target. "
    "IVT: this particular instrument-action-target relation, not three unrelated visible components. "
    "Grasp and retract are distinct ontology labels: gripping tissue alone does not prove retract, "
    "and visible displacement alone does not identify the tissue at the tool tip. "
    "Do not infer the action from phase or instrument identity. Do not replace an unsupported relation with a "
    "nearby legal triplet merely to avoid null; use the supplied ontology boundaries for null labels. "
    "An IVT's rejection does not by itself refute its component labels: another tool/relation may support them."
)


def semantic_body(seat, base, selected, pool):
    body = body_for(seat, base, selected, pool, gemini_json_object=True)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    schema = semantic.review_schema(pool, len(base.images))
    packet.update(task="Audit each CURRENT-frame proposition against the images independently.",
        proposition_semantics=BOUNDARIES,
        rating_scale={"1": "clearly refuted", "2": "likely refuted", "3": "uncertain",
                      "4": "likely supported", "5": "clearly supported"},
        instructions=(
            "Return judgments only, with ONE authoritative rating per candidate. MATCH requires rating 4/5. "
            "REFUTED requires 1/2; UNCLEAR requires 3. For Target only, VISIBLE_ONLY and INDIRECT_EFFECT "
            "mean 1/2: anatomy is seen or affected but is not the actual tool-action object. "
            "A conclusive judgment must cite the target image index; history-only events are insufficient. "
            "Deletion is a frame-level claim: any 1/2 requires WHOLE_FRAME scope after checking all tools. "
            "A negative observation of just one tool/region requires UNCLEAR and 3. "
            "For MATCH give the observed tool/action/object and its image location in one short sentence. "
            "For REFUTED state what contradicts the proposition; occlusion is uncertainty, not refutation. "
            "Keep observation under about 20 words where possible (hard limit 1000 characters). "
            "Do not supply a second score or bounding box. Candidate membership is not visual evidence. "
            "No confidence in another judge's answer or majority vote is available here."),
        response_schema=schema)
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    body["max_tokens"] = 8192
    if body["response_format"]["type"] == "json_schema":
        body["response_format"]["json_schema"].update(name="candidate_semantic_v1", schema=schema)
    return body


def gemini_proposal(base, selected, state, pool, issues):
    body = proposal_body(base, selected, state, pool, issues)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    packet["proposition_semantics"] = BOUNDARIES
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    body.update(model=PROPOSER, provider={"only": ["google-ai-studio"], "allow_fallbacks": False,
                                        "require_parameters": True})
    # Gemini 3.8 Flash requires thinking. This is the proposer, not a reviewer.
    body["reasoning"] = {"effort": "low"}
    return body


def prepare(output, adapter):
    if output.exists():
        raise ValueError("new output directory required")
    selected = []
    for video in ("VID04", "VID13"):
        samples = list(adapter.iter_inference_video(video))
        for numerator in (1, 2):
            s = samples[len(samples) * numerator // 3]
            if s.source_split is not DatasetSplit.TRAINING:
                raise ValueError("Training only")
            item = {"video_id": video, "frame_id": s.target_frame_id,
                    "key": f"{video}_{s.target_frame_id}", "causal_frame_ids": list(s.causal_frame_ids),
                    "sampling_rule": f"canonical timeline quantile {numerator}/3",
                    "images": [{"frame_id": f, "path": str(p), "sha256": sha(p)}
                               for f, p in zip(s.causal_frame_ids, s.media_refs, strict=True)]}
            base = build_base(adapter, item)
            item["request_metadata"] = canonical_request_metadata(base).to_mapping()
            selected.append(item)
    previous = read(PREVIOUS / "budget.json")["occupied"]
    limits = {k: str(Decimal(previous[k]) + value) for k, value in ALLOWANCE.items()}
    sources = [Path(__file__), ROOT / "scripts/run_candidate_panel_trial.py",
               ROOT / "scripts/check_candidate_panel_providers.py", ROOT / "scripts/run_prior_panel_trial.py",
               ROOT / "scripts/run_presence_review_trial.py", ROOT / "scripts/run_grounded_api_pipeline.py",
               *sorted((ROOT / "src/surgical_agent/research/verification").glob("*.py")),
               *sorted((ROOT / "src/surgical_agent/perception").rglob("*.py")),
               *sorted((ROOT / "src/surgical_agent/perception/prompts").glob("*")),
               ROOT / "configs/perception/joint_openrouter_h0.yaml",
               ROOT / "src/surgical_agent/evaluation/repair_comparison.py"]
    sources = [p for p in sources if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    plan = {"created_utc": now(), "selection": selected, "max_calls": 96, "round_cap": 3,
            "h0": "original frozen qwen/qwen3.8-max-0902 Alibaba final-only",
            "proposer": PROPOSER, "reviewers": MODELS, "previous_budget": str(PREVIOUS),
            "carried_occupied": previous, "limits": limits, "incremental_allowances": {k: str(v) for k, v in ALLOWANCE.items()},
            "source_sha256": hashes, "no_gt_in_selection_or_inference": True,
            "comparison": "old scores vs semantic judgments share first pool; extra new-arm rounds reported separately",
            "stopping": "3 rounds or no new candidates or failure/budget; no retry",
            "caveats": ["Development sample, not an independent held-out effectiveness claim.",
                        "Explicit conflicts block edits; this does not certify visual correctness."]}
    save(output / "plan.json", plan)
    for p in sources:
        destination = output / "frozen_source" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    print(json.dumps({"selection": [(s["key"], s["causal_frame_ids"]) for s in selected],
                      "max_calls": 96, "incremental_allowances": plan["incremental_allowances"]}), flush=True)


def execute(output, adapter, *, review_builder=semantic_body, review_normalizer=None,
            base_builder=build_base, h0_wire_builder=wire_body):
    plan = read(output / "plan.json")
    if (output / "budget.json").exists() or (output / "predictions.json").exists():
        raise ValueError("paid replay/overwrite prohibited; use a separately frozen follow-up")
    if any(sha(ROOT / p) != value for p, value in plan["source_sha256"].items()):
        raise ValueError("prepared source changed")
    calls = Calls(output, plan["carried_occupied"], limits={k: Decimal(v) for k, v in plan["limits"].items()},
                  providers={**PROVIDERS, PROPOSER: "Google AI Studio"}, rates=RATES, max_calls=plan["max_calls"])
    rows, results, comparisons, failed_h0 = [], [], [], []
    compare_old = plan.get("compare_old", True)
    for selected in plan["selection"]:
        key, base = selected["key"], base_builder(adapter, selected)
        if canonical_request_metadata(base).to_mapping() != selected["request_metadata"]:
            raise ValueError("H0 request differs from prepared model/input metadata")
        raw_h0 = calls.call(key, "h0", "base", h0_wire_builder(base))
        try:
            validate_final_only(raw_h0)
        except ApiSchemaError:
            failed_h0.append(key)
            save(output / "failed_h0.json", failed_h0)
            continue
        h0 = {q: [raw_h0[q]["selected_id"]] if q == "phase" else raw_h0[q]["selected_ids"]
              for q in (*old.TASKS, "phase")}
        def propose(n, state, pool, issues, key=key, base=base, selected=selected):
            return calls.call(key, f"proposal_{n}", "base", gemini_proposal(base, selected, state, pool, issues))
        old_record = {}
        def panel(pool, n, semantic_mode, key=key, base=base, selected=selected):
            def one(seat):
                body = (review_builder(seat, base, selected, pool) if semantic_mode
                        else body_for(seat, base, selected, pool, gemini_json_object=True))
                if not semantic_mode:
                    body["max_tokens"] = 1536
                raw = calls.call(key, f"{'semantic' if semantic_mode else 'old'}_{n}", seat, body)
                return review_normalizer(seat, raw, pool) if semantic_mode and review_normalizer else raw
            with ThreadPoolExecutor(max_workers=5) as workers:
                return dict(zip(old.SEATS, workers.map(one, old.SEATS), strict=True))
        def review(n, state, pool, panel=panel, old_record=old_record, h0=h0, key=key):
            del state
            if n == 1 and compare_old:
                raw = panel(pool, n, False)
                old_record.update(pool=pool, raw=raw, final=h0, status="FAILED")
                try:
                    means, _ = old.aggregate(raw, pool)
                    old_record.update(final=old.select(h0, pool, means), means=means, status="VALID")
                except (ApiSchemaError, ValueError, TypeError, KeyError):
                    pass
                save(output / "targets" / key / "old_review.json", old_record)
            return panel(pool, n, True)
        result = old.run(h0, propose, review,
            lambda stage, value, key=key: save(output / "targets" / key / f"{stage}.json", value),
            aggregate_review=partial(semantic.aggregate, image_count=len(base.images)),
            max_rounds=plan["round_cap"])
        result.update(video_id=selected["video_id"], frame_id=selected["frame_id"])
        save(output / "targets" / key / "result.json", result)
        results.append(result)
        rows.append({"video_id": selected["video_id"], "frame_id": selected["frame_id"],
                     "h0": h0, "h1": None, "final": result["final"], "status": result["stop_reason"]})
        if compare_old:
            comparisons.append({**rows[-1], "final": old_record.get("final", h0)})
        save(output / "predictions.json", rows)
        if compare_old:
            save(output / "old_predictions.json", comparisons)
        print(json.dumps({"target": key, "stop": result["stop_reason"], "rounds": len(result["history"]),
                          "changed": old.labels(result["final"]) != old.labels(h0), "calls": len(calls.rows)}), flush=True)
    calls.stopped = True
    calls.persist()
    frozen_hash = sha(output / "predictions.json") if rows else None
    if any(sha(ROOT / p) != value for p, value in plan["source_sha256"].items()):
        raise ValueError("source changed during inference")
    # Only here is GT passed to evaluation, after inference is irrevocably closed.
    report, details = score(output, adapter, rows, results) if rows else ({}, [])
    old_report, old_truth = score_saved(adapter, comparisons) if comparisons else (None, [])
    if compare_old:
        save(output / "old_comparison.json", old_report)
        save(output / "old_scored_predictions.json", old_truth)
    if rows and sha(output / "predictions.json") != frozen_hash:
        raise ValueError("predictions changed while scoring")
    summary = {"targets": len(rows), "failed_h0": failed_h0, "post_calls": len(calls.rows),
               "old_arm_run": compare_old,
               "prediction_sha256": frozen_hash, "details": details,
               "native_costs": {k: str(sum(Decimal(r["charge"]) for r in calls.rows
                                           if r["account"] == k and r["charge_kind"] == "native")) for k in ALLOWANCE},
               "unpriced_calls": [r["index"] for r in calls.rows if r["charge_kind"] != "native"],
               "comparison": report, "old_comparison": old_report}
    save(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("comparison", "old_comparison", "details")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, adapter)
    else:
        execute(args.output, adapter)


if __name__ == "__main__":
    main()
