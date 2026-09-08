"""Mask-checked four-target Training confirmation of the corrected review wire.

Candidate samples come from 1/4 and 3/4 of each video's canonical timeline.
If unavailable, use the nearest fully annotated sample (earlier breaks ties).
Only GT availability is inspected before inference, never annotation labels.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.replay_semantic_review import review_body_v2
from scripts.run_candidate_panel_trial import Calls, sha
from scripts.run_presence_review_trial import _mask_only
from scripts.run_prior_panel_trial import build_base, now, read, save
from scripts.run_semantic_candidate_trial import execute
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.verification import semantic_coordinator as semantic


def review_body_v3(seat, base, selected, pool):
    body = review_body_v2(seat, base, selected, pool)
    if seat == "gemini":
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        # Nested ID-keyed strict objects produced empty objects in a paid probe.
        # Use a uniform row array; exact ID uniqueness/completeness stays local.
        item = semantic.item_schema(len(base.images))
        item["properties"]["candidate_id"] = {"type": "string", "enum": [p["id"] for p in pool["propositions"]]}
        item["required"].append("candidate_id")
        schema = {"type": "object", "properties": {"rows": {"type": "array", "items": item,
                  "minItems": len(pool["propositions"]), "maxItems": len(pool["propositions"])}},
                  "required": ["rows"], "additionalProperties": False}
        packet["response_schema"] = schema
        packet["wire_output"] = ("Return rows array, exactly one object per candidate_id, with these six fields: "
                                 "candidate_id, rating, finding, scope, image_indices, observation.")
        packet["instructions"] = packet["instructions"].replace("Return judgments only", "Return rows only")
        body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
        # This provider returned empty nested objects in both strict probes.
        # Uniform JSON-object rows plus exact local validation are the wire contract.
        body["response_format"] = {"type": "json_object"}
    return body


def normalize_review_wire(seat, raw, pool):
    if seat != "gemini" or raw is None:
        return raw
    if not isinstance(raw, dict) or set(raw) != {"rows"} or not isinstance(raw["rows"], list):
        return {"wire_error": "INVALID_ROW_ENVELOPE"}
    wanted = {p["id"] for p in pool["propositions"]}
    judgments = {}
    for row in raw["rows"]:
        if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
            return {"wire_error": "INVALID_ROW_ID"}
        pid = row["candidate_id"]
        if pid not in wanted or pid in judgments:
            return {"wire_error": "UNKNOWN_OR_DUPLICATE_ROW_ID"}
        judgments[pid] = {k: v for k, v in row.items() if k != "candidate_id"}
    if set(judgments) != wanted:
        return {"wire_error": "MISSING_ROW_ID"}
    return {"judgments": judgments}


def choose_complete_samples(adapter):
    selected = []
    for video in ("VID103", "VID23"):
        samples = list(adapter.iter_inference_video(video))
        availability = {r.inference.target_frame_id: _mask_only(r) for r in adapter.iter_video(video)}
        eligible = [s for s in samples if availability.get(s.target_frame_id)
                    and all(availability[s.target_frame_id].values())]
        if not eligible:
            raise ValueError("no fully annotated Training targets")
        for n in (1, 3):
            anchor = samples[len(samples) * n // 4].target_frame_id
            chosen = min(eligible, key=lambda s: (abs(s.target_frame_id - anchor), s.target_frame_id))
            if chosen.source_split is not DatasetSplit.TRAINING:
                raise ValueError("Training only")
            item = {"key": f"{video}_{chosen.target_frame_id}", "video_id": video,
                    "frame_id": chosen.target_frame_id, "anchor_frame_id": anchor,
                    "sampling_rule": f"nearest full-mask canonical Training target to {n}/4 timeline; earlier tie break",
                    "causal_frame_ids": list(chosen.causal_frame_ids),
                    "gt_availability_only": availability[chosen.target_frame_id],
                    "images": [{"frame_id": f, "path": str(p), "sha256": sha(p)}
                               for f, p in zip(chosen.causal_frame_ids, chosen.media_refs, strict=True)]}
            base = build_base(adapter, item)
            item["request_metadata"] = canonical_request_metadata(base).to_mapping()
            selected.append(item)
    if len({s["key"] for s in selected}) != 4:
        raise ValueError("duplicate selected target")
    return selected


def prepare(output, previous, adapter):
    if output.exists():
        raise ValueError("new output directory required")
    prior = read(previous / "budget.json")
    if not prior["stopped"]:
        raise ValueError("previous inference still running")
    selected = choose_complete_samples(adapter)
    sources = [Path(__file__), ROOT / "scripts/replay_semantic_review.py",
               ROOT / "scripts/run_semantic_candidate_trial.py", ROOT / "scripts/run_candidate_panel_trial.py",
               ROOT / "scripts/check_candidate_panel_providers.py", ROOT / "scripts/run_prior_panel_trial.py",
               ROOT / "scripts/run_presence_review_trial.py", ROOT / "scripts/run_grounded_api_pipeline.py",
               *sorted((ROOT / "src/surgical_agent/research/verification").glob("*.py")),
               *sorted((ROOT / "src/surgical_agent/perception").rglob("*.py")),
               *sorted((ROOT / "src/surgical_agent/perception/prompts").glob("*")),
               ROOT / "configs/perception/joint_openrouter_h0.yaml",
               ROOT / "src/surgical_agent/evaluation/repair_comparison.py"]
    sources = [p for p in sources if p.is_file()]
    plan = {"created_utc": now(), "selection": selected, "max_calls": 48, "round_cap": 1,
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "previous_budget": str(previous), "carried_occupied": prior["occupied"], "limits": prior["limits"],
            "proposer": "google/gemini-3.8-flash", "review_wire": "semantic_v1_with_explicit_contract_v3_gemini_rows",
            "h0": "frozen original qwen/qwen3.8-max-0902 Alibaba",
            "comparison": "H0 vs old vs new; exact shared first-round pool, one round only",
            "gt_policy": "selection uses availability masks only; GT labels scored after all inference stops",
            "caveats": ["Four Training targets are a small development check, not proof of generalization."]}
    save(output / "plan.json", plan)
    for path in sources:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    print(json.dumps({"selection": [s["key"] for s in selected], "masks": [s["gt_availability_only"] for s in selected],
                      "max_calls": 48, "limits": prior["limits"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "probe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "probe":
        from decimal import Decimal
        source = ROOT / "artifacts/preflight/semantic_candidate_20260908_v2"
        s = read(source / "plan.json")["selection"][1]
        base = build_base(adapter, s)
        pool = read(source / "targets" / s["key"] / "round_1.json")["pool"]
        if args.output.exists() or args.previous is None:
            raise ValueError("new output and previous ledger required")
        prior = read(args.previous / "budget.json")
        if not prior["stopped"]:
            raise ValueError("previous run still active")
        save(args.output / "plan.json", {"source": str(source), "selection": s, "max_calls": 1,
                                        "source_sha256": sha(Path(__file__))})
        calls = Calls(args.output, prior["occupied"], limits={k: Decimal(v) for k, v in prior["limits"].items()}, max_calls=1)
        raw = calls.call(s["key"], "gemini_strict_probe", "gemini", review_body_v3("gemini", base, s, pool))
        normalized = normalize_review_wire("gemini", raw, pool)
        _, clean = semantic.aggregate({seat: normalized for seat in ("grok", "qwen", "gpt", "gemini", "deepseek")}, pool)
        calls.stopped = True
        calls.persist()
        save(args.output / "result.json", {"raw": raw, "validation": clean["gemini"]})
        print(json.dumps({"parsed": raw is not None, "errors": clean["gemini"]["errors"]}), flush=True)
    elif args.command == "prepare":
        if args.previous is None:
            raise ValueError("previous ledger is required")
        prepare(args.output, args.previous, adapter)
    else:
        execute(args.output, adapter, review_builder=review_body_v3, review_normalizer=normalize_review_wire)
