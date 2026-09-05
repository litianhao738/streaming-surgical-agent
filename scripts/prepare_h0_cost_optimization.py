"""Prepare two isolated H0 cost contrasts; this module never sends API requests."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_h0_frame_strategy_study import VIDEOS, validate_request
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.schema import schema_for
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file

ARMS = ("baseline_low", "lean_low", "original_minimal")
TRACK_TEXT = (
    "The track summary and workflow summary are fallible predictions, never ground\n"
    "truth. Prefer current visual evidence when they conflict with the image.\n"
)
FRAME_TEXT = (
    "The causal_frame_ids describe the full window consumed locally. The uploaded\n"
    "images are only selected_image_frame_ids, remain chronological, and always end\n"
    "at the target frame. Use older images and temporal_evidence to assess motion and\n"
    "causal continuity; use the target image to decide which labels are active now.\n"
)
LEAN_FRAME_TEXT = (
    "Uploaded images follow frame_ids in chronological order and end at target_frame_id.\n"
    "relative_seconds gives each image's time relative to the target. Use older images\n"
    "to assess motion and causal continuity; use the target image to decide which labels are active now.\n"
)


def transform(payload, generation, schema_version, arm):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    payload, generation = thaw_json(payload), thaw_json(generation)
    data = json.loads(payload["input_text"])
    if (generation.get("reasoning") != {"effort": "low"}
            or data["prior_finalized_prediction"] is not None
            or data["track_summary"]["status"] != "UNAVAILABLE"
            or data["track_summary"]["frames"]
            or data["workflow_summary"]["source_max_frame_id"] is not None
            or data["workflow_summary"]["recent_finalized_phases"]
            or data["workflow_summary"]["observed_transitions"]):
        raise ValueError("requires original pure H0 low baseline without state")
    if arm == "original_minimal":
        generation["reasoning"] = {"effort": "minimal"}
    elif arm == "lean_low":
        system = payload["system_text"]
        schema_suffix = "Return exactly this JSON schema:\n" + json.dumps(schema_for(schema_version), sort_keys=True)
        if not system.endswith(schema_suffix) or system.count(TRACK_TEXT) != 1 or system.count(FRAME_TEXT) != 1:
            raise ValueError("unexpected original prompt; do not remove text heuristically")
        system = system[:-len(schema_suffix)] + "Use the supplied response_format JSON schema."
        system = system.replace(TRACK_TEXT, "").replace(FRAME_TEXT, LEAN_FRAME_TEXT)
        payload["system_text"] = system
        payload["input_text"] = json.dumps({
            "frame_ids": data["selected_image_frame_ids"],
            "relative_seconds": data["relative_seconds"],
            "target_frame_id": data["target_frame_id"],
        }, sort_keys=True, separators=(",", ":"))
    return payload, generation


def build_variant(base, arm):
    data = json.loads(base.payload["input_text"])
    validate_request(base, "B", data["target_frame_id"])
    payload, generation = transform(base.payload, base.generation_parameters, base.response_schema_version, arm)
    return replace(base, payload=payload, generation_parameters=generation,
                   prompt_version="joint_final_only_lean_input_v1" if arm == "lean_low" else base.prompt_version)


def prepare(source, output):
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    # Fixed positions outside the prior 32-target paid pilot; no GT labels are read here.
    positions = (8, 14)
    samples = [{"video_id": video, "frame_id": plan["selection"][video]["selected_targets"][position]}
               for position in positions for video in VIDEOS]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "PREPARED_NOT_EXECUTED", "source_plan_sha256": plan["plan_sha256"],
        "source_plan_file_sha256": sha256_file(source / "plan.json"), "builder_sha256": sha256_file(__file__),
        "model": plan["model"], "source_split": "Training", "arms": ARMS,
        "contrasts": {"input_only": ["baseline_low", "lean_low"], "reasoning_only": ["baseline_low", "original_minimal"]},
        "sample_positions_zero_based": positions, "samples": samples, "selection_uses_gt_labels": False,
        "prior_exposure": "prior stopped runs produced offline GT scores for frozen targets; this preparation reads only the frozen request plan, never GT values",
        "planned_requests": len(samples) * len(ARMS), "planning_budget_usd": 0.50,
        "budget_note": "proposal only; estimate approximately 0.29 USD at prior baseline average, not a quoted charge or executed budget guard",
        "paid_calls_in_preparation": 0, "retries": 0, "suggested_concurrency": 2,
        "max_tokens": 4096, "temperature": 0, "frame_offsets_raw": [-50, -25, 0],
        "image_detail": ["low", "low", "high"], "tracker": False, "repair": False, "memory": False,
        "output_schema": "joint_perception_final_only_v1", "call_order": [], "files_sha256": {},
        "evaluation": "same targets and task masks; all-target exact and micro-PRF plus shared-success paired metrics; report input/cache/reasoning/visible tokens, cost, latency and failures",
    }
    sizes = {}
    for index, sample in enumerate(samples):
        key = f"{sample['video_id']}_{sample['frame_id']}_B"
        original = json.loads((source / "requests" / f"{key}.json").read_text(encoding="utf-8"))
        if original["metadata"] != plan["requests"][key]:
            raise ValueError("source request differs from frozen plan")
        rotation = index % len(ARMS)
        for arm in ARMS[rotation:] + ARMS[:rotation]:
            payload, generation = transform(original["payload"], original["metadata"]["generation_parameters"],
                                            original["metadata"]["response_schema_version"], arm)
            name = f"{sample['video_id']}_{sample['frame_id']}_{arm}"
            path = output / "requests" / f"{name}.json"
            atomic_write_json(path, {
                "status": "REQUEST_SPEC_NOT_SENT", "source_request_metadata": original["metadata"],
                "source_request_file_sha256": sha256_file(source / "requests" / f"{key}.json"),
                "payload": payload, "generation_parameters": generation,
                "prompt_version": "joint_final_only_lean_input_v1" if arm == "lean_low" else original["metadata"]["prompt_version"],
                "images_note": "same three source images bound by source metadata; execution must reload bytes, validate provenance and compute new canonical request hash",
            })
            manifest["call_order"].append({**sample, "arm": arm, "key": name})
            manifest["files_sha256"][str(path.relative_to(output))] = sha256_file(path)
            sizes.setdefault(arm, {"system_characters": len(payload["system_text"]),
                                    "dynamic_input_characters": len(payload["input_text"])})
            if index == 0:
                (output / f"{arm}_system.txt").write_text(payload["system_text"], encoding="utf-8")
                atomic_write_json(output / f"{arm}_input_example.json", json.loads(payload["input_text"]))
    manifest["example_character_counts_not_tokens"] = sizes
    atomic_write_json(output / "output_schema.json", schema_for("joint_perception_final_only_v1"))
    atomic_write_json(output / "plan.json", manifest)
    print(json.dumps({"status": manifest["status"], "requests": len(manifest["call_order"]), "samples": samples,
                      "character_counts_not_tokens": sizes, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/preflight/h0_frame_strategy_qwen0902_training80_20260905")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preflight/h0_cost_two_proposals_20260905")
    args = parser.parse_args()
    prepare(args.source, args.output)
