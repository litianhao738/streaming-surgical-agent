"""Freeze and run 24 additional paired baseline/schema-only H0 targets.

The earlier runner is loaded into a private module object so its audited sender,
accounting, scoring and restoration can be reused without changing its globals
for any existing importer. Existing experiment files are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("baseline", "schema_only")
POSITIONS = (10, 11, 13, 16, 17, 19)
CALL_LIMIT, CONCURRENCY, CAP, RESERVE = 48, 4, 0.85, 0.05
PREREGISTRATION = "docs/H0_SCHEMA_CONFIRMATION_SMOKE_2026-09-06.md"
PRIOR_PLANS = (
    "artifacts/preflight/h0_conservative_input_qwen0902_20260905/plan.json",
    "artifacts/preflight/h0_prompt_refinement_qwen0902_20260906/plan.json",
)


def load_private_runner():
    spec = importlib.util.spec_from_file_location(
        "_private_h0_schema_confirmation_helpers",
        ROOT / "scripts/run_h0_prompt_refinement_smoke.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load existing audited helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ARMS = ARMS
    module.POSITIONS = POSITIONS
    module.CALL_LIMIT, module.CONCURRENCY = CALL_LIMIT, CONCURRENCY
    module.CAP, module.RESERVE = CAP, RESERVE
    return module


prior = load_private_runner()
study = prior.study
_paired_analysis = prior.analyze
_restore = prior.restore


def selected_samples(source):
    return [
        {"video_id": video,
         "frame_id": source["selection"][video]["selected_targets"][position],
         "source_position_zero_based": position}
        for position in POSITIONS for video in study.VIDEOS
    ]


def exclusion_precheck(source, samples):
    # All first-eight positions count as exposed, including unsuccessful calls.
    exposed = {(video, frame) for video in study.VIDEOS
               for frame in source["selection"][video]["selected_targets"][:8]}
    provenance = {}
    for name in PRIOR_PLANS:
        path = ROOT / name
        previous = prior.read_json(path)
        prior.verify_digest(previous)
        exposed.update((s["video_id"], s["frame_id"]) for s in previous["samples"])
        provenance[name] = study.sha256_file(path)
    chosen = {(s["video_id"], s["frame_id"]) for s in samples}
    if len(chosen) != 24 or chosen & exposed:
        raise ValueError("fixed confirmation targets duplicate a prior paid cohort")
    return provenance


def prepare(args):
    if (args.output / "plan.json").exists():
        result = restore(args)
        if (args.output / "execution.lock").exists():
            raise ValueError("execution already started; preserve frozen experiment")
        return result
    source = prior.read_json(args.source / "plan.json")
    prior.verify_digest(source)
    config = prior.configured(args, source)
    for name, expected in source["source_sha256"].items():
        if study.sha256_file(ROOT / name) != expected:
            raise ValueError("original study dependency changed")
    samples = selected_samples(source)
    exclusions = exclusion_precheck(source, samples)
    masks, annotation_sources = prior.mask_precheck(args, samples)
    source_index = {
        f"cholectrack20:{Path(p).parent.parent.name}:frame:{int(Path(p).stem)}": (p, sha)
        for p, sha in source["source_png_sha256"].items()
    }
    saved_bases, selected_paths = {}, {}
    for sample in samples:
        key = f"{sample['video_id']}_{sample['frame_id']}_B"
        saved = prior.read_json(args.source / "requests" / f"{key}.json")
        if saved["metadata"] != source["requests"][key]:
            raise ValueError("original baseline metadata changed")
        saved_bases[key] = saved
        for image in saved["metadata"]["images"]:
            path, expected = source_index[image["identifier"]]
            selected_paths[path] = expected
    images = prior.encode_images(selected_paths)
    requests, order = {}, []
    for index, sample in enumerate(samples):
        base = prior.reconstruct(
            saved_bases[f"{sample['video_id']}_{sample['frame_id']}_B"], images
        )
        for arm in ARMS if index % 2 == 0 else ARMS[::-1]:
            item = {**sample, "arm": arm, "group": "B",
                    "key": f"{sample['video_id']}_{sample['frame_id']}_{arm}"}
            candidate = prior.variant_request(base, arm)
            if (candidate.images != base.images
                    or candidate.generation_parameters != base.generation_parameters
                    or candidate.response_schema_version != base.response_schema_version
                    or candidate.payload["input_text"] != base.payload["input_text"]):
                raise ValueError("candidate alters something beyond system schema suffix")
            requests[item["key"]] = candidate
            order.append(item)
    dependencies = dict(source["source_sha256"])
    for name in (
        "scripts/run_h0_schema_confirmation_smoke.py",
        "scripts/run_h0_prompt_refinement_smoke.py", "scripts/h0_prompt_refinement.py",
        "scripts/run_h0_conservative_input_smoke.py", "scripts/prepare_h0_cost_optimization.py",
        "scripts/resume_h0_frame_strategy_study.py",
        "src/surgical_agent/perception/final_only.py", "src/surgical_agent/api/schema.py",
    ):
        dependencies[name] = study.sha256_file(ROOT / name)
    preregistration = (ROOT / PREREGISTRATION).read_text(encoding="utf-8")
    plan = {
        "schema_version": "h0_schema_confirmation_smoke_v1", "status": "FROZEN_BEFORE_INFERENCE",
        "model": config.requested_model_identifier, "source_split": "Training", "samples": samples,
        "source_plan_sha256": source["plan_sha256"],
        "source_plan_path": str((args.source / "plan.json").resolve()),
        "selection_policy": "Fixed positions 10,11,13,16,17,19 per video; masks only; abort without substitution",
        "prior_exposure": "24 targets not previously paid in the 32-window, 8-conservative, or 16-refinement cohorts. Original 80-target annotations were parsed/scored offline; development data, not a sealed test set.",
        "excluded_prior_plans_sha256": exclusions,
        "excluded_window_positions_zero_based": list(range(8)),
        "label_access_policy": "Adapter parses annotations; precheck inspects masks only. Freeze every request before current scoring. No GT in inputs.",
        "all_five_head_valid_masks": masks, "annotation_sources_sha256": annotation_sources,
        "arms": list(ARMS), "arm_changes": {
            "baseline": "Exact retained three-frame low request",
            "schema_only": "Existing candidate: replace only complete system schema suffix with short response_format reference",
        }, "call_order": order,
        "order_policy": "Alternate AB/BA by sample; four concurrent dispatches; completion order uncontrolled",
        "paid_call_limit": CALL_LIMIT, "concurrency": CONCURRENCY, "retries": 0,
        "planning_budget_usd": CAP, "inflight_reserve_usd": RESERVE,
        "spend_policy": "Known spend plus 0.05 USD per outstanding new call <= min(0.85, balance minus 0.10); planning cap not provider-enforced; stop on unknown cost",
        "adoption_rule": "All five task micro-F1 and exact accuracy, all-five exact accuracy, complete response rate must not decrease on all-target and shared-success cohorts; actual cost and uncached price-equivalent mean must decrease. Require 48 valid wire audits and 24 native usage/raw/schema-complete records per arm. Pilot pass is not statistical noninferiority.",
        "preregistration_snapshot": {"path": "preregistration.md",
                                     "sha256": hashlib.sha256(preregistration.encode()).hexdigest()},
        "tracker": False, "repair": False, "gate": False, "memory": False,
        "config_sha256": source["config_sha256"],
        "repair_manifest_sha256": source["repair_manifest_sha256"],
        "source_sha256": dependencies, "source_png_sha256": selected_paths,
        "requests": {key: study.canonical_request_metadata(req).to_mapping()
                     for key, req in requests.items()},
    }
    plan["plan_sha256"] = hashlib.sha256(prior.canonical_json_bytes(plan)).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("execution.lock", "calls", "cache", "predictions.json")):
        raise ValueError("fresh experiment output required")
    snapshot = args.output / "preregistration.md"
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != preregistration:
        raise ValueError("preexisting preregistration differs")
    if not snapshot.exists():
        snapshot.write_text(preregistration, encoding="utf-8", newline="")
    for key, request in requests.items():
        path = args.output / "requests" / f"{key}.json"
        record = {"payload": prior.thaw_json(request.payload), "metadata": plan["requests"][key]}
        if path.exists() and prior.read_json(path) != record:
            raise ValueError("preexisting request differs")
        if not path.exists():
            study.atomic_write_json(path, record)
    study.atomic_write_json(args.output / "plan.json", plan)
    return config, plan, requests


def restore(args):
    config, plan, requests = _restore(args)
    source = prior.read_json(Path(plan["source_plan_path"]))
    if (plan["samples"] != selected_samples(source) or plan["arms"] != list(ARMS)
            or plan["paid_call_limit"] != CALL_LIMIT or len(requests) != CALL_LIMIT
            or plan["concurrency"] != CONCURRENCY or plan["planning_budget_usd"] != CAP):
        raise ValueError("frozen confirmation protocol differs")
    if exclusion_precheck(source, plan["samples"]) != plan["excluded_prior_plans_sha256"]:
        raise ValueError("frozen prior cohort provenance differs")
    return config, plan, requests


def analyze(args, plan, rows, requests):
    # This function's private module globals contain exactly the two current arms.
    summary = _paired_analysis(args, plan, rows, requests)
    summary["schema_version"] = "h0_schema_confirmation_result_v1"
    decision = summary["adoption_assessment"]["schema_only"]
    failures = decision["failed_requirements"]
    wire = summary["wire_audit"]
    if wire["status"] != "PASS" or wire["checked"] != CALL_LIMIT:
        failures.append("complete_valid_wire_audit")
    expected_keys = {item["key"] for item in plan["call_order"]}
    if len(rows) != CALL_LIMIT or {row["key"] for row in rows} != expected_keys:
        failures.append("complete_unique_frozen_rows")
    for arm in ARMS:
        group = summary["groups"][arm]
        shape = group["complete_schema"]
        if (group["native_usage_count"] != 24 or shape["raw_parsed_responses"] != 24
                or shape["valid"] != 24 or shape["scheduled"] != 24):
            failures.append(f"{arm}.complete_native_raw_schema_records")
    decision["pilot_rule_pass"] = not failures
    study.atomic_write_json(args.output / "summary.json", summary)
    return summary


prior.prepare, prior.restore, prior.analyze = prepare, restore, analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/preflight/h0_frame_strategy_qwen0902_training80_20260905")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preflight/h0_schema_confirmation_qwen0902_20260906")
    parser.add_argument("--dataset", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml")
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "docs/API.txt")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    try:
        prior.run(args)
    except Exception as exc:  # noqa: BLE001 - never print credentials/provider bodies
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__,
                          "detail": str(exc) if isinstance(exc, (ValueError, FileExistsError))
                          else "See preserved artifacts; no inference retry"}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
