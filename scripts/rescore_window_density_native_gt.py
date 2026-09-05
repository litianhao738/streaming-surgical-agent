"""Offline-only native Testing GT scoring for a completed density diagnostic.

The runtime Testing adapter deliberately hides supervision. This explicit
post-inference evaluator parses native labels after verifying frozen prediction
identities and annotation SHA. It never constructs or calls an API client.
Original inference artifacts, plan, and the runtime-adapter summary are preserved.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_pure_h0_smoke import TASKS, evaluate_offline
from scripts.run_window_density_h0_smoke import GROUPS, augment_metrics
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.data.parser import parse_annotation_file
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.frame_ground_truth import aggregate_frame_target


def score_native_rows(adapter, video_id, rows):
    evaluation = augment_metrics(evaluate_offline(adapter, video_id, rows))
    unavailable = [task for task, metrics in evaluation["tasks"].items() if metrics["valid_gt"] == 0]
    if len(unavailable) == len(TASKS):
        raise ValueError(
            "inference_adapter_hidden_gt: all task valid_gt counts are zero. "
            "The runtime Testing adapter intentionally hides supervision; use explicitly "
            "parsed native offline targets and inspect their masks. Annotation keys alone "
            "do not establish label availability; this is not a model accuracy result."
        )
    evaluation["tasks_without_valid_gt"] = unavailable
    evaluation["availability_note"] = (
        "Some tasks have no valid aggregated GT; their metrics remain null, not zero accuracy"
        if unavailable else "All five tasks have valid exact-target native supervision"
    )
    return evaluation


def run(args):
    if args.output.exists():
        raise ValueError("Use a fresh output directory; preserve previous evaluations")
    plan_path = args.run_dir / "plan.json"
    predictions_path = args.run_dir / "predictions.json"
    summary_path = args.run_dir / "summary.json"
    status_path = args.run_dir / "run_status.json"
    hashes = {path.name: sha256_file(path) for path in (plan_path, predictions_path, summary_path, status_path)}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    original_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] != "COMPLETE":
        raise ValueError("This six-target native rescore requires a COMPLETE frozen run")
    expected = [(item["frame_id"], item["group"]) for item in plan["call_order"]]
    if [(row["frame_id"], row["group"]) for row in predictions] != expected:
        raise ValueError("Persisted prediction identities differ from the frozen plan")
    if len(predictions) != 18 or len(plan["targets"]) != 6:
        raise ValueError("This frozen diagnostic requires all 18 prediction rows on six targets")
    if original_summary["predictions_sha256_before_gt"] != hashes["predictions.json"]:
        raise ValueError("Inference predictions have changed since completion")
    annotation_path = args.dataset / "Testing" / plan["video_id"] / (plan["video_id"].lower() + ".json")
    annotation_sha = sha256_file(annotation_path)
    if annotation_sha != plan["annotation_index_precheck"]["annotation_sha256"]:
        raise ValueError("Annotation source changed after the pre-inference key check")
    native = parse_annotation_file(annotation_path, expected_split=DatasetSplit.TESTING)
    if native.video_id != plan["video_id"]:
        raise ValueError("Native annotation video identity mismatch")
    targets = {
        frame.frame_id: aggregate_frame_target(frame, allowed_tasks=frozenset(TASKS),
            source=f"Testing/{plan['video_id']}/{plan['video_id'].lower()}.json")
        for frame in native.frames if frame.frame_id in plan["targets"]
    }
    if sorted(targets) != plan["targets"]:
        raise ValueError("An exact frozen target annotation is missing")

    def iter_native_targets(video, frame_ids):
        if video != native.video_id:
            raise ValueError("Evaluation video identity mismatch")
        for fid in frame_ids:
            yield SimpleNamespace(inference=SimpleNamespace(target_frame_id=fid),
                frame_supervision=targets[fid], evaluation=None)

    adapter = SimpleNamespace(iter_video=iter_native_targets)
    summary = copy.deepcopy(original_summary)
    for group in GROUPS:
        rows = [row for row in predictions if row["group"] == group]
        evaluation = score_native_rows(adapter, native.video_id, rows)
        summary["groups"][group]["evaluation"] = evaluation
        by_frame = {row["frame_id"]: row for row in evaluation["frames"]}
        for row in rows:
            row["evaluation"] = by_frame[row["frame_id"]]
    summary["evaluation_correction"] = {
        "reason": "Runtime Testing adapter returns inference_only and no GT; original zero-valid-GT summary is not an accuracy result",
        "native_parser": "surgical_agent.data.parser.parse_annotation_file",
        "frame_aggregation": "aggregate_frame_target; all-instances per-task validity; no interpolation",
        "annotation_sha256": annotation_sha,
        "original_artifact_sha256": hashes,
        "evaluator_script_sha256": sha256_file(Path(__file__)),
        "purpose": "diagnostic_not_sealed_test_result",
        "extra_provider_calls": 0,
    }
    args.output.mkdir(parents=True)
    atomic_write_json(args.output / "summary.json", summary)
    atomic_write_json(args.output / "predictions_with_evaluation.json", predictions)
    for name, digest in hashes.items():
        if sha256_file(args.run_dir / name) != digest:
            raise ValueError("Original artifact changed during offline evaluation")
    print(json.dumps({
        "output": str(args.output), "original_artifacts_unchanged": True,
        "groups": {group: summary["groups"][group]["evaluation"]["tasks"] for group in GROUPS},
    }), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
