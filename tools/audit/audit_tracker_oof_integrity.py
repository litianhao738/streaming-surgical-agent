"""Read existing tracker OOF provenance without training or inference."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.tracking.config import load_tracker_training_config
from surgical_agent.tracking.predicted_provider import PrecomputedPredictedTrackProvider
from surgical_agent.tracking.training_data import build_detection_training_records, deterministic_video_folds, supervision_qualified_training_video_ids


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    base = ROOT / "artifacts/training/tracker_oof5"
    output = ROOT / "artifacts/preflight/tracker_oof_integrity_20260905.json"
    if output.exists():
        raise FileExistsError(output)
    adapter = CholecTrack20DatasetAdapter("D:/cholec_dataset")
    qualified = supervision_qualified_training_video_ids(adapter)
    train_ids = sorted(v for v,e in adapter.entries.items() if e.split is DatasetSplit.TRAINING)
    nontraining = sorted(v for v,e in adapter.entries.items() if e.split is not DatasetSplit.TRAINING)
    index = read(base / "oof/index.json")
    summary = read(base / "oof_run_summary.json")
    prior = read(base / "oof/heldout_evaluation.json")
    expected_folds = deterministic_video_folds(qualified, 5)
    confpaths = {"oof": ROOT / "configs/tracker/fasterrcnn_mobilenet_v3_5090_oof5.yaml", "full": ROOT / "configs/tracker/fasterrcnn_mobilenet_v3_5090.yaml"}
    configs = {k: load_tracker_training_config(p) for k,p in confpaths.items()}
    configshas = {k: sha(p) for k,p in confpaths.items()}
    repairsha = sha(adapter.dataset_root / "repair_manifest.json")
    print("Reconstructing currently eligible detector-training records (no images decoded)", flush=True)
    records = build_detection_training_records(adapter, qualified, progress_enabled=False)
    counts = Counter(r.video_id for r in records)
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    report = {"schema_version": "tracker_oof_integrity_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha(__file__), "base": str(base), "no_training_or_inference": True,
              "qualified_training_videos": list(qualified), "nontraining_videos": nontraining,
              "current_detection_record_counts": dict(counts), "dataset_repair_manifest_sha256": repairsha,
              "config_sources": {k: {"path": str(p), "sha256": configshas[k], "config": asdict(configs[k])} for k,p in confpaths.items()},
              "index_sha256": sha(base / "oof/index.json"), "summary_sha256": sha(base / "oof_run_summary.json"),
              "checks": {}, "models": {}, "prediction_artifacts": {}, "video_routes": {}, "limitations": []}
    checks = report["checks"]
    checks["index_fold_count_is_five"] = index["fold_count"] == 5
    checks["summary_has_five_folds"] = len(summary["folds"]) == 5
    checks["index_and_summary_cover_exact_training_videos"] = sorted(index["video_to_artifact"]) == summary["oof_video_ids"] == train_ids
    heldout_all = [v for f in summary["folds"] for v in f["held_out_video_ids"]]
    checks["heldout_sets_partition_nine_qualified_videos_once"] = sorted(heldout_all) == list(qualified) and len(set(heldout_all)) == len(heldout_all)
    for name in ["full"] + [f"oof/fold_{i}" for i in range(5)]:
        print(f"Hash and metadata audit {name}", flush=True)
        directory = base / name
        manifestpath = directory / "training_manifest.json"
        manifest = read(manifestpath)
        kind = "full" if name == "full" else "oof"
        expected_held = () if kind == "full" else expected_folds[int(name[-1])]
        expected_train = tuple(v for v in qualified if v not in expected_held)
        checkpointpath = directory / "checkpoint.pt"
        checkpointsha = sha(checkpointpath)
        payload = torch.load(checkpointpath, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        local_checks = {
            "checkpoint_file_sha_matches_manifest": checkpointsha == manifest["checkpoint_sha256"],
            "checkpoint_metadata_matches_all_manifest_shared_fields": all(manifest.get(k) == v for k,v in metadata.items()),
            "train_ids_exact_complement_of_heldout": tuple(manifest["training_video_ids"]) == expected_train,
            "heldout_ids_exact_expected_fold": tuple(manifest["excluded_video_ids"]) == expected_held,
            "train_heldout_disjoint": not set(manifest["training_video_ids"]) & set(manifest["excluded_video_ids"]),
            "no_validation_testing_in_training": not set(manifest["training_video_ids"]) & set(nontraining),
            "vid31_not_used_in_detector_training": "VID31" not in manifest["training_video_ids"],
            "config_file_sha_matches_manifest_and_checkpoint": manifest["tracker_config_sha256"] == metadata["tracker_config_sha256"] == configshas[kind],
            "embedded_config_matches_actual_yaml": manifest["config"] == asdict(configs[kind]),
            "repair_sha_matches_current_and_checkpoint": manifest["dataset_repair_manifest_sha256"] == metadata["dataset_repair_manifest_sha256"] == repairsha,
            "training_samples_match_current_eligible_records": manifest["training_samples"] == sum(counts[v] for v in expected_train),
            "optimizer_steps_match_epochs_batches": manifest["optimizer_steps"] == math.ceil(manifest["training_samples"] / manifest["config"]["batch_size"]) * manifest["epochs_completed"],
        }
        if kind == "oof":
            foldsummary = summary["folds"][int(name[-1])]
            local_checks["run_summary_matches_manifest_partition"] = foldsummary["training_video_ids"] == manifest["training_video_ids"] and foldsummary["held_out_video_ids"] == manifest["excluded_video_ids"]
            local_checks["prior_heldout_audit_manifest_hash_matches"] = prior["leakage_audit"]["folds"][name.split("/")[-1]]["training_manifest_sha256"] == sha(manifestpath)
        modelstate = payload["model_state"]
        report["models"][name] = {"manifest_sha256": sha(manifestpath), "checkpoint_sha256": checkpointsha, "checkpoint_size_bytes": checkpointpath.stat().st_size,
                                  "checkpoint_schema": payload["schema_version"], "metadata": metadata, "created_at_utc": manifest["created_at_utc"],
                                  "training_samples": manifest["training_samples"], "checks": local_checks,
                                  "state_tensor_count": len(modelstate), "classifier_weight_shape": list(modelstate["roi_heads.box_predictor.cls_score.weight"].shape)}
        del payload, modelstate
    checks["six_distinct_checkpoint_files"] = len({v["checkpoint_sha256"] for v in report["models"].values()}) == 6
    for relative, indexedsha in index["artifacts"].items():
        artifactpath = base / "oof" / relative
        raw = read(artifactpath)
        provider = PrecomputedPredictedTrackProvider.from_json(artifactpath)
        mapped = sorted(v for v,p in index["video_to_artifact"].items() if p == relative)
        modelname = "full" if relative.startswith("vid31/") else "oof/" + relative.split("/")[0]
        model = report["models"][modelname]
        artifact_checks = {
            "artifact_sha_matches_index": sha(artifactpath) == indexedsha,
            "artifact_checkpoint_sha_matches_actual_weights": raw["checkpoint_sha256"] == model["checkpoint_sha256"],
            "artifact_config_sha_matches_oof_inference_yaml": raw["inference_config_sha256"] == configshas["oof"],
            "artifact_repair_sha_matches_current": raw["dataset_repair_manifest_sha256"] == repairsha,
            "artifact_videos_match_index_mapping": sorted(raw["videos"]) == mapped,
            "artifact_schema_provider_validation_passed": True,
            "declared_causal_online_forward": raw["causal"] is True and raw["inference_mode"] == "online_forward_only",
        }
        report["prediction_artifacts"][relative] = {"sha256": sha(artifactpath), "checkpoint_source": modelname, "checks": artifact_checks}
        for video_id in mapped:
            video = provider.video(video_id)
            expected_ids = [s.target_frame_id for s in adapter.iter_inference_video(video_id)]
            actual_ids = [f.frame_id for f in video.frames]
            report["video_routes"][video_id] = {"artifact": relative, "model": modelname, "training_video_ids": model["metadata"]["training_video_ids"],
                "frame_count": len(actual_ids), "first_frame_id": actual_ids[0], "last_frame_id": actual_ids[-1],
                "track_detections": sum(len(f.tracks) for f in video.frames),
                "checks": {"self_video_excluded_from_optimizer_manifest_and_checkpoint": video_id not in model["metadata"]["training_video_ids"],
                           "exact_all_png_runtime_frame_coverage": actual_ids == expected_ids,
                           "source_split_is_training": video.source_split is DatasetSplit.TRAINING}}
    checks["prior_heldout_index_hash_matches_current"] = prior["provenance"]["oof_index_sha256"] == report["index_sha256"]
    diff = {k: {"full": asdict(configs["full"])[k], "oof": asdict(configs["oof"])[k]} for k in asdict(configs["full"]) if asdict(configs["full"])[k] != asdict(configs["oof"])[k]}
    report["full_vs_oof_config_difference"] = diff
    checks["full_vs_oof_difference_only_fold_count"] = set(diff) == {"oof_folds"}
    all_checks = list(checks.values()) + [ok for collection in (report["models"], report["prediction_artifacts"], report["video_routes"]) for row in collection.values() for ok in row["checks"].values()]
    report["all_recorded_checks_passed"] = all(all_checks)
    report["failed_checks"] = [k for k,v in checks.items() if not v] + [f"{name}.{k}" for collection in (report["models"], report["prediction_artifacts"], report["video_routes"]) for name,row in collection.items() for k,v in row["checks"].items() if not v]
    report["notes"] = [
        "Full checkpoint was trained earlier with oof_folds=3; this parameter controls partitioning in OOF mode, not full detector optimization or inference. Five actual OOF checkpoints use oof_folds=5. All other stored config fields are identical.",
        "VID31 has no instance box supervision. Its out-of-training predictions use the full nine-qualified-video checkpoint and are excluded from held-out detection metrics.",
        "Run summary and training_state_path retain the historical AutoDL absolute paths. Portable index uses relative paths; historical absolute paths are provenance, not current local file locators.",
    ]
    report["limitations"] = [
        "Hashes bind existing checkpoint bytes, manifests, predictions and index; they do not independently reconstruct the historical GPU optimizer execution or prove every batch content.",
        "No rerun of detector/association and no independent reproduction of predictions or heldout metrics in this audit.",
        "Causal=true and online_forward_only are validated artifact declarations and current code contract; hash provenance alone cannot prove historical no-future use.",
        "No historical source commit/code snapshot is bound by these training manifests; current code is supporting evidence rather than proof of the exact AutoDL implementation used.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha(output), "all_checks_passed": report["all_recorded_checks_passed"], "failed_checks": report["failed_checks"], "model_count": len(report["models"]), "artifact_count": len(report["prediction_artifacts"]), "predicted_frame_count": sum(v["frame_count"] for v in report["video_routes"].values()), "full_vs_oof_config_difference": diff}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
