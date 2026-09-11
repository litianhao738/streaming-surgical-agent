"""Offline final-only Gate preparation: verify OOF, freeze features, join GT.

No transport, credential loading, model training or paid API call is performed.
The latest paid three-round panel is a development seed, not an independent test.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgical_agent.artifacts.manifest import (
    atomic_write_json,
    sha256_file,
    sha256_mapping,
)
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.gate.final_only_training import (
    BASE_FEATURES,
    FEATURE_ORDER,
    FEATURE_VERSION,
    LABEL_VERSION,
    TARGETS,
    TASKS,
    TRACKER_FEATURES,
    canonical_labels,
    extract_features,
    label_outcome,
    mask_tracker,
    readiness,
)
from surgical_agent.research.gate.supervision import gate_training_target
from surgical_agent.tracking.oof_index import load_tracker_oof_index

DEFAULT_SOURCE = ROOT / "artifacts/preflight/recent_mean_panel_20260908_v2"
DEFAULT_OOF = ROOT / "artifacts/training/tracker_clip_v2_oof5_20260906/oof/index.json"
ATTRS = dict(zip(TASKS, ("instrument_ids", "verb_ids", "target_ids", "triplet_ids", "phase_id"), strict=True))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def masks_only(resolved):
    evaluation = resolved.evaluation
    if evaluation is not None and evaluation.instance_supervision_available:
        return {task: bool(evaluation.instances) and all(getattr(i.mask, task) for i in evaluation.instances)
                for task in TASKS}
    target = resolved.frame_supervision
    return {task: target is not None and getattr(target.mask, task) for task in TASKS}


def verified_oof(index_path, dataset_root):
    index = load_tracker_oof_index(index_path)
    providers, audit, sources = {}, {}, {str(index.path): sha256_file(index.path)}
    repair_hash = sha256_file(dataset_root / "repair_manifest.json")
    unique = {}
    for video, artifact in index.video_to_artifact.items():
        if artifact not in unique:
            manifest_path = artifact.with_name("training_manifest.json")
            if video == "VID31":
                manifest_path = index.path.parent.parent / "full/training_manifest.json"
            manifest = read(manifest_path)
            provider = index.provider_for(video)
            checkpoint = manifest_path.with_name("checkpoint.pt")
            if (manifest["bbox_policy"] != "clip_to_frame_v2"
                    or manifest["dataset_repair_manifest_sha256"] != repair_hash
                    or provider.dataset_repair_manifest_sha256 != repair_hash
                    or manifest["checkpoint_sha256"] != provider.checkpoint_sha256
                    or sha256_file(checkpoint) != provider.checkpoint_sha256):
                raise ValueError("corrected OOF checkpoint or dataset provenance mismatch")
            unique[artifact] = (manifest, provider)
            sources.update({str(manifest_path): sha256_file(manifest_path),
                            str(artifact): sha256_file(artifact), str(checkpoint): provider.checkpoint_sha256})
        manifest, provider = unique[artifact]
        if video in manifest["training_video_ids"]:
            raise ValueError(f"Tracker for {video} was trained on that video")
        if provider.video(video).source_split is not DatasetSplit.TRAINING:
            raise ValueError("OOF target video must be Training")
        providers[video] = provider
        audit[video] = {"artifact": str(artifact), "artifact_sha256": provider.artifact_sha256,
                        "checkpoint_sha256": provider.checkpoint_sha256,
                        "training_video_ids": manifest["training_video_ids"],
                        "query_video_excluded": True, "bbox_policy": manifest["bbox_policy"]}
    return providers, audit, sources


def historical_targets(root):
    """Use identities only, including previously planned targets; no label values."""
    excluded, hashes = {}, {}
    for path in sorted(root.glob("*/plan.json")):
        try:
            data = read(path)
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        selections = data.get("selection", [])
        identities = []
        if isinstance(selections, list):
            for row in selections:
                if isinstance(row, dict):
                    identities.append((row.get("video_id"), row.get("frame_id", row.get("target_frame_id"))))
        for video, frames in data.get("selected_frames_by_video", {}).items():
            identities.extend((video, frame) for frame in frames)
        usable = [(v, f) for v, f in identities if isinstance(v, str) and type(f) is int]
        if usable:
            hashes[str(path)] = sha256_file(path)
        for video, frame in usable:
            excluded.setdefault(video, set()).add(frame)
    return excluded, hashes


def select_pilot(adapter, videos, *, per_video, excluded):
    selected, inventory = [], {}
    for video in videos:
        if adapter.entries[video].split is not DatasetSplit.TRAINING:
            raise ValueError("selection is restricted to Training")
        masks = {r.inference.target_frame_id: masks_only(r) for r in adapter.iter_video(video)}
        samples = list(adapter.iter_inference_video(video))
        eligible = [s for s in samples if all(masks.get(s.target_frame_id, {}).get(t, False) for t in TASKS)
                    and all(abs(s.target_frame_id - old) > 75 for old in excluded.get(video, ()))]
        inventory[video] = {"annotated_targets": len(masks),
                            "valid_by_task": {t: sum(m[t] for m in masks.values()) for t in TASKS},
                            "all_heads_valid": sum(all(m.values()) for m in masks.values()),
                            "eligible_after_history_exclusion": len(eligible)}
        if not eligible:
            continue
        used = set()
        for i in range(1, per_video + 1):
            anchor = samples[len(samples) * i // (per_video + 1)].target_frame_id
            choices = [s for s in eligible if all(abs(s.target_frame_id - frame) > 75 for frame in used)]
            if not choices:
                raise ValueError("insufficient distinct temporally spaced pilot targets")
            sample = min(choices, key=lambda s: (abs(s.target_frame_id - anchor), s.target_frame_id))
            used.add(sample.target_frame_id)
            selected.append({"video_id": video, "frame_id": sample.target_frame_id,
                             "source_split": "Training", "anchor_frame_id": anchor,
                             "causal_frame_ids": list(sample.causal_frame_ids),
                             "gt_availability_only": masks[sample.target_frame_id],
                             "images": [{"path": str(p), "sha256": sha256_file(p)} for p in sample.media_refs]})
    return selected, inventory


def prepare(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError("a new output directory is required; previous preparations are immutable")
    source = args.source.resolve()
    plan, saved, audit = (read(source / p) for p in ("plan.json", "round_3_predictions.json", "independent_audit.json"))
    if (plan["profile"] not in {"recent_five_family_mean_20260908_v1", "recent_five_family_mean_20260908_v2_empty_pool_guard"}
            or plan["h0"] != "google/gemini-3.8-flash"
            or plan["threshold_primary"] != 4 or plan["round_cap"] != 3
            or audit["verified"] is not True):
        raise ValueError("this importer requires the audited, frozen latest paid policy")
    source_paths = [source / p for p in ("plan.json", "round_1_predictions.json", "round_2_predictions.json",
                                       "round_3_predictions.json", "independent_audit.json")]
    sources = {str(p): sha256_file(p) for p in source_paths}
    if read(source / "scores/round_3.json")["prediction_sha256"] != sha256_file(source / "round_3_predictions.json"):
        raise ValueError("scored predictions no longer match the frozen output")
    selected = {f"{s['video_id']}:{s['frame_id']}": s for s in plan["selection"]}
    if len(selected) != len(saved) or len(selected) != len(plan["selection"]):
        raise ValueError("source plan or predictions contain missing/duplicate identities")
    saved_by_id = {f"{r['video_id']}:{r['frame_id']}": r for r in saved}
    if len(saved_by_id) != len(saved) or set(saved_by_id) != set(selected):
        raise ValueError("source predictions do not exactly cover frozen selection")
    if any(row["h0"] is None for row in saved):
        raise ValueError("source collection has unfinished/failed H0; keep it as a partial diagnostic")
    for number in (1, 2):
        earlier = read(source / f"round_{number}_predictions.json")
        if len(earlier) != len(saved):
            raise ValueError("round snapshot target count changed")
        for row in earlier:
            identity = f"{row['video_id']}:{row['frame_id']}"
            if row["h0"] != saved_by_id[identity]["h0"]:
                raise ValueError("H0 changed between repair rounds")
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    providers, oof_audit, oof_sources = verified_oof(args.tracker_oof_index, args.dataset_root)
    sources.update(oof_sources)
    policy = {k: plan[k] for k in ("profile", "h0", "models", "threshold_primary", "round_cap")}
    # Sampling files and budget ledgers are provenance, not changes to the policy.
    policy["source_sha256"] = {p: digest for p, digest in plan["source_sha256"].items()
                               if p.replace("\\", "/").startswith(("src/", "scripts/", "configs/"))}
    policy_id = sha256_mapping(policy)
    rows = []
    for row in saved:
        video, frame = row["video_id"], row["frame_id"]
        identity = f"{video}:{frame}"
        selection = selected[identity]
        if adapter.entries[video].split is not DatasetSplit.TRAINING:
            raise ValueError("source contains a non-Training target")
        sample = next(s for s in adapter.iter_inference_video(video) if s.target_frame_id == frame)
        if list(sample.causal_frame_ids) != selection["causal_frame_ids"]:
            raise ValueError("source H0 causal inputs changed")
        for recorded, path in zip(selection["images"], sample.media_refs, strict=True):
            if sha256_file(path) != recorded["sha256"]:
                raise ValueError("source image hash mismatch")
        provider = providers[video]
        provider.reset(video)
        features = extract_features(row["h0"], target_frame_id=frame, causal_frame_ids=sample.causal_frame_ids,
                                    tracker_snapshot=provider.snapshot(sample))
        rows.append({"sample_id": identity, "video_id": video, "frame_id": frame, "source_split": "Training",
                     "policy_id": policy_id, "feature_version": FEATURE_VERSION,
                     "features_no_tracker": mask_tracker(features), "features_with_tracker": features,
                     "oof_artifact_sha256": provider.artifact_sha256})
    # Persist feature-only inputs before loading any GT label values.
    output.mkdir(parents=True)
    atomic_write_json(output / "features_before_gt.json", rows)
    feature_hash = sha256_file(output / "features_before_gt.json")
    by_id = {f"{r['video_id']}:{r['frame_id']}": r for r in saved}
    for row in rows:
        resolved = next(adapter.iter_video(row["video_id"], frame_ids=[row["frame_id"]]))
        target = gate_training_target(resolved)
        mask = {t: getattr(target.mask, t) for t in TASKS}
        truth = {t: ([getattr(target, ATTRS[t])] if t == "phase" else list(getattr(target, ATTRS[t])))
                 if mask[t] else None for t in TASKS}
        original = by_id[row["sample_id"]]
        state = audit["audited_statuses"][f"{row['video_id']}_{row['frame_id']}"]
        row["labels"] = label_outcome(original["h0"], original["primary"], gt=truth, mask=mask,
                                      repair_observed=state in {"UNRESOLVED", "MODEL_PASS"})
        row["label_version"] = LABEL_VERSION
        row["source_audited_status"] = state
        row["h0_labels"] = canonical_labels(original["h0"])
        row["final_labels"] = canonical_labels(original["primary"])
        row["gt"] = truth
        row["task_mask"] = mask
    if sha256_file(output / "features_before_gt.json") != feature_hash:
        raise ValueError("feature artifact changed after GT join")
    atomic_write_json(output / "training_examples.json", rows)
    checks = {target: readiness(rows, target=target) for target in TARGETS}
    exclusions, exclusion_hashes = historical_targets(ROOT / "artifacts/preflight")
    pilot, inventory = select_pilot(adapter, sorted(providers), per_video=args.pilot_per_video, excluded=exclusions)
    for row in pilot:
        provider = providers[row["video_id"]]
        available = {f.frame_id for f in provider.video(row["video_id"]).frames}
        if set(row["causal_frame_ids"]) - available:
            raise ValueError("planned target missing exact OOF causal frame coverage")
    atomic_write_json(output / "pilot_selection.json", {
        "schema_version": "final_only_gate_collection_selection_v1", "status": "SELECTION_ONLY_NO_API",
        "selection_rule": "time quantiles; nearest full-mask target; earlier tie; >75 original frames from historical/planned targets",
        "per_video": args.pilot_per_video, "selection": pilot, "training_inventory": inventory,
        "historical_identity_source_sha256": exclusion_hashes,
        "maximum_calls_if_using_three_round_five_seat_episode": len(pilot) * 19,
        "no_gt_label_values_used_for_selection": True,
        "pricing_and_provider_preflight_required_before_dispatch": True,
    })
    manifest = {
        "schema_version": "final_only_gate_preparation_v1",
        "status": "PILOT_FIT_READY" if checks["benefit"]["can_fit_pilot"] else "PREPARED_DATA_INSUFFICIENT",
        "feature_version": FEATURE_VERSION, "feature_order": list(FEATURE_ORDER),
        "base_features": list(BASE_FEATURES), "tracker_features": list(TRACKER_FEATURES),
        "label_version": LABEL_VERSION, "source_policy": policy, "policy_id": policy_id,
        "source_sha256": sources, "tracker_oof": oof_audit,
        "features_before_gt_sha256": feature_hash,
        "training_examples_sha256": sha256_file(output / "training_examples.json"),
        "pilot_selection_sha256": sha256_file(output / "pilot_selection.json"),
        "readiness": checks, "source_examples": len(rows),
        "source_videos": dict(Counter(row["video_id"] for row in rows)),
        "source_is_previously_analyzed_development_data": True,
        "source_repair_policy_used_tracker": False,
        "paired_features_do_not_create_extra_independent_observations": True,
        "upstream_oof_caveat": "Per-video Tracker exclusion is verified. Gate-only held-out diagnostics do not establish nested end-to-end OOF independence: other Gate-fit rows can use detectors trained on the Gate-held-out video.",
        "feature_snapshot_created_before_gt_join": True, "new_api_calls": 0,
        "testing_used": False, "validation_used": False, "deployable": False,
        "implementation_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                                  [Path(__file__), ROOT / "src/surgical_agent/research/gate/final_only_training.py"]},
    }
    for path, digest in sources.items():
        if sha256_file(path) != digest:
            raise ValueError("source changed during offline preparation")
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "source_examples": len(rows),
                      "pilot_targets_prepared": len(pilot), "readiness": checks}), flush=True)
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    result.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    result.add_argument("--tracker-oof-index", type=Path, default=DEFAULT_OOF)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--pilot-per-video", type=int, default=10)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.pilot_per_video < 1:
        raise SystemExit("pilot-per-video must be positive")
    prepare(args)
