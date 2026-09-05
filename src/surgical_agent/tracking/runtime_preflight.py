"""Validate Tracker coverage and held-out model routing before any API calls."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from surgical_agent.artifacts.manifest import sha256_file
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.tracking.box_policy import BBoxPolicy
from surgical_agent.tracking.contracts import PredictedTrackVideo
from surgical_agent.tracking.oof_index import load_tracker_oof_index
from surgical_agent.tracking.predicted_provider import (
    PrecomputedPredictedTrackProvider,
    PredictedTrackArtifactError,
)
from surgical_agent.tracking.training_data import (
    deterministic_video_folds,
    supervision_qualified_training_video_ids,
)


def validate_gap_resets(video: PredictedTrackVideo, *, max_frame_id_gap: int) -> int:
    """Reject old IDs revived across a gap, including after an empty first frame."""
    if type(max_frame_id_gap) is not int or max_frame_id_gap <= 0:
        raise ValueError("max_frame_id_gap must be a positive integer")
    seen: set[str] = set()
    forbidden: set[str] = set()
    previous: int | None = None
    gaps = 0
    for frame in video.frames:
        after_gap = previous is not None and frame.frame_id - previous > max_frame_id_gap
        if after_gap:
            forbidden.update(seen)
            gaps += 1
        for track in frame.tracks:
            if track.track_id in forbidden or (after_gap and track.age != 1):
                raise PredictedTrackArtifactError(
                    f"{video.video_id} Tracker association violates gap reset at {frame.frame_id}"
                )
            seen.add(track.track_id)
        previous = frame.frame_id
    return gaps


def _training_source(
    directory: Path,
    provider: PrecomputedPredictedTrackProvider,
    *,
    expected_training: set[str],
    expected_excluded: set[str],
    expected_mode: str,
    repair_sha256: str,
    check_config: bool,
) -> dict[str, object]:
    manifest_path = directory / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("Tracker training manifest must be an object")
    training = manifest.get("training_video_ids")
    excluded = manifest.get("excluded_video_ids")
    if (
        manifest.get("schema_version") != "predicted_tracker_training_manifest_v1"
        or manifest.get("mode") != expected_mode
        or not isinstance(training, list)
        or not isinstance(excluded, list)
        or any(not isinstance(value, str) for value in training + excluded)
        or len(training) != len(set(training))
        or len(excluded) != len(set(excluded))
        or set(training) != expected_training
        or set(excluded) != expected_excluded
    ):
        raise ValueError("Tracker model training partition does not match the selected route")
    if manifest.get("dataset_repair_manifest_sha256") != repair_sha256:
        raise ValueError("Tracker training manifest uses a different dataset repair source")
    if manifest.get("checkpoint_sha256") != provider.checkpoint_sha256:
        raise ValueError("Tracker artifact does not match its model training manifest")
    if sha256_file(directory / "checkpoint.pt") != provider.checkpoint_sha256:
        raise ValueError("Tracker checkpoint digest does not match its prediction artifact")
    if check_config and manifest.get("tracker_config_sha256") != provider.inference_config_sha256:
        raise ValueError("OOF Tracker model and prediction configs differ")
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": provider.checkpoint_sha256,
        "training_video_ids": sorted(expected_training),
        "excluded_video_ids": sorted(expected_excluded),
        "bbox_policy": BBoxPolicy(manifest.get("bbox_policy", "legacy_strict_v1")).value,
    }


class ValidatedTrackRouter:
    """Route each selected video to a provider whose source and coverage passed."""

    provider_name = "validated_predicted_track_router"

    def __init__(
        self,
        providers: Mapping[str, PrecomputedPredictedTrackProvider],
        *,
        source_sha256: str,
        audit: Mapping[str, object],
    ) -> None:
        self._providers = dict(providers)
        self._current: PrecomputedPredictedTrackProvider | None = None
        self.artifact_sha256 = source_sha256
        self.audit = dict(audit)

    def reset(self, video_id: str) -> None:
        try:
            self._current = self._providers[video_id]
        except KeyError as exc:
            raise PredictedTrackArtifactError("Tracker route was not validated for this video") from exc
        self._current.reset(video_id)

    def snapshot(self, sample: InferenceSample) -> Mapping[str, object]:
        if self._current is None:
            raise PredictedTrackArtifactError("Tracker router must reset before snapshot")
        return self._current.snapshot(sample)


def build_validated_track_router(
    *,
    adapter: Any,
    samples: Sequence[InferenceSample],
    artifact_path: str | Path | None = None,
    oof_index_path: str | Path | None = None,
) -> ValidatedTrackRouter:
    """Read prediction/model provenance only; never inspect target label values."""
    if (artifact_path is None) == (oof_index_path is None):
        raise ValueError("select exactly one Tracker artifact or OOF index")
    if not samples:
        raise ValueError("Tracker preflight requires selected observations")
    selected: dict[str, list[InferenceSample]] = defaultdict(list)
    for sample in samples:
        entry = adapter.entries.get(sample.video_id)
        if entry is None or sample.source_split is not entry.split:
            raise ValueError("Tracker observation split differs from the canonical dataset entry")
        selected[sample.video_id].append(sample)
    repair_sha = sha256_file(Path(adapter.dataset_root) / "repair_manifest.json")
    qualified = set(supervision_qualified_training_video_ids(adapter))
    step = adapter.expected_frame_id_step
    providers: dict[str, PrecomputedPredictedTrackProvider] = {}
    routes: dict[str, object] = {}

    if oof_index_path is not None:
        if any(sample.source_split is not DatasetSplit.TRAINING for sample in samples):
            raise ValueError("Tracker OOF index is only for Training observations")
        source = Path(oof_index_path).expanduser().resolve()
        index = load_tracker_oof_index(source)
        raw_index = json.loads(source.read_text(encoding="utf-8"))
        all_training = {
            video_id for video_id, entry in adapter.entries.items()
            if entry.split is DatasetSplit.TRAINING
        }
        if set(index.video_to_artifact) != all_training:
            raise ValueError("Tracker OOF index must exactly cover the Training videos")
        folds = deterministic_video_folds(tuple(sorted(qualified)), raw_index["fold_count"])
        by_path: dict[Path, set[str]] = defaultdict(set)
        for video_id, path in index.video_to_artifact.items():
            by_path[path].add(video_id)
        # Audit all mapped model routes, even when this rollout selects one video.
        inference_configs: set[str] = set()
        bbox_policies: set[str] = set()
        for path, held_out in by_path.items():
            provider = index.provider_for(min(held_out))
            inference_configs.add(provider.inference_config_sha256)
            if provider.dataset_repair_manifest_sha256 != repair_sha:
                raise ValueError("Tracker artifact uses a different dataset repair source")
            if set(provider.available_video_ids) != held_out:
                raise ValueError("OOF artifact video scope differs from its index mapping")
            if held_out == {"VID31"} and "VID31" not in qualified:
                model_directory = source.parent.parent / "full"
                training, excluded, mode = qualified, set(), "full"
                check_config = False
            else:
                expected = next((set(fold) for fold in folds if set(fold) == held_out), None)
                if expected is None:
                    raise ValueError("OOF artifact does not match a complete held-out fold")
                model_directory = path.parent
                training, excluded, mode = qualified - held_out, held_out, "oof"
                check_config = True
            evidence = _training_source(
                model_directory, provider, expected_training=training,
                expected_excluded=excluded, expected_mode=mode,
                repair_sha256=repair_sha, check_config=check_config,
            )
            bbox_policies.add(str(evidence["bbox_policy"]))
            for video_id in held_out:
                if video_id in selected:
                    providers[video_id] = provider
                    routes[video_id] = {
                        "artifact": str(path), "artifact_sha256": provider.artifact_sha256,
                        "model_source": evidence,
                    }
        if len(inference_configs) != 1 or len(bbox_policies) != 1:
            raise ValueError("OOF routes mix different inference or training box policies")
    else:
        if any(sample.source_split is DatasetSplit.TRAINING for sample in samples):
            raise ValueError("Training Tracker rollouts require --tracker-oof-index")
        source = Path(artifact_path).expanduser().resolve()  # type: ignore[arg-type]
        provider = PrecomputedPredictedTrackProvider.from_json(source)
        # Native training stores predictions at the run root and weights in
        # full/. Reassociated bundles keep predictions beside those weights.
        model_directory = source.parent
        if not (model_directory / "training_manifest.json").is_file():
            model_directory = model_directory / "full"
        evidence = _training_source(
            model_directory, provider, expected_training=qualified,
            expected_excluded=set(), expected_mode="full",
            repair_sha256=repair_sha, check_config=False,
        )
        for video_id in selected:
            providers[video_id] = provider
            routes[video_id] = {
                "artifact": str(source), "artifact_sha256": provider.artifact_sha256,
                "model_source": evidence,
            }

    coverage: dict[str, object] = {}
    for video_id, observations in selected.items():
        provider = providers[video_id]
        if provider.dataset_repair_manifest_sha256 != repair_sha:
            raise ValueError("Tracker artifact uses a different dataset repair source")
        video = provider.video(video_id)
        gaps = validate_gap_resets(video, max_frame_id_gap=step)
        by_frame = {frame.frame_id for frame in video.frames}
        for sample in observations:
            if video.source_split is not sample.source_split:
                raise ValueError("Tracker artifact and selected observation splits differ")
            if sample.target_frame_id not in by_frame or set(sample.causal_frame_ids) - by_frame:
                raise ValueError(f"Tracker coverage is incomplete for {video_id}/{sample.target_frame_id}")
        coverage[video_id] = {
            "selected_observations": len(observations), "coverage": 1.0,
            "artifact_frames": len(video.frames), "gap_resets_checked": gaps,
        }
    audit = {
        "schema_version": "tracker_runtime_preflight_v1", "status": "PASS",
        "source_kind": "oof_index" if oof_index_path is not None else "full_artifact",
        "source_sha256": sha256_file(source), "max_frame_id_gap": step,
        "dataset_repair_manifest_sha256": repair_sha,
        "label_values_accessed": False, "label_access_scope": "TRACKER_PREFLIGHT_ONLY",
        "checkpoint_provenance_check": "file_sha256_and_training_manifest",
        "routes": routes, "coverage": coverage,
    }
    return ValidatedTrackRouter(providers, source_sha256=sha256_file(source), audit=audit)


__all__ = ["ValidatedTrackRouter", "build_validated_track_router", "validate_gap_resets"]
