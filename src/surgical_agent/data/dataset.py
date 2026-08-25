"""P2 dataset adapter with explicit derived-source and granularity routing."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from surgical_agent.data.derived_supervision import (
    DerivedSupervisionManifest,
    DerivedVideoSource,
    load_derived_supervision_manifest,
    load_frame_level_ivt_supervision,
    load_image_phase_supervision,
)
from surgical_agent.data.media_backend import (
    ExactFrameFolderResolver,
    Mp4FrameResolver,
)
from surgical_agent.data.parser import parse_annotation_file
from surgical_agent.data.schemas import (
    DatasetSplit,
    EvaluationInstanceTarget,
    EvaluationTarget,
    FrameSupervisionTarget,
    FrameTaskMask,
    InferenceSample,
    LabelMask,
)
from surgical_agent.data.splits import (
    SplitManifestEntry,
    discover_official_split_manifest,
)
from surgical_agent.data.targets import build_evaluation_target, build_inference_sample

PNG_ALIGNMENT_VERSION = "ct20_exact_png_stem_v1"
MP4_ALIGNMENT_VERSION = "ct20_test_mp4_annotation_id_minus_1_v1"


class DatasetContractError(RuntimeError):
    """Raised when a sample cannot be built without guessing data semantics."""


def _causal_from_sorted(
    available_frame_ids: tuple[int, ...],
    *,
    target_frame_id: int,
    max_frames: int,
) -> tuple[int, ...]:
    """Build a window in logarithmic time from an already validated index."""

    position = bisect_right(available_frame_ids, target_frame_id)
    if position == 0 or available_frame_ids[position - 1] != target_frame_id:
        raise DatasetContractError(f"No exact media for frame {target_frame_id}")
    return available_frame_ids[max(0, position - max_frames) : position]


@dataclass(frozen=True)
class SampleProvenance:
    """Auditable source route kept separate from both model input and GT."""

    video_id: str
    split: DatasetSplit
    route: str
    media_source: str
    annotation_source: str | None
    phase_source: str | None
    frame_action_source: str | None
    manifest_source: str | None
    supervision_granularity: str


@dataclass(frozen=True)
class ResolvedSample:
    """Dataset output before the runtime/evaluation branches are separated."""

    inference: InferenceSample
    evaluation: EvaluationTarget | None
    frame_supervision: FrameSupervisionTarget | None
    provenance: SampleProvenance


@dataclass(frozen=True)
class TensorResolvedSample:
    """One current-frame tensor paired with non-model bookkeeping objects."""

    image: Any
    resolved: ResolvedSample


@dataclass(frozen=True)
class SmokeBatch:
    """Collated P2 batch; targets never enter the model call."""

    images: Any
    inference_samples: tuple[InferenceSample, ...]
    frame_targets: tuple[FrameSupervisionTarget | None, ...]
    evaluation_targets: tuple[EvaluationTarget | None, ...]
    provenance: tuple[SampleProvenance, ...]


def _read_test_frame_ids(path: Path, expected_video_id: str) -> tuple[int, ...]:
    """Read only test frame keys and identity; never construct test GT objects."""

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    video = raw.get("video")
    annotations = raw.get("annotations")
    if not isinstance(video, dict) or not isinstance(annotations, dict):
        raise DatasetContractError(f"Malformed test metadata: {path}")
    if str(video.get("name", "")).upper() != expected_video_id:
        raise DatasetContractError(f"Test video identity mismatch in {path}")
    if DatasetSplit.parse(str(video.get("split"))) is not DatasetSplit.TESTING:
        raise DatasetContractError(f"Test split mismatch in {path}")
    try:
        frame_ids = tuple(sorted(int(value) for value in annotations))
    except ValueError as exc:
        raise DatasetContractError(f"Non-numeric test frame key in {path}") from exc
    if not frame_ids or len(set(frame_ids)) != len(frame_ids):
        raise DatasetContractError(f"Invalid test frame keys in {path}")
    return frame_ids


def _phase_only_target(
    evaluation: EvaluationTarget,
    *,
    source: str,
    phase_allowed: bool,
) -> FrameSupervisionTarget:
    phases = {
        instance.phase_id
        for instance in evaluation.instances
        if instance.mask.phase and phase_allowed
    }
    if len(phases) > 1:
        raise DatasetContractError(
            f"{evaluation.video_id} frame {evaluation.frame_id} has conflicting phases"
        )
    phase_id = next(iter(phases)) if phases else None
    return FrameSupervisionTarget(
        video_id=evaluation.video_id,
        frame_id=evaluation.frame_id,
        instrument_ids=(),
        verb_ids=(),
        target_ids=(),
        triplet_ids=(),
        phase_id=phase_id,
        mask=FrameTaskMask(
            instrument=False,
            verb=False,
            target=False,
            ivt=False,
            phase=phase_id is not None,
        ),
        source_granularity="phase_only",
        source=source,
    )


def _mask_evaluation_target(
    target: EvaluationTarget,
    source: DerivedVideoSource | None,
) -> EvaluationTarget:
    if source is None:
        return target
    instances = tuple(
        EvaluationInstanceTarget(
            instrument_id=instance.instrument_id,
            verb_id=instance.verb_id,
            target_id=instance.target_id,
            triplet_id=instance.triplet_id,
            phase_id=instance.phase_id,
            operator_id=instance.operator_id,
            bbox=instance.bbox,
            tracks=instance.tracks,
            mask=LabelMask(
                instrument=instance.mask.instrument and source.allows("instrument"),
                verb=instance.mask.verb and source.allows("verb"),
                target=instance.mask.target and source.allows("target"),
                ivt=instance.mask.ivt and source.allows("triplet"),
                phase=instance.mask.phase and source.allows("phase"),
            ),
        )
        for instance in target.instances
    )
    return EvaluationTarget(
        video_id=target.video_id,
        frame_id=target.frame_id,
        instances=instances,
        instance_supervision_available=any(
            source.allows(field)
            for field in ("instrument", "verb", "target", "triplet", "bbox")
        ),
        source_granularity="instance",
    )


class CholecTrack20DatasetAdapter:
    """Deterministic read-only adapter for official and explicit sidecar routes."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        derived_manifest_path: str | Path | None = None,
        causal_window_size: int = 3,
    ) -> None:
        if causal_window_size <= 0:
            raise ValueError("causal_window_size must be positive")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.causal_window_size = causal_window_size
        entries = discover_official_split_manifest(self.dataset_root)
        self.entries = {entry.video_id: entry for entry in entries}
        split_counts = {
            split: sum(entry.split is split for entry in entries)
            for split in DatasetSplit
        }
        expected = {
            DatasetSplit.TRAINING: 10,
            DatasetSplit.VALIDATION: 2,
            DatasetSplit.TESTING: 8,
        }
        if split_counts != expected:
            raise DatasetContractError(
                f"Expected official 10/2/8 split, observed {split_counts}"
            )
        manifest_path = (
            self.dataset_root / "repair_manifest.json"
            if derived_manifest_path is None
            else Path(derived_manifest_path).expanduser().resolve()
        )
        if not manifest_path.is_file():
            raise DatasetContractError(f"Required repair manifest is missing: {manifest_path}")
        self.derived_manifest: DerivedSupervisionManifest = (
            load_derived_supervision_manifest(
                manifest_path,
                dataset_root_override=self.dataset_root,
            )
        )
        for required in ("VID30", "VID31"):
            if required not in self.derived_manifest.videos:
                raise DatasetContractError(f"Repair manifest omits required {required} route")
        self._verify_repair_manifest_integrity(manifest_path)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_repair_manifest_integrity(self, manifest_path: Path) -> None:
        """Fail before sampling when raw or materialized evidence has drifted."""

        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = raw.get("dataset_contract")
        if not isinstance(contract, dict):
            raise DatasetContractError("Repair manifest lacks dataset_contract")
        if contract.get("official_video_count") != 20:
            raise DatasetContractError("Repair manifest does not preserve 20 videos")
        for key in ("added_video_directories", "added_splits", "added_classes"):
            if contract.get(key) != []:
                raise DatasetContractError(f"Repair manifest violates {key}=[]")
        materialized = raw.get("materialized_sha256")
        source_hashes = raw.get("source_sha256")
        if not isinstance(materialized, dict) or not isinstance(source_hashes, dict):
            raise DatasetContractError("Repair manifest lacks required SHA-256 mappings")
        vid30 = self.derived_manifest.video("VID30")
        vid31 = self.derived_manifest.video("VID31")
        if (
            vid30.annotation_source is None
            or vid31.phase_source is None
            or vid31.frame_level_action_source is None
        ):
            raise DatasetContractError("Repair manifest runtime sources are incomplete")
        expected_paths = {
            "vid30_repaired": vid30.annotation_source,
            "vid31_phase_repaired": vid31.phase_source,
            "vid31_frame_ivt_repaired": vid31.frame_level_action_source,
        }
        raw_paths = {
            "track20_vid30_json": self.dataset_root / "Validation/VID30/vid30.json",
            "track20_vid31_json": self.dataset_root / "Training/VID31/vid31.json",
        }
        for name, path in (*expected_paths.items(), *raw_paths.items()):
            expected = (
                materialized.get(name)
                if name in expected_paths
                else source_hashes.get(name)
            )
            if not isinstance(expected, str) or len(expected) != 64:
                raise DatasetContractError(f"Repair manifest lacks valid hash for {name}")
            observed = self._sha256(path)
            if observed != expected:
                raise DatasetContractError(
                    f"Repair artifact hash mismatch for {name}: {path}"
                )

    def _entry(self, video_id: str) -> SplitManifestEntry:
        normalized = video_id.upper()
        try:
            return self.entries[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown CholecTrack20 video: {normalized}") from exc

    def _derived(self, video_id: str) -> DerivedVideoSource | None:
        return self.derived_manifest.videos.get(video_id.upper())

    def _png_resolver(
        self,
        entry: SplitManifestEntry,
        derived: DerivedVideoSource | None,
    ) -> ExactFrameFolderResolver:
        media = derived.media_source if derived is not None else Path(entry.media_source)
        if not media.is_dir():
            raise DatasetContractError(f"Expected PNG directory for {entry.video_id}: {media}")
        return ExactFrameFolderResolver(
            video_id=entry.video_id,
            split=entry.split,
            frames_dir=media,
            alignment_version=PNG_ALIGNMENT_VERSION,
        )

    def iter_video(
        self,
        video_id: str,
        *,
        frame_ids: Iterable[int] | None = None,
        max_samples: int | None = None,
    ) -> Iterator[ResolvedSample]:
        """Yield a deterministic causal sequence for one official video."""

        entry = self._entry(video_id)
        if max_samples is not None and max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        requested = None if frame_ids is None else set(frame_ids)
        if entry.split is DatasetSplit.TESTING:
            yield from self._iter_test(entry, requested=requested, max_samples=max_samples)
            return
        derived = self._derived(entry.video_id)
        if entry.video_id == "VID31":
            yield from self._iter_vid31(
                entry,
                derived,
                requested=requested,
                max_samples=max_samples,
            )
            return
        yield from self._iter_instance_video(
            entry,
            derived,
            requested=requested,
            max_samples=max_samples,
        )

    def _iter_instance_video(
        self,
        entry: SplitManifestEntry,
        derived: DerivedVideoSource | None,
        *,
        requested: set[int] | None,
        max_samples: int | None,
    ) -> Iterator[ResolvedSample]:
        annotation_path = (
            derived.annotation_source
            if derived is not None and derived.annotation_source is not None
            else Path(entry.annotation_file)
        )
        video = parse_annotation_file(
            annotation_path,
            expected_split=entry.split,
            manifest_source=(
                str(self.derived_manifest.source_path) if derived is not None else None
            ),
        )
        resolver = self._png_resolver(entry, derived)
        available = resolver.available_frame_ids
        selected = [
            frame
            for frame in video.frames
            if requested is None or frame.frame_id in requested
        ]
        if max_samples is not None:
            selected = selected[:max_samples]
        for frame in selected:
            causal_ids = _causal_from_sorted(
                available,
                target_frame_id=frame.frame_id,
                max_frames=self.causal_window_size,
            )
            refs = tuple(resolver.resolve(value).media_path for value in causal_ids)
            inference = build_inference_sample(
                frame,
                causal_frame_ids=causal_ids,
                media_refs=refs,
                alignment_version=PNG_ALIGNMENT_VERSION,
            )
            evaluation = _mask_evaluation_target(
                build_evaluation_target(frame),
                derived,
            )
            phase_allowed = derived is None or derived.allows("phase")
            frame_target = _phase_only_target(
                evaluation,
                source=str(annotation_path),
                phase_allowed=phase_allowed,
            )
            route = "derived_vid30" if entry.video_id == "VID30" else "official_raw"
            yield ResolvedSample(
                inference=inference,
                evaluation=evaluation,
                frame_supervision=frame_target,
                provenance=SampleProvenance(
                    video_id=entry.video_id,
                    split=entry.split,
                    route=route,
                    media_source=str(resolver.frames_dir),
                    annotation_source=str(annotation_path),
                    phase_source=None,
                    frame_action_source=None,
                    manifest_source=(
                        str(self.derived_manifest.source_path)
                        if derived is not None
                        else None
                    ),
                    supervision_granularity="instance_plus_phase_only",
                ),
            )

    def _iter_vid31(
        self,
        entry: SplitManifestEntry,
        derived: DerivedVideoSource | None,
        *,
        requested: set[int] | None,
        max_samples: int | None,
    ) -> Iterator[ResolvedSample]:
        if (
            derived is None
            or derived.phase_source is None
            or derived.frame_level_action_source is None
            or derived.frame_level_action_granularity != "FRAME_LEVEL_MULTI_LABEL"
        ):
            raise DatasetContractError("VID31 requires explicit phase and frame-IVT sidecars")
        phases = load_image_phase_supervision(derived.phase_source)
        actions = load_frame_level_ivt_supervision(derived.frame_level_action_source)
        resolver = self._png_resolver(entry, derived)
        available = resolver.available_frame_ids
        eligible = sorted(set(available) & phases.keys() & actions.keys())
        if requested is not None:
            eligible = [frame_id for frame_id in eligible if frame_id in requested]
        if max_samples is not None:
            eligible = eligible[:max_samples]
        for frame_id in eligible:
            action = actions[frame_id]
            phase_id = phases[frame_id]
            if action.cholect80_phase_id != phase_id:
                raise DatasetContractError(
                    f"VID31 frame {frame_id} phase sidecars disagree"
                )
            causal_ids = _causal_from_sorted(
                available,
                target_frame_id=frame_id,
                max_frames=self.causal_window_size,
            )
            refs = tuple(resolver.resolve(value).media_path for value in causal_ids)
            inference = InferenceSample(
                video_id=entry.video_id,
                target_frame_id=frame_id,
                causal_frame_ids=causal_ids,
                media_refs=refs,
                source_split=entry.split,
                alignment_version=PNG_ALIGNMENT_VERSION,
            )
            evaluation = EvaluationTarget(
                video_id=entry.video_id,
                frame_id=frame_id,
                instances=(),
                instance_supervision_available=False,
                source_granularity="none",
            )
            frame_target = FrameSupervisionTarget(
                video_id=entry.video_id,
                frame_id=frame_id,
                instrument_ids=action.instrument_ids,
                verb_ids=action.verb_ids,
                target_ids=action.target_ids,
                triplet_ids=action.triplet_ids,
                phase_id=phase_id,
                mask=FrameTaskMask(True, True, True, True, True),
                source_granularity="frame_multilabel",
                source=str(derived.frame_level_action_source),
            )
            yield ResolvedSample(
                inference=inference,
                evaluation=evaluation,
                frame_supervision=frame_target,
                provenance=SampleProvenance(
                    video_id=entry.video_id,
                    split=entry.split,
                    route="derived_vid31_frame_multilabel",
                    media_source=str(resolver.frames_dir),
                    annotation_source=None,
                    phase_source=str(derived.phase_source),
                    frame_action_source=str(derived.frame_level_action_source),
                    manifest_source=str(self.derived_manifest.source_path),
                    supervision_granularity="frame_multilabel_plus_phase",
                ),
            )

    def _iter_test(
        self,
        entry: SplitManifestEntry,
        *,
        requested: set[int] | None,
        max_samples: int | None,
    ) -> Iterator[ResolvedSample]:
        frame_ids = _read_test_frame_ids(Path(entry.annotation_file), entry.video_id)
        selected = [value for value in frame_ids if requested is None or value in requested]
        if max_samples is not None:
            selected = selected[:max_samples]
        resolver = Mp4FrameResolver(
            video_id=entry.video_id,
            split=entry.split,
            media_path=entry.media_source,
            frame_count=max(frame_ids),
            decoder_index_offset=-1,
            alignment_version=MP4_ALIGNMENT_VERSION,
        )
        for frame_id in selected:
            causal_ids = _causal_from_sorted(
                frame_ids,
                target_frame_id=frame_id,
                max_frames=self.causal_window_size,
            )
            refs = tuple(resolver.resolve(value).media_path for value in causal_ids)
            yield ResolvedSample(
                inference=InferenceSample(
                    video_id=entry.video_id,
                    target_frame_id=frame_id,
                    causal_frame_ids=causal_ids,
                    media_refs=refs,
                    source_split=entry.split,
                    alignment_version=MP4_ALIGNMENT_VERSION,
                ),
                evaluation=None,
                frame_supervision=None,
                provenance=SampleProvenance(
                    video_id=entry.video_id,
                    split=entry.split,
                    route="official_test_gold_free",
                    media_source=entry.media_source,
                    annotation_source=None,
                    phase_source=None,
                    frame_action_source=None,
                    manifest_source=None,
                    supervision_granularity="inference_only",
                ),
            )

    def collect(
        self,
        video_ids: Sequence[str],
        *,
        samples_per_video: int,
    ) -> tuple[ResolvedSample, ...]:
        """Collect a deterministic, video-major P2 subset."""

        if samples_per_video <= 0:
            raise ValueError("samples_per_video must be positive")
        records = tuple(
            sample
            for video_id in video_ids
            for sample in self.iter_video(video_id, max_samples=samples_per_video)
        )
        if not records:
            raise DatasetContractError("No samples matched the requested videos")
        return records

    def target_granularity_audit(
        self,
        video_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Count eligibility without choosing future paper metric granularity."""

        selected = sorted(video_ids or self.entries)
        result: dict[str, Any] = {
            "schema_version": "p2_target_granularity_audit_v1",
            "paper_metric_decision": "DEFERRED_TO_P4",
            "videos": {},
        }
        for video_id in selected:
            counters = {
                "samples": 0,
                "instance_eligible": {task: 0 for task in ("instrument", "verb", "target", "ivt", "phase")},
                "frame_eligible": {task: 0 for task in ("instrument", "verb", "target", "ivt", "phase")},
                "masked": {task: 0 for task in ("instrument", "verb", "target", "ivt", "phase")},
                "phase_conflicts": 0,
            }
            for sample in self.iter_video(video_id):
                counters["samples"] += 1
                evaluation = sample.evaluation
                frame_target = sample.frame_supervision
                for task in counters["instance_eligible"]:
                    eligible = bool(
                        evaluation
                        and evaluation.instance_supervision_available
                        and any(getattr(instance.mask, task) for instance in evaluation.instances)
                    )
                    counters["instance_eligible"][task] += int(eligible)
                    frame_eligible = bool(frame_target and getattr(frame_target.mask, task))
                    counters["frame_eligible"][task] += int(frame_eligible)
                    counters["masked"][task] += int(not eligible and not frame_eligible)
            result["videos"][video_id] = counters
        return result


class CurrentFrameTensorDataset:
    """Torch-compatible current-frame view over resolved P2 samples."""

    def __init__(self, records: Sequence[ResolvedSample], *, image_size: int = 96) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.records = tuple(records)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TensorResolvedSample:
        import numpy as np
        import torch
        from PIL import Image

        resolved = self.records[index]
        if resolved.inference.source_split is DatasetSplit.TESTING:
            raise DatasetContractError("P2 tensor dataset forbids test samples")
        media_path = Path(resolved.inference.media_refs[-1])
        with Image.open(media_path) as image:
            rgb = image.convert("RGB").resize((self.image_size, self.image_size))
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return TensorResolvedSample(image=tensor, resolved=resolved)


def collate_smoke_batch(items: Sequence[TensorResolvedSample]) -> SmokeBatch:
    """Stack model inputs while preserving targets outside the model payload."""

    import torch

    if not items:
        raise ValueError("Cannot collate an empty batch")
    return SmokeBatch(
        images=torch.stack([item.image for item in items], dim=0),
        inference_samples=tuple(item.resolved.inference for item in items),
        frame_targets=tuple(item.resolved.frame_supervision for item in items),
        evaluation_targets=tuple(item.resolved.evaluation for item in items),
        provenance=tuple(item.resolved.provenance for item in items),
    )
