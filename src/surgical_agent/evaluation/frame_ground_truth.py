"""Split-local frame ground truth loading for offline evaluation only."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from surgical_agent.data.media_backend import ExactFrameFolderResolver
from surgical_agent.data.parser import parse_annotation_file
from surgical_agent.data.schemas import (
    CanonicalFrameAnnotation,
    DatasetSplit,
    EvaluationTarget,
    FrameSupervisionTarget,
    FrameTaskMask,
)
from surgical_agent.evaluation.offline_artifacts import (
    CompletedRun,
    OfflineEvaluationError,
)

_ALL_TASKS = frozenset({"instrument", "verb", "target", "ivt", "phase"})
_TEST_VIDEO_COUNT = 8


@dataclass(frozen=True)
class GroundTruthSource:
    relative_path: str
    sha256: str
    provenance_status: str


@dataclass(frozen=True)
class EvaluationData:
    split: DatasetSplit
    runtime_identities: tuple[tuple[str, int], ...]
    targets: Mapping[tuple[str, int], FrameSupervisionTarget]
    sources: Mapping[str, GroundTruthSource]
    provenance_by_video: Mapping[str, str]
    repair_manifest_sha256: str


def aggregate_frame_target(
    frame: CanonicalFrameAnnotation,
    *,
    allowed_tasks: frozenset[str],
    source: str,
) -> FrameSupervisionTarget:
    """Union instance labels only when the whole frame is supervised for a task."""

    return _aggregate_instances(
        video_id=frame.video_id,
        frame_id=frame.frame_id,
        instances=tuple(frame.instances),
        allowed_tasks=allowed_tasks,
        source=source,
    )


def aggregate_evaluation_target(
    target: EvaluationTarget,
    *,
    allowed_tasks: frozenset[str] = _ALL_TASKS,
    source: str,
) -> FrameSupervisionTarget:
    """Conservatively aggregate an instance-level Training target for Gate labels."""

    if not isinstance(target, EvaluationTarget):
        raise TypeError("target must be an EvaluationTarget")
    if not target.instance_supervision_available:
        raise OfflineEvaluationError("instance supervision is unavailable")
    return _aggregate_instances(
        video_id=target.video_id,
        frame_id=target.frame_id,
        instances=tuple(target.instances),
        allowed_tasks=allowed_tasks,
        source=source,
    )


def _aggregate_instances(
    *,
    video_id: str,
    frame_id: int,
    instances: tuple[Any, ...],
    allowed_tasks: frozenset[str],
    source: str,
) -> FrameSupervisionTarget:
    """Shared all-instances mask rule for canonical and isolated targets."""

    if not allowed_tasks <= _ALL_TASKS:
        raise OfflineEvaluationError("allowed_tasks contains an unknown task")
    if not instances:
        mask = FrameTaskMask(False, False, False, False, False)
        return FrameSupervisionTarget(
            video_id=video_id,
            frame_id=frame_id,
            instrument_ids=(),
            verb_ids=(),
            target_ids=(),
            triplet_ids=(),
            phase_id=None,
            mask=mask,
            source_granularity="frame_multilabel",
            source=source,
        )

    availability = {
        task: task in allowed_tasks
        and all(getattr(instance.mask, task) for instance in instances)
        for task in _ALL_TASKS
    }

    def ids(task: str, attribute: str) -> tuple[int, ...]:
        if not availability[task]:
            return ()
        return tuple(sorted({getattr(instance, attribute) for instance in instances}))

    phase_id: int | None = None
    if availability["phase"]:
        phases = {instance.phase_id for instance in instances}
        if len(phases) != 1:
            raise OfflineEvaluationError(
                f"conflicting phase labels for {video_id} frame {frame_id}"
            )
        phase_id = next(iter(phases))
    return FrameSupervisionTarget(
        video_id=video_id,
        frame_id=frame_id,
        instrument_ids=ids("instrument", "instrument_id"),
        verb_ids=ids("verb", "verb_id"),
        target_ids=ids("target", "target_id"),
        triplet_ids=ids("ivt", "triplet_id"),
        phase_id=phase_id,
        mask=FrameTaskMask(
            availability["instrument"],
            availability["verb"],
            availability["target"],
            availability["ivt"],
            availability["phase"],
        ),
        source_granularity="frame_multilabel",
        source=source,
    )


def load_evaluation_data(
    run: CompletedRun,
    dataset_root: str | Path,
    *,
    authorize_test_gt_evaluation: bool,
) -> EvaluationData:
    """Reconstruct only the run's split selection and load its authorized GT."""

    split = run.effective_split
    if split is DatasetSplit.TESTING:
        if not authorize_test_gt_evaluation:
            raise OfflineEvaluationError(
                "Testing GT evaluation requires explicit authorization"
            )
        if (
            run.mode != "paper"
            or run.declared_split is not DatasetSplit.TESTING
            or len(run.video_ids) != _TEST_VIDEO_COUNT
        ):
            raise OfflineEvaluationError(
                "Testing evaluation requires a complete paper-mode Testing run"
            )
    elif split is not DatasetSplit.VALIDATION:
        raise OfflineEvaluationError("offline evaluation supports Validation or Testing")

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise OfflineEvaluationError("dataset_root is not an existing directory")
    repair_path = root / "repair_manifest.json"
    if not repair_path.is_file():
        raise OfflineEvaluationError("repair_manifest.json is missing")
    repair_sha = _sha256(repair_path)
    if repair_sha != run.repair_manifest_sha256:
        raise OfflineEvaluationError("repair manifest SHA-256 does not match rollout")

    predictions_by_video: dict[str, tuple[int, ...]] = {}
    for video_id in run.video_ids:
        predictions_by_video[video_id] = tuple(
            prediction.frame_id
            for prediction in run.predictions
            if prediction.video_id == video_id
        )

    runtime: list[tuple[str, int]] = []
    targets: dict[tuple[str, int], FrameSupervisionTarget] = {}
    sources: dict[str, GroundTruthSource] = {}
    provenance: dict[str, str] = {}
    repair_payload: dict[str, Any] | None = None

    for video_id in run.video_ids:
        allowed_tasks = _ALL_TASKS
        if split is DatasetSplit.VALIDATION:
            if video_id == "VID30":
                if repair_payload is None:
                    repair_payload = _read_repair_manifest(repair_path)
                media_dir, annotation_path, allowed_tasks = _vid30_sources(
                    repair_payload, root
                )
                provenance_status = "candidate_repaired_validation"
            else:
                video_dir = root / "Validation" / video_id
                media_dir = video_dir / "Frames"
                annotation_path = video_dir / f"{video_id.lower()}.json"
                provenance_status = "official_raw"
            resolver = ExactFrameFolderResolver(
                video_id=video_id,
                split=split,
                frames_dir=media_dir,
            )
            available_ids = resolver.available_frame_ids
            requested_count = len(predictions_by_video[video_id])
            start_index = 0
            if run.mode == "engineering" and requested_count:
                # The engineering CLI permits an explicit start target. Rebuild
                # its contiguous media slice, not an assumed video prefix.
                first_id = predictions_by_video[video_id][0]
                if first_id not in available_ids:
                    raise OfflineEvaluationError(
                        f"prediction start is outside canonical selection for {video_id}"
                    )
                start_index = available_ids.index(first_id)
            selected_ids = (
                available_ids
                if run.mode == "paper"
                else available_ids[start_index:start_index + requested_count]
            )
        else:
            video_dir = root / "Testing" / video_id
            annotation_path = video_dir / f"{video_id.lower()}.json"
            provenance_status = "official_raw"
            selected_ids = ()

        annotation_path = _contained_file(annotation_path, root, "annotation source")
        annotation = parse_annotation_file(
            annotation_path,
            expected_split=split,
            manifest_source="repair_manifest.json" if video_id == "VID30" else None,
        )
        if annotation.video_id != video_id:
            raise OfflineEvaluationError("annotation video ID does not match selection")
        if split is DatasetSplit.TESTING:
            selected_ids = annotation.frame_ids

        prediction_ids = predictions_by_video[video_id]
        if prediction_ids != tuple(selected_ids):
            raise OfflineEvaluationError(
                f"prediction identities do not match canonical selection for {video_id}"
            )
        selected_set = set(selected_ids)
        if run.mode == "paper" and not set(annotation.frame_ids) <= selected_set:
            raise OfflineEvaluationError(
                f"GT-bearing frames fall outside the paper runtime selection for {video_id}"
            )

        relative_source = annotation_path.relative_to(root).as_posix()
        runtime.extend((video_id, frame_id) for frame_id in selected_ids)
        for frame in annotation.frames:
            if frame.frame_id in selected_set:
                identity = (video_id, frame.frame_id)
                targets[identity] = aggregate_frame_target(
                    frame,
                    allowed_tasks=allowed_tasks,
                    source=relative_source,
                )
        sources[video_id] = GroundTruthSource(
            relative_path=relative_source,
            sha256=_sha256(annotation_path),
            provenance_status=provenance_status,
        )
        provenance[video_id] = provenance_status

    runtime_identities = tuple(runtime)
    if runtime_identities != tuple(run.prediction_identities):
        raise OfflineEvaluationError("prediction ordering differs from canonical selection")
    return EvaluationData(
        split=split,
        runtime_identities=runtime_identities,
        targets=MappingProxyType(targets),
        sources=MappingProxyType(sources),
        provenance_by_video=MappingProxyType(provenance),
        repair_manifest_sha256=repair_sha,
    )


def _read_repair_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OfflineEvaluationError("repair manifest is not valid JSON") from error
    if not isinstance(value, dict):
        raise OfflineEvaluationError("repair manifest must be a JSON object")
    if value.get("schema_version") != "ct20_vid30_vid31_derived_supervision_v1":
        raise OfflineEvaluationError("unsupported repair manifest schema")
    return value


def _vid30_sources(
    manifest: Mapping[str, Any], root: Path
) -> tuple[Path, Path, frozenset[str]]:
    videos = manifest.get("videos")
    if not isinstance(videos, dict) or not isinstance(videos.get("VID30"), dict):
        raise OfflineEvaluationError("repair manifest has no VID30 source")
    entry = videos["VID30"]
    if entry.get("split") != "validation":
        raise OfflineEvaluationError("VID30 repair source must declare validation")
    media = _rebase_manifest_path(entry.get("media_source"), root, directory=True)
    annotation = _rebase_manifest_path(
        entry.get("annotation_source"), root, directory=False
    )
    fields = entry.get("field_supervision")
    if not isinstance(fields, dict):
        raise OfflineEvaluationError("VID30 field_supervision must be an object")
    task_map = {
        "instrument": "instrument",
        "verb": "verb",
        "target": "target",
        "triplet": "ivt",
        "phase": "phase",
    }
    if any(type(fields.get(field)) is not bool for field in task_map):
        raise OfflineEvaluationError("VID30 task supervision flags must be booleans")
    allowed = frozenset(
        task_map[field] for field in task_map if fields[field]
    )
    return media, annotation, allowed


def _rebase_manifest_path(value: object, root: Path, *, directory: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise OfflineEvaluationError("repair manifest path must be non-empty text")
    normalized = value.replace("\\", "/")
    marker = "/Validation/VID30/"
    position = normalized.casefold().find(marker.casefold())
    if position < 0:
        raise OfflineEvaluationError("VID30 repair path is outside its split directory")
    relative = normalized[position + 1 :].split("/")
    candidate = root.joinpath(*relative).resolve()
    _require_contained(candidate, root)
    exists = candidate.is_dir() if directory else candidate.is_file()
    if not exists:
        raise OfflineEvaluationError("VID30 repair source is missing")
    return candidate


def _contained_file(path: Path, root: Path, name: str) -> Path:
    resolved = path.resolve()
    _require_contained(resolved, root)
    if not resolved.is_file():
        raise OfflineEvaluationError(f"{name} is missing")
    return resolved


def _require_contained(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise OfflineEvaluationError("dataset source escapes dataset_root") from error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
