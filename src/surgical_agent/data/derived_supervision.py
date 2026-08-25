"""Strict runtime contract for explicitly derived VID30/VID31 supervision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from surgical_agent.data.qualification import SUPERVISION_FIELDS

DERIVED_SCHEMA_VERSION = "ct20_vid30_vid31_derived_supervision_v1"
FRAME_LEVEL_IVT_SCHEMA_VERSION = "ct20_vid31_cholect50_frame_ivt_v1"


class DerivedSupervisionError(ValueError):
    """Raised when a derived source is missing or violates its frozen contract."""


@dataclass(frozen=True)
class DerivedVideoSource:
    """Resolved media and field-specific supervision for one anomalous video."""

    video_id: str
    split: str
    media_source: Path
    annotation_source: Path | None
    phase_source: Path | None
    frame_level_action_source: Path | None
    frame_level_action_granularity: str | None
    field_supervision: dict[str, bool]
    status: str

    def allows(self, field: str) -> bool:
        if field not in SUPERVISION_FIELDS:
            raise KeyError(f"Unknown supervision field: {field}")
        return self.field_supervision[field]


@dataclass(frozen=True)
class DerivedSupervisionManifest:
    """Explicit source substitutions; raw parser behavior remains untouched."""

    source_path: Path
    raw_dataset_root: Path
    videos: dict[str, DerivedVideoSource]

    def video(self, video_id: str) -> DerivedVideoSource:
        normalized = video_id.upper()
        try:
            return self.videos[normalized]
        except KeyError as error:
            raise DerivedSupervisionError(
                f"No derived supervision source for {normalized}"
            ) from error


@dataclass(frozen=True)
class FrameLevelIvtTarget:
    """One image-aligned CholecT50 multi-label target without instance geometry."""

    frame_id: int
    cholect50_frame_id: int
    instrument_ids: tuple[int, ...]
    verb_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    triplet_ids: tuple[int, ...]
    cholect50_phase_id: int
    cholect80_phase_id: int
    phase_agreement: bool
    action_present: bool


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DerivedSupervisionError(f"{location} must be a JSON object")
    return value


def _existing_path(value: object, location: str, *, directory: bool) -> Path:
    if not isinstance(value, str) or not value:
        raise DerivedSupervisionError(f"{location} must be a non-empty path")
    path = Path(value).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        expected = "directory" if directory else "file"
        raise DerivedSupervisionError(f"{location} is not an existing {expected}: {path}")
    return path


def _optional_file(value: object, location: str) -> Path | None:
    if value is None:
        return None
    return _existing_path(value, location, directory=False)


def _rebased_existing_path(
    value: object,
    location: str,
    *,
    directory: bool,
    recorded_root: str,
    runtime_root: Path,
) -> Path:
    """Resolve a manifest path after safely rebasing its recorded dataset root."""

    if not isinstance(value, str) or not value:
        raise DerivedSupervisionError(f"{location} must be a non-empty path")
    normalized_value = value.replace("\\", "/").rstrip("/")
    normalized_root = recorded_root.replace("\\", "/").rstrip("/")
    value_folded = normalized_value.casefold()
    root_folded = normalized_root.casefold()
    if value_folded == root_folded:
        relative_parts: tuple[str, ...] = ()
    elif value_folded.startswith(f"{root_folded}/"):
        suffix = normalized_value[len(normalized_root) + 1 :]
        relative_parts = tuple(part for part in suffix.split("/") if part)
    else:
        return _existing_path(value, location, directory=directory)
    if any(part in {".", ".."} for part in relative_parts):
        raise DerivedSupervisionError(f"{location} escapes the dataset root")
    candidate = runtime_root.joinpath(*relative_parts).resolve()
    try:
        candidate.relative_to(runtime_root)
    except ValueError as exc:
        raise DerivedSupervisionError(f"{location} escapes the dataset root") from exc
    exists = candidate.is_dir() if directory else candidate.is_file()
    if not exists:
        expected = "directory" if directory else "file"
        raise DerivedSupervisionError(
            f"{location} is not an existing {expected} after rebasing: {candidate}"
        )
    return candidate


def _rebased_optional_file(
    value: object,
    location: str,
    *,
    recorded_root: str,
    runtime_root: Path,
) -> Path | None:
    if value is None:
        return None
    return _rebased_existing_path(
        value,
        location,
        directory=False,
        recorded_root=recorded_root,
        runtime_root=runtime_root,
    )


def load_derived_supervision_manifest(
    path: str | Path,
    *,
    dataset_root_override: str | Path | None = None,
) -> DerivedSupervisionManifest:
    """Load the source-aware manifest and reject implicit or missing inputs."""

    source_path = Path(path).expanduser().resolve()
    raw = _mapping(
        json.loads(source_path.read_text(encoding="utf-8")),
        "manifest",
    )
    if raw.get("schema_version") != DERIVED_SCHEMA_VERSION:
        raise DerivedSupervisionError(
            f"Unsupported derived schema: {raw.get('schema_version')!r}"
        )
    if raw.get("raw_dataset_modified") is not False:
        raise DerivedSupervisionError("Derived manifest must preserve the raw dataset")
    if raw.get("policy") != "EXPLICIT_DERIVED_SOURCES_ONLY":
        raise DerivedSupervisionError("Derived substitutions must be explicit")
    recorded_root = raw.get("raw_dataset_root")
    if not isinstance(recorded_root, str) or not recorded_root:
        raise DerivedSupervisionError("raw_dataset_root must be a non-empty path")
    dataset_root = (
        _existing_path(recorded_root, "raw_dataset_root", directory=True)
        if dataset_root_override is None
        else Path(dataset_root_override).expanduser().resolve()
    )
    if not dataset_root.is_dir():
        raise DerivedSupervisionError(
            f"dataset_root_override is not an existing directory: {dataset_root}"
        )

    videos: dict[str, DerivedVideoSource] = {}
    for raw_video_id, raw_video in _mapping(raw.get("videos"), "videos").items():
        video_id = str(raw_video_id).upper()
        value = _mapping(raw_video, f"videos.{video_id}")
        fields = _mapping(
            value.get("field_supervision"),
            f"videos.{video_id}.field_supervision",
        )
        if set(fields) != SUPERVISION_FIELDS:
            raise DerivedSupervisionError(
                f"videos.{video_id} must define exactly {sorted(SUPERVISION_FIELDS)}"
            )
        if any(not isinstance(enabled, bool) for enabled in fields.values()):
            raise DerivedSupervisionError(
                f"videos.{video_id} field decisions must be booleans"
            )
        split = value.get("split")
        status = value.get("status")
        if not isinstance(split, str) or not split:
            raise DerivedSupervisionError(f"videos.{video_id}.split must be text")
        if not isinstance(status, str) or not status:
            raise DerivedSupervisionError(f"videos.{video_id}.status must be text")
        path_resolver = (
            _optional_file
            if dataset_root_override is None
            else lambda item, item_location: _rebased_optional_file(
                item,
                item_location,
                recorded_root=recorded_root,
                runtime_root=dataset_root,
            )
        )
        annotation_source = path_resolver(
            value.get("annotation_source"), f"videos.{video_id}.annotation_source"
        )
        phase_source = path_resolver(
            value.get("phase_source"), f"videos.{video_id}.phase_source"
        )
        frame_level_action_source = path_resolver(
            value.get("frame_level_action_source"),
            f"videos.{video_id}.frame_level_action_source",
        )
        frame_level_action_granularity = value.get("frame_level_action_granularity")
        if frame_level_action_source is None:
            if frame_level_action_granularity is not None:
                raise DerivedSupervisionError(
                    f"videos.{video_id} declares frame-level granularity without a source"
                )
        elif frame_level_action_granularity != "FRAME_LEVEL_MULTI_LABEL":
            raise DerivedSupervisionError(
                f"videos.{video_id} frame-level action source has invalid granularity"
            )
        if fields["phase"] and annotation_source is None and phase_source is None:
            raise DerivedSupervisionError(
                f"videos.{video_id} enables phase without an explicit source"
            )
        non_phase_enabled = any(
            enabled for field, enabled in fields.items() if field != "phase"
        )
        if non_phase_enabled and annotation_source is None:
            raise DerivedSupervisionError(
                f"videos.{video_id} enables core fields without annotation_source"
            )
        videos[video_id] = DerivedVideoSource(
            video_id=video_id,
            split=split,
            media_source=(
                _existing_path(
                    value.get("media_source"),
                    f"videos.{video_id}.media_source",
                    directory=True,
                )
                if dataset_root_override is None
                else _rebased_existing_path(
                    value.get("media_source"),
                    f"videos.{video_id}.media_source",
                    directory=True,
                    recorded_root=recorded_root,
                    runtime_root=dataset_root,
                )
            ),
            annotation_source=annotation_source,
            phase_source=phase_source,
            frame_level_action_source=frame_level_action_source,
            frame_level_action_granularity=frame_level_action_granularity,
            field_supervision=dict(fields),
            status=status,
        )
    return DerivedSupervisionManifest(
        source_path=source_path,
        raw_dataset_root=dataset_root,
        videos=videos,
    )


def load_image_phase_supervision(path: str | Path) -> dict[int, int]:
    """Load a derived image-frame-to-phase mapping with strict task ranges."""

    source_path = Path(path).expanduser().resolve()
    raw = _mapping(
        json.loads(source_path.read_text(encoding="utf-8")),
        "phase_supervision",
    )
    if raw.get("schema_version") != DERIVED_SCHEMA_VERSION:
        raise DerivedSupervisionError(
            f"Unsupported phase supervision schema: {raw.get('schema_version')!r}"
        )
    values = _mapping(
        raw.get("phase_by_track20_image_frame_id"),
        "phase_by_track20_image_frame_id",
    )
    result: dict[int, int] = {}
    for raw_frame_id, raw_value in values.items():
        frame_id = int(raw_frame_id)
        value = _mapping(raw_value, f"phase[{raw_frame_id}]")
        phase_id = value.get("phase_id")
        if not isinstance(phase_id, int) or not 0 <= phase_id <= 6:
            raise DerivedSupervisionError(
                f"phase[{raw_frame_id}].phase_id is outside 0..6"
            )
        result[frame_id] = phase_id
    return result


def load_frame_level_ivt_supervision(
    path: str | Path,
) -> dict[int, FrameLevelIvtTarget]:
    """Load image-level IVT labels while rejecting instance-level interpretation."""

    source_path = Path(path).expanduser().resolve()
    raw = _mapping(
        json.loads(source_path.read_text(encoding="utf-8")),
        "frame_level_ivt_supervision",
    )
    if raw.get("schema_version") != FRAME_LEVEL_IVT_SCHEMA_VERSION:
        raise DerivedSupervisionError(
            f"Unsupported frame-level IVT schema: {raw.get('schema_version')!r}"
        )
    if raw.get("supervision_granularity") != "FRAME_LEVEL_MULTI_LABEL":
        raise DerivedSupervisionError("CholecT50 supervision must remain frame-level")
    if raw.get("instance_level_bbox_available") is not False:
        raise DerivedSupervisionError("Frame-level IVT must not expose instance bboxes")
    if raw.get("track_ids_available") is not False:
        raise DerivedSupervisionError("Frame-level IVT must not expose track IDs")

    limits = {
        "instrument_ids": (0, 5),
        "verb_ids": (0, 9),
        "target_ids": (0, 14),
        "triplet_ids": (0, 99),
    }
    result: dict[int, FrameLevelIvtTarget] = {}
    for raw_frame_id, raw_record in _mapping(raw.get("frames"), "frames").items():
        frame_id = int(raw_frame_id)
        record = _mapping(raw_record, f"frames.{raw_frame_id}")
        labels: dict[str, tuple[int, ...]] = {}
        for field, (minimum, maximum) in limits.items():
            values = record.get(field)
            if not isinstance(values, list) or any(
                not isinstance(value, int) or not minimum <= value <= maximum
                for value in values
            ):
                raise DerivedSupervisionError(
                    f"frames.{raw_frame_id}.{field} is outside {minimum}..{maximum}"
                )
            if values != sorted(set(values)):
                raise DerivedSupervisionError(
                    f"frames.{raw_frame_id}.{field} must be sorted and unique"
                )
            labels[field] = tuple(values)
        action_present = record.get("action_present")
        if not isinstance(action_present, bool):
            raise DerivedSupervisionError(
                f"frames.{raw_frame_id}.action_present must be boolean"
            )
        if action_present != bool(labels["triplet_ids"]):
            raise DerivedSupervisionError(
                f"frames.{raw_frame_id}.action_present contradicts triplet labels"
            )
        phases = (record.get("cholect50_phase_id"), record.get("cholect80_phase_id"))
        if any(not isinstance(value, int) or not 0 <= value <= 6 for value in phases):
            raise DerivedSupervisionError(
                f"frames.{raw_frame_id} phase IDs must be in 0..6"
            )
        phase_agreement = record.get("phase_agreement")
        if not isinstance(phase_agreement, bool) or phase_agreement != (
            phases[0] == phases[1]
        ):
            raise DerivedSupervisionError(
                f"frames.{raw_frame_id}.phase_agreement is inconsistent"
            )
        result[frame_id] = FrameLevelIvtTarget(
            frame_id=frame_id,
            cholect50_frame_id=int(record["cholect50_frame_id"]),
            instrument_ids=labels["instrument_ids"],
            verb_ids=labels["verb_ids"],
            target_ids=labels["target_ids"],
            triplet_ids=labels["triplet_ids"],
            cholect50_phase_id=phases[0],
            cholect80_phase_id=phases[1],
            phase_agreement=phase_agreement,
            action_present=action_present,
        )
    return result
