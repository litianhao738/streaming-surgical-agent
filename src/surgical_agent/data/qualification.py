"""Field-level supervision qualification loaded from derived audit evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPERVISION_FIELDS = frozenset(
    {
        "instrument",
        "verb",
        "target",
        "triplet",
        "phase",
        "bbox",
        "operator",
        "track_ids",
        "visual_conditions",
    }
)


class SupervisionQualificationError(ValueError):
    """Raised when unresolved supervision is requested."""


@dataclass(frozen=True)
class VideoSupervisionQualification:
    """Per-field decision for one release video."""

    video_id: str
    field_supervision: dict[str, bool]
    status: str
    raw_assets_retained: bool

    def allows(self, field: str) -> bool:
        if field not in SUPERVISION_FIELDS:
            raise KeyError(f"Unknown supervision field: {field}")
        return self.field_supervision[field]


@dataclass(frozen=True)
class FieldLevelSupervisionManifest:
    """Read-only manifest; unlisted videos continue through canonical task masks."""

    schema_version: str
    source_path: Path
    unlisted_video_policy: str
    videos: dict[str, VideoSupervisionQualification]

    def allows(self, video_id: str, field: str) -> bool:
        normalized = video_id.upper()
        qualification = self.videos.get(normalized)
        if qualification is None:
            if self.unlisted_video_policy != "USE_CANONICAL_TASK_MASKS":
                raise SupervisionQualificationError(
                    f"No field-level qualification for {normalized}"
                )
            if field not in SUPERVISION_FIELDS:
                raise KeyError(f"Unknown supervision field: {field}")
            return True
        return qualification.allows(field)

    def require(self, video_id: str, field: str) -> None:
        if not self.allows(video_id, field):
            qualification = self.videos[video_id.upper()]
            raise SupervisionQualificationError(
                f"{video_id.upper()} field {field!r} is disabled: {qualification.status}"
            )


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{location} must be a JSON object")
    return value


def load_field_level_supervision_manifest(
    path: str | Path,
) -> FieldLevelSupervisionManifest:
    """Load and strictly validate the derived field-supervision decisions."""

    source_path = Path(path).expanduser().resolve()
    raw = _mapping(
        json.loads(source_path.read_text(encoding="utf-8")),
        "manifest",
    )
    schema_version = raw.get("schema_version")
    if schema_version != "ct20_field_level_supervision_manifest_v1":
        raise ValueError(f"Unsupported supervision manifest: {schema_version!r}")
    unlisted_policy = raw.get("unlisted_video_policy")
    if unlisted_policy != "USE_CANONICAL_TASK_MASKS":
        raise ValueError(f"Unsupported unlisted-video policy: {unlisted_policy!r}")

    videos: dict[str, VideoSupervisionQualification] = {}
    for raw_video_id, raw_value in _mapping(raw.get("videos"), "videos").items():
        video_id = str(raw_video_id).upper()
        value = _mapping(raw_value, f"videos.{video_id}")
        field_mapping = _mapping(
            value.get("field_supervision"),
            f"videos.{video_id}.field_supervision",
        )
        if set(field_mapping) != SUPERVISION_FIELDS:
            raise ValueError(
                f"videos.{video_id}.field_supervision must contain exactly "
                f"{sorted(SUPERVISION_FIELDS)}"
            )
        if any(not isinstance(enabled, bool) for enabled in field_mapping.values()):
            raise TypeError(f"videos.{video_id} field decisions must be booleans")
        status = value.get("status")
        if not isinstance(status, str) or not status:
            raise TypeError(f"videos.{video_id}.status must be non-empty text")
        retained = value.get("raw_assets_retained")
        if retained is not True:
            raise ValueError(f"videos.{video_id} must retain raw assets")
        videos[video_id] = VideoSupervisionQualification(
            video_id=video_id,
            field_supervision=dict(field_mapping),
            status=status,
            raw_assets_retained=retained,
        )
    return FieldLevelSupervisionManifest(
        schema_version=schema_version,
        source_path=source_path,
        unlisted_video_policy=unlisted_policy,
        videos=videos,
    )
