"""Read-only CholecTrack20 canonical JSON parser."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from surgical_agent.data.masks import canonical_label_mask
from surgical_agent.data.schemas import (
    BoundingBox,
    CanonicalFrameAnnotation,
    CanonicalToolInstance,
    CanonicalVideoAnnotation,
    DatasetSplit,
    OntologyTerm,
    SourceProvenance,
    TrackIds,
    VisualConditions,
)

REQUIRED_INSTANCE_FIELDS = frozenset(
    {
        "instrument",
        "verb",
        "target",
        "phase",
        "triplet",
        "tool_bbox",
        "operator",
        "iscrowd",
        "area",
        "score",
        "intraoperative_track",
        "intracorporeal_track",
        "visibility_track",
        "visibility",
        "crowded",
        "visible",
        "occluded",
        "bleeding",
        "smoke",
        "blurred",
        "undercoverage",
        "reflection",
        "stainedlens",
    }
)


class AnnotationSchemaError(ValueError):
    """Raised when raw annotation data violates the observed schema contract."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnnotationSchemaError(f"{location} must be a JSON object")
    return value


def _integer(value: object, location: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AnnotationSchemaError(f"{location} must be an integer")
    return value


def _number(value: object, location: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AnnotationSchemaError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AnnotationSchemaError(f"{location} must be finite")
    return result


def _binary(value: object, location: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise AnnotationSchemaError(f"{location} must be boolean or 0/1")


def _bbox(value: object, location: str) -> BoundingBox:
    if not isinstance(value, list) or len(value) != 4:
        raise AnnotationSchemaError(f"{location} must contain four numbers")
    return BoundingBox(
        *(_number(item, f"{location}[{index}]") for index, item in enumerate(value))
    )


def _ontology_terms(
    categories: Mapping[str, Any],
    *,
    source_path: Path,
    source_version: str,
) -> tuple[OntologyTerm, ...]:
    terms: list[OntologyTerm] = []
    for raw_task, canonical_task in (
        ("tools", "instrument"),
        ("operators", "operator"),
    ):
        raw_terms = categories.get(raw_task, [])
        if not isinstance(raw_terms, list):
            raise AnnotationSchemaError(f"categories.{raw_task} must be a list")
        for index, item in enumerate(raw_terms):
            term = _mapping(item, f"categories.{raw_task}[{index}]")
            numeric_id = _integer(term.get("id"), f"categories.{raw_task}[{index}].id")
            name = term.get("name")
            if not isinstance(name, str) or not name:
                raise AnnotationSchemaError(
                    f"categories.{raw_task}[{index}].name must be text"
                )
            terms.append(
                OntologyTerm(
                    task=canonical_task,
                    numeric_id=numeric_id,
                    canonical_name=name,
                    component_relation=None,
                    source="CholecTrack20 raw JSON categories",
                    source_version=source_version,
                    source_location=str(source_path),
                    verification_method="parsed identically from local release JSON",
                    verified=True,
                    notes="Dataset-embedded mapping; no inferred medical name.",
                )
            )
    return tuple(terms)


def _parse_instance(raw: object, location: str) -> CanonicalToolInstance:
    data = _mapping(raw, location)
    missing = sorted(REQUIRED_INSTANCE_FIELDS - data.keys())
    if missing:
        raise AnnotationSchemaError(f"{location} is missing required fields: {missing}")

    instrument_id = _integer(data["instrument"], f"{location}.instrument")
    verb_id = _integer(data["verb"], f"{location}.verb")
    target_id = _integer(data["target"], f"{location}.target")
    triplet_id = _integer(data["triplet"], f"{location}.triplet")
    phase_id = _integer(data["phase"], f"{location}.phase")
    extras = {
        key: value for key, value in data.items() if key not in REQUIRED_INSTANCE_FIELDS
    }

    return CanonicalToolInstance(
        instrument_id=instrument_id,
        verb_id=verb_id,
        target_id=target_id,
        triplet_id=triplet_id,
        phase_id=phase_id,
        operator_id=_integer(data["operator"], f"{location}.operator"),
        bbox=_bbox(data["tool_bbox"], f"{location}.tool_bbox"),
        tracks=TrackIds(
            intraoperative=_integer(
                data["intraoperative_track"], f"{location}.intraoperative_track"
            ),
            intracorporeal=_integer(
                data["intracorporeal_track"], f"{location}.intracorporeal_track"
            ),
            visibility=_integer(
                data["visibility_track"], f"{location}.visibility_track"
            ),
        ),
        conditions=VisualConditions(
            visibility=_binary(data["visibility"], f"{location}.visibility"),
            visible=_binary(data["visible"], f"{location}.visible"),
            crowded=_binary(data["crowded"], f"{location}.crowded"),
            occluded=_binary(data["occluded"], f"{location}.occluded"),
            bleeding=_binary(data["bleeding"], f"{location}.bleeding"),
            smoke=_binary(data["smoke"], f"{location}.smoke"),
            blurred=_binary(data["blurred"], f"{location}.blurred"),
            undercoverage=_binary(data["undercoverage"], f"{location}.undercoverage"),
            reflection=_binary(data["reflection"], f"{location}.reflection"),
            stained_lens=_binary(data["stainedlens"], f"{location}.stainedlens"),
        ),
        mask=canonical_label_mask(
            instrument_id=instrument_id,
            verb_id=verb_id,
            target_id=target_id,
            triplet_id=triplet_id,
            phase_id=phase_id,
        ),
        score=_number(data["score"], f"{location}.score"),
        area=_number(data["area"], f"{location}.area"),
        is_crowd=_binary(data["iscrowd"], f"{location}.iscrowd"),
        extras=extras,
    )


def parse_annotation_file(
    path: str | Path,
    *,
    expected_split: DatasetSplit | None = None,
    manifest_source: str | None = None,
) -> CanonicalVideoAnnotation:
    """Parse one raw JSON file without modifying it or resolving media."""

    annotation_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationSchemaError(
            f"Cannot read valid JSON from {annotation_path}: {exc}"
        ) from exc

    root = _mapping(raw, "root")
    for required in ("info", "annotations", "categories", "video"):
        if required not in root:
            raise AnnotationSchemaError(f"root is missing required field: {required}")

    info = _mapping(root["info"], "info")
    video = _mapping(root["video"], "video")
    annotations = _mapping(root["annotations"], "annotations")
    categories = _mapping(root["categories"], "categories")
    dataset_name = info.get("dataset")
    if dataset_name != "CholecTrack20":
        raise AnnotationSchemaError(f"Unsupported dataset identity: {dataset_name!r}")

    video_name = video.get("name")
    if not isinstance(video_name, str) or not video_name.upper().startswith("VID"):
        raise AnnotationSchemaError("video.name must be a VID-prefixed string")
    video_id = video_name.upper()
    try:
        split = DatasetSplit.parse(str(video.get("split")))
    except ValueError as exc:
        raise AnnotationSchemaError(str(exc)) from exc
    if expected_split is not None and split is not expected_split:
        raise AnnotationSchemaError(
            f"{video_id} declares {split.value}, expected {expected_split.value}"
        )

    source_version = str(info.get("version", "UNKNOWN"))
    provenance = SourceProvenance(
        dataset="CholecTrack20",
        dataset_version=source_version,
        split=split,
        annotation_path=str(annotation_path),
        manifest_source=manifest_source,
    )

    parsed_frame_ids: dict[int, str] = {}
    for raw_frame_id in annotations:
        try:
            frame_id = int(raw_frame_id)
        except (TypeError, ValueError) as exc:
            raise AnnotationSchemaError(
                f"Invalid annotation frame key: {raw_frame_id!r}"
            ) from exc
        if frame_id < 0:
            raise AnnotationSchemaError(f"Frame ID must be non-negative: {frame_id}")
        if frame_id in parsed_frame_ids:
            raise AnnotationSchemaError(
                f"Frame keys {parsed_frame_ids[frame_id]!r} and {raw_frame_id!r} collide"
            )
        parsed_frame_ids[frame_id] = str(raw_frame_id)

    frames: list[CanonicalFrameAnnotation] = []
    for frame_id in sorted(parsed_frame_ids):
        raw_key = parsed_frame_ids[frame_id]
        raw_instances = annotations[raw_key]
        if not isinstance(raw_instances, list):
            raise AnnotationSchemaError(f"annotations.{raw_key} must be a list")
        instances = tuple(
            _parse_instance(item, f"annotations.{raw_key}[{index}]")
            for index, item in enumerate(raw_instances)
        )
        valid_phases = {item.phase_id for item in instances if item.mask.phase}
        if len(valid_phases) > 1:
            raise AnnotationSchemaError(
                f"annotations.{raw_key} contains inconsistent per-instance phase IDs"
            )
        frames.append(
            CanonicalFrameAnnotation(
                video_id=video_id,
                frame_id=frame_id,
                instances=instances,
                provenance=provenance,
            )
        )

    declared_frames = _integer(video.get("num_frames"), "video.num_frames")
    if declared_frames != len(frames):
        raise AnnotationSchemaError(
            f"video.num_frames={declared_frames} but annotations contains {len(frames)} frames"
        )

    return CanonicalVideoAnnotation(
        video_id=video_id,
        split=split,
        width=_integer(video.get("width"), "video.width"),
        height=_integer(video.get("height"), "video.height"),
        declared_annotated_frames=declared_frames,
        frames=tuple(frames),
        ontology_terms=_ontology_terms(
            categories,
            source_path=annotation_path,
            source_version=source_version,
        ),
        provenance=provenance,
    )
