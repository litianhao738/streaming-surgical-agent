"""Canonical CholecTrack20 records and Gold-free runtime boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from surgical_agent.data.constants import TASK_ID_BOUNDS


class DatasetSplit(str, Enum):
    """Official CholecTrack20 split names."""

    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"

    @classmethod
    def parse(cls, value: str) -> DatasetSplit:
        normalized = value.strip().lower()
        aliases = {
            "train": cls.TRAINING,
            "training": cls.TRAINING,
            "val": cls.VALIDATION,
            "validation": cls.VALIDATION,
            "test": cls.TESTING,
            "testing": cls.TESTING,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported dataset split: {value!r}") from exc


@dataclass(frozen=True)
class SourceProvenance:
    """Origin of one canonical annotation record."""

    dataset: str
    dataset_version: str
    split: DatasetSplit
    annotation_path: str
    manifest_source: str | None = None


@dataclass(frozen=True)
class OntologyTerm:
    """One numeric ontology term with explicit evidence provenance."""

    task: str
    numeric_id: int
    canonical_name: str | None
    component_relation: Mapping[str, int] | None
    source: str
    source_version: str
    source_location: str
    verification_method: str
    verified: bool
    notes: str = ""


@dataclass(frozen=True)
class BoundingBox:
    """Raw normalized TLWH box, including release-native boundary values."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bounding-box values must be finite")

    @property
    def is_all_negative_one(self) -> bool:
        """Return whether the raw release uses the unresolved-looking -1 tuple."""

        return (self.x, self.y, self.width, self.height) == (-1.0, -1.0, -1.0, -1.0)

    @property
    def has_positive_extent(self) -> bool:
        return self.width > 0 and self.height > 0

    @property
    def is_inside_unit_frame(self) -> bool:
        return (
            self.x >= 0
            and self.y >= 0
            and self.x + self.width <= 1
            and self.y + self.height <= 1
        )


@dataclass(frozen=True)
class LabelMask:
    """Per-instance task availability from the canonical mask implementation."""

    instrument: bool
    verb: bool
    target: bool
    ivt: bool
    phase: bool


@dataclass(frozen=True)
class TrackIds:
    """Raw multi-perspective track identifiers."""

    intraoperative: int
    intracorporeal: int
    visibility: int


@dataclass(frozen=True)
class VisualConditions:
    """Observed binary visual-state attributes for one tool instance."""

    visibility: bool
    visible: bool
    crowded: bool
    occluded: bool
    bleeding: bool
    smoke: bool
    blurred: bool
    undercoverage: bool
    reflection: bool
    stained_lens: bool


@dataclass(frozen=True)
class CanonicalToolInstance:
    """One raw tool record normalized without inventing missing semantics."""

    instrument_id: int
    verb_id: int
    target_id: int
    triplet_id: int
    phase_id: int
    operator_id: int
    bbox: BoundingBox
    tracks: TrackIds
    conditions: VisualConditions
    mask: LabelMask
    score: float
    area: float
    is_crowd: bool
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalFrameAnnotation:
    """All canonical tool instances for one annotated source frame."""

    video_id: str
    frame_id: int
    instances: tuple[CanonicalToolInstance, ...]
    provenance: SourceProvenance


@dataclass(frozen=True)
class CanonicalVideoAnnotation:
    """Validated canonical representation of one raw JSON annotation file."""

    video_id: str
    split: DatasetSplit
    width: int
    height: int
    declared_annotated_frames: int
    frames: tuple[CanonicalFrameAnnotation, ...]
    ontology_terms: tuple[OntologyTerm, ...]
    provenance: SourceProvenance

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(frame.frame_id for frame in self.frames)


@dataclass(frozen=True)
class InferenceSample:
    """Runtime input containing visual references and no annotation labels."""

    video_id: str
    target_frame_id: int
    causal_frame_ids: tuple[int, ...]
    media_refs: tuple[str, ...]
    source_split: DatasetSplit
    alignment_version: str

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must not be empty")
        if self.target_frame_id < 0:
            raise ValueError("target_frame_id must be non-negative")
        if not self.causal_frame_ids:
            raise ValueError("At least one causal frame is required")
        if tuple(sorted(set(self.causal_frame_ids))) != self.causal_frame_ids:
            raise ValueError("causal_frame_ids must be unique and increasing")
        if self.causal_frame_ids[-1] != self.target_frame_id:
            raise ValueError("The causal window must end at target_frame_id")
        if len(self.media_refs) != len(self.causal_frame_ids):
            raise ValueError("Each causal frame must have exactly one media reference")


@dataclass(frozen=True)
class EvaluationInstanceTarget:
    """GT-bearing instance record restricted to the evaluation branch."""

    instrument_id: int
    verb_id: int
    target_id: int
    triplet_id: int
    phase_id: int
    operator_id: int
    bbox: BoundingBox
    tracks: TrackIds
    mask: LabelMask


@dataclass(frozen=True)
class EvaluationTarget:
    """Ground truth consumed only by the isolated evaluation path."""

    video_id: str
    frame_id: int
    instances: tuple[EvaluationInstanceTarget, ...]
    instance_supervision_available: bool = True
    source_granularity: str = "instance"


@dataclass(frozen=True)
class FrameTaskMask:
    """Availability of complete frame-level supervision for each task."""

    instrument: bool
    verb: bool
    target: bool
    ivt: bool
    phase: bool


@dataclass(frozen=True)
class FrameSupervisionTarget:
    """Evaluation-only frame labels with explicit granularity and task masks."""

    video_id: str
    frame_id: int
    instrument_ids: tuple[int, ...]
    verb_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    triplet_ids: tuple[int, ...]
    phase_id: int | None
    mask: FrameTaskMask
    source_granularity: str
    source: str

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        for name, values in (
            ("instrument_ids", self.instrument_ids),
            ("verb_ids", self.verb_ids),
            ("target_ids", self.target_ids),
            ("triplet_ids", self.triplet_ids),
        ):
            if tuple(sorted(set(values))) != values or any(value < 0 for value in values):
                raise ValueError(f"{name} must contain sorted unique non-negative IDs")
        fields_to_tasks = {
            "instrument_ids": (self.instrument_ids, "instrument"),
            "verb_ids": (self.verb_ids, "verb"),
            "target_ids": (self.target_ids, "target"),
            "triplet_ids": (self.triplet_ids, "ivt"),
        }
        for name, (values, task) in fields_to_tasks.items():
            lower, upper = TASK_ID_BOUNDS[task]
            if any(not lower <= value <= upper for value in values):
                raise ValueError(f"{name} contains an ID outside {lower}..{upper}")
        if self.mask.phase != (self.phase_id is not None):
            raise ValueError("phase mask must exactly match phase_id availability")
        if self.phase_id is not None:
            lower, upper = TASK_ID_BOUNDS["phase"]
            if not lower <= self.phase_id <= upper:
                raise ValueError(f"phase_id is outside {lower}..{upper}")
        if self.source_granularity not in {"phase_only", "frame_multilabel"}:
            raise ValueError(f"Unsupported frame target granularity: {self.source_granularity}")
