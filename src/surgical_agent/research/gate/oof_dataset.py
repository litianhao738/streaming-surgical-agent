"""Video-grouped counterfactual records for one shared formal Benefit Gate."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from surgical_agent.artifacts.manifest import atomic_write_json, atomic_write_text
from surgical_agent.research.gate.contracts import (
    FORMAL_GATE_FEATURE_ORDER,
    REPAIR_SCOPE_ORDER,
    RepairScope,
)


@dataclass(frozen=True)
class CounterfactualGateRecord:
    """One Training-only H0 with observed verification benefit per scope."""

    sample_id: str
    video_id: str
    frame_id: int
    features: Mapping[str, float]
    benefit_by_scope: Mapping[RepairScope, int | None]
    tracker_artifact_sha256: str
    source_split: str = "Training"

    def __post_init__(self) -> None:
        if not self.sample_id or not self.video_id:
            raise ValueError("sample_id and video_id must be non-empty")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.source_split != "Training":
            raise ValueError("Gate counterfactuals must be Training-only")
        features = dict(self.features)
        if set(features) != set(FORMAL_GATE_FEATURE_ORDER):
            raise ValueError("counterfactual features do not match the formal schema")
        labels = dict(self.benefit_by_scope)
        if set(labels) != set(REPAIR_SCOPE_ORDER):
            raise ValueError("benefit labels must contain exactly the three scopes")
        if any(value not in {0, 1, None} for value in labels.values()):
            raise ValueError("benefit labels must be 0, 1 or null when unobserved")
        if (
            len(self.tracker_artifact_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.tracker_artifact_sha256)
        ):
            raise ValueError("tracker_artifact_sha256 must be lowercase SHA-256")
        object.__setattr__(
            self,
            "features",
            MappingProxyType({name: float(features[name]) for name in FORMAL_GATE_FEATURE_ORDER}),
        )
        object.__setattr__(self, "benefit_by_scope", MappingProxyType(labels))


def counterfactual_record_as_json(
    item: CounterfactualGateRecord,
) -> dict[str, object]:
    return {
        "schema_version": "formal_gate_counterfactual_v1",
        "sample_id": item.sample_id,
        "video_id": item.video_id,
        "frame_id": item.frame_id,
        "features": dict(item.features),
        "benefit_by_scope": dict(item.benefit_by_scope),
        "tracker_artifact_sha256": item.tracker_artifact_sha256,
        "source_split": item.source_split,
    }


def _record_from_json(raw: Mapping[str, object]) -> CounterfactualGateRecord:
    return CounterfactualGateRecord(
        sample_id=raw["sample_id"],  # type: ignore[arg-type]
        video_id=raw["video_id"],  # type: ignore[arg-type]
        frame_id=raw["frame_id"],  # type: ignore[arg-type]
        features=raw["features"],  # type: ignore[arg-type]
        benefit_by_scope=raw["benefit_by_scope"],  # type: ignore[arg-type]
        tracker_artifact_sha256=raw["tracker_artifact_sha256"],  # type: ignore[arg-type]
        source_split=raw.get("source_split", "Training"),  # type: ignore[arg-type]
    )


class CounterfactualCollectionStore:
    """Atomic per-observation ledger for resumable paid collection."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_observation(
        self,
        *,
        video_id: str,
        frame_id: int,
        safety_class: Literal["HARD_VALID", "HARD_INVALID"],
        record: CounterfactualGateRecord | None,
    ) -> Path:
        if safety_class == "HARD_VALID":
            if record is None:
                raise ValueError("hard-valid observation requires a record")
            if record.video_id != video_id or record.frame_id != frame_id:
                raise ValueError("counterfactual record identity mismatch")
        elif safety_class == "HARD_INVALID":
            if record is not None:
                raise ValueError("hard-invalid observation cannot carry a record")
        else:
            raise ValueError("unsupported counterfactual safety class")
        payload: dict[str, object] = {
            "schema_version": "formal_gate_collection_item_v1",
            "sample_id": f"{video_id}:{frame_id}",
            "video_id": video_id,
            "frame_id": frame_id,
            "safety_class": safety_class,
            "record": None if record is None else counterfactual_record_as_json(record),
        }
        destination = self.root / video_id / f"{frame_id}.json"
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError("resumed counterfactual observation conflicts with ledger")
            return destination
        return atomic_write_json(destination, payload)

    def observations(self) -> tuple[Mapping[str, object], ...]:
        values: list[Mapping[str, object]] = []
        for path in sorted(self.root.glob("*/*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("invalid counterfactual collection item") from exc
            if not isinstance(raw, Mapping):
                raise TypeError("counterfactual collection item must be an object")
            values.append(raw)
        return tuple(
            sorted(values, key=lambda value: (str(value["video_id"]), int(value["frame_id"])))
        )

    def records(self) -> tuple[CounterfactualGateRecord, ...]:
        result: list[CounterfactualGateRecord] = []
        for item in self.observations():
            record = item.get("record")
            if record is None:
                continue
            if not isinstance(record, Mapping):
                raise TypeError("counterfactual collection record must be an object")
            result.append(_record_from_json(record))
        return tuple(result)


@dataclass(frozen=True)
class GateOOFExample:
    """One fixed-shape T0 or T1 view with identical counterfactual labels."""

    sample_id: str
    video_id: str
    frame_id: int
    tracker_view: Literal["T0", "T1"]
    fold: int
    feature_order: tuple[str, ...]
    feature_values: tuple[float, ...]
    benefit_labels: tuple[int | None, ...]
    tracker_artifact_sha256: str

    def __post_init__(self) -> None:
        if self.tracker_view not in {"T0", "T1"}:
            raise ValueError("tracker_view must be T0 or T1")
        if self.fold < 0:
            raise ValueError("fold must be non-negative")
        if self.feature_order != FORMAL_GATE_FEATURE_ORDER:
            raise ValueError("feature_order must equal the formal Gate schema")
        if len(self.feature_values) != len(self.feature_order):
            raise ValueError("feature values and names must align")
        if len(self.benefit_labels) != len(REPAIR_SCOPE_ORDER):
            raise ValueError("benefit labels must align with the three scopes")

    def as_json(self) -> dict[str, object]:
        return {
            "schema_version": "formal_gate_oof_example_v1",
            "sample_id": self.sample_id,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "tracker_view": self.tracker_view,
            "fold": self.fold,
            "feature_order": list(self.feature_order),
            "feature_values": list(self.feature_values),
            "scope_order": list(REPAIR_SCOPE_ORDER),
            "benefit_labels": list(self.benefit_labels),
            "tracker_artifact_sha256": self.tracker_artifact_sha256,
        }


def deterministic_video_folds(video_ids: Sequence[str], fold_count: int) -> Mapping[str, int]:
    videos = tuple(sorted(set(video_ids)))
    if fold_count < 2 or fold_count > len(videos):
        raise ValueError("fold_count must be in 2..number of videos")
    return MappingProxyType({video_id: index % fold_count for index, video_id in enumerate(videos)})


def build_paired_oof_examples(
    records: Sequence[CounterfactualGateRecord],
    *,
    fold_count: int,
) -> tuple[GateOOFExample, ...]:
    values = tuple(records)
    if not values:
        raise ValueError("at least one counterfactual record is required")
    identities = tuple(item.sample_id for item in values)
    if len(set(identities)) != len(identities):
        raise ValueError("counterfactual sample IDs must be unique")
    folds = deterministic_video_folds([item.video_id for item in values], fold_count)
    tracker_fields = {
        "tracker_available",
        "tracker_conflict",
        "tracker_instrument_count",
    }
    examples: list[GateOOFExample] = []
    for item in sorted(values, key=lambda value: (value.video_id, value.frame_id)):
        for tracker_view in ("T0", "T1"):
            features = dict(item.features)
            if tracker_view == "T0":
                for name in tracker_fields:
                    features[name] = 0.0
            examples.append(
                GateOOFExample(
                    sample_id=f"{item.sample_id}:{tracker_view}",
                    video_id=item.video_id,
                    frame_id=item.frame_id,
                    tracker_view=tracker_view,
                    fold=folds[item.video_id],
                    feature_order=FORMAL_GATE_FEATURE_ORDER,
                    feature_values=tuple(features[name] for name in FORMAL_GATE_FEATURE_ORDER),
                    benefit_labels=tuple(
                        item.benefit_by_scope[scope] for scope in REPAIR_SCOPE_ORDER
                    ),
                    tracker_artifact_sha256=item.tracker_artifact_sha256,
                )
            )
    return tuple(examples)


def partition_oof_examples(
    examples: Sequence[GateOOFExample],
    *,
    calibration_fold: int,
) -> tuple[tuple[GateOOFExample, ...], tuple[GateOOFExample, ...]]:
    """Reserve whole videos for threshold calibration; never split T0/T1 pairs."""

    values = tuple(examples)
    folds = {item.fold for item in values}
    if len(folds) < 2 or calibration_fold not in folds:
        raise ValueError("calibration_fold must reserve one of at least two folds")
    by_observation: dict[tuple[str, int], list[GateOOFExample]] = {}
    for item in values:
        by_observation.setdefault((item.video_id, item.frame_id), []).append(item)
    if any(
        {item.tracker_view for item in pair} != {"T0", "T1"}
        or len(pair) != 2
        or len({item.fold for item in pair}) != 1
        for pair in by_observation.values()
    ):
        raise ValueError("every observation must contain one same-fold T0/T1 pair")
    training = tuple(item for item in values if item.fold != calibration_fold)
    calibration = tuple(item for item in values if item.fold == calibration_fold)
    training_videos = {item.video_id for item in training}
    calibration_videos = {item.video_id for item in calibration}
    if not training or not calibration or training_videos & calibration_videos:
        raise ValueError("Gate fit and calibration must use disjoint non-empty videos")
    return training, calibration


def load_counterfactual_records(path: str | Path) -> tuple[CounterfactualGateRecord, ...]:
    source = Path(path).expanduser().resolve()
    records: list[CounterfactualGateRecord] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            records.append(_record_from_json(raw))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid counterfactual JSONL line {line_number}") from exc
    return tuple(records)


def write_counterfactual_records(
    path: str | Path,
    records: Iterable[CounterfactualGateRecord],
) -> Path:
    destination = Path(path).expanduser().resolve()
    values = tuple(records)
    if not values:
        raise ValueError("at least one formal counterfactual record is required")
    identities = tuple(item.sample_id for item in values)
    if len(set(identities)) != len(identities):
        raise ValueError("counterfactual sample IDs must be unique")
    content = "".join(
        json.dumps(counterfactual_record_as_json(item), sort_keys=True, separators=(",", ":"))
        + "\n"
        for item in values
    )
    atomic_write_text(destination, content)
    return destination


def write_oof_examples(path: str | Path, examples: Iterable[GateOOFExample]) -> Path:
    destination = Path(path).expanduser().resolve()
    values = tuple(examples)
    content = "".join(
        json.dumps(item.as_json(), sort_keys=True, separators=(",", ":")) + "\n"
        for item in values
    )
    atomic_write_text(destination, content)
    return destination


def load_oof_examples(path: str | Path) -> tuple[GateOOFExample, ...]:
    source = Path(path).expanduser().resolve()
    examples: list[GateOOFExample] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if raw["schema_version"] != "formal_gate_oof_example_v1":
                raise ValueError("unsupported Gate OOF schema")
            if tuple(raw["scope_order"]) != REPAIR_SCOPE_ORDER:
                raise ValueError("scope order mismatch")
            examples.append(
                GateOOFExample(
                    sample_id=raw["sample_id"],
                    video_id=raw["video_id"],
                    frame_id=raw["frame_id"],
                    tracker_view=raw["tracker_view"],
                    fold=raw["fold"],
                    feature_order=tuple(raw["feature_order"]),
                    feature_values=tuple(raw["feature_values"]),
                    benefit_labels=tuple(raw["benefit_labels"]),
                    tracker_artifact_sha256=raw["tracker_artifact_sha256"],
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Gate OOF JSONL line {line_number}") from exc
    return tuple(examples)


__all__ = [
    "CounterfactualCollectionStore",
    "CounterfactualGateRecord",
    "GateOOFExample",
    "build_paired_oof_examples",
    "counterfactual_record_as_json",
    "deterministic_video_folds",
    "load_counterfactual_records",
    "load_oof_examples",
    "partition_oof_examples",
    "write_counterfactual_records",
    "write_oof_examples",
]
