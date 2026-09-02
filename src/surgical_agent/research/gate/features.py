"""Deterministic, gold-free features for the demo Learned Benefit Gate."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import TASK_NAMES, JointPerceptionResult
from surgical_agent.research.signals.contracts import EvidenceProfile
from surgical_agent.research.signals.frame_evidence import load_ivt_components

_TASK_FEATURE_SUFFIXES = (
    "selected_count",
    "selected_min",
    "selected_mean",
    "top1",
    "margin",
)
BASE_GATE_FEATURE_NAMES = tuple(
    f"{task}_{suffix}" for task in TASK_NAMES for suffix in _TASK_FEATURE_SUFFIXES
) + (
    "global_selected_min",
    "global_selected_mean",
    "global_min_margin",
    "global_mean_margin",
    "ivt_closure_conflict",
    "triplet_compatibility_conflict",
    "phase_triplet_conflict",
    "phase_transition_conflict",
    "temporal_jump",
)
TRACKER_GATE_FEATURE_NAMES = (
    "tracker_gpt_instrument_conflict",
    "tracker_support_count",
    "tracker_missing_tool_conflict",
    "tracker_new_tool_count",
    "tracker_disappeared_tool_count",
    "track_continuity",
)
NO_TRACKER_FEATURE_ORDER = BASE_GATE_FEATURE_NAMES
WITH_TRACKER_FEATURE_ORDER = BASE_GATE_FEATURE_NAMES + TRACKER_GATE_FEATURE_NAMES
ALL_DEMO_GATE_FEATURE_NAMES = frozenset(WITH_TRACKER_FEATURE_ORDER)


@dataclass(frozen=True)
class GateFeatureVector:
    feature_order: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.feature_order or len(self.feature_order) != len(self.values):
            raise ValueError("Gate feature names and values must align")
        if len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("Gate feature names must be unique")
        if any(name not in ALL_DEMO_GATE_FEATURE_NAMES for name in self.feature_order):
            raise ValueError("Gate feature order contains an unknown feature")
        if any(not math.isfinite(float(value)) for value in self.values):
            raise ValueError("Gate features must be finite")
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))

    def as_mapping(self) -> Mapping[str, float]:
        return MappingProxyType(dict(zip(self.feature_order, self.values, strict=True)))


def _selected(prediction: InitialPrediction, task: str) -> tuple[int, ...]:
    return {
        "instrument": prediction.instrument_ids,
        "verb": prediction.verb_ids,
        "target": prediction.target_ids,
        "ivt": prediction.triplet_ids,
        "phase": (prediction.phase_id,),
    }[task]


def _value(profile: EvidenceProfile, task: str, name: str) -> float:
    evidence = profile.task_values[task][name]
    return (
        float(evidence.value)
        if evidence.available and evidence.value is not None
        else 0.0
    )


def _phase_triplet_conflict(
    prediction: InitialPrediction,
    phase_allowed_ivt: Mapping[int, tuple[int, ...]] | None,
) -> bool:
    if phase_allowed_ivt is None:
        return False
    allowed = phase_allowed_ivt.get(prediction.phase_id, ())
    return any(ivt_id not in allowed for ivt_id in prediction.triplet_ids)


def _track_frames(snapshot: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    if snapshot.get("status") != "AVAILABLE":
        return ()
    frames = snapshot.get("frames", ())
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return ()
    return tuple(frame for frame in frames if isinstance(frame, Mapping))


def _track_ids(frame: Mapping[str, object]) -> tuple[str, ...]:
    tracks = frame.get("tracks", ())
    if not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
        return ()
    return tuple(
        str(track["track_id"])
        for track in tracks
        if isinstance(track, Mapping) and isinstance(track.get("track_id"), str)
    )


def _instrument_ids(frame: Mapping[str, object]) -> tuple[int, ...]:
    tracks = frame.get("tracks", ())
    if not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
        return ()
    return tuple(
        sorted(
            {
                int(track["instrument_id"])
                for track in tracks
                if isinstance(track, Mapping)
                and isinstance(track.get("instrument_id"), int)
                and not isinstance(track.get("instrument_id"), bool)
            }
        )
    )


def extract_gate_features(
    signals: EvidenceProfile,
    *,
    context: PerceptionContext,
    perception_result: JointPerceptionResult,
    include_tracker: bool,
    phase_allowed_ivt: Mapping[int, tuple[int, ...]] | None = None,
) -> GateFeatureVector:
    """Extract one frozen-order vector without reading any supervision."""

    if not isinstance(signals, EvidenceProfile):
        raise TypeError("signals must be an EvidenceProfile")
    if not isinstance(context, PerceptionContext):
        raise TypeError("context must be a PerceptionContext")
    if not isinstance(perception_result, JointPerceptionResult):
        raise TypeError("perception_result must be a JointPerceptionResult")
    prediction = perception_result.prediction
    evidence = perception_result.raw_evidence
    values: dict[str, float] = {}
    all_selected_confidences: list[float] = []
    all_margins: list[float] = []
    for task in TASK_NAMES:
        rankings = evidence.ranked_candidates[task]
        by_id = {candidate.class_id: candidate.confidence for candidate in rankings}
        selected_confidences = [
            float(by_id.get(class_id, 0.0)) for class_id in _selected(prediction, task)
        ]
        all_selected_confidences.extend(selected_confidences)
        values[f"{task}_selected_count"] = float(len(_selected(prediction, task)))
        values[f"{task}_selected_min"] = (
            min(selected_confidences) if selected_confidences else 0.0
        )
        values[f"{task}_selected_mean"] = (
            sum(selected_confidences) / len(selected_confidences)
            if selected_confidences
            else 0.0
        )
        values[f"{task}_top1"] = float(rankings[0].confidence) if rankings else 0.0
        values[f"{task}_margin"] = (
            float(rankings[0].confidence - rankings[1].confidence)
            if len(rankings) > 1
            else values[f"{task}_top1"]
        )
        all_margins.append(values[f"{task}_margin"])
    values["global_selected_min"] = (
        min(all_selected_confidences) if all_selected_confidences else 0.0
    )
    values["global_selected_mean"] = (
        sum(all_selected_confidences) / len(all_selected_confidences)
        if all_selected_confidences
        else 0.0
    )
    values["global_min_margin"] = min(all_margins) if all_margins else 0.0
    values["global_mean_margin"] = (
        sum(all_margins) / len(all_margins) if all_margins else 0.0
    )

    components = load_ivt_components()
    represented = {"instrument": set(), "verb": set(), "target": set()}
    closure = False
    for ivt_id in prediction.triplet_ids:
        instrument_id, verb_id, target_id = components[ivt_id]
        for task, class_id in (
            ("instrument", instrument_id),
            ("verb", verb_id),
            ("target", target_id),
        ):
            represented[task].add(class_id)
            if class_id not in _selected(prediction, task):
                closure = True
    compatibility = any(
        set(_selected(prediction, task)) - represented[task] for task in represented
    )
    values["ivt_closure_conflict"] = float(
        closure or _value(signals, "ivt", "ivt_internal_conflict") > 0.0
    )
    values["triplet_compatibility_conflict"] = float(compatibility)
    values["phase_triplet_conflict"] = float(
        _phase_triplet_conflict(prediction, phase_allowed_ivt)
    )
    values["phase_transition_conflict"] = float(
        _value(signals, "phase", "phase_change_anomaly") > 0.0
    )
    values["temporal_jump"] = float(
        max(
            _value(signals, task, "temporal_set_change")
            for task in ("instrument", "verb", "target", "ivt")
        )
        >= 0.8
    )

    if include_tracker:
        frames = _track_frames(context.track_snapshot)
        current = frames[-1] if frames else {}
        previous = frames[-2] if len(frames) > 1 else {}
        current_instruments = set(_instrument_ids(current))
        previous_instruments = set(_instrument_ids(previous))
        predicted_instruments = set(prediction.instrument_ids)
        current_track_ids = set(_track_ids(current))
        previous_track_ids = set(_track_ids(previous))
        values.update(
            tracker_gpt_instrument_conflict=float(
                current_instruments != predicted_instruments
            ),
            tracker_support_count=float(
                len(current_instruments & predicted_instruments)
            ),
            tracker_missing_tool_conflict=float(
                bool(current_instruments - predicted_instruments)
            ),
            tracker_new_tool_count=float(
                len(current_instruments - previous_instruments)
            ),
            tracker_disappeared_tool_count=float(
                len(previous_instruments - current_instruments)
            ),
            track_continuity=(
                len(current_track_ids & previous_track_ids) / len(current_track_ids)
                if current_track_ids
                else 0.0
            ),
        )
    order = WITH_TRACKER_FEATURE_ORDER if include_tracker else NO_TRACKER_FEATURE_ORDER
    return GateFeatureVector(order, tuple(values[name] for name in order))


__all__ = [
    "ALL_DEMO_GATE_FEATURE_NAMES",
    "NO_TRACKER_FEATURE_ORDER",
    "WITH_TRACKER_FEATURE_ORDER",
    "GateFeatureVector",
    "extract_gate_features",
]
