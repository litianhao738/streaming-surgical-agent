"""Deterministic, no-LLM transforms from joint perception to evidence."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from importlib.resources import files
from types import MappingProxyType

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import (
    JointPerceptionResult,
    PerceptionEvidence,
    RankedCandidate,
)
from surgical_agent.research.signals.contracts import (
    EvidenceProfile,
    EvidenceValue,
    PhaseTransitionGraph,
    unavailable,
)

IVT_MAP_SOURCE_COMMIT = "c0a565cef6e09fc090b2b68c3b967d4140d3b91b"
IVT_MAP_SOURCE_URL = (
    "https://raw.githubusercontent.com/CAMMA-public/ivtmetrics/"
    "c0a565cef6e09fc090b2b68c3b967d4140d3b91b/ivtmetrics/maps.txt"
)
IVT_MAP_SOURCE_SHA256 = "e031ce8646491ddabfd37a23804e46cf4775d257b569deb7702eafd7e9119092"


def load_ivt_components() -> Mapping[int, tuple[int, int, int]]:
    """Load and validate the pinned CAMMA IVT-to-component map."""

    resource = files("surgical_agent.research.signals.resources").joinpath(
        "ivt_components_v1.csv"
    )
    with resource.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0] != ["ivt", "instrument", "verb", "target"]:
        raise ValueError("IVT component map must have the expected CSV header")
    components: dict[int, tuple[int, int, int]] = {}
    for row in rows[1:]:
        if len(row) != 4:
            raise ValueError("IVT component map rows must contain four numeric values")
        try:
            ivt_id, instrument_id, verb_id, target_id = (int(value) for value in row)
        except ValueError as error:
            raise ValueError("IVT component map rows must contain integers") from error
        if ivt_id in components:
            raise ValueError("IVT component map IDs must occur exactly once")
        _validate_component_id("instrument", instrument_id)
        _validate_component_id("verb", verb_id)
        _validate_component_id("target", target_id)
        components[ivt_id] = (instrument_id, verb_id, target_id)
    if set(components) != set(range(100)):
        raise ValueError("IVT component map must contain IDs 0 through 99 exactly once")
    return MappingProxyType(components)


def _validate_component_id(task: str, value: int) -> None:
    lower, upper = TASK_ID_BOUNDS[task]
    if not lower <= value <= upper:
        raise ValueError(f"{task} component ID is outside {lower}..{upper}")


class FrameEvidenceSignalExtractor:
    """Derive the approved evidence-frame-v1 signals with no external calls."""

    def __init__(self, *, phase_transition_graph: PhaseTransitionGraph | None = None) -> None:
        if phase_transition_graph is not None and not isinstance(
            phase_transition_graph, PhaseTransitionGraph
        ):
            raise TypeError("phase_transition_graph must be a PhaseTransitionGraph")
        self._phase_transition_graph = phase_transition_graph
        self._ivt_components = load_ivt_components()

    def extract(
        self,
        context: PerceptionContext,
        result: JointPerceptionResult,
    ) -> EvidenceProfile:
        if not isinstance(context, PerceptionContext):
            raise TypeError("context must be a PerceptionContext")
        if not isinstance(result, JointPerceptionResult):
            raise TypeError("result must be a JointPerceptionResult")
        frame_id = context.sample.target_frame_id
        source_max_frame_id = result.raw_evidence.source_max_frame_id
        if source_max_frame_id > frame_id:
            raise ValueError("raw evidence provenance must not reference a future frame")

        prediction = result.prediction
        raw_evidence = result.raw_evidence
        _validate_evidence_score_semantics(prediction.score_semantics, raw_evidence)
        task_values: dict[str, dict[str, EvidenceValue]] = {
            task: {
                "candidate_ambiguity": _ambiguity(
                    raw_evidence.ranked_candidates[task], source_max_frame_id
                ),
                "self_reported_uncertainty": _self_reported_uncertainty(
                    raw_evidence.self_reported_confidence[task], source_max_frame_id
                ),
            }
            for task in raw_evidence.ranked_candidates
        }

        ivt_conflict = _ivt_internal_conflict(
            prediction.instrument_ids,
            prediction.verb_ids,
            prediction.target_ids,
            prediction.triplet_ids,
            self._ivt_components,
            source_max_frame_id,
        )
        for task in ("instrument", "verb", "target", "ivt"):
            task_values[task]["ivt_internal_conflict"] = ivt_conflict

        prior = context.prior_finalized_prediction
        for task, current_ids, previous_ids in (
            ("instrument", prediction.instrument_ids, None if prior is None else prior.instrument_ids),
            ("verb", prediction.verb_ids, None if prior is None else prior.verb_ids),
            ("target", prediction.target_ids, None if prior is None else prior.target_ids),
            ("ivt", prediction.triplet_ids, None if prior is None else prior.triplet_ids),
        ):
            task_values[task]["temporal_set_change"] = _temporal_set_change(
                current_ids, previous_ids, source_max_frame_id
            )
        task_values["phase"]["phase_change_anomaly"] = _phase_change_anomaly(
            previous_phase=None if prior is None else prior.phase_id,
            current_phase=prediction.phase_id,
            graph=self._phase_transition_graph,
            source_max_frame_id=source_max_frame_id,
        )
        return EvidenceProfile(
            video_id=context.sample.video_id,
            frame_id=frame_id,
            task_values=task_values,
            global_values={"ivt_internal_conflict": ivt_conflict},
        )


def _validate_evidence_score_semantics(
    score_semantics: str,
    raw_evidence: PerceptionEvidence,
) -> None:
    if score_semantics == "uncalibrated_rank_v1":
        return
    if (
        raw_evidence.source == "local_smoke"
        and all(not candidates for candidates in raw_evidence.ranked_candidates.values())
        and all(
            confidence is None
            for confidence in raw_evidence.self_reported_confidence.values()
        )
    ):
        return
    raise ValueError(
        "rank-bearing evidence requires score_semantics=uncalibrated_rank_v1"
    )


def _ambiguity(
    candidates: tuple[RankedCandidate, ...], source_max_frame_id: int
) -> EvidenceValue:
    if len(candidates) < 2:
        return unavailable("joint_rank_margin", source_max_frame_id)
    margin = candidates[0].score - candidates[1].score
    return EvidenceValue(
        1.0 - min(max(margin, 0.0), 1.0),
        True,
        "joint_rank_margin",
        source_max_frame_id,
    )


def _ivt_internal_conflict(
    instrument_ids: tuple[int, ...],
    verb_ids: tuple[int, ...],
    target_ids: tuple[int, ...],
    triplet_ids: tuple[int, ...],
    components: Mapping[int, tuple[int, int, int]],
    source_max_frame_id: int,
) -> EvidenceValue:
    if not triplet_ids:
        return unavailable("ivt_component_map_v1", source_max_frame_id)
    inconsistent = 0
    selected_instruments = set(instrument_ids)
    selected_verbs = set(verb_ids)
    selected_targets = set(target_ids)
    for triplet_id in triplet_ids:
        try:
            instrument_id, verb_id, target_id = components[triplet_id]
        except KeyError as error:
            raise ValueError(f"selected IVT ID {triplet_id} has no component-map entry") from error
        if (
            instrument_id not in selected_instruments
            or verb_id not in selected_verbs
            or target_id not in selected_targets
        ):
            inconsistent += 1
    return EvidenceValue(
        inconsistent / len(triplet_ids),
        True,
        "ivt_component_map_v1",
        source_max_frame_id,
    )


def _temporal_set_change(
    current_ids: tuple[int, ...],
    previous_ids: tuple[int, ...] | None,
    source_max_frame_id: int,
) -> EvidenceValue:
    if previous_ids is None:
        return unavailable("finalized_prior_jaccard", source_max_frame_id)
    union = set(current_ids) | set(previous_ids)
    if not union:
        distance = 0.0
    else:
        distance = 1.0 - len(set(current_ids) & set(previous_ids)) / len(union)
    return EvidenceValue(distance, True, "finalized_prior_jaccard", source_max_frame_id)


def _phase_change_anomaly(
    *,
    previous_phase: int | None,
    current_phase: int,
    graph: PhaseTransitionGraph | None,
    source_max_frame_id: int,
) -> EvidenceValue:
    if previous_phase is None:
        return unavailable("frozen_phase_transition_graph", source_max_frame_id)
    if graph is None:
        return unavailable("frozen_phase_transition_graph", source_max_frame_id)
    return EvidenceValue(
        0.0 if graph.allows(previous_phase, current_phase) else 1.0,
        True,
        "frozen_phase_transition_graph",
        source_max_frame_id,
    )


def _self_reported_uncertainty(
    confidence: float | None, source_max_frame_id: int
) -> EvidenceValue:
    if confidence is None:
        return unavailable("joint_self_reported_confidence", source_max_frame_id)
    return EvidenceValue(
        1.0 - confidence,
        True,
        "joint_self_reported_confidence",
        source_max_frame_id,
    )
