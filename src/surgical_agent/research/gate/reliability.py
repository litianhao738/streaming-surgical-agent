"""Deterministic reliability Gate over parsed, gold-free perception output."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import (
    FIELD_UNCERTAINTY_PATHS,
    JointPerceptionResult,
)
from surgical_agent.research.reliability.state import (
    CANONICAL_TASK_ORDER,
    TASK_PATHS,
    GateFinding,
)
from surgical_agent.research.signals.contracts import (
    EvidenceProfile,
    PhaseTransitionGraph,
)
from surgical_agent.research.signals.frame_evidence import load_ivt_components
from surgical_agent.systems.pipeline import GateDecision


def _selected(prediction: InitialPrediction, task: str) -> tuple[int, ...]:
    if task == "instrument":
        return tuple(prediction.instrument_ids)
    if task == "verb":
        return tuple(prediction.verb_ids)
    if task == "target":
        return tuple(prediction.target_ids)
    if task == "ivt":
        return tuple(prediction.triplet_ids)
    if task == "phase":
        return (prediction.phase_id,)
    raise KeyError(task)


def _jaccard_distance(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    union = set(first) | set(second)
    if not union:
        return 0.0
    return 1.0 - len(set(first) & set(second)) / len(union)


def _finding_key(finding: GateFinding) -> tuple[int, str, tuple[int, ...]]:
    task = FIELD_UNCERTAINTY_PATHS[finding.path]
    return (
        CANONICAL_TASK_ORDER.index(task),
        finding.reason,
        finding.alternative_ids,
    )


class ReliabilityGatePolicy:
    """Route only field-scoped deterministic conflicts to verification."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.60,
        temporal_jaccard_threshold: float = 0.80,
        phase_transition_graph: PhaseTransitionGraph | None = None,
        phase_allowed_ivt: Mapping[int, tuple[int, ...]] | None = None,
        ivt_components: Mapping[int, tuple[int, int, int]] | None = None,
        use_tracker: bool = False,
    ) -> None:
        for name, value in (
            ("confidence_threshold", confidence_threshold),
            ("temporal_jaccard_threshold", temporal_jaccard_threshold),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite in [0, 1]")
        if phase_transition_graph is not None and not isinstance(
            phase_transition_graph, PhaseTransitionGraph
        ):
            raise TypeError("phase_transition_graph must be a PhaseTransitionGraph")
        self.confidence_threshold = float(confidence_threshold)
        self.temporal_jaccard_threshold = float(temporal_jaccard_threshold)
        self.phase_transition_graph = phase_transition_graph
        self.phase_allowed_ivt = self._normalize_phase_allowed_ivt(
            phase_allowed_ivt
        )
        self.ivt_components = self._normalize_ivt_components(ivt_components)
        if type(use_tracker) is not bool:
            raise TypeError("use_tracker must be a boolean")
        self.use_tracker = use_tracker

    @staticmethod
    def _normalize_phase_allowed_ivt(
        value: Mapping[int, tuple[int, ...]] | None,
    ) -> Mapping[int, tuple[int, ...]] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("phase_allowed_ivt must be a mapping")
        phase_lower, phase_upper = TASK_ID_BOUNDS["phase"]
        ivt_lower, ivt_upper = TASK_ID_BOUNDS["ivt"]
        normalized: dict[int, tuple[int, ...]] = {}
        for phase_id, raw_ids in value.items():
            if (
                not isinstance(phase_id, int)
                or isinstance(phase_id, bool)
                or not phase_lower <= phase_id <= phase_upper
            ):
                raise ValueError("phase_allowed_ivt phase ID is outside the ontology")
            ids = tuple(raw_ids)
            if tuple(sorted(set(ids))) != ids or any(
                not isinstance(ivt_id, int)
                or isinstance(ivt_id, bool)
                or not ivt_lower <= ivt_id <= ivt_upper
                for ivt_id in ids
            ):
                raise ValueError(
                    "phase_allowed_ivt values must be sorted unique IVT IDs"
                )
            normalized[phase_id] = ids
        return MappingProxyType(normalized)

    @staticmethod
    def _normalize_ivt_components(
        value: Mapping[int, tuple[int, int, int]] | None,
    ) -> Mapping[int, tuple[int, int, int]]:
        source = load_ivt_components() if value is None else value
        if not isinstance(source, Mapping):
            raise TypeError("ivt_components must be a mapping")
        if set(source) != set(range(TASK_ID_BOUNDS["ivt"][1] + 1)):
            raise ValueError("ivt_components must contain IVT IDs 0 through 99")
        normalized: dict[int, tuple[int, int, int]] = {}
        component_tasks = ("instrument", "verb", "target")
        for ivt_id in range(100):
            raw_components = tuple(source[ivt_id])
            if len(raw_components) != 3:
                raise ValueError("ivt_components values must contain I, V and T")
            for task, component_id in zip(component_tasks, raw_components):
                lower, upper = TASK_ID_BOUNDS[task]
                if (
                    not isinstance(component_id, int)
                    or isinstance(component_id, bool)
                    or not lower <= component_id <= upper
                ):
                    raise ValueError(
                        f"ivt_components {task} ID is outside the ontology"
                    )
            normalized[ivt_id] = (
                raw_components[0],
                raw_components[1],
                raw_components[2],
            )
        return MappingProxyType(normalized)

    def decide(
        self,
        signals: EvidenceProfile,
        *,
        context: PerceptionContext,
        perception_result: JointPerceptionResult,
    ) -> GateDecision:
        if not isinstance(signals, EvidenceProfile):
            raise TypeError("signals must be an EvidenceProfile")
        if not isinstance(context, PerceptionContext):
            raise TypeError("context must be a PerceptionContext")
        if not isinstance(perception_result, JointPerceptionResult):
            raise TypeError("perception_result must be a JointPerceptionResult")

        prediction = perception_result.prediction
        findings: list[GateFinding] = []
        invalid_tasks = self._ontology_findings(prediction, findings)
        if "ivt" not in invalid_tasks:
            self._ivt_findings(prediction, findings)
            self._phase_ivt_findings(prediction, findings)
        self._temporal_findings(prediction, context, findings)
        self._tracker_findings(prediction, context, findings)
        confidence_floor = self._confidence_findings(
            prediction,
            perception_result,
            findings,
        )
        findings.extend(
            GateFinding(item.path, item.reason, item.alternative_ids)
            for item in perception_result.raw_evidence.field_uncertainties
        )
        ordered = tuple(sorted(set(findings), key=_finding_key))
        flagged = tuple(
            task
            for task in CANONICAL_TASK_ORDER
            if any(FIELD_UNCERTAINTY_PATHS[item.path] == task for item in ordered)
        )
        if not ordered:
            return GateDecision(
                action="ACCEPT",
                scope=None,
                reason="RELIABILITY_CHECKS_PASSED",
                selected_confidence_floor=confidence_floor,
            )
        return GateDecision(
            action="VERIFY",
            scope="targeted",
            reason="RELIABILITY_FINDINGS",
            findings=ordered,
            flagged_fields=flagged,
            selected_confidence_floor=confidence_floor,
        )

    @staticmethod
    def _ontology_findings(
        prediction: InitialPrediction,
        findings: list[GateFinding],
    ) -> set[str]:
        invalid: set[str] = set()
        for task in CANONICAL_TASK_ORDER:
            lower, upper = TASK_ID_BOUNDS[task]
            values = _selected(prediction, task)
            if (
                tuple(sorted(set(values))) != values
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not lower <= value <= upper
                    for value in values
                )
            ):
                invalid.add(task)
                findings.append(GateFinding(TASK_PATHS[task], "ONTOLOGY_VIOLATION"))
        return invalid

    def _ivt_findings(
        self,
        prediction: InitialPrediction,
        findings: list[GateFinding],
    ) -> None:
        if not prediction.triplet_ids:
            return
        selected_components = {
            "instrument": set(prediction.instrument_ids),
            "verb": set(prediction.verb_ids),
            "target": set(prediction.target_ids),
        }
        represented = {task: set() for task in selected_components}
        for ivt_id in prediction.triplet_ids:
            components = self.ivt_components.get(ivt_id)
            if components is None:
                findings.append(GateFinding(TASK_PATHS["ivt"], "ONTOLOGY_VIOLATION"))
                continue
            for task, component in zip(selected_components, components):
                represented[task].add(component)
                if component not in selected_components[task]:
                    findings.append(GateFinding(TASK_PATHS[task], "IVT_CLOSURE"))
                    findings.append(GateFinding(TASK_PATHS["ivt"], "IVT_CLOSURE"))
        for task, values in selected_components.items():
            if values - represented[task]:
                findings.append(
                    GateFinding(TASK_PATHS[task], "TRIPLET_COMPATIBILITY")
                )
                findings.append(
                    GateFinding(TASK_PATHS["ivt"], "TRIPLET_COMPATIBILITY")
                )

    def _phase_ivt_findings(
        self,
        prediction: InitialPrediction,
        findings: list[GateFinding],
    ) -> None:
        if self.phase_allowed_ivt is None or not prediction.triplet_ids:
            return
        allowed = set(self.phase_allowed_ivt.get(prediction.phase_id, ()))
        if not set(prediction.triplet_ids).issubset(allowed):
            findings.append(GateFinding(TASK_PATHS["ivt"], "PHASE_TRIPLET_CONFLICT"))
            findings.append(GateFinding(TASK_PATHS["phase"], "PHASE_TRIPLET_CONFLICT"))

    def _temporal_findings(
        self,
        prediction: InitialPrediction,
        context: PerceptionContext,
        findings: list[GateFinding],
    ) -> None:
        prior = context.prior_finalized_prediction
        if prior is None:
            return
        previous = {
            "instrument": prior.instrument_ids,
            "verb": prior.verb_ids,
            "target": prior.target_ids,
            "ivt": prior.triplet_ids,
        }
        for task in ("instrument", "verb", "target", "ivt"):
            if _jaccard_distance(_selected(prediction, task), previous[task]) >= (
                self.temporal_jaccard_threshold
            ):
                findings.append(GateFinding(TASK_PATHS[task], "TEMPORAL_JUMP"))
        if (
            self.phase_transition_graph is not None
            and not self.phase_transition_graph.allows(prior.phase_id, prediction.phase_id)
        ):
            findings.append(GateFinding(TASK_PATHS["phase"], "TEMPORAL_JUMP"))

    def _confidence_findings(
        self,
        prediction: InitialPrediction,
        result: JointPerceptionResult,
        findings: list[GateFinding],
    ) -> float | None:
        selected_confidences: list[float] = []
        for task in CANONICAL_TASK_ORDER:
            rankings = result.raw_evidence.ranked_candidates[task]
            by_id = {candidate.class_id: candidate.confidence for candidate in rankings}
            selected_ids = _selected(prediction, task)
            alternatives = tuple(
                candidate.class_id
                for candidate in rankings
                if candidate.class_id not in selected_ids
            )
            for selected_id in selected_ids:
                confidence = by_id.get(selected_id)
                if confidence is None or confidence < self.confidence_threshold:
                    findings.append(
                        GateFinding(TASK_PATHS[task], "LOW_CONFIDENCE", alternatives)
                    )
                if confidence is not None:
                    selected_confidences.append(confidence)
        return min(selected_confidences) if selected_confidences else None

    def _tracker_findings(
        self,
        prediction: InitialPrediction,
        context: PerceptionContext,
        findings: list[GateFinding],
    ) -> None:
        if not self.use_tracker or context.track_snapshot.get("status") != "AVAILABLE":
            return
        frames = context.track_snapshot.get("frames", ())
        if not isinstance(frames, (tuple, list)) or not frames:
            return
        current = frames[-1]
        if not isinstance(current, Mapping):
            return
        tracks = current.get("tracks", ())
        if not isinstance(tracks, (tuple, list)):
            return
        tracker_ids = {
            int(track["instrument_id"])
            for track in tracks
            if isinstance(track, Mapping)
            and isinstance(track.get("instrument_id"), int)
            and not isinstance(track.get("instrument_id"), bool)
        }
        if tracker_ids != set(prediction.instrument_ids):
            findings.append(
                GateFinding(TASK_PATHS["instrument"], "TRACKER_CONFLICT")
            )
