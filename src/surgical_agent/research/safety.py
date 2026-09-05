"""Deterministic hard-safety validation and soft decision support.

Tracker disagreement, confidence and temporal variation are intentionally soft
features.  Only ontology/IVT closure and explicitly supplied strict constraints
can make a hypothesis hard-invalid.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.contracts import TASK_NAMES, JointPerceptionResult
from surgical_agent.research.gate.contracts import (
    REPAIR_SCOPE_ORDER,
    SCOPE_TASKS,
    RepairScope,
    RouteDecision,
    SafetySupport,
    SafetyViolation,
    SoftRisk,
)
from surgical_agent.research.signals.frame_evidence import load_ivt_components
from surgical_agent.research.verification.contracts import CandidateSet


def _selected(prediction: InitialPrediction, task: str) -> tuple[int, ...]:
    return {
        "instrument": prediction.instrument_ids,
        "verb": prediction.verb_ids,
        "target": prediction.target_ids,
        "ivt": prediction.triplet_ids,
        "phase": (prediction.phase_id,),
    }[task]


def _jaccard_distance(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    union = set(first) | set(second)
    return 0.0 if not union else 1.0 - len(set(first) & set(second)) / len(union)


class SafetyValidator:
    """One reusable pure validator for both H0 and repaired hypotheses."""

    def __init__(
        self,
        *,
        ivt_components: Mapping[int, tuple[int, int, int]] | None = None,
        strict_phase_allowed_ivt: Mapping[int, tuple[int, ...]] | None = None,
        require_component_coverage: bool = False,
    ) -> None:
        self.ivt_components = dict(
            load_ivt_components() if ivt_components is None else ivt_components
        )
        if set(self.ivt_components) != set(range(100)):
            raise ValueError("ivt_components must define IDs 0 through 99")
        self.strict_phase_allowed_ivt = (
            None
            if strict_phase_allowed_ivt is None
            else {
                phase: tuple(values)
                for phase, values in strict_phase_allowed_ivt.items()
            }
        )
        if type(require_component_coverage) is not bool:
            raise TypeError("require_component_coverage must be boolean")
        self.require_component_coverage = require_component_coverage

    def validate(self, hypothesis: InitialPrediction) -> tuple[SafetyViolation, ...]:
        if not isinstance(hypothesis, InitialPrediction):
            raise TypeError("SafetyValidator requires an InitialPrediction")
        findings: set[tuple[str, tuple[str, ...]]] = set()
        represented = {"instrument": set(), "verb": set(), "target": set()}
        selected = {
            "instrument": set(hypothesis.instrument_ids),
            "verb": set(hypothesis.verb_ids),
            "target": set(hypothesis.target_ids),
        }
        for ivt_id in hypothesis.triplet_ids:
            components = self.ivt_components.get(ivt_id)
            if components is None:
                findings.add(("UNKNOWN_IVT", ("ivt",)))
                continue
            for task, component in zip(("instrument", "verb", "target"), components):
                represented[task].add(component)
                if component not in selected[task]:
                    findings.add(("IVT_CLOSURE", (task, "ivt")))

        if self.require_component_coverage and hypothesis.triplet_ids:
            for task in ("instrument", "verb", "target"):
                if selected[task] - represented[task]:
                    findings.add(("TRIPLET_COMPATIBILITY", (task, "ivt")))

        if self.strict_phase_allowed_ivt is not None and hypothesis.triplet_ids:
            allowed = set(self.strict_phase_allowed_ivt.get(hypothesis.phase_id, ()))
            if not set(hypothesis.triplet_ids).issubset(allowed):
                findings.add(("STRICT_PHASE_CONSTRAINT", ("ivt", "phase")))

        order = {task: index for index, task in enumerate(TASK_NAMES)}
        return tuple(
            SafetyViolation(code, tasks)
            for code, tasks in sorted(
                findings,
                key=lambda item: (
                    min(order[task] for task in item[1]),
                    item[0],
                    item[1],
                ),
            )
        )


def legal_repair_scopes(
    violations: tuple[SafetyViolation, ...],
    candidates: CandidateSet,
) -> tuple[RepairScope, ...]:
    """Return scopes with at least one admissible alternative for affected tasks."""

    if not isinstance(candidates, CandidateSet):
        raise TypeError("legal scope construction requires a CandidateSet")
    affected = {task for item in violations for task in item.tasks}
    scopes: list[RepairScope] = []
    for scope in REPAIR_SCOPE_ORDER:
        scope_tasks = SCOPE_TASKS[scope]
        relevant = affected & scope_tasks if affected else scope_tasks
        if affected and not _scope_covers_violations(violations, scope_tasks):
            continue
        if any(
            set(candidates.allowed_ids[task])
            - set(_selected(candidates.initial_prediction, task))
            for task in relevant
        ):
            scopes.append(scope)
    return tuple(scopes)


def choose_covering_scope(
    violations: tuple[SafetyViolation, ...],
    legal_scopes: tuple[RepairScope, ...],
) -> RepairScope | None:
    return next(
        (
            scope
            for scope in REPAIR_SCOPE_ORDER
            if scope in legal_scopes
            and _scope_covers_violations(violations, SCOPE_TASKS[scope])
        ),
        None,
    )


def _scope_covers_violations(violations, scope_tasks) -> bool:
    # A relation can be repaired by changing either endpoint. It does not
    # require one specialist to own both the phase and interaction heads.
    return all(
        bool(set(item.tasks) & scope_tasks)
        if item.code == "STRICT_PHASE_CONSTRAINT"
        else set(item.tasks).issubset(scope_tasks)
        for item in violations
    )


@dataclass(frozen=True)
class DecisionSupportBuilder:
    """Build conservative soft risks and fixed-shape Gate features without GT."""

    confidence_threshold: float = 0.60
    temporal_jump_threshold: float = 0.80
    phase_ivt_support: Mapping[int, tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("temporal_jump_threshold", self.temporal_jump_threshold),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite in [0, 1]")

    def build(
        self,
        *,
        perception: JointPerceptionResult,
        candidates: CandidateSet,
        violations: tuple[SafetyViolation, ...],
        tracker_snapshot: Mapping[str, object],
        previous: InitialPrediction | None,
        previous_verified_tasks: tuple[str, ...] | None = None,
    ) -> SafetySupport:
        if not isinstance(perception, JointPerceptionResult):
            raise TypeError("decision support requires JointPerceptionResult")
        prediction = perception.prediction
        if prediction.score_semantics == "hard_label_v1":
            raise ValueError(
                "legacy Gate decision support requires confidence rankings; "
                "hard_label_v1 H0 must use the single_pass pipeline"
            )
        risks: list[SoftRisk] = []
        features: dict[str, float] = {}
        selected_confidences: list[float] = []
        margins: list[float] = []
        if self.phase_ivt_support is not None and prediction.triplet_ids:
            supported = set(self.phase_ivt_support.get(prediction.phase_id, ()))
            unsupported = set(prediction.triplet_ids) - supported
            if unsupported:
                risks.append(
                    SoftRisk(
                        "PHASE_IVT_UNOBSERVED",
                        ("ivt", "phase"),
                        len(unsupported) / len(prediction.triplet_ids),
                    )
                )
        for task in TASK_NAMES:
            ranking = perception.raw_evidence.ranked_candidates[task]
            by_id = {item.class_id: float(item.confidence) for item in ranking}
            values = [by_id.get(value, 0.0) for value in _selected(prediction, task)]
            floor = min(values) if values else 0.0
            margin = (
                float(ranking[0].confidence - ranking[1].confidence)
                if len(ranking) > 1
                else (float(ranking[0].confidence) if ranking else 0.0)
            )
            features[f"{task}_selected_confidence"] = floor
            features[f"{task}_candidate_margin"] = margin
            selected_confidences.extend(values)
            margins.append(margin)
            if floor < self.confidence_threshold:
                risks.append(SoftRisk("LOW_CONFIDENCE", (task,), 1.0 - floor))

        uncertainty_tasks = tuple(
            dict.fromkeys(
                item.path.split("/")[1]
                for item in perception.raw_evidence.field_uncertainties
            )
        )
        for task in uncertainty_tasks:
            risks.append(SoftRisk("EXPLICIT_UNCERTAINTY", (task,), 1.0))

        temporal_tasks = ("instrument", "verb", "target", "ivt")
        if previous_verified_tasks is not None:
            if any(task not in TASK_NAMES for task in previous_verified_tasks):
                raise ValueError("previous_verified_tasks contains an unknown task")
            reliable_temporal_tasks = tuple(
                task for task in temporal_tasks if task in previous_verified_tasks
            )
        else:
            # Callers without task-wise Memory retain the legacy all-task contract.
            reliable_temporal_tasks = temporal_tasks
        temporal_jump = 0.0
        if previous is not None and reliable_temporal_tasks:
            temporal_jump = max(
                _jaccard_distance(
                    _selected(previous, task), _selected(prediction, task)
                )
                for task in reliable_temporal_tasks
            )
            if temporal_jump >= self.temporal_jump_threshold:
                risks.append(
                    SoftRisk(
                        "TEMPORAL_JUMP",
                        reliable_temporal_tasks,
                        temporal_jump,
                    )
                )
        features["temporal_jump"] = temporal_jump

        tracker_status = str(
            tracker_snapshot.get(
                "runtime_status",
                tracker_snapshot.get("status", "ERROR"),
            )
        )
        tracker_available = tracker_status in {"OK", "AVAILABLE", "AVAILABLE_EMPTY"}
        tracker_instruments: set[int] = set()
        frames = tracker_snapshot.get("frames", ())
        if tracker_available and isinstance(frames, (tuple, list)) and frames:
            current = frames[-1]
            if isinstance(current, Mapping):
                tracks = current.get("tracks", ())
                if isinstance(tracks, (tuple, list)):
                    tracker_instruments = {
                        int(track["instrument_id"])
                        for track in tracks
                        if isinstance(track, Mapping)
                        and isinstance(track.get("instrument_id"), int)
                        and not isinstance(track.get("instrument_id"), bool)
                    }
        conflict = float(
            tracker_available and tracker_instruments != set(prediction.instrument_ids)
        )
        if conflict:
            risks.append(SoftRisk("TRACKER_CONFLICT", ("instrument",), conflict))
        features.update(
            tracker_available=float(tracker_available),
            tracker_conflict=conflict,
            tracker_instrument_count=float(len(tracker_instruments)),
            selected_confidence_floor=(
                min(selected_confidences) if selected_confidences else 0.0
            ),
            candidate_margin_floor=min(margins) if margins else 0.0,
        )
        return SafetySupport(
            hard_violations=violations,
            soft_risks=tuple(risks),
            gate_features=features,
            legal_scopes=legal_repair_scopes(violations, candidates),
        )


class MandatorySafetyGuard:
    """Prevent hard-invalid H0 from ever reaching the optional Gate."""

    def route(self, support: SafetySupport) -> RouteDecision | None:
        if not isinstance(support, SafetySupport):
            raise TypeError("MandatorySafetyGuard requires SafetySupport")
        if support.safety_class == "HARD_VALID":
            return None
        scope = choose_covering_scope(support.hard_violations, support.legal_scopes)
        if scope is None:
            return RouteDecision(
                kind="PENDING",
                source="MANDATORY_GUARD",
                reason="NO_LEGAL_REPAIR_SCOPE",
            )
        return RouteDecision(
            kind="VERIFY",
            source="MANDATORY_GUARD",
            reason="HARD_INVALID_REQUIRES_VERIFY",
            scope=scope,
            priority="MANDATORY",
            fallback_h0_allowed=False,
        )


__all__ = [
    "DecisionSupportBuilder",
    "MandatorySafetyGuard",
    "SafetyValidator",
    "choose_covering_scope",
    "legal_repair_scopes",
]
