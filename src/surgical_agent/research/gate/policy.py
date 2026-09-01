"""Causal no-training policies and the future learned-policy runtime boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import JointPerceptionResult
from surgical_agent.research.gate.benefit_model import FrozenLinearBenefitArtifact
from surgical_agent.research.gate.features import (
    ALL_DEMO_GATE_FEATURE_NAMES,
    WITH_TRACKER_FEATURE_ORDER,
    extract_gate_features,
)
from surgical_agent.research.gate.reliability import ReliabilityGatePolicy
from surgical_agent.research.reliability.state import ALL_TASK_FIELDS, TASK_PATHS
from surgical_agent.research.signals.contracts import EvidenceProfile
from surgical_agent.systems.pipeline import GateDecision


def _selected_ids(prediction: InitialPrediction, task: str) -> tuple[int, ...]:
    return {
        "instrument": tuple(prediction.instrument_ids),
        "verb": tuple(prediction.verb_ids),
        "target": tuple(prediction.target_ids),
        "ivt": tuple(prediction.triplet_ids),
        "phase": (prediction.phase_id,),
    }[task]


def _task_confidence_floors(
    perception_result: JointPerceptionResult,
) -> dict[str, float]:
    prediction = perception_result.prediction
    floors: dict[str, float] = {}
    for task in ALL_TASK_FIELDS:
        ranked = perception_result.raw_evidence.ranked_candidates[task]
        by_id = {item.class_id: item.confidence for item in ranked}
        selected = _selected_ids(prediction, task)
        floors[task] = min(
            (by_id.get(class_id, 0.0) for class_id in selected),
            default=0.0,
        )
    return floors


def _lowest_confidence_task(
    task_floors: Mapping[str, float],
    eligible: tuple[str, ...],
) -> str:
    return min(
        eligible,
        key=lambda task: (task_floors[task], ALL_TASK_FIELDS.index(task)),
    )


class AlwaysVerifyPolicy:
    """P8 capability policy: one configured verification call per state."""

    def __init__(self, *, scope: str = "joint") -> None:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scope must not be empty")
        self.scope = scope

    def decide(
        self,
        signals: EvidenceProfile,
        *,
        context: PerceptionContext | None = None,
        perception_result: JointPerceptionResult | None = None,
    ) -> GateDecision:
        del context, perception_result
        if not isinstance(signals, EvidenceProfile):
            raise TypeError("signals must be an EvidenceProfile")
        return GateDecision(
            action="VERIFY",
            scope=self.scope,
            reason=f"ALWAYS_VERIFY_{self.scope.upper()}",
            flagged_fields=ALL_TASK_FIELDS,
        )


class ForcedTargetedVerificationPolicy:
    """Collect counterfactual labels by verifying one uncertain head per example."""

    def __init__(self, finding_policy: ReliabilityGatePolicy) -> None:
        if not isinstance(finding_policy, ReliabilityGatePolicy):
            raise TypeError("finding_policy must be a ReliabilityGatePolicy")
        self.finding_policy = finding_policy

    def decide(
        self,
        signals: EvidenceProfile,
        *,
        context: PerceptionContext,
        perception_result: JointPerceptionResult,
    ) -> GateDecision:
        task_floors = _task_confidence_floors(perception_result)
        routed = self.finding_policy.decide(
            signals,
            context=context,
            perception_result=perception_result,
        )
        eligible = (
            routed.flagged_fields
            if routed.action == "VERIFY" and routed.flagged_fields
            else ALL_TASK_FIELDS
        )
        selected_task = _lowest_confidence_task(task_floors, eligible)
        selected_findings = tuple(
            finding
            for finding in routed.findings
            if finding.path == TASK_PATHS[selected_task]
        )
        return GateDecision(
            action="VERIFY",
            scope="joint",
            reason=(
                "FORCED_TARGETED_RELIABILITY_FIELD"
                if routed.action == "VERIFY"
                else "FORCED_TARGETED_LOWEST_CONFIDENCE_FIELD"
            ),
            findings=selected_findings,
            flagged_fields=(selected_task,),
            selected_confidence_floor=routed.selected_confidence_floor,
        )


class EvidenceThresholdPolicy:
    """Frozen rule baseline over available causal evidence, not a learned Gate."""

    def __init__(self, *, threshold: float, scope: str = "joint") -> None:
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("threshold must be finite in [0, 1]")
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scope must not be empty")
        self.threshold = float(threshold)
        self.scope = scope

    def decide(
        self,
        signals: EvidenceProfile,
        *,
        context: PerceptionContext | None = None,
        perception_result: JointPerceptionResult | None = None,
    ) -> GateDecision:
        del context, perception_result
        if not isinstance(signals, EvidenceProfile):
            raise TypeError("signals must be an EvidenceProfile")
        available = [
            float(value.value)
            for task_values in signals.task_values.values()
            for value in task_values.values()
            if value.available and value.value is not None
        ]
        available.extend(
            float(value.value)
            for value in signals.global_values.values()
            if value.available and value.value is not None
        )
        if available and max(available) >= self.threshold:
            return GateDecision(
                action="VERIFY",
                scope=self.scope,
                reason="EVIDENCE_THRESHOLD_TRIGGERED",
            )
        return GateDecision(
            action="ACCEPT",
            scope=None,
            reason="EVIDENCE_THRESHOLD_NOT_TRIGGERED",
        )


def _feature_value(profile: EvidenceProfile, name: str) -> float:
    task, signal_name, attribute = name.split(".")
    values = (
        profile.global_values
        if task == "global"
        else profile.task_values.get(task)
    )
    if values is None or signal_name not in values:
        raise ValueError(f"Gate artifact references unknown feature: {name}")
    evidence = values[signal_name]
    if attribute == "available":
        return 1.0 if evidence.available else 0.0
    if attribute == "value":
        return 0.0 if evidence.value is None else float(evidence.value)
    raise ValueError(f"Gate artifact feature has unknown attribute: {name}")


class FrozenLinearBenefitGate:
    """Read-only final-G1 inference; training is intentionally outside runtime."""

    def __init__(
        self,
        artifact: FrozenLinearBenefitArtifact,
        *,
        finding_policy: ReliabilityGatePolicy | None = None,
        phase_allowed_ivt: Mapping[int, tuple[int, ...]] | None = None,
    ) -> None:
        if not isinstance(artifact, FrozenLinearBenefitArtifact):
            raise TypeError("artifact must be a FrozenLinearBenefitArtifact")
        self.artifact = artifact
        self.finding_policy = finding_policy or ReliabilityGatePolicy()
        self.phase_allowed_ivt = phase_allowed_ivt

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        finding_policy: ReliabilityGatePolicy | None = None,
        phase_allowed_ivt: Mapping[int, tuple[int, ...]] | None = None,
    ) -> FrozenLinearBenefitGate:
        return cls(
            FrozenLinearBenefitArtifact.from_json(path),
            finding_policy=finding_policy,
            phase_allowed_ivt=phase_allowed_ivt,
        )

    def decide(
        self,
        signals: EvidenceProfile,
        *,
        context: PerceptionContext | None = None,
        perception_result: JointPerceptionResult | None = None,
    ) -> GateDecision:
        if not isinstance(signals, EvidenceProfile):
            raise TypeError("signals must be an EvidenceProfile")
        is_demo = set(self.artifact.feature_order) <= ALL_DEMO_GATE_FEATURE_NAMES
        if is_demo:
            if context is None or perception_result is None:
                raise ValueError(
                    "demo Learned Gate requires context and perception_result"
                )
            vector = extract_gate_features(
                signals,
                context=context,
                perception_result=perception_result,
                include_tracker=(
                    self.artifact.feature_order == WITH_TRACKER_FEATURE_ORDER
                ),
                phase_allowed_ivt=self.phase_allowed_ivt,
            ).values
        else:
            vector = tuple(
                _feature_value(signals, name)
                for name in self.artifact.feature_order
            )
        scores = {
            scope: self.artifact.bias_by_scope[scope]
            + sum(
                weight * value
                for weight, value in zip(
                    self.artifact.weights_by_scope[scope],
                    vector,
                )
            )
            for scope in self.artifact.scope_order
        }
        selected_scope = max(
            self.artifact.scope_order,
            key=lambda scope: (scores[scope], -self.artifact.scope_order.index(scope)),
        )
        bounded_score = max(-709.0, min(709.0, scores[selected_scope]))
        probability = 1.0 / (1.0 + math.exp(-bounded_score))
        if scores[selected_scope] > self.artifact.threshold:
            findings = ()
            flagged_fields = ()
            selected_confidence_floor = None
            if context is not None and perception_result is not None:
                finding_decision = self.finding_policy.decide(
                    signals,
                    context=context,
                    perception_result=perception_result,
                )
                findings = finding_decision.findings
                flagged_fields = finding_decision.flagged_fields
                selected_confidence_floor = (
                    finding_decision.selected_confidence_floor
                )
                if not flagged_fields:
                    task_floors = _task_confidence_floors(perception_result)
                    selected_task = _lowest_confidence_task(
                        task_floors, ALL_TASK_FIELDS
                    )
                    flagged_fields = (selected_task,)
                    selected_confidence_floor = task_floors[selected_task]
            return GateDecision(
                action="VERIFY",
                scope=selected_scope,
                reason="FINAL_G1_BENEFIT_ABOVE_THRESHOLD",
                findings=findings,
                flagged_fields=flagged_fields,
                selected_confidence_floor=selected_confidence_floor,
                benefit_probability=probability,
            )
        return GateDecision(
            action="ACCEPT",
            scope=None,
            reason="FINAL_G1_BENEFIT_NOT_ABOVE_THRESHOLD",
            benefit_probability=probability,
        )
