"""Rule and learned Benefit Gates for hard-valid hypotheses only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from surgical_agent.research.gate.benefit_model import FrozenLinearBenefitArtifact
from surgical_agent.research.gate.contracts import (
    REPAIR_SCOPE_ORDER,
    SCOPE_TASKS,
    GateAction,
    RepairScope,
    SafetySupport,
)


class BenefitGate(Protocol):
    def decide(self, support: SafetySupport) -> GateAction: ...


def _scope_for_risks(support: SafetySupport) -> tuple[RepairScope | None, float]:
    legal = tuple(
        scope for scope in REPAIR_SCOPE_ORDER if scope in support.legal_scopes
    )
    if not legal:
        return None, 0.0
    scores = {
        scope: max(
            (
                risk.value
                for risk in support.soft_risks
                if set(risk.tasks) & SCOPE_TASKS[scope]
            ),
            default=0.0,
        )
        for scope in legal
    }
    scope = max(legal, key=lambda value: (scores[value], -legal.index(value)))
    return scope, scores[scope]


@dataclass(frozen=True)
class RuleBenefitGate:
    """Deterministic G0: verify when a soft risk exceeds the frozen threshold."""

    risk_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.risk_threshold) <= 1.0:
            raise ValueError("risk_threshold must lie in [0, 1]")

    def decide(self, support: SafetySupport) -> GateAction:
        if support.safety_class != "HARD_VALID":
            raise ValueError("Benefit Gate may only process HARD_VALID H0")
        scope, scoped_risk = _scope_for_risks(support)
        if scoped_risk >= self.risk_threshold and scope is not None:
            return GateAction(
                "REQUEST_VERIFY",
                scope=scope,
                reason="RULE_RISK_THRESHOLD",
            )
        return GateAction("USE_H0", reason="RULE_USE_H0")


@dataclass(frozen=True)
class LearnedBenefitGate:
    """Use one frozen multi-scope artifact for both Tracker ablation views."""

    artifact: FrozenLinearBenefitArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, FrozenLinearBenefitArtifact):
            raise TypeError("artifact must be FrozenLinearBenefitArtifact")
        if any(scope not in REPAIR_SCOPE_ORDER for scope in self.artifact.scope_order):
            raise ValueError("formal Gate artifact contains a legacy verification scope")

    def decide(self, support: SafetySupport) -> GateAction:
        if support.safety_class != "HARD_VALID":
            raise ValueError("Benefit Gate may only process HARD_VALID H0")
        values = tuple(
            float(support.gate_features.get(name, 0.0))
            for name in self.artifact.feature_order
        )
        legal = tuple(
            scope
            for scope in self.artifact.scope_order
            if scope in support.legal_scopes
        )
        if not legal:
            return GateAction("USE_H0", reason="NO_LEGAL_OPTIONAL_SCOPE")
        scores = {
            scope: self.artifact.bias_by_scope[scope]
            + sum(
                weight * value
                for weight, value in zip(
                    self.artifact.weights_by_scope[scope], values, strict=True
                )
            )
            for scope in legal
        }
        scope = max(legal, key=lambda value: (scores[value], -legal.index(value)))
        if scores[scope] < self.artifact.threshold:
            return GateAction("USE_H0", reason="LEARNED_BENEFIT_BELOW_THRESHOLD")
        return GateAction(
            "REQUEST_VERIFY",
            scope=scope,
            reason="LEARNED_BENEFIT_ABOVE_THRESHOLD",
        )


__all__ = ["BenefitGate", "LearnedBenefitGate", "RuleBenefitGate"]
