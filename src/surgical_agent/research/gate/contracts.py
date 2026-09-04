"""Contracts shared by safety routing and the Benefit Gate.

The Gate is deliberately downstream of deterministic safety validation.  It can
only choose whether a hard-valid hypothesis should use ``H0`` or request one
bounded verification scope; it never assigns a final semantic state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from surgical_agent.perception.contracts import TASK_NAMES

RepairScope = Literal["instrument_presence", "interaction", "workflow"]
VerificationPriority = Literal["MANDATORY", "OPTIONAL"]
GateActionKind = Literal["USE_H0", "REQUEST_VERIFY"]
RouteKind = Literal["USE_H0", "VERIFY", "PENDING"]

REPAIR_SCOPE_ORDER: tuple[RepairScope, ...] = (
    "instrument_presence",
    "interaction",
    "workflow",
)
SCOPE_TASKS: Mapping[RepairScope, frozenset[str]] = MappingProxyType(
    {
        "instrument_presence": frozenset({"instrument"}),
        "interaction": frozenset({"instrument", "verb", "target", "ivt"}),
        "workflow": frozenset({"phase"}),
    }
)
FORMAL_GATE_FEATURE_ORDER: tuple[str, ...] = tuple(
    name
    for task in TASK_NAMES
    for name in (f"{task}_selected_confidence", f"{task}_candidate_margin")
) + (
    "temporal_jump",
    "tracker_available",
    "tracker_conflict",
    "tracker_instrument_count",
    "selected_confidence_floor",
    "candidate_margin_floor",
)


def _canonical_tasks(values: tuple[str, ...]) -> tuple[str, ...]:
    if any(value not in TASK_NAMES for value in values):
        raise ValueError("tasks contain an unknown prediction head")
    return tuple(task for task in TASK_NAMES if task in set(values))


@dataclass(frozen=True)
class SafetyViolation:
    """One deterministic hard-invalid finding."""

    code: str
    tasks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("safety violation code must be non-empty")
        tasks = _canonical_tasks(tuple(self.tasks))
        if not tasks:
            raise ValueError("a safety violation must identify at least one task")
        object.__setattr__(self, "tasks", tasks)


@dataclass(frozen=True)
class SoftRisk:
    """One non-binding uncertainty feature for Gate input."""

    code: str
    tasks: tuple[str, ...]
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("soft risk code must be non-empty")
        tasks = _canonical_tasks(tuple(self.tasks))
        if not math.isfinite(float(self.value)) or not 0.0 <= float(self.value) <= 1.0:
            raise ValueError("soft risk value must be finite in [0, 1]")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class SafetySupport:
    """Frozen output of deterministic precheck and decision-support building."""

    hard_violations: tuple[SafetyViolation, ...]
    soft_risks: tuple[SoftRisk, ...]
    gate_features: Mapping[str, float]
    legal_scopes: tuple[RepairScope, ...]

    def __post_init__(self) -> None:
        hard = tuple(self.hard_violations)
        soft = tuple(self.soft_risks)
        if any(not isinstance(item, SafetyViolation) for item in hard):
            raise TypeError("hard_violations must contain SafetyViolation values")
        if any(not isinstance(item, SoftRisk) for item in soft):
            raise TypeError("soft_risks must contain SoftRisk values")
        features = dict(self.gate_features)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for name, value in features.items()
        ):
            raise ValueError("gate_features must contain named finite numbers")
        scopes = tuple(self.legal_scopes)
        if len(set(scopes)) != len(scopes) or any(
            scope not in REPAIR_SCOPE_ORDER for scope in scopes
        ):
            raise ValueError("legal_scopes contain an unknown or duplicate scope")
        scopes = tuple(scope for scope in REPAIR_SCOPE_ORDER if scope in scopes)
        object.__setattr__(self, "hard_violations", hard)
        object.__setattr__(self, "soft_risks", soft)
        object.__setattr__(
            self,
            "gate_features",
            MappingProxyType({name: float(value) for name, value in features.items()}),
        )
        object.__setattr__(self, "legal_scopes", scopes)

    @property
    def safety_class(self) -> Literal["HARD_VALID", "HARD_INVALID"]:
        return "HARD_INVALID" if self.hard_violations else "HARD_VALID"


@dataclass(frozen=True)
class GateAction:
    """Optional routing decision for an already hard-valid H0."""

    kind: GateActionKind
    scope: RepairScope | None = None
    benefit_probability: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"USE_H0", "REQUEST_VERIFY"}:
            raise ValueError("unsupported Gate action")
        if (self.kind == "REQUEST_VERIFY") != (self.scope is not None):
            raise ValueError("REQUEST_VERIFY requires one scope; USE_H0 forbids it")
        if self.scope is not None and self.scope not in REPAIR_SCOPE_ORDER:
            raise ValueError("Gate requested an unknown scope")
        if self.benefit_probability is not None and (
            not math.isfinite(float(self.benefit_probability))
            or not 0.0 <= float(self.benefit_probability) <= 1.0
        ):
            raise ValueError("benefit_probability must be finite in [0, 1]")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Gate actions require an auditable reason")


@dataclass(frozen=True)
class RouteDecision:
    """Resolved mandatory-or-optional route before budget consumption."""

    kind: RouteKind
    source: Literal["MANDATORY_GUARD", "BENEFIT_GATE"]
    reason: str
    scope: RepairScope | None = None
    priority: VerificationPriority | None = None
    fallback_h0_allowed: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"USE_H0", "VERIFY", "PENDING"}:
            raise ValueError("unsupported route kind")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("route decisions require a reason")
        if self.kind == "VERIFY":
            if self.scope not in REPAIR_SCOPE_ORDER or self.priority is None:
                raise ValueError("VERIFY requires a legal scope and priority")
        elif self.scope is not None or self.priority is not None:
            raise ValueError("non-VERIFY routes cannot carry scope or priority")
        if self.source == "MANDATORY_GUARD" and self.fallback_h0_allowed:
            raise ValueError("mandatory routes can never fall back to hard-invalid H0")
        if self.kind == "USE_H0" and not self.fallback_h0_allowed:
            raise ValueError("USE_H0 must explicitly permit H0")


__all__ = [
    "FORMAL_GATE_FEATURE_ORDER",
    "REPAIR_SCOPE_ORDER",
    "SCOPE_TASKS",
    "GateAction",
    "GateActionKind",
    "RepairScope",
    "RouteDecision",
    "RouteKind",
    "SafetySupport",
    "SafetyViolation",
    "SoftRisk",
    "VerificationPriority",
]
