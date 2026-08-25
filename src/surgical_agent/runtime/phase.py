"""Phase-gate contracts for P0-P12 implementation progress."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum


class Phase(str, Enum):
    """Ordered phases from the V3.1 task book."""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"
    P6 = "P6"
    P7 = "P7"
    P8 = "P8"
    P9 = "P9"
    P10 = "P10"
    P11 = "P11"
    P12 = "P12"


class PhaseStatus(str, Enum):
    """Allowed evidence-backed phase states."""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


class PhaseGateError(RuntimeError):
    """Raised when work attempts to skip an incomplete phase."""


def require_prior_phases_passed(
    target: Phase,
    statuses: Mapping[Phase, PhaseStatus],
) -> None:
    """Reject entry into ``target`` unless every preceding phase passed."""

    phases = list(Phase)
    for prior in phases[: phases.index(target)]:
        status = statuses.get(prior, PhaseStatus.NOT_STARTED)
        if status is not PhaseStatus.PASS:
            raise PhaseGateError(
                f"Cannot enter {target.value}: {prior.value} is {status.value}"
            )
