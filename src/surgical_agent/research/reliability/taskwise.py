"""Closed task-wise reliability states for five-head semantic outputs."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from surgical_agent.perception.contracts import TASK_NAMES

TaskReliabilityState = Literal[
    "Accepted",
    "Checked",
    "Verified",
    "Derived",
    "Pending",
    "Rejected",
]

TASK_RELIABILITY_STATES = frozenset(
    {"Accepted", "Checked", "Verified", "Derived", "Pending", "Rejected"}
)


def uniform_task_states(state: TaskReliabilityState) -> Mapping[str, str]:
    """Return one immutable state for every canonical prediction head."""

    if state not in TASK_RELIABILITY_STATES:
        raise ValueError("unsupported task reliability state")
    return MappingProxyType({task: state for task in TASK_NAMES})


def normalize_task_states(
    values: Mapping[str, str] | None,
    *,
    default: TaskReliabilityState,
) -> Mapping[str, str]:
    """Validate and freeze exactly one reliability state per prediction head."""

    raw = dict(uniform_task_states(default) if values is None else values)
    if set(raw) != set(TASK_NAMES):
        raise ValueError("task_states must contain exactly the five prediction heads")
    if any(value not in TASK_RELIABILITY_STATES for value in raw.values()):
        raise ValueError("task_states contains an unsupported reliability state")
    return MappingProxyType({task: raw[task] for task in TASK_NAMES})


def tasks_in_state(values: Mapping[str, str], state: str) -> tuple[str, ...]:
    """Return matching tasks in the canonical five-head order."""

    return tuple(task for task in TASK_NAMES if values[task] == state)


__all__ = [
    "TASK_RELIABILITY_STATES",
    "TaskReliabilityState",
    "normalize_task_states",
    "tasks_in_state",
    "uniform_task_states",
]
