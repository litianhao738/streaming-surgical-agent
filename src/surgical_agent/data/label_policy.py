"""Task-wise label validity and cross-task IVT consistency policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from surgical_agent.data.constants import TASK_ID_BOUNDS

LABEL_POLICY_VERSION = "ct20_task_range_and_ivt_consistency_v1"
SPECIMEN_BAG_SCOPE_EXCEPTION = (6, 0, 13, 12)


class IvtConsistencyStatus(str, Enum):
    """Relationship between raw I/V/T components and a mapped triplet."""

    EXACT = "EXACT"
    NOT_APPLICABLE_PARTIAL_LABEL = "NOT_APPLICABLE_PARTIAL_LABEL"
    SCOPE_DIFFERENCE = "SCOPE_DIFFERENCE"
    UNRESOLVED_CONFLICT = "UNRESOLVED_CONFLICT"
    MAPPING_UNAVAILABLE = "MAPPING_UNAVAILABLE"


@dataclass(frozen=True)
class IvtConsistencyResult:
    """Auditable result used by losses, metrics, and future Gate targets."""

    status: IvtConsistencyStatus
    consistency_applicable: bool
    use_for_component_consistency: bool
    notes: str


def is_valid_task_id(task: str, value: object) -> bool:
    """Return whether a value lies inside the verified non-negative task range."""

    normalized_task = "ivt" if task == "triplet" else task
    if normalized_task not in TASK_ID_BOUNDS:
        raise KeyError(f"Unknown CholecTrack20 task: {task}")
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    lower, upper = TASK_ID_BOUNDS[normalized_task]
    return lower <= value <= upper


def classify_ivt_consistency(
    *,
    instrument_id: int,
    verb_id: int,
    target_id: int,
    triplet_id: int,
    triplet_mapping: Mapping[int, tuple[int, int, int]],
) -> IvtConsistencyResult:
    """Classify consistency without rewriting any release-native label."""

    if not all(
        (
            is_valid_task_id("instrument", instrument_id),
            is_valid_task_id("verb", verb_id),
            is_valid_task_id("target", target_id),
            is_valid_task_id("triplet", triplet_id),
        )
    ):
        return IvtConsistencyResult(
            status=IvtConsistencyStatus.NOT_APPLICABLE_PARTIAL_LABEL,
            consistency_applicable=False,
            use_for_component_consistency=False,
            notes="At least one raw component is unavailable for joint consistency.",
        )

    observed = (instrument_id, verb_id, target_id)
    expected = triplet_mapping.get(triplet_id)
    if expected is None:
        return IvtConsistencyResult(
            status=IvtConsistencyStatus.MAPPING_UNAVAILABLE,
            consistency_applicable=False,
            use_for_component_consistency=False,
            notes="No verified component mapping is available for this triplet ID.",
        )
    if observed == expected:
        return IvtConsistencyResult(
            status=IvtConsistencyStatus.EXACT,
            consistency_applicable=True,
            use_for_component_consistency=True,
            notes="Raw I/V/T components exactly match the verified triplet relation.",
        )
    if (*observed, triplet_id) == SPECIMEN_BAG_SCOPE_EXCEPTION:
        return IvtConsistencyResult(
            status=IvtConsistencyStatus.SCOPE_DIFFERENCE,
            consistency_applicable=False,
            use_for_component_consistency=False,
            notes=(
                "Track20 boxes specimen-bag while triplet 12 describes the grasper "
                "grasping that bag; keep task labels independent."
            ),
        )
    return IvtConsistencyResult(
        status=IvtConsistencyStatus.UNRESOLVED_CONFLICT,
        consistency_applicable=True,
        use_for_component_consistency=False,
        notes="Raw components disagree with the verified triplet relation; do not repair.",
    )
