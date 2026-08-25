"""Single canonical partial-label validity implementation."""

from __future__ import annotations

from surgical_agent.data.label_policy import LABEL_POLICY_VERSION, is_valid_task_id
from surgical_agent.data.schemas import LabelMask

PARTIAL_LABEL_SEMANTICS_VERSION = LABEL_POLICY_VERSION
UNRESOLVED_MISSING_SENTINEL = -1


def is_valid_categorical_id(value: object) -> bool:
    """Return whether a raw value is usable categorical supervision.

    Negative IDs are conservatively excluded. This does not name ``-1`` as
    background, none, or a negative class; its authoritative semantic name is
    still unresolved.
    """

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def canonical_label_mask(
    *,
    instrument_id: int,
    verb_id: int,
    target_id: int,
    triplet_id: int,
    phase_id: int,
) -> LabelMask:
    """Build task-wise masks independently from the five observed raw IDs."""

    return LabelMask(
        instrument=is_valid_task_id("instrument", instrument_id),
        verb=is_valid_task_id("verb", verb_id),
        target=is_valid_task_id("target", target_id),
        ivt=is_valid_task_id("triplet", triplet_id),
        phase=is_valid_task_id("phase", phase_id),
    )
