"""Deterministic status-aware event report templates."""

from __future__ import annotations

from surgical_agent.systems.pipeline import FinalizedEvent

_STATUS_PREFIXES = {
    "Verified": "Confirmed facts",
    "Accepted": "High-confidence observation",
    "Pending": "Unresolved possible observations that may require review",
}
_STATUS_ORDER = ("Verified", "Accepted", "Pending")


def _unique(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(values))


def _ids(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values) or "none"


class TemplateReportGenerator:
    """Generate local prose without an API or mutable state."""

    def generate(
        self,
        events: tuple[FinalizedEvent, ...],
        *,
        trigger_reasons: tuple[str, ...],
    ) -> str:
        del trigger_reasons
        clauses: list[str] = []
        for status in _STATUS_ORDER:
            selected = tuple(event for event in events if event.final_status == status)
            if not selected:
                continue
            clauses.append(
                f"{_STATUS_PREFIXES[status]} at frames "
                f"{_ids(tuple(event.frame_id for event in selected))}: "
                f"phase IDs {_ids(_unique(tuple(event.phase_id for event in selected)))}, "
                f"instrument IDs {_ids(_unique(tuple(value for event in selected for value in event.instrument_ids)))}, "
                f"verb IDs {_ids(_unique(tuple(value for event in selected for value in event.verb_ids)))}, "
                f"target IDs {_ids(_unique(tuple(value for event in selected for value in event.target_ids)))}, "
                f"and triplet IDs {_ids(_unique(tuple(value for event in selected for value in event.triplet_ids)))}"
            )
        if not clauses:
            raise ValueError("template report requires at least one eligible event")
        return ". ".join(clauses) + "."


__all__ = ["TemplateReportGenerator"]
