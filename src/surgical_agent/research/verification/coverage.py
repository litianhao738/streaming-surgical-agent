"""Bounded verification coverage after one Gate admission decision."""

from dataclasses import replace

from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.research.gate.contracts import SCOPE_TASKS, RepairScope
from surgical_agent.research.verification.contracts import changed_tasks
from surgical_agent.research.verification.repair import (
    BoundedVerifyRepairLoop,
    RepairProposal,
)


def run_scope_coverage(
    *,
    h0,
    candidates,
    initial_scope: RepairScope,
    specialist,
    validator,
    budget,
    max_scopes: int = 3,
) -> RepairProposal:
    """Keep admitted patches, visit uncovered tasks, and never reuse GT.

    One specialist attempt may make two model calls. The provider budget still
    counts each actual call; the scope budget counts specialist attempts.
    """
    if max_scopes not in {1, 2, 3}:
        raise ValueError("coverage max_scopes must be between one and three")
    current: InitialPrediction = h0
    attempts = []
    covered: set[str] = set()
    order = tuple(
        dict.fromkeys((initial_scope, "interaction", "workflow", "instrument_presence"))
    )
    count = 0
    for scope in order:
        if count >= max_scopes:
            break
        if SCOPE_TASKS[scope].issubset(covered):
            continue
        count += 1
        valid = not validator.validate(current)
        result = BoundedVerifyRepairLoop(
            specialist=specialist,
            validator=validator,
            budget=budget,
            max_attempts=1,
            require_repair_evidence=True,
        ).run(
            h0=current,
            candidates=replace(candidates, initial_prediction=current),
            scope=scope,
            priority="OPTIONAL" if valid else "MANDATORY",
            fallback_h0_allowed=valid,
        )
        base_attempt = len(attempts)
        attempts.extend(
            replace(item, attempt=base_attempt + index + 1)
            for index, item in enumerate(result.attempts)
        )
        if result.status == "REJECTED_BY_VERIFY":
            return replace(result, attempts=tuple(attempts))
        if result.status in {"VERIFIED_KEEP", "VERIFIED_REPAIR"}:
            current = result.hypothesis
            covered.update(SCOPE_TASKS[scope])
        # A provider refusal is terminal for this frame. Do not try another
        # scope to get the same blocked visual material through the provider.
        if any(
            item.specialist_reason == "content_moderation" for item in result.attempts
        ):
            break
    if validator.validate(current):
        return RepairProposal(
            "PENDING_UNRESOLVED",
            current,
            "COVERAGE_HARD_CONFLICT_REMAINS",
            tuple(attempts),
        )
    if changed_tasks(h0, current):
        status = "VERIFIED_REPAIR"
    elif any(item.accepted for item in attempts):
        status = "VERIFIED_KEEP"
    else:
        status = "FALLBACK_KEEP"
    return RepairProposal(status, current, "BOUNDED_SCOPE_COVERAGE", tuple(attempts))
