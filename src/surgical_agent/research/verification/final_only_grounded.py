"""Checked final-label boundary for the existing grounded-contact experiment.

The historical v1 implementation and prompts remain reproducible. This adapter
accepts the published H0 wire payload or Batch-imported label sets; it does not
invent scores, call a provider, or make model-assessed evidence independent.
"""

from collections.abc import Mapping

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.final_only import (
    FINAL_ONLY_SCHEMA_VERSION,
    TASKS,
    validate_final_only,
)
from surgical_agent.research.verification.grounded_repair import (
    REVIEW_VERSION,
    admit,
    proposed_labels,
)

CHECKED_REPAIR_VERSION = "grounded_final_only_checked_v1"


def final_labels(payload):
    """Validate five heads without imposing equality to the IVT projection."""
    if not isinstance(payload, Mapping):
        raise ApiSchemaError("H0 must contain final labels")
    if "schema_version" in payload:
        validate_final_only(payload)
        return {
            task: [payload[task]["selected_id"]]
            if task == "phase" else list(payload[task]["selected_ids"])
            for task in TASKS
        }
    if set(payload) != set(TASKS):
        raise ApiSchemaError("Final labels require exactly five heads")
    if any(not isinstance(payload[task], (list, tuple)) for task in TASKS):
        raise ApiSchemaError("Each final-label head must be an ID sequence")
    if len(payload["phase"]) != 1:
        raise ApiSchemaError("Phase requires exactly one final ID")
    wire = {"schema_version": FINAL_ONLY_SCHEMA_VERSION}
    for task in TASKS:
        wire[task] = (
            {"selected_id": payload[task][0]}
            if task == "phase" else {"selected_ids": payload[task]}
        )
    validate_final_only(wire)
    return {task: list(payload[task]) for task in TASKS}


def _same_labels(left, right):
    return all(set(left[task]) == set(right[task]) for task in TASKS)


def prepare_grounded_review(h0, locator, proposal, *, proposal_slot):
    """Check a candidate before spending a contrast-review call.

Invalid baseline input is an error, because KEEP cannot make it valid. An
invalid optional proposal instead preserves the valid H0. Label permutations
do not constitute a repair and therefore do not require another model call.
    """
    if proposal_slot not in {"FIRST", "SECOND"}:
        raise ValueError("invalid blinded hypothesis slot")
    baseline = final_labels(h0)
    result = {
        "schema_version": CHECKED_REPAIR_VERSION,
        "h0": baseline,
        "h1": None,
        "review_required": False,
        "hypotheses": None,
        "proposal_slot": proposal_slot,
        "reason": "INCOMPLETE_OR_INVALID_GROUNDING",
    }
    if locator is None or proposal is None:
        return result
    try:
        candidate = proposed_labels(baseline, locator, proposal)
    except (ApiSchemaError, OverflowError):
        return result
    if candidate is None:
        return result
    try:
        candidate = final_labels(candidate)
    except ApiSchemaError:
        result["reason"] = "CANDIDATE_OUTSIDE_FINAL_ONLY_CONTRACT"
        return result
    result["h1"] = candidate
    if _same_labels(baseline, candidate):
        result["reason"] = "NO_LABEL_CHANGE"
        return result
    other_slot = "SECOND" if proposal_slot == "FIRST" else "FIRST"
    result.update(
        review_required=True,
        hypotheses={proposal_slot: candidate, other_slot: baseline},
        reason="REVIEW_REQUIRED",
    )
    return result


def finalize_grounded_repair(h0, locator, proposal, review, *, proposal_slot,
                             review_schema_version=REVIEW_VERSION):
    """Apply one optional review; never relax the historical evidence checks."""
    prepared = prepare_grounded_review(
        h0, locator, proposal, proposal_slot=proposal_slot
    )
    decision = {"decision": "KEEP", "reason": prepared["reason"]}
    if prepared["review_required"]:
        try:
            decision = admit(
                prepared["h0"], prepared["h1"], locator, proposal, review,
                proposal_slot=proposal_slot, review_schema_version=review_schema_version,
            )
        except (ApiSchemaError, OverflowError):
            decision = {"decision": "KEEP", "reason": "INVALID_REVIEW"}
    final = prepared["h1"] if decision["decision"] == "ACCEPT" else prepared["h0"]
    return {**prepared, **decision, "final": final_labels(final)}
