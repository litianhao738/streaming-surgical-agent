"""One-pass final-only H0, grounded proposal, visual review and checked repair.

The caller supplies the frozen H0 request and a synchronous, audited API callback.
This module never reads GT, opens a network connection, retries, or changes H0.
Its visual admission is uncalibrated and requires paired semantic evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any

from surgical_agent.api.contracts import ApiRequest, thaw_json
from surgical_agent.api.errors import ApiContractError, ApiError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.schema import schema_for, validator_for
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION
from surgical_agent.perception.main_h0 import _LABEL_BOUNDARY, main_h0_input
from surgical_agent.perception.ontology_prompt import (
    ACADEMIC_MEDICAL_CONTEXT,
    load_prompt_ontology_text,
)
from surgical_agent.research.verification.final_only_grounded import (
    CHECKED_REPAIR_VERSION,
    final_labels,
    finalize_grounded_repair,
    prepare_grounded_review,
)
from surgical_agent.research.verification.grounded_repair import (
    LOCATOR_VERSION,
    PROPOSAL_VERSION,
    REVIEW_VERSION,
    make_contact_crops,
)

GROUNDED_PIPELINE_VERSION = "grounded_final_only_pipeline_v1"

# Retain the original experiment's stage instructions. Only the image position
# wording is generalized so an actual one- or two-frame history stays causal.
LOCATOR_INSTRUCTION = (
    "Locate each currently visible surgical instrument TIP and contact region in the "
    "LAST full original image (the target frame). "
    "Return normalized [left,top,right,bottom] boxes using the full target image dimensions. "
    "Assign distinct instance IDs 1..3, left-to-right. Do not classify actions, anatomy, phase or instrument type. "
    "If more than three tools exist, or visibility prevents full coverage, all_visible_tools_covered must be false. "
    "Boxes are proposals, not trusted detections; do not invent hidden tools."
)
PROPOSAL_INSTRUCTION = (
    "Recognize each localized instrument and its CURRENT interaction using full causal images plus target crops. "
    "The location proposals can be wrong: return INSUFFICIENT if a box is irrelevant or identity/contact is unclear. "
    "Return one instance record per supplied instance ID. Use the full 100-class ontology, not a ranked candidate pool. "
    "Select only supported tuples belonging to that instance. Describe a short factual contact observation. "
    "Nearby visible anatomy alone is not an interaction target. Do not predict the phase. "
    "For valid tools acting outside the ontology use the defined null tuple only when visually supported; "
    "do not use null as an uncertainty fallback."
)
REVIEW_INSTRUCTION = (
    "Compare two alternative scene hypotheses against the original causal frames and localized target crops. "
    "Their order is arbitrary; neither is ground truth. Do not favor one for being more structured or more detailed. "
    "Prefer a hypothesis only if visual contact, anatomy identity and action evidence distinguish it. "
    "Check each supplied region is relevant and all visible tools are covered; no new answer is requested. "
    "If the evidence cannot distinguish the alternatives, return TIE or INSUFFICIENT. "
    "For null interactions require evidence of the ontology boundary, not just uncertainty. "
    "Report brief observable distinctions, no chain of thought. Phase is held fixed and is not under review."
)


def _visual_input(base: ApiRequest) -> dict[str, Any]:
    """Rebuild an allowlisted image-only context; never copy predictions or GT."""
    if base.response_schema_version != FINAL_ONLY_SCHEMA_VERSION:
        raise ApiContractError("grounded pipeline requires a final-only H0 request")
    try:
        original = json.loads(base.payload["input_text"])
        video_id = original["video_id"]
        target = original["target_frame_id"]
        frames = tuple(original["causal_frame_ids"])
        selected = tuple(original["selected_image_frame_ids"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiContractError("grounded pipeline requires causal H0 image metadata") from exc
    if (
        not isinstance(video_id, str) or not video_id
        or type(target) is not int or target < 1
        or any(type(frame) is not int or frame < 1 for frame in frames)
        or selected != frames or len(base.images) != len(frames)
    ):
        raise ApiContractError("grounded pipeline requires matching ordered causal images")
    visual = main_h0_input(video_id=video_id, target_frame_id=target, frame_ids=frames)
    for key in ("prior_finalized_prediction", "workflow_summary", "track_summary"):
        del visual[key]
    return visual


def _request(base, visual, version, instruction, extra=None, crops=()):
    # Transport routing is retained, but all language-model-visible text is
    # rebuilt. In particular, the blind locator cannot inherit H0 predictions.
    payload = thaw_json(base.payload)
    payload["system_text"] = (
        ACADEMIC_MEDICAL_CONTEXT + "\n" + instruction + "\n"
        + (load_prompt_ontology_text() + _LABEL_BOUNDARY if version != LOCATOR_VERSION else "")
        + f'\nReturn JSON only. schema_version must be "{version}". '
        + "Include every required field. No markdown. Observations must be brief visible facts, not reasoning traces.\n"
        + "Required output schema:\n" + json.dumps(schema_for(version), separators=(",", ":"))
    )
    payload["input_text"] = json.dumps({**visual, **(extra or {})}, sort_keys=True, separators=(",", ":"))
    payload["image_details"] = ["low"] * (len(base.images) - 1) + ["high"] * (1 + len(crops))
    payload["openrouter_image_detail_mode"] = "explicit_v1"
    return replace(
        base, payload=payload, images=base.images + tuple(crops),
        response_schema_version=version,
        prompt_version=version + "_causal_checked_v1",
    )


def run_grounded_target(
    base: ApiRequest,
    call: Callable[[str, ApiRequest], dict | None],
    *,
    proposal_slot: str,
    review_schema_version: str = REVIEW_VERSION,
) -> dict[str, Any]:
    """Run at most four calls, preserving valid H0 when optional stages fail.

``call(stage, request)`` returns that exact request's parsed response or None;
the caller owns transport, schema/HTTP audit, cache, cost and exception logging.
No externally prepared review or candidate can enter this function. Hypotheses,
their order, images and target are bound by the saved canonical review hash.
Phase always remains the H0 value. A valid H0 gives status OK even if the repair
is unavailable; stage_errors and admission explain every optional fallback.
    """
    if proposal_slot not in {"FIRST", "SECOND"}:
        raise ValueError("invalid blinded hypothesis slot")
    from surgical_agent.research.verification.grounded_repair import REVIEW_1000_VERSION

    if review_schema_version not in (REVIEW_VERSION, REVIEW_1000_VERSION):
        raise ValueError("unsupported contrast-review schema version")
    visual = _visual_input(base)
    row: dict[str, Any] = {
        "schema_version": GROUNDED_PIPELINE_VERSION,
        "checked_repair_version": CHECKED_REPAIR_VERSION,
        "video_id": visual["video_id"],
        "frame_id": visual["target_frame_id"],
        "status": "H0_FAILURE",
        "h0_payload": None, "h0": None, "h1": None, "final": None,
        "locator": None, "proposal": None, "review": None,
        "crop_manifest": [], "proposal_slot": proposal_slot,
        "review_binding": None, "review_request_hash": None,
        "request_hashes": {}, "stage_errors": {},
        "admission": {"decision": "KEEP", "reason": "H0_FAILURE"},
        "phase_policy": "keep_H0",
    }

    def invoke(stage, request):
        row["request_hashes"][stage] = canonical_request_metadata(request).request_hash
        try:
            payload = call(stage, request)
            if payload is None:
                row["stage_errors"][stage] = "NO_VALID_RESPONSE"
                return None
            validator_for(request.response_schema_version)(payload)
        except ApiError as exc:
            row["stage_errors"][stage] = getattr(getattr(exc, "cause", exc), "code", "api_error")
            return None
        except OverflowError:
            row["stage_errors"][stage] = "schema_error"
            return None
        return deepcopy(payload)

    h0_payload = invoke("h0", base)
    row["h0_payload"] = h0_payload
    if h0_payload is None:
        return row
    h0 = final_labels(h0_payload)
    row.update(status="OK", h0=h0, final=deepcopy(h0))
    locator = invoke("locator", _request(base, visual, LOCATOR_VERSION, LOCATOR_INSTRUCTION))
    row["locator"] = locator
    proposal = review = None
    if locator and locator["instances"] and locator["all_visible_tools_covered"]:
        try:
            crops, manifest = make_contact_crops(
                base.images[-1], locator, target_frame_id=row["frame_id"]
            )
        except (OSError, ValueError, ApiError, OverflowError):
            row["stage_errors"]["crop"] = "CROP_UNAVAILABLE"
        else:
            row["crop_manifest"] = manifest
            extra = {
                "predicted_instance_regions": locator,
                "crop_manifest": manifest,
                "image_order": (
                    f"first {len(base.images)} full causal images; the last full image is the target; "
                    "remaining target-frame crops in manifest order"
                ),
            }
            proposal = invoke("proposal", _request(
                base, visual, PROPOSAL_VERSION, PROPOSAL_INSTRUCTION, extra, crops
            ))
            row["proposal"] = proposal
            prepared = prepare_grounded_review(h0, locator, proposal, proposal_slot=proposal_slot)
            row["h1"] = prepared["h1"]
            if prepared["review_required"]:
                request = _request(
                    base, visual, review_schema_version, REVIEW_INSTRUCTION,
                    {**extra, "hypotheses": prepared["hypotheses"]}, crops,
                )
                request_hash = canonical_request_metadata(request).request_hash
                hypotheses_json = json.dumps(prepared["hypotheses"], sort_keys=True, separators=(",", ":"))
                row["review_request_hash"] = request_hash
                row["review_binding"] = {
                    "request_hash": request_hash,
                    "h0_request_hash": row["request_hashes"]["h0"],
                    "hypotheses_sha256": hashlib.sha256(hypotheses_json.encode()).hexdigest(),
                    "hypotheses": deepcopy(prepared["hypotheses"]),
                    "proposal_slot": proposal_slot,
                    "video_id": row["video_id"], "target_frame_id": row["frame_id"],
                }
                review = invoke("review", request)
                row["review"] = review
    result = finalize_grounded_repair(h0, locator, proposal, review, proposal_slot=proposal_slot,
                                      review_schema_version=review_schema_version)
    row.update(
        h1=result["h1"], final=result["final"],
        admission={"decision": result["decision"], "reason": result["reason"]},
    )
    return row
