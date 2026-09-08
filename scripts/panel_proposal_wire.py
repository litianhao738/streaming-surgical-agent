"""Isolated review-plus-proposal wire; proposals never constitute accepted edits.

The established five reviewers keep their images, routes, judgment semantics
and evidence schema. A new IVT is only a hypothesis for the next panel round.
This module performs no requests and has no access to GT or accepted answers.
"""

import json
from copy import deepcopy
from itertools import pairwise

from scripts.run_recent_mean_panel_trial import review_wire
from surgical_agent.perception.ontology_prompt import (
    _TASK_NAMES,
    load_prompt_ontology_text,
)
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import BOUNDS, COMPONENTS, TASKS

MAX_PROPOSED_IVTS = 2


def _request_pool(pool):
    """Rebuild canonical propositions so caller metadata cannot expose origins."""
    if not isinstance(pool, dict) or not isinstance(pool.get("propositions"), list):
        raise TypeError("candidate pool must contain a proposition list")
    propositions, seen = [], set()
    for item in pool["propositions"]:
        if not isinstance(item, dict):
            raise TypeError("invalid candidate proposition")
        task, label_id = item.get("task"), item.get("label_id")
        if (not isinstance(task, str) or task not in TASKS or type(label_id) is not int
                or not 0 <= label_id < BOUNDS[task]):
            raise ValueError("candidate outside ontology")
        pid = f"{task}_{label_id}"
        if item.get("id") != pid or pid in seen:
            raise ValueError("candidate ID must be unique and bound to its task and label")
        seen.add(pid)
        components = deepcopy(COMPONENTS[label_id]) if task == "ivt" else None
        name = ("/".join(_TASK_NAMES[q][c] for q, c in components.items())
                if components else _TASK_NAMES[task][label_id])
        propositions.append({"id": pid, "task": task, "label_id": label_id,
                             "name": name, "components": components})
    return {"propositions": propositions}


def _proposal_schema():
    return {
        "type": "object",
        "properties": {"ivt": {"type": "array", "items": {
            "type": "integer", "minimum": 0, "maximum": BOUNDS["ivt"] - 1},
            "maxItems": MAX_PROPOSED_IVTS, "uniqueItems": True}},
        "required": ["ivt"],
        "additionalProperties": False,
    }


def review_and_propose_wire(seat, base, selected, pool):
    """Extend the existing provider wire without exposing H0 or other votes.

The input response_schema and strict transport schema are the same object in
meaning. JSON-object providers retain local validation. Empty current pools
can still propose; an empty review is not proof that the scene is correct.
"""
    if seat not in SEATS:
        raise ValueError("unknown reviewer seat")
    frames = selected.get("causal_frame_ids")
    if (not isinstance(frames, (list, tuple)) or not 1 <= len(base.images) <= 3
            or len(frames) != len(base.images) or any(type(f) is not int for f in frames)
            or type(selected.get("frame_id")) is not int or frames[-1] != selected["frame_id"]
            or any(b - a != 25 for a, b in pairwise(frames))):
        raise ValueError("images must match the real contiguous causal target window")
    request_pool = _request_pool(pool)
    body = review_wire(seat, base, selected, request_pool)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    review_key = "rows" if seat == "gemini" else "judgments"
    schema = packet["response_schema"]
    # An empty enum is invalid JSON Schema even when maxItems=0 prevents rows.
    if seat == "gemini" and not request_pool["propositions"]:
        schema["properties"]["rows"]["items"]["properties"]["candidate_id"] = {"type": "string"}
    schema["properties"]["proposals"] = _proposal_schema()
    if seat == "gpt":
        # The actual OpenAI route rejects uniqueItems in strict response_format.
        # Distinctness remains in the instructions and the local validator.
        schema["properties"]["proposals"]["properties"]["ivt"].pop("uniqueItems")
    schema["required"].append("proposals")
    packet["instructions"] = packet["instructions"].replace(
        f"Return {review_key} only", f"Return {review_key} and proposals only")
    packet["ontology"] = load_prompt_ontology_text()
    packet["proposal_instructions"] = (
        "After independently reviewing every supplied candidate, inspect the same CURRENT frame "
        "for omitted instrument-action-target relations. You may propose at most 2 distinct IVT IDs "
        "from the supplied full ontology that are NOT already in propositions. Return fewer or [] "
        "when visual evidence is insufficient; do not fill a quota. Propose only a visually supported "
        "relation, not a merely legal combination, background organ, phase prior or history-only event. "
        "Output proposals as {\"ivt\": [IDs]}. Do not add a proposal to judgments or rows: only the "
        "supplied pool is reviewed this round. A proposal is not a repair or evidence of correctness. "
        "New proposals are eligible only for a later round where all five reviewers see the same "
        "expanded pool; the proposer's own score cannot accept them in this round. "
        "If the supplied pool is empty, return an empty review container and still check for proposals. "
        "Use English for observation text and the exact existing finding and scope enums."
    )
    if seat == "gemini":
        packet["wire_output"] = (
            "Return one object with rows and proposals. rows must contain exactly one object per "
            "supplied candidate_id, with candidate_id, rating, finding, scope, image_indices, observation. "
            "If no candidates are supplied, rows must be []. proposals must contain only an ivt list."
        )
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    if body["response_format"]["type"] == "json_schema":
        body["response_format"]["json_schema"].update(
            name="candidate_review_proposals_v1", schema=deepcopy(schema))
    return body


def split_review_and_proposals(raw, seat, pool):
    """Separate containers without letting a broken proposal erase a review.

    Returns (review_raw, proposal_dict, diagnostics). Preserve all actual review
    containers for the normalizer; never hide ambiguous containers by seat.
    Item-level visual/evidence validation remains the coordinator's job. Invalid
    or missing proposals yield empty lists with an explicit non-VALID status,
    never a claim that no omitted relation exists or that repair succeeded.
    """
    if seat not in SEATS:
        raise ValueError("unknown reviewer seat")
    request_pool = _request_pool(pool)
    proposals = {task: [] for task in TASKS}
    diagnostics = {
        "review_status": "MISSING",
        "proposal_status": "MISSING",
        "proposal_errors": [],
        "ignored_top_level_fields": [],
        "review_requires_separate_validation": True,
        "proposals_require_next_round_review": True,
    }
    if not isinstance(raw, dict):
        diagnostics.update(review_status="MALFORMED_RESPONSE", proposal_status="MALFORMED_RESPONSE")
        diagnostics["proposal_errors"].append("RESPONSE_NOT_OBJECT")
        return None, proposals, diagnostics
    review_keys = set(raw) & {"rows", "judgments"}
    review_raw = {key: deepcopy(raw[key]) for key in sorted(review_keys)} if review_keys else None
    if len(review_keys) == 1:
        review_key = next(iter(review_keys))
        expected_type = list if review_key == "rows" else dict
        diagnostics["review_status"] = (
            "EXTRACTED" if isinstance(raw[review_key], expected_type) else "MALFORMED_REVIEW_CONTAINER")
    elif len(review_keys) > 1:
        diagnostics["review_status"] = "AMBIGUOUS_REVIEW_CONTAINERS"
    diagnostics["ignored_top_level_fields"] = sorted(str(k) for k in set(raw) - review_keys - {"proposals"})
    if "proposals" not in raw:
        diagnostics["proposal_errors"].append("MISSING_PROPOSALS")
        return review_raw, proposals, diagnostics
    proposed = raw["proposals"]
    error = None
    if not isinstance(proposed, dict) or set(proposed) != {"ivt"}:
        error = "INVALID_PROPOSAL_FIELDS"
    else:
        ids = proposed["ivt"]
        existing = {p["label_id"] for p in request_pool["propositions"] if p["task"] == "ivt"}
        if not isinstance(ids, list):
            error = "PROPOSAL_IDS_NOT_LIST"
        elif len(ids) > MAX_PROPOSED_IVTS:
            error = "TOO_MANY_PROPOSALS"
        elif any(type(i) is not int or not 0 <= i < BOUNDS["ivt"] for i in ids):
            error = "INVALID_PROPOSAL_ID"
        elif len(set(ids)) != len(ids):
            error = "DUPLICATE_PROPOSAL_ID"
        elif existing.intersection(ids):
            error = "PROPOSAL_ALREADY_IN_POOL"
    if error:
        diagnostics.update(proposal_status="INVALID", proposal_errors=[error])
    else:
        proposals["ivt"] = list(proposed["ivt"])
        diagnostics["proposal_status"] = "VALID"
    return review_raw, proposals, diagnostics
