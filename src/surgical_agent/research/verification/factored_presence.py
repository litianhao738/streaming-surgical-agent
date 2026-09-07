"""Optional, offline-only decomposition of a frozen names1000 presence request.

Nothing imports this adapter into a production or experiment runner by default.
Native response/request binding and dispatch accounting remain runner duties.
One call per proposition increases calls and total available generation tokens;
any future improvement cannot be attributed solely to context isolation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import replace

from surgical_agent.api.contracts import ApiRequest, thaw_json
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.ontology_prompt import _TASK_NAMES, _ivt_rows
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    validate_presence_review_shape,
)

NAMES1000_PROMPT_VERSION = PRESENCE_REVIEW_1000_VERSION + "_causal_v1_decoded_names_v1"
FACTORED_PRESENCE_PROMPT_VERSION = NAMES1000_PROMPT_VERSION + "_factored_v1"
_BOUNDS = {"instrument": 6, "verb": 9, "target": 14, "ivt": 99}
_PROPOSITION_ID = re.compile(r"p0(?:0[1-9]|[1-3][0-9]|40)")


def _source_body(request: ApiRequest) -> dict:
    """Reject a wrong source contract before any dispatch or response merge."""
    if (not isinstance(request, ApiRequest)
            or request.response_schema_version != PRESENCE_REVIEW_1000_VERSION
            or request.prompt_version != NAMES1000_PROMPT_VERSION):
        raise ValueError("Factored presence requires the original names1000 v3 request")
    try:
        body = json.loads(request.payload["input_text"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid names1000 input JSON") from exc
    if not isinstance(body, dict):
        raise TypeError("Names1000 input must be an object")
    propositions = body.get("propositions")
    if not isinstance(propositions, list) or not 1 <= len(propositions) <= 40:
        raise ValueError("Factored presence requires one to forty propositions")
    refs = body.get("allowed_evidence_refs")
    if (not isinstance(refs, list) or not refs
            or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
            or len(set(refs)) != len(refs)
            or not isinstance(body.get("full_frame_ref"), str)
            or body["full_frame_ref"] not in refs):
        raise ValueError("Names1000 source has invalid evidence references")
    identities, labels = set(), set()
    for proposition in propositions:
        if not isinstance(proposition, dict):
            raise TypeError("A proposition must be an object")
        task, label = proposition.get("task"), proposition.get("label_id")
        if (not isinstance(task, str) or task not in _BOUNDS
                or type(label) is not int or not 0 <= label <= _BOUNDS[task]):
            raise ValueError("A proposition has an invalid task or label ID")
        name_key = "decoded_components" if task == "ivt" else "decoded_label_name"
        if set(proposition) != {"proposition_id", "task", "label_id", "statement", name_key}:
            raise ValueError("Propositions must follow the neutral names1000 contract")
        identity, statement = proposition["proposition_id"], proposition["statement"]
        if (not isinstance(identity, str) or _PROPOSITION_ID.fullmatch(identity) is None
                or not isinstance(statement, str) or not statement.strip()):
            raise ValueError("Invalid proposition identity or statement")
        if identity in identities or (task, label) in labels:
            raise ValueError("Duplicate source proposition")
        identities.add(identity)
        labels.add((task, label))
        if task == "ivt":
            _, instrument, verb, target = _ivt_rows()[label]
            expected = {"instrument": _TASK_NAMES["instrument"][instrument],
                        "verb": _TASK_NAMES["verb"][verb], "target": _TASK_NAMES["target"][target]}
        else:
            expected = _TASK_NAMES[task][label]
        if proposition[name_key] != expected:
            raise ValueError("Decoded names disagree with the frozen ontology")
    return body


def split_factored_presence_request(request: ApiRequest) -> tuple[ApiRequest, ...]:
    """Copy one names1000 request per original opaque proposition, in order.

    Images, evidence manifests, system text, complete ontology, response schema
    and per-request generation parameters are retained verbatim. Only the
    propositions array and prompt version change. Invalid source inputs raise.
    """
    body = _source_body(request)
    result = []
    for proposition in body["propositions"]:
        payload = thaw_json(request.payload)
        payload["input_text"] = json.dumps(
            {**body, "propositions": [proposition]}, sort_keys=True, separators=(",", ":"))
        result.append(replace(request, payload=payload,
                              prompt_version=FACTORED_PRESENCE_PROMPT_VERSION))
    return tuple(result)


def merge_factored_presence_reviews(request: ApiRequest, reviews: Sequence) -> dict | None:
    """Merge complete valid v3 reviews in the corresponding split-request order.

    Any missing/invalid response returns None for the existing whole-review
    failure fallback. No model answer is synthesized. Evidence admission and
    A/B repair remain the responsibility of evaluate_presence_review; this
    adapter checks only structure, proposition association and allowed refs.
    Invalid source requests raise before response processing.
    """
    body = _source_body(request)
    propositions = body["propositions"]
    if (not isinstance(reviews, Sequence) or isinstance(reviews, (str, bytes))
            or len(reviews) != len(propositions)):
        return None
    refs, merged = set(body["allowed_evidence_refs"]), []
    for proposition, review in zip(propositions, reviews, strict=True):
        try:
            validate_presence_review_shape(review, schema_version=PRESENCE_REVIEW_1000_VERSION)
        except (ApiSchemaError, TypeError, ValueError, OverflowError):
            return None
        if len(review["assessments"]) != 1:
            return None
        assessment = review["assessments"][0]
        if (assessment["proposition_id"] != proposition["proposition_id"]
                or not set(assessment["evidence_refs"]) <= refs):
            return None
        # All other fields are immutable primitives under the validated schema.
        merged.append({key: list(value) if key == "evidence_refs" else value
                       for key, value in assessment.items()})
    return {"schema_version": PRESENCE_REVIEW_1000_VERSION, "assessments": merged}
