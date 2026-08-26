"""Behavioral contracts for strict joint-perception response parsing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from surgical_agent.api.contracts import ApiResponseRecord
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.schema import schema_for, validator_for
from surgical_agent.perception.parser import parse_joint_perception_response
from surgical_agent.perception.schema import JOINT_PERCEPTION_SCHEMA_VERSION

TASK_COUNTS = {
    "instrument": 7,
    "verb": 10,
    "target": 15,
    "ivt": 20,
    "phase": 7,
}


def api_record(*, parsed_payload: dict[str, Any]) -> ApiResponseRecord:
    return ApiResponseRecord(
        provider="openrouter",
        endpoint_identifier="https://openrouter.ai/api/v1/chat/completions",
        request_hash="a" * 64,
        requested_model_identifier="openai/gpt-5.6-sol",
        returned_model_identifier="openai/gpt-5.6-sol",
        parsed_payload=parsed_payload,
    )


def ranked_candidates(count: int) -> list[dict[str, float | int]]:
    return [
        {"id": class_id, "score": round(0.99 - class_id * 0.01, 2)}
        for class_id in range(count)
    ]


def valid_joint_payload() -> dict[str, Any]:
    return {
        "schema_version": JOINT_PERCEPTION_SCHEMA_VERSION,
        "instrument": {"selected_ids": [0], "topk": ranked_candidates(7)},
        "verb": {"selected_ids": [1], "topk": ranked_candidates(10)},
        "target": {"selected_ids": [2], "topk": ranked_candidates(15)},
        "ivt": {"selected_ids": [12], "topk": ranked_candidates(20)},
        "phase": {"selected_id": 3, "topk": ranked_candidates(7)},
        "evidence_refs": [
            {"frame_id": 12, "code": "CURRENT_VISUAL_SUPPORT"},
        ],
        "self_reported_confidence": {
            "instrument": 0.8,
            "verb": 0.7,
            "target": 0.6,
            "ivt": 0.5,
            "phase": 0.4,
        },
    }


def mutate(payload: dict[str, Any], mutation: str) -> dict[str, Any]:
    value = deepcopy(payload)
    if mutation == "unknown_field":
        value["unexpected"] = "must not enter a result"
    elif mutation == "wrong_count":
        value["instrument"]["topk"].pop()
    elif mutation == "unsorted_scores":
        value["verb"]["topk"][:2] = value["verb"]["topk"][1::-1]
    elif mutation == "duplicate_id":
        value["target"]["topk"][1]["id"] = 0
    elif mutation == "selected_not_ranked":
        value["ivt"]["selected_ids"] = [99]
    elif mutation == "unknown_id":
        value["instrument"]["topk"][0]["id"] = 7
    elif mutation == "bad_evidence_code":
        value["evidence_refs"][0]["code"] = "FREE_FORM_REASONING"
    elif mutation == "future_ref":
        value["evidence_refs"][0]["frame_id"] = 13
    else:
        raise AssertionError(f"unknown test mutation: {mutation}")
    return value


def test_parser_reconstructs_dense_scores_and_selected_sets() -> None:
    """Removing zero-fill or selected-ID reconstruction must break this result."""

    result = parse_joint_perception_response(
        api_record(parsed_payload=valid_joint_payload()),
        video_id="VID02",
        frame_id=12,
        backend="joint_openrouter_gpt56sol",
    )

    assert len(result.prediction.probabilities["ivt"]) == 100
    assert sum(score > 0 for score in result.prediction.probabilities["ivt"]) == 20
    assert result.prediction.probabilities["ivt"][20:] == (0.0,) * 80
    assert result.prediction.instrument_ids == (0,)
    assert result.prediction.verb_ids == (1,)
    assert result.prediction.target_ids == (2,)
    assert result.prediction.triplet_ids == (12,)
    assert result.prediction.phase_id == 3
    assert result.prediction.score_semantics == "uncalibrated_rank_v1"
    assert result.raw_evidence.source_max_frame_id == 12
    assert result.raw_evidence.evidence_refs[0].frame_id == 12
    assert result.api_provenance.request_hash == "a" * 64


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_field",
        "wrong_count",
        "unsorted_scores",
        "duplicate_id",
        "selected_not_ranked",
        "unknown_id",
        "bad_evidence_code",
        "future_ref",
    ],
)
def test_parser_fails_closed_on_invalid_payload(mutation: str) -> None:
    """Dropping any schema or semantic guard must admit an invalid response."""

    with pytest.raises(ApiSchemaError):
        parse_joint_perception_response(
            api_record(parsed_payload=mutate(valid_joint_payload(), mutation)),
            video_id="VID02",
            frame_id=12,
            backend="joint_openrouter_gpt56sol",
        )


def test_parser_schema_failure_does_not_echo_untrusted_payload_content() -> None:
    """Echoing a provider's free-form content in errors must fail this boundary."""

    payload = valid_joint_payload()
    payload["unexpected"] = "provider secret chain of thought"

    with pytest.raises(ApiSchemaError) as error:
        parse_joint_perception_response(
            api_record(parsed_payload=payload),
            video_id="VID02",
            frame_id=12,
            backend="joint_openrouter_gpt56sol",
        )

    assert "provider secret chain of thought" not in str(error.value)


def test_joint_schema_is_registered_without_replacing_p3_registry_behavior() -> None:
    """Removing the joint registry entry must make schema lookup fail closed."""

    schema = schema_for(JOINT_PERCEPTION_SCHEMA_VERSION)

    assert schema["additionalProperties"] is False
    assert schema["properties"]["phase"]["required"] == ["selected_id", "topk"]
    assert validator_for(JOINT_PERCEPTION_SCHEMA_VERSION) is not None
