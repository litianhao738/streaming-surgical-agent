"""Behavioral contracts for strict joint-perception response parsing."""

from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from surgical_agent.api.contracts import ApiResponseRecord
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.schema import schema_for, validator_for
from surgical_agent.perception.parser import parse_joint_perception_response
from surgical_agent.perception.schema import (
    COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    COMPACT_TASK_LAYOUT,
    JOINT_PERCEPTION_SCHEMA_VERSION,
    RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    validate_compact_joint_perception_payload,
    validate_joint_perception_payload,
    validate_reliability_compact_joint_perception_payload,
)

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


def valid_compact_joint_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    }
    for task, count in COMPACT_TASK_LAYOUT:
        topk = ranked_candidates(count)
        payload[task] = (
            {"selected_id": topk[0]["id"], "topk": topk}
            if task == "phase"
            else {"selected_ids": [topk[0]["id"]], "topk": topk}
        )
    payload["evidence_refs"] = [
        {"frame_id": 12, "code": "CURRENT_VISUAL_SUPPORT"}
    ]
    payload["self_reported_confidence"] = {
        task: 0.8 for task, _count in COMPACT_TASK_LAYOUT
    }
    payload["ivt"] = {
        "selected_ids": [99],
        "topk": [
            {"id": class_id, "score": score}
            for class_id, score in zip(
                (99, 42, 7, 11, 35, 61, 2, 0),
                (0.99, 0.91, 0.83, 0.75, 0.67, 0.59, 0.51, 0.43),
                strict=True,
            )
        ],
    }
    return payload


def valid_reliability_compact_joint_payload() -> dict[str, Any]:
    """Return a hand-authored v2 payload with one bounded uncertainty finding."""

    payload: dict[str, Any] = {
        "schema_version": RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    }
    for task, count in COMPACT_TASK_LAYOUT:
        topk = [
            {"id": class_id, "confidence": round(0.99 - class_id * 0.01, 2)}
            for class_id in range(count)
        ]
        payload[task] = (
            {"selected_id": topk[0]["id"], "topk": topk}
            if task == "phase"
            else {"selected_ids": [topk[0]["id"]], "topk": topk}
        )
    payload["ivt"] = {
        "selected_ids": [99],
        "topk": [
            {"id": class_id, "confidence": confidence}
            for class_id, confidence in zip(
                (99, 42, 7, 11, 35, 61, 2, 0),
                (0.99, 0.91, 0.83, 0.75, 0.67, 0.59, 0.51, 0.43),
                strict=True,
            )
        ],
    }
    payload["uncertainty"] = [
        {
            "path": "/ivt/selected_ids",
            "reason": "CLOSE_ALTERNATIVES",
            "alternative_ids": [42],
        }
    ]
    return payload


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
    elif mutation == "duplicate_selected_id":
        value["target"]["selected_ids"] = [2, 2]
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


def test_compact_parser_zero_fills_full_evaluator_vectors() -> None:
    """Compact wire rankings must preserve the full downstream task contract."""

    payload = valid_compact_joint_payload()
    result = parse_joint_perception_response(
        api_record(parsed_payload=payload),
        video_id="VID30",
        frame_id=12,
        backend="joint_openrouter_gpt56sol",
    )

    assert {
        task: len(scores)
        for task, scores in result.prediction.probabilities.items()
    } == {
        "instrument": 7,
        "verb": 10,
        "target": 15,
        "ivt": 100,
        "phase": 7,
    }
    assert COMPACT_TASK_LAYOUT == (
        ("instrument", 3),
        ("verb", 4),
        ("target", 5),
        ("ivt", 8),
        ("phase", 3),
    )
    for task, compact_count in COMPACT_TASK_LAYOUT:
        scores = result.prediction.probabilities[task]
        assert sum(score > 0 for score in scores) == compact_count
        assert len(result.raw_evidence.ranked_candidates[task]) == compact_count
    assert result.prediction.probabilities["ivt"][99] == 0.99
    assert result.prediction.probabilities["ivt"][98] == 0.0
    assert result.prediction.triplet_ids == (99,)


def test_compact_schema_is_registered_and_bounds_evidence() -> None:
    schema = schema_for(COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION)

    assert schema["properties"]["ivt"]["properties"]["topk"]["maxItems"] == 8
    assert schema["properties"]["evidence_refs"]["maxItems"] == 6
    assert validator_for(COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION) is not None

    payload = valid_compact_joint_payload()
    validate_compact_joint_perception_payload(payload)
    payload["evidence_refs"] *= 7
    with pytest.raises(ApiSchemaError):
        validate_compact_joint_perception_payload(payload)


def test_reliability_compact_v2_parses_confidences_and_field_uncertainty() -> None:
    """Replacing v2 label confidences with task confidences must break this result."""

    result = parse_joint_perception_response(
        api_record(parsed_payload=valid_reliability_compact_joint_payload()),
        video_id="VID31",
        frame_id=12,
        backend="joint_openrouter_gpt56sol",
    )

    assert result.prediction.probabilities["ivt"][99] == 0.99
    assert result.prediction.probabilities["ivt"][98] == 0.0
    assert result.raw_evidence.ranked_candidates["ivt"][1].confidence == 0.91
    assert result.raw_evidence.self_reported_confidence == {
        "instrument": None,
        "verb": None,
        "target": None,
        "ivt": None,
        "phase": None,
    }
    assert result.raw_evidence.evidence_refs == ()
    assert result.raw_evidence.field_uncertainties[0].path == "/ivt/selected_ids"
    assert result.raw_evidence.field_uncertainties[0].alternative_ids == (42,)


@pytest.mark.parametrize(
    "mutation",
    [
        "status",
        "score_key",
        "duplicate_uncertainty_path",
        "invalid_uncertainty_alternative",
    ],
)
def test_reliability_compact_v2_rejects_forbidden_or_unbounded_content(
    mutation: str,
) -> None:
    """Dropping v2 root or uncertainty guards must admit model prose or bad pools."""

    payload = valid_reliability_compact_joint_payload()
    if mutation == "status":
        payload["status"] = "ACCEPTED"
    elif mutation == "score_key":
        candidate = payload["instrument"]["topk"][0]
        candidate["score"] = candidate.pop("confidence")
    elif mutation == "duplicate_uncertainty_path":
        payload["uncertainty"].append(
            {
                "path": "/ivt/selected_ids",
                "reason": "OCCLUSION",
                "alternative_ids": [],
            }
        )
    elif mutation == "invalid_uncertainty_alternative":
        payload["uncertainty"][0]["alternative_ids"] = [98]
    else:
        raise AssertionError(f"unknown mutation: {mutation}")

    with pytest.raises(ApiSchemaError):
        validate_reliability_compact_joint_perception_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_field",
        "wrong_count",
        "unsorted_scores",
        "duplicate_id",
        "duplicate_selected_id",
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


def test_provider_facing_joint_schema_omits_unsupported_unique_items() -> None:
    """Duplicate rejection remains semantic because provider schemas omit this keyword."""

    def mappings(value: object) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(value, dict):
            found.append(value)
            for child in value.values():
                found.extend(mappings(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(mappings(child))
        return found

    schema = schema_for(JOINT_PERCEPTION_SCHEMA_VERSION)

    assert all("uniqueItems" not in mapping for mapping in mappings(schema))


def test_public_schema_module_imports_in_a_clean_interpreter() -> None:
    """Restoring the API-to-schema import cycle must fail this public import."""

    repository_root = Path(__file__).parents[2]
    result = subprocess.run(
        [sys.executable, "-c", "import surgical_agent.perception.schema"],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_extreme_integer_score_fails_closed_at_validator_and_parser_boundaries() -> None:
    """Removing overflow handling must leak an OverflowError from either boundary."""

    payload = valid_joint_payload()
    payload["instrument"]["topk"][0]["score"] = 10**400

    for validate in (
        lambda: validate_joint_perception_payload(payload),
        lambda: parse_joint_perception_response(
            api_record(parsed_payload=payload),
            video_id="VID02",
            frame_id=12,
            backend="joint_openrouter_gpt56sol",
        ),
    ):
        with pytest.raises(ApiSchemaError) as error:
            validate()
        assert "100000" not in str(error.value)
