"""Deterministic non-network transport for P3 tests and smoke artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from surgical_agent.api.contracts import (
    ApiRequest,
    CompletionTokenDetails,
    ProviderResponse,
)
from surgical_agent.api.errors import ApiContractError, ApiTransportError
from surgical_agent.api.schema import (
    P3_SMOKE_SCHEMA_VERSION,
    TARGETED_VERIFICATION_SCHEMA_VERSION,
)
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION, TASKS
from surgical_agent.perception.schema import (
    COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    JOINT_PERCEPTION_SCHEMA_VERSION,
    RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    task_layout_for_schema_version,
)

if TYPE_CHECKING:
    from surgical_agent.config.schema import ApiConfig


class MockProviderTransport:
    provider = "mock"
    endpoint_identifier = "mock://local/p3"

    def __init__(
        self,
        *,
        returned_model_identifier: str = "mock-model-returned-v1",
        retryable_failures_before_success: int = 0,
        malformed_payload: bool = False,
        provider_cost: float | None = 0.0,
    ) -> None:
        if (
            not isinstance(retryable_failures_before_success, int)
            or isinstance(retryable_failures_before_success, bool)
        ):
            raise TypeError("retryable_failures_before_success must be an integer")
        if retryable_failures_before_success < 0:
            raise ValueError("retryable_failures_before_success must be non-negative")
        if not isinstance(returned_model_identifier, str) or not returned_model_identifier:
            raise ValueError("returned_model_identifier must be non-empty text")
        if type(malformed_payload) is not bool:
            raise TypeError("malformed_payload must be a boolean")
        if provider_cost is not None and (
            not isinstance(provider_cost, (int, float))
            or isinstance(provider_cost, bool)
            or not math.isfinite(float(provider_cost))
            or provider_cost < 0
        ):
            raise ValueError("provider_cost must be non-negative and finite")
        self.returned_model_identifier = returned_model_identifier
        self.retryable_failures_before_success = retryable_failures_before_success
        self.malformed_payload = malformed_payload
        self.provider_cost = provider_cost
        self.provider_call_count = 0

    @classmethod
    def from_config(
        cls,
        config: ApiConfig,
        overrides: Mapping[str, object],
    ) -> MockProviderTransport:
        """Construct a deterministic mock from validated provider options."""

        from surgical_agent.config.schema import ApiConfig as ConcreteApiConfig

        if not isinstance(config, ConcreteApiConfig):
            raise TypeError("config must be ApiConfig")
        if config.provider != cls.provider:
            raise ApiContractError("mock config provider does not match mock transport")
        if config.endpoint_identifier != cls.endpoint_identifier:
            raise ApiContractError("mock config endpoint does not match mock transport")
        if not isinstance(overrides, Mapping):
            raise TypeError("mock options must be a mapping")
        values = {**dict(config.provider_options), **dict(overrides)}
        transport_keys = {
            "returned_model_identifier",
            "retryable_failures_before_success",
            "malformed_payload",
            "provider_cost",
        }
        runtime_keys = {
            "frame_selection_strategy",
            "history_image_detail",
            "target_image_detail",
            "initial_prompt_profile",
            "verification_prompt_profile",
        }
        unknown = set(values) - transport_keys - runtime_keys
        if unknown:
            raise ApiContractError("mock options contain unknown fields")
        return cls(
            returned_model_identifier=values.get(
                "returned_model_identifier", "mock-model-returned-v1"
            ),
            retryable_failures_before_success=values.get(
                "retryable_failures_before_success", 0
            ),
            malformed_payload=values.get("malformed_payload", False),
            provider_cost=values.get("provider_cost", 0.0),
        )

    def send(self, request: ApiRequest) -> ProviderResponse:
        if request.provider != self.provider:
            raise ApiContractError("Request provider does not match mock transport")
        if request.endpoint_identifier != self.endpoint_identifier:
            raise ApiContractError("Request endpoint does not match mock transport")
        self.provider_call_count += 1
        if self.provider_call_count <= self.retryable_failures_before_success:
            raise ApiTransportError(
                "Injected transient mock failure",
                code="mock_transient",
                retryable=True,
            )
        if self.malformed_payload:
            payload: dict[str, object] = {"schema_version": "malformed"}
        elif request.response_schema_version == P3_SMOKE_SCHEMA_VERSION:
            payload = {
                "schema_version": P3_SMOKE_SCHEMA_VERSION,
                "message": "deterministic mock multimodal response",
                "image_observed": bool(request.images),
                "structured": True,
            }
        elif request.response_schema_version == FINAL_ONLY_SCHEMA_VERSION:
            selected = {"instrument": 0, "verb": 2, "target": 1, "ivt": 0, "phase": 0}
            payload = {
                "schema_version": FINAL_ONLY_SCHEMA_VERSION,
                **{
                    task: (
                        {"selected_id": selected[task]}
                        if task == "phase"
                        else {"selected_ids": [selected[task]]}
                    )
                    for task in TASKS
                },
            }
        elif request.response_schema_version in {
            JOINT_PERCEPTION_SCHEMA_VERSION,
            COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
            RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
            GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        }:
            payload = _joint_perception_payload(request)
        elif request.response_schema_version == TARGETED_VERIFICATION_SCHEMA_VERSION:
            payload = _targeted_verification_payload(request)
        else:
            raise ApiContractError("mock transport does not support response schema")
        return ProviderResponse(
            provider=self.provider,
            returned_model_identifier=self.returned_model_identifier,
            parsed_payload=payload,
            input_tokens=12,
            output_tokens=7,
            total_tokens=19,
            completion_tokens_details=CompletionTokenDetails(reasoning_tokens=0),
            visible_output_tokens=7,
            time_to_first_token_ms=0.0,
            total_latency_ms=0.0,
            image_count=len(request.images),
            provider_request_id=f"mock-request-{self.provider_call_count}",
            timestamp=datetime.now(UTC).isoformat(),
            provider_cost=self.provider_cost,
            safe_metadata={"finish_reason": "stop"},
        )


def _joint_perception_payload(request: ApiRequest) -> dict[str, object]:
    """Produce the one stable mock response admitted by the joint schema."""

    target_frame_id = 0
    input_text = request.payload.get("input_text")
    if isinstance(input_text, str):
        try:
            decoded = json.loads(input_text)
            candidate = (
                decoded.get("target_frame_id") if isinstance(decoded, Mapping) else None
            )
            if (
                isinstance(candidate, int)
                and not isinstance(candidate, bool)
                and candidate >= 0
            ):
                target_frame_id = candidate
        except (TypeError, ValueError):
            pass
    schema_version = request.response_schema_version
    task_layout = task_layout_for_schema_version(schema_version)
    confidence_key = (
        "confidence"
        if schema_version == RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
        else "score"
    )
    payload: dict[str, object] = {"schema_version": schema_version}
    gate_owned_selection = {
        "instrument": 0,
        "verb": 2,
        "target": 1,
        "ivt": 0,
        "phase": 0,
    }
    for task, count in task_layout:
        topk = [
            {"id": index, confidence_key: 1.0 - index / (count + 1)}
            for index in range(count)
        ]
        selected = (
            gate_owned_selection[task]
            if schema_version == GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
            else 0
        )
        payload[task] = (
            {"selected_id": selected, "topk": topk}
            if task == "phase"
            else {"selected_ids": [selected], "topk": topk}
        )
    if schema_version == RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
        payload["uncertainty"] = []
    elif schema_version != GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
        payload["evidence_refs"] = [
            {"frame_id": target_frame_id, "code": "CURRENT_VISUAL_SUPPORT"}
        ]
        payload["self_reported_confidence"] = {
            task: 0.8 for task, _count in task_layout
        }
    return payload


def _targeted_verification_payload(request: ApiRequest) -> dict[str, object]:
    """Select the first blind candidate as deterministic Verified mock output."""

    input_text = request.payload.get("input_text")
    try:
        decoded = json.loads(input_text) if isinstance(input_text, str) else None
    except (TypeError, ValueError):
        decoded = None
    if not isinstance(decoded, Mapping):
        raise ApiContractError("targeted mock request input must be an object")
    flagged_fields = decoded.get("flagged_fields")
    candidate_fields = decoded.get("candidate_fields")
    required_fields = decoded.get("required_fields", [])
    if (
        not isinstance(flagged_fields, list)
        or not isinstance(candidate_fields, Mapping)
        or not isinstance(required_fields, list)
    ):
        raise ApiContractError("targeted mock request fields are malformed")
    required_by_path = {
        item.get("path"): item.get("required_ids")
        for item in required_fields
        if isinstance(item, Mapping)
    }
    try:
        fields = []
        for path in flagged_fields:
            candidate_ids = candidate_fields[path]
            if (
                not isinstance(candidate_ids, list)
                or not candidate_ids
                or any(
                    not isinstance(candidate_id, int)
                    or isinstance(candidate_id, bool)
                    for candidate_id in candidate_ids
                )
            ):
                raise TypeError
            denominator = len(candidate_ids) + 1
            selected_ids = required_by_path.get(path, [candidate_ids[0]])
            if (
                not isinstance(selected_ids, list)
                or not selected_ids
                or not set(selected_ids).issubset(candidate_ids)
            ):
                raise TypeError
            fields.append(
                {
                    "path": path,
                    "selected_ids": selected_ids,
                    "topk": [
                        {
                            "id": candidate_id,
                            "confidence": 1.0 - rank / denominator,
                        }
                        for rank, candidate_id in enumerate(candidate_ids)
                    ],
                    "status": "Verified",
                    "uncertainty": None,
                }
            )
    except (KeyError, TypeError):
        raise ApiContractError("targeted mock request paths are inconsistent") from None
    return {
        "schema_version": TARGETED_VERIFICATION_SCHEMA_VERSION,
        "fields": fields,
    }
