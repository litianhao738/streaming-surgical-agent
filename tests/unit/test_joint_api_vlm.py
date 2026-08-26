"""Behavioral tests for the causal joint-perception API boundary."""

from __future__ import annotations

import json

import pytest

from surgical_agent.api.contracts import ApiImageInput, ApiResponseRecord
from surgical_agent.api.errors import ApiContractError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.config.schema import ApiConfig
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.joint_api_vlm import (
    JointApiVlm,
    JointPerceptionRequestBuilder,
)
from surgical_agent.perception.schema import (
    JOINT_PERCEPTION_SCHEMA_VERSION,
    TASK_LAYOUT,
)


def _config(**changes: object) -> ApiConfig:
    values: dict[str, object] = {
        "enabled": True,
        "mode": "mock",
        "provider": "mock",
        "endpoint_identifier": "mock://local/p3",
        "requested_model_identifier": "mock-joint-v1",
        "prompt_version": "joint_perception_prompt_v1",
        "response_schema_version": JOINT_PERCEPTION_SCHEMA_VERSION,
        "generation_parameters": {"max_output_tokens": 256},
    }
    values.update(changes)
    return ApiConfig.from_mapping(values)


def _context(
    *,
    image_ids: tuple[str, ...] = ("synthetic:10", "synthetic:11", "synthetic:12"),
    image_order: tuple[int, ...] = (0, 1, 2),
    workflow_snapshot: dict[str, object] | None = None,
) -> PerceptionContext:
    images = tuple(
        ApiImageInput(identifier, "image/png", f"image-{index}".encode())
        for index, identifier in enumerate(image_ids)
    )
    selected_images = tuple(images[index] for index in image_order)
    sample = InferenceSample(
        video_id="VID01",
        target_frame_id=12,
        causal_frame_ids=(10, 11, 12),
        media_refs=("synthetic:10", "synthetic:11", "synthetic:12"),
        source_split=DatasetSplit.TESTING,
        alignment_version="unit-test",
    )
    return PerceptionContext(
        sample=sample,
        frames=None,  # The request boundary consumes the already-materialized images.
        images=selected_images,
        workflow_snapshot=workflow_snapshot
        if workflow_snapshot is not None
        else {
            "source_max_frame_id": 11,
            "recent_finalized_phases": ("preparation",),
        },
        memory_snapshot={},
        prior_finalized_prediction=None,
    )


def _joint_payload() -> dict[str, object]:
    payload: dict[str, object] = {"schema_version": JOINT_PERCEPTION_SCHEMA_VERSION}
    for task, count in TASK_LAYOUT:
        topk = [{"id": index, "score": 1.0 - index / (count + 1)} for index in range(count)]
        payload[task] = (
            {"selected_id": 0, "topk": topk}
            if task == "phase"
            else {"selected_ids": [0], "topk": topk}
        )
    payload["evidence_refs"] = [
        {"frame_id": 12, "code": "CURRENT_VISUAL_SUPPORT"}
    ]
    payload["self_reported_confidence"] = {
        task: 0.8 for task, _count in TASK_LAYOUT
    }
    return payload


class _SpyClient:
    def __init__(self) -> None:
        self.call_count = 0
        self.requests: list[object] = []

    def call(self, request: object) -> ApiResponseRecord:
        self.call_count += 1
        self.requests.append(request)
        return ApiResponseRecord(
            provider="mock",
            endpoint_identifier="mock://local/p3",
            request_hash="a" * 64,
            requested_model_identifier="mock-joint-v1",
            returned_model_identifier="mock-joint-v1",
            parsed_payload=_joint_payload(),
            image_count=3,
            provider_call_count=1,
        )


def test_three_image_request_hash_depends_on_order_and_context() -> None:
    """Catches a canonical builder that loses causal image ordering."""

    builder = JointPerceptionRequestBuilder(config=_config())

    first = builder.build(_context(image_order=(0, 1, 2)))
    reordered = builder.build(_context(image_order=(1, 0, 2)))

    assert canonical_request_metadata(first).request_hash != (
        canonical_request_metadata(reordered).request_hash
    )
    assert [image.identifier for image in first.images] == [
        "synthetic:10",
        "synthetic:11",
        "synthetic:12",
    ]
    assert json.loads(first.payload["input_text"]) == {
        "causal_frame_ids": [10, 11, 12],
        "ontology_version": "cholectrack20_v1",
        "prior_finalized_prediction": None,
        "target_frame_id": 12,
        "video_id": "VID01",
        "workflow_summary": {
            "recent_finalized_phases": ["preparation"],
            "source_max_frame_id": 11,
        },
    }


def test_false_upload_authorization_rejects_dataset_identifier_before_call() -> None:
    """Catches a backend that sends a dataset image despite the default policy."""

    client = _SpyClient()
    backend = JointApiVlm(
        client=client,
        request_builder=JointPerceptionRequestBuilder(config=_config()),
        data_upload_authorized=False,
    )

    with pytest.raises(ApiContractError, match="synthetic"):
        backend.predict(_context(image_ids=("D:/dataset/frame.png",), image_order=(0,)))

    assert client.call_count == 0


def test_request_builder_rejects_workflow_state_at_the_target_frame() -> None:
    """Catches a request that admits non-finalized workflow state as causal input."""

    builder = JointPerceptionRequestBuilder(config=_config())

    with pytest.raises(ApiContractError, match="strictly earlier"):
        builder.build(_context(workflow_snapshot={"source_max_frame_id": 12}))


def test_backend_calls_client_once_and_parses_the_joint_result() -> None:
    """Catches a backend that bypasses the cached client or returns raw payloads."""

    client = _SpyClient()
    backend = JointApiVlm(
        client=client,
        request_builder=JointPerceptionRequestBuilder(config=_config()),
        data_upload_authorized=False,
    )

    result = backend.predict(_context())

    assert client.call_count == 1
    assert result.prediction.backend == "joint_mock"
    assert result.raw_evidence.evidence_refs[0].frame_id == 12
    assert result.api_provenance.provider == "mock"
