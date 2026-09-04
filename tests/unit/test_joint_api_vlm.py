"""Behavioral tests for the causal joint-perception API boundary."""

from __future__ import annotations

import json

import pytest

from surgical_agent.api.contracts import ApiImageInput, ApiResponseRecord
from surgical_agent.api.errors import ApiContractError
from surgical_agent.api.openrouter_routing import (
    LATENCY_FALLBACK_ROUTING_PROFILE,
    OPENROUTER_ROUTING_PAYLOAD_KEY,
    STRICT_OPENAI_ROUTING_PROFILE,
)
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.schema import validator_for
from surgical_agent.config.schema import ApiConfig
from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.joint_api_vlm import (
    JointApiVlm,
    JointPerceptionRequestBuilder,
)
from surgical_agent.perception.ontology_prompt import (
    ACADEMIC_MEDICAL_CONTEXT,
    load_prompt_ontology_text,
)
from surgical_agent.perception.schema import (
    COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    JOINT_PERCEPTION_SCHEMA_VERSION,
    RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    TASK_LAYOUT,
)


def _config(**changes: object) -> ApiConfig:
    values: dict[str, object] = {
        "enabled": True,
        "mode": "mock",
        "provider": "mock",
        "endpoint_identifier": "mock://local/p3",
        "requested_model_identifier": "mock-joint-v1",
        "prompt_version": JOINT_PERCEPTION_SCHEMA_VERSION,
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
    track_snapshot: dict[str, object] | None = None,
    prior_finalized_prediction: PredictionRecord | None = None,
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
        prior_finalized_prediction=prior_finalized_prediction,
        track_snapshot=track_snapshot
        if track_snapshot is not None
        else {
            "status": "UNAVAILABLE",
            "source_max_frame_id": None,
            "frames": (),
        },
    )


def _joint_payload() -> dict[str, object]:
    payload: dict[str, object] = {"schema_version": JOINT_PERCEPTION_SCHEMA_VERSION}
    for task, count in TASK_LAYOUT:
        topk = [
            {"id": index, "score": 1.0 - index / (count + 1)} for index in range(count)
        ]
        payload[task] = (
            {"selected_id": 0, "topk": topk}
            if task == "phase"
            else {"selected_ids": [0], "topk": topk}
        )
    payload["evidence_refs"] = [{"frame_id": 12, "code": "CURRENT_VISUAL_SUPPORT"}]
    payload["self_reported_confidence"] = {task: 0.8 for task, _count in TASK_LAYOUT}
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


def _prior(
    *,
    frame_id: int,
    video_id: str = "VID01",
    causal_frame_ids: tuple[int, ...] | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        run_id="unit-run",
        video_id=video_id,
        frame_id=frame_id,
        source_split=DatasetSplit.TESTING,
        causal_frame_ids=(frame_id,) if causal_frame_ids is None else causal_frame_ids,
        instrument_ids=(0,),
        verb_ids=(0,),
        target_ids=(0,),
        triplet_ids=(0,),
        phase_id=0,
        granularity="frame_multilabel",
        backend="unit-test",
        gate_action="ACCEPT",
        verification_status="SKIPPED",
        alignment_version="unit-test",
        probabilities={
            task: (0.0,) * class_count
            for task, class_count in TASK_CLASS_COUNTS.items()
        },
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
        "selected_image_frame_ids": [10, 11, 12],
        "target_frame_id": 12,
        "temporal_evidence": {},
        "track_summary": {
            "frames": [],
            "source_max_frame_id": None,
            "status": "UNAVAILABLE",
        },
        "video_id": "VID01",
        "workflow_summary": {
            "observed_transitions": [],
            "phase_stability": None,
            "recent_finalized_phases": ["preparation"],
            "source_max_frame_id": 11,
        },
    }


def test_compact_request_builder_selects_compact_prompt_and_schema() -> None:
    config = _config(
        prompt_version=COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        response_schema_version=COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        generation_parameters={"max_output_tokens": 4096},
    )

    request = JointPerceptionRequestBuilder(config=config).build(_context())

    assert request.prompt_version == COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    assert request.response_schema_version == COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    normalized_prompt = " ".join(request.payload["system_text"].split())
    assert "3 instruments, 4 verbs, 5 targets, 8 IVT triplets" in normalized_prompt
    assert "at most 6 evidence references" in normalized_prompt


def test_reliability_compact_request_loads_v2_prompt_and_mock_payload() -> None:
    """Losing the v2 prompt registration or mock shape must break the API boundary."""

    config = _config(
        prompt_version=RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        response_schema_version=RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    )

    request = JointPerceptionRequestBuilder(config=config).build(_context())
    response = MockProviderTransport().send(request)

    assert request.prompt_version == RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    assert request.response_schema_version == (
        RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    )
    assert "Return JSON only" in request.payload["system_text"]
    assert set(response.parsed_payload) == {
        "schema_version",
        "instrument",
        "verb",
        "target",
        "ivt",
        "phase",
        "uncertainty",
    }
    assert set(response.parsed_payload["instrument"]["topk"][0]) == {
        "id",
        "confidence",
    }
    validator_for(RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION)(
        response.parsed_payload
    )


def test_gate_owned_request_excludes_model_uncertainty_and_status() -> None:
    config = _config(
        prompt_version=GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        response_schema_version=GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    )

    request = JointPerceptionRequestBuilder(config=config).build(_context())
    response = MockProviderTransport().send(request)
    normalized_prompt = " ".join(request.payload["system_text"].split())

    assert "selected_ids are sparse final predictions" in normalized_prompt
    assert "must never be padded" in normalized_prompt
    assert "separate local Gate" in normalized_prompt
    assert set(response.parsed_payload) == {
        "schema_version",
        "instrument",
        "verb",
        "target",
        "ivt",
        "phase",
    }
    assert set(response.parsed_payload["instrument"]["topk"][0]) == {
        "id",
        "score",
    }
    assert response.parsed_payload["instrument"]["selected_ids"] == (0,)
    assert response.parsed_payload["verb"]["selected_ids"] == (2,)
    assert response.parsed_payload["target"]["selected_ids"] == (1,)
    assert response.parsed_payload["ivt"]["selected_ids"] == (0,)
    validator_for(GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION)(
        response.parsed_payload
    )


def test_openrouter_prompt_adds_medical_context_without_replacing_ontology() -> None:
    config = _config(
        mode="real",
        provider="openrouter",
        endpoint_identifier="https://openrouter.ai/api/v1/chat/completions",
        requested_model_identifier="openai/gpt-5.6-sol",
        prompt_version=RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        response_schema_version=RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        generation_parameters={
            "max_output_tokens": 1536,
            "reasoning": {"effort": "none"},
        },
        provider_options={
            "timeout_seconds": 120.0,
            "routing_profile": STRICT_OPENAI_ROUTING_PROFILE,
        },
    )

    request = JointPerceptionRequestBuilder(config=config).build(_context())
    prompt = request.payload["system_text"]

    assert (
        request.payload[OPENROUTER_ROUTING_PAYLOAD_KEY]
        == STRICT_OPENAI_ROUTING_PROFILE
    )
    assert prompt.startswith(ACADEMIC_MEDICAL_CONTEXT)
    assert "5=cut" in prompt
    assert "5=blood_vessel" in prompt
    assert "2=clipping_and_cutting" in prompt
    assert "5=divide" not in prompt
    assert "5=vascular_structure" not in prompt
    canonical_ivt = load_prompt_ontology_text().splitlines()[5].split(":", 1)[1]
    assert len(canonical_ivt.split(";")) == 100


def test_openrouter_routing_profile_is_cache_bound_but_not_prompted() -> None:
    common = {
        "mode": "real",
        "provider": "openrouter",
        "endpoint_identifier": "https://openrouter.ai/api/v1/chat/completions",
        "requested_model_identifier": "openai/gpt-5.6-sol",
        "prompt_version": RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        "response_schema_version": (
            RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
        ),
        "generation_parameters": {
            "max_output_tokens": 1536,
            "reasoning": {"effort": "none"},
        },
    }
    strict = JointPerceptionRequestBuilder(
        config=_config(
            **common,
            provider_options={
                "timeout_seconds": 120.0,
                "routing_profile": STRICT_OPENAI_ROUTING_PROFILE,
            },
        )
    ).build(_context())
    efficient = JointPerceptionRequestBuilder(
        config=_config(
            **common,
            provider_options={
                "timeout_seconds": 120.0,
                "routing_profile": LATENCY_FALLBACK_ROUTING_PROFILE,
            },
        )
    ).build(_context())

    assert strict.payload[OPENROUTER_ROUTING_PAYLOAD_KEY] == (
        STRICT_OPENAI_ROUTING_PROFILE
    )
    assert efficient.payload[OPENROUTER_ROUTING_PAYLOAD_KEY] == (
        LATENCY_FALLBACK_ROUTING_PROFILE
    )
    assert json.loads(strict.payload["input_text"]) == json.loads(
        efficient.payload["input_text"]
    )
    assert OPENROUTER_ROUTING_PAYLOAD_KEY not in json.loads(
        strict.payload["input_text"]
    )
    assert canonical_request_metadata(strict).request_hash != (
        canonical_request_metadata(efficient).request_hash
    )


def test_request_includes_structured_workflow_and_causal_predicted_tracks() -> None:
    builder = JointPerceptionRequestBuilder(config=_config())
    request = builder.build(
        _context(
            workflow_snapshot={
                "source_max_frame_id": 11,
                "recent_finalized_phases": ("1", "2"),
                "phase_stability": 0.5,
                "observed_transitions": (("1", "2"),),
            },
            track_snapshot={
                "status": "AVAILABLE",
                "source_max_frame_id": 12,
                "frames": (
                    {
                        "frame_id": 12,
                        "tracks": (
                            {
                                "track_id": "tool-1",
                                "instrument_id": 0,
                                "bbox_tlwh": (0.1, 0.2, 0.3, 0.4),
                                "score": 0.9,
                                "age": 3,
                            },
                        ),
                    },
                ),
            },
        )
    )

    payload = json.loads(request.payload["input_text"])
    assert payload["workflow_summary"] == {
        "source_max_frame_id": 11,
        "recent_finalized_phases": ["1", "2"],
        "phase_stability": 0.5,
        "observed_transitions": [["1", "2"]],
    }
    assert payload["track_summary"]["source_max_frame_id"] == 12
    assert payload["track_summary"]["frames"][0]["tracks"][0]["track_id"] == ("tool-1")


def test_request_rejects_future_predicted_track_context() -> None:
    builder = JointPerceptionRequestBuilder(config=_config())

    with pytest.raises(ApiContractError, match="track source_max_frame_id"):
        builder.build(
            _context(
                track_snapshot={
                    "status": "AVAILABLE",
                    "source_max_frame_id": 13,
                    "frames": (
                        {
                            "frame_id": 13,
                            "tracks": (),
                        },
                    ),
                }
            )
        )


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


def test_backend_rejects_same_frame_prior_before_client_call() -> None:
    """Catches a target-frame prior being sent as causal context."""

    client = _SpyClient()
    backend = JointApiVlm(
        client=client,
        request_builder=JointPerceptionRequestBuilder(config=_config()),
    )

    with pytest.raises(ApiContractError, match="strictly earlier"):
        backend.predict(_context(prior_finalized_prediction=_prior(frame_id=12)))

    assert client.call_count == 0


def test_backend_rejects_future_prior_history_before_client_call() -> None:
    """Catches a prior whose causal history contains a future frame."""

    client = _SpyClient()
    backend = JointApiVlm(
        client=client,
        request_builder=JointPerceptionRequestBuilder(config=_config()),
    )

    with pytest.raises(ApiContractError, match="strictly earlier"):
        backend.predict(
            _context(
                prior_finalized_prediction=_prior(
                    frame_id=11,
                    causal_frame_ids=(10, 13, 11),
                )
            )
        )

    assert client.call_count == 0


def test_backend_rejects_cross_video_prior_before_client_call() -> None:
    """Catches a prior finalized for a different video entering the prompt."""

    client = _SpyClient()
    backend = JointApiVlm(
        client=client,
        request_builder=JointPerceptionRequestBuilder(config=_config()),
    )

    with pytest.raises(ApiContractError, match="same video"):
        backend.predict(
            _context(prior_finalized_prediction=_prior(frame_id=11, video_id="VID02"))
        )

    assert client.call_count == 0


@pytest.mark.parametrize(
    ("config_changes", "message"),
    [
        ({"requested_model_identifier": "other/model"}, "model"),
        ({"response_schema_version": "p3_multimodal_smoke_v1"}, "schema"),
        ({"prompt_version": "wrong-prompt"}, "prompt"),
    ],
    ids=("wrong-model", "wrong-schema", "wrong-prompt"),
)
def test_backend_rejects_non_joint_openrouter_config_before_client_call(
    config_changes: dict[str, object],
    message: str,
) -> None:
    """Catches an OpenRouter joint call labeled as GPT-5.6-Sol without its contract."""

    client = _SpyClient()
    openrouter_changes = {
        "requested_model_identifier": "openai/gpt-5.6-sol",
        **config_changes,
    }
    config = _config(
        mode="real",
        provider="openrouter",
        endpoint_identifier="https://openrouter.ai/api/v1/chat/completions",
        **openrouter_changes,
    )
    backend = JointApiVlm(
        client=client,
        request_builder=JointPerceptionRequestBuilder(config=config),
    )

    with pytest.raises(ApiContractError, match=message):
        backend.predict(_context())

    assert client.call_count == 0


def test_openrouter_joint_builder_admits_only_sol_and_luna() -> None:
    common = {
        "mode": "real",
        "provider": "openrouter",
        "endpoint_identifier": "https://openrouter.ai/api/v1/chat/completions",
        "prompt_version": RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        "response_schema_version": RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    }
    sol = _config(requested_model_identifier="openai/gpt-5.6-sol", **common)
    luna = _config(requested_model_identifier="openai/gpt-5.6-luna", **common)
    unknown = _config(requested_model_identifier="openai/gpt-5.6-mini", **common)

    assert JointPerceptionRequestBuilder(config=sol).build(
        _context()
    ).model_identifier == ("openai/gpt-5.6-sol")
    assert JointPerceptionRequestBuilder(config=luna).build(
        _context()
    ).model_identifier == ("openai/gpt-5.6-luna")
    with pytest.raises(ApiContractError, match="Sol/Luna"):
        JointPerceptionRequestBuilder(config=unknown).build(_context())


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
