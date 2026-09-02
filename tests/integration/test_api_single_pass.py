"""Configured synthetic single-pass coverage for the canonical pipeline."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.run_api_single_pass import _synthetic_input, run_single_pass
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import ApiRequest, ProviderResponse
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import (
    ApiCallFailure,
    ApiContractError,
    ApiTransportError,
)
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.api.providers.openrouter import HttpResponse, OpenRouterTransport
from surgical_agent.api.registry import build_validator
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.usage import UsageLedger
from surgical_agent.config.loader import load_api_config, load_yaml
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.joint_api_vlm import JointPerceptionRequestBuilder
from surgical_agent.perception.schema import (
    JOINT_PERCEPTION_SCHEMA_VERSION,
    TASK_LAYOUT,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOCK_CONFIG = PROJECT_ROOT / "configs/perception/joint_mock.yaml"
REAL_CONFIG = PROJECT_ROOT / "configs/perception/joint_openrouter.yaml"
GATE_OWNED_MOCK_CONFIG = (
    PROJECT_ROOT / "configs/perception/joint_mock_gate_owned_smoke.yaml"
)
GATE_OWNED_REAL_CONFIG = (
    PROJECT_ROOT / "configs/perception/joint_openrouter_gate_owned_smoke.yaml"
)

EXPECTED_IMAGES = [
    {
        "identifier": "synthetic:joint:0",
        "sha256": "f72ef93fe46f9e67f2a9cb65217bfa3e436f481516626d53d98523367dd3bb14",
    },
    {
        "identifier": "synthetic:joint:1",
        "sha256": "e6e5416e5a29857028468fcd5bb64eafc9865fa8a9d9a85c243109a7ab8c0b93",
    },
    {
        "identifier": "synthetic:joint:2",
        "sha256": "d6e61eba692a158a889ab50388ad05853708a20b160850b10c029b1956dfc870",
    },
]


def _joint_payload(frame_id: int = 2) -> dict[str, object]:
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
    payload["evidence_refs"] = [
        {"frame_id": frame_id, "code": "CURRENT_VISUAL_SUPPORT"}
    ]
    payload["self_reported_confidence"] = {task: 0.8 for task, _count in TASK_LAYOUT}
    return payload


class CountingOpenRouterTransport:
    provider = "openrouter"
    endpoint_identifier = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        self.call_count = 0
        self.input_tokens: int | None = 101
        self.output_tokens: int | None = 37
        self.total_tokens: int | None = 138
        self.provider_cost: float | None = 0.00125

    def send(self, request: ApiRequest) -> ProviderResponse:
        self.call_count += 1
        return ProviderResponse(
            provider=self.provider,
            returned_model_identifier="openai/gpt-5.6-sol:injected",
            parsed_payload=_joint_payload(),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            image_count=len(request.images),
            provider_request_id="injected-response-1",
            timestamp=datetime.now(UTC).isoformat(),
            provider_cost=self.provider_cost,
            safe_metadata={},
        )


class RetryableFailureOpenRouterTransport:
    provider = "openrouter"
    endpoint_identifier = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        self.call_count = 0

    def send(self, request: ApiRequest) -> ProviderResponse:
        del request
        self.call_count += 1
        raise ApiTransportError(
            "provider detail must remain private",
            code="provider_busy",
            retryable=True,
        )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _persisted_text(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def test_joint_configs_freeze_exact_single_pass_identity_and_policy() -> None:
    real = load_api_config(REAL_CONFIG)
    mock = load_api_config(MOCK_CONFIG)

    assert {
        "enabled": real.enabled,
        "mode": real.mode,
        "provider": real.provider,
        "endpoint_identifier": real.endpoint_identifier,
        "requested_model_identifier": real.requested_model_identifier,
        "prompt_version": real.prompt_version,
        "response_schema_version": real.response_schema_version,
        "generation_parameters": dict(real.generation_parameters),
        "provider_options": dict(real.provider_options),
        "synthetic_input_required": real.synthetic_input_required,
        "cache_required": real.cache_required,
        "data_upload_authorized": real.data_upload_authorized,
        "max_causal_frames": real.max_causal_frames,
    } == {
        "enabled": True,
        "mode": "real",
        "provider": "openrouter",
        "endpoint_identifier": "https://openrouter.ai/api/v1/chat/completions",
        "requested_model_identifier": "openai/gpt-5.6-sol",
        "prompt_version": "joint_perception_frame_v1",
        "response_schema_version": "joint_perception_frame_v1",
        "generation_parameters": {
            "max_output_tokens": 4096,
            "reasoning": {"effort": "low"},
        },
        "provider_options": {"timeout_seconds": 120.0},
        "synthetic_input_required": True,
        "cache_required": True,
        "data_upload_authorized": False,
        "max_causal_frames": 3,
    }
    assert {
        "mode": mock.mode,
        "provider": mock.provider,
        "endpoint_identifier": mock.endpoint_identifier,
        "requested_model_identifier": mock.requested_model_identifier,
        "provider_options": dict(mock.provider_options),
    } == {
        "mode": "mock",
        "provider": "mock",
        "endpoint_identifier": "mock://local/p3",
        "requested_model_identifier": "mock-joint-perception-v1",
        "provider_options": {},
    }
    for field in (
        "enabled",
        "prompt_version",
        "response_schema_version",
        "generation_parameters",
        "synthetic_input_required",
        "cache_required",
        "data_upload_authorized",
        "max_causal_frames",
    ):
        assert getattr(mock, field) == getattr(real, field)


def test_gate_owned_synthetic_configs_are_exact_and_executable(
    tmp_path: Path,
) -> None:
    mock = load_api_config(GATE_OWNED_MOCK_CONFIG)
    real = load_api_config(GATE_OWNED_REAL_CONFIG)
    expected_version = "joint_perception_gate_owned_compact_v1"

    assert mock.prompt_version == expected_version
    assert mock.response_schema_version == expected_version
    assert real.prompt_version == expected_version
    assert real.response_schema_version == expected_version
    assert dict(mock.generation_parameters) == {
        "max_output_tokens": 1536,
        "reasoning": {"effort": "none"},
    }
    assert dict(real.generation_parameters) == dict(mock.generation_parameters)
    assert dict(real.provider_options) == {
        "timeout_seconds": 120.0,
        "routing_profile": "strict_openai",
    }
    assert mock.synthetic_input_required is True
    assert real.synthetic_input_required is True
    assert mock.data_upload_authorized is False
    assert real.data_upload_authorized is False

    artifact = run_single_pass(
        config=mock,
        output_dir=tmp_path / "gate-owned-smoke",
        api_key=None,
    )

    assert artifact["status"] == "MOCK_COMPLETE"
    assert artifact["prompt_version"] == expected_version
    assert artifact["response_schema_version"] == expected_version
    assert artifact["first"]["cache_hit"] is False
    assert artifact["second"]["cache_hit"] is True
    assert artifact["second"]["provider_call_count"] == 0

    experiment = load_yaml(PROJECT_ROOT / "configs/experiments/api_single_pass.yaml")
    assert experiment["comparison_group"] == "main_accuracy"
    assert experiment["backbone_policy"] == "shared"
    assert experiment["perception_config"] == (
        "configs/perception/joint_openrouter_dataset.yaml"
    )
    assert experiment["initial_model_requested"] == "openai/gpt-5.6-sol"
    assert experiment["verification_model_requested"] == "openai/gpt-5.6-sol"
    assert experiment["runtime"] == {
        "entrypoint": "scripts/run_dataset_api_pipeline.py",
        "pipeline_profile": "single_pass",
        "context_profile": "workflow",
        "event_memory_enabled": True,
        "phase_transition_graph": "artifacts/training/phase_transition_graph.json",
        "predicted_track_artifact": None,
        "allow_oracle_track_provider": False,
    }


def test_single_pass_mock_writes_one_safe_prediction_evidence_pair(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"

    artifact = run_single_pass(
        config=load_api_config(MOCK_CONFIG),
        output_dir=output_dir,
        api_key=None,
        run_id="integration-mock",
    )

    assert artifact["status"] == "MOCK_COMPLETE"
    assert artifact["model_requested"] == "mock-joint-perception-v1"
    assert artifact["model_returned"] == "mock-model-returned-v1"
    assert artifact["response_id"] == "mock-request-1"
    assert artifact["causal_frame_ids"] == [0, 1, 2]
    assert artifact["frame_tensor_shape"] == [3, 3, 32, 32]
    assert artifact["images"] == EXPECTED_IMAGES
    assert artifact["first"]["cache_hit"] is False
    assert artifact["first"]["provider_call_count"] == 1
    assert artifact["second"]["cache_hit"] is True
    assert artifact["second"]["provider_call_count"] == 0
    assert artifact["second"]["provider_cost"] == 0.0
    assert artifact["first"]["request_hash"] == artifact["request_hash"]
    assert artifact["second"]["request_hash"] == artifact["request_hash"]
    assert artifact["pipeline"] == {
        "run_count": 1,
        "gate_policy": "NeverVerify",
        "gate_action": "ACCEPT",
        "coordination_action": "KEEP",
        "verification_status": "NOT_REQUESTED",
    }
    assert artifact["paper_metric_eligible"] is False
    assert artifact["track20_image_uploaded"] is False

    prediction_path = output_dir / "predictions/SYNTHETIC01.jsonl"
    evidence_path = output_dir / "evidence/SYNTHETIC01.jsonl"
    assert len(prediction_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(evidence_path.read_text(encoding="utf-8").splitlines()) == 1
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert prediction["gate_action"] == "ACCEPT"
    assert prediction["verification_status"] == "NOT_REQUESTED"
    assert "10_keep_initial_prediction" in prediction["trace"]
    manifest = _read_json(output_dir / "manifest.json")
    assert manifest["record_count"] == 1
    assert manifest["metadata"] == {"paper_metric_eligible": False}
    assert _read_json(output_dir / artifact["artifact_file"]) == artifact
    assert artifact["cache_provenance"] == {
            "schema_version": "api_cache_entry_v3",
        "directory": "api_cache",
        "entry_file": f"api_cache/{artifact['request_hash']}.json",
        "request_hash": artifact["request_hash"],
    }
    assert (output_dir / artifact["cache_provenance"]["entry_file"]).is_file()

    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and "api_cache" not in path.relative_to(output_dir).parts
    ).lower()
    for forbidden in (
        "raw_response",
        "parsed_payload",
        "system_text",
        "input_text",
        "authorization",
        "bearer ",
        '"headers"',
        '"argv"',
        "ground_truth",
        "evaluation_target",
    ):
        assert forbidden not in persisted


def test_mock_single_pass_cache_is_safe_and_replays_with_a_fresh_client(
    tmp_path: Path,
) -> None:
    config = load_api_config(MOCK_CONFIG)
    output_dir = tmp_path / "run"
    artifact = run_single_pass(
        config=config,
        output_dir=output_dir,
        api_key=None,
        run_id="persistent-cache",
    )
    sample, frames = _synthetic_input()
    context_builder = CausalPerceptionContextBuilder(max_frames=3)
    request_builder = JointPerceptionRequestBuilder(config=config)
    request = request_builder.build(
        context_builder.build(
            sample,
            frames,
            workflow_snapshot={},
            memory_snapshot={},
            prior_finalized_prediction=None,
        )
    )
    metadata = canonical_request_metadata(request)
    cache = FileApiCache(output_dir / "api_cache")
    cache_files = sorted((output_dir / "api_cache").glob("*.json"))

    assert metadata.request_hash == artifact["request_hash"]
    assert cache_files == [
        output_dir / "api_cache" / f"{metadata.request_hash}.json"
    ]
    envelope = _read_json(cache_files[0])
    assert set(envelope) == {"schema_version", "request", "response"}
    assert envelope["schema_version"] == "api_cache_entry_v3"
    assert envelope["request"] == metadata.to_mapping()
    cached = cache.get(metadata)
    assert cached is not None
    assert cached.request_hash == metadata.request_hash
    assert cached.cache_hit is False
    assert cached.provider_call_count == 1

    cache_text = cache_files[0].read_text(encoding="utf-8").lower()
    assert "parsed_payload" in cache_text
    for forbidden in (
        "api_key",
        "raw_response",
        "system_text",
        "input_text",
        "authorization",
        "bearer ",
        '"headers"',
        "refusal",
        "data:image",
        "base64",
        "ivbor",
    ):
        assert forbidden not in cache_text

    replay_transport = MockProviderTransport()
    replay_usage = UsageLedger(tmp_path / "fresh_client/api_usage.jsonl")
    replay_client = CachedMultimodalApiClient(
        transport=replay_transport,
        cache=FileApiCache(output_dir / "api_cache"),
        usage=replay_usage,
        validator=build_validator(config),
        retry_policy=RetryPolicy(
            max_attempts=1,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
        ),
        sleep=lambda _seconds: None,
    )

    replay = replay_client.call(request)

    assert replay.request_hash == metadata.request_hash
    assert replay.cache_hit is True
    assert replay.provider_call_count == 0
    assert replay.retry_count == 0
    assert replay.provider_cost == 0.0
    assert replay_transport.provider_call_count == 0
    replay_rows = replay_usage.records()
    assert len(replay_rows) == 1
    assert replay_rows[0]["cache_hit"] is True
    assert replay_rows[0]["provider_call_count"] == 0


def test_injected_real_single_pass_uses_one_provider_call_and_reports_identity(
    tmp_path: Path,
) -> None:
    transport = CountingOpenRouterTransport()
    secret_text = "-".join(  # noqa: FLY002 - keep scan fixture non-contiguous
        ("injected", "credential", "material")
    )
    secret = SecretValue(secret_text)

    artifact = run_single_pass(
        config=load_api_config(REAL_CONFIG),
        output_dir=tmp_path / "run",
        api_key=secret,
        transport=transport,
        run_id="integration-real",
    )

    assert transport.call_count == 1
    assert artifact["status"] == "REAL_RESPONSE_RECEIVED"
    assert artifact["model_requested"] == "openai/gpt-5.6-sol"
    assert artifact["model_returned"] == "openai/gpt-5.6-sol:injected"
    assert artifact["response_id"] == "injected-response-1"
    assert artifact["first"]["provider_cost"] == 0.00125
    assert artifact["second"]["provider_call_count"] == 0
    assert artifact["second"]["provider_cost"] == 0.0
    assert secret_text not in _persisted_text(tmp_path / "run")


def test_actual_openrouter_transport_sends_exact_joint_request_body(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def sender(
        url: str,
        headers: object,
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse:
        del headers
        captured.update(url=url, body=body, timeout_seconds=timeout_seconds)
        response = {
            "id": "injected-http-response-1",
            "object": "chat.completion",
            "model": "openai/gpt-5.6-sol:injected-http",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(_joint_payload())},
                }
            ],
            "usage": {
                "prompt_tokens": 101,
                "completion_tokens": 37,
                "total_tokens": 138,
                "cost": 0.00125,
            },
        }
        return HttpResponse(
            status_code=200,
            headers={},
            body=json.dumps(response).encode("utf-8"),
        )

    secret = SecretValue(
        "-".join(  # noqa: FLY002 - keep scan fixture non-contiguous
            ("actual", "transport", "credential")
        )
    )
    transport = OpenRouterTransport(
        api_key=secret,
        endpoint_identifier="https://openrouter.ai/api/v1/chat/completions",
        sender=sender,
        timeout_seconds=120.0,
    )

    artifact = run_single_pass(
        config=load_api_config(REAL_CONFIG),
        output_dir=tmp_path / "run",
        api_key=secret,
        transport=transport,
        run_id="actual-openrouter-body",
    )

    sent = json.loads(captured["body"])
    content = sent["messages"][1]["content"]
    encoded_images = [item["image_url"]["url"] for item in content[1:]]
    ordered_hashes = [
        hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()
        for url in encoded_images
    ]
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["timeout_seconds"] == 120.0
    assert sent["model"] == "openai/gpt-5.6-sol"
    assert sent["max_tokens"] == 4096
    assert sent["reasoning"] == {"effort": "low"}
    assert "temperature" not in sent
    assert "top_p" not in sent
    assert sent["provider"] == {
        "only": ["openai"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert len(encoded_images) == 3
    assert ordered_hashes == [row["sha256"] for row in EXPECTED_IMAGES]
    assert artifact["first"]["provider_call_count"] == 1
    assert artifact["second"]["provider_call_count"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("input_tokens", None, id="missing-input-tokens"),
        pytest.param("output_tokens", None, id="missing-output-tokens"),
        pytest.param("total_tokens", None, id="missing-total-tokens"),
        pytest.param("provider_cost", None, id="missing-provider-cost"),
    ],
)
def test_real_single_pass_records_incomplete_accounting_before_pair_completion(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    transport = CountingOpenRouterTransport()
    setattr(transport, field, value)
    output_dir = tmp_path / "run"
    secret_text = "-".join(  # noqa: FLY002 - keep scan fixture non-contiguous
        ("incomplete", "accounting", "credential")
    )

    with pytest.raises(ApiCallFailure) as caught:
        run_single_pass(
            config=load_api_config(REAL_CONFIG),
            output_dir=output_dir,
            api_key=SecretValue(secret_text),
            transport=transport,
            run_id="incomplete-accounting",
        )

    assert caught.value.cause.code == "response_usage_invalid"
    assert caught.value.cause.retryable is False
    assert caught.value.attempt_count == 1
    assert caught.value.provider_call_count == 1
    assert caught.value.retry_count == 0
    assert transport.call_count == 1
    usage_lines = (output_dir / "api_usage.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(usage_lines) == 1
    usage_row = json.loads(usage_lines[0])
    assert usage_row["provider_call_count"] == 1
    assert usage_row["retry_count"] == 0
    assert usage_row["error"] == {
        "code": "response_usage_invalid",
        "retryable": False,
        "status_code": None,
    }
    assert usage_row["returned_model_identifier"] is None
    assert usage_row["provider_request_id"] is None
    assert usage_row["input_tokens"] is None
    assert usage_row["output_tokens"] is None
    assert usage_row["total_tokens"] is None
    assert usage_row["provider_cost"] is None
    persisted = "\n".join(usage_lines).lower()
    for forbidden in (
        "raw_response",
        "parsed_payload",
        "system_text",
        "input_text",
        "authorization",
        "bearer ",
        '"headers"',
        "refusal",
        secret_text,
    ):
        assert forbidden not in persisted
    assert not (output_dir / "single_pass_artifact.json").exists()
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / "predictions/SYNTHETIC01.jsonl").exists()
    assert not (output_dir / "evidence/SYNTHETIC01.jsonl").exists()


def test_real_single_pass_never_retries_the_authorized_provider_call(
    tmp_path: Path,
) -> None:
    transport = RetryableFailureOpenRouterTransport()

    with pytest.raises(ApiCallFailure) as caught:
        run_single_pass(
            config=load_api_config(REAL_CONFIG),
            output_dir=tmp_path / "run",
            api_key=SecretValue(
                "-".join(  # noqa: FLY002 - keep scan fixture non-contiguous
                    ("one", "attempt", "credential")
                )
            ),
            transport=transport,
            run_id="single-attempt-real",
        )

    assert caught.value.provider_call_count == 1
    assert transport.call_count == 1


def test_real_requires_credential_before_output_mutation(tmp_path: Path) -> None:
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ApiContractError, match="credential"):
        run_single_pass(
            config=load_api_config(REAL_CONFIG),
            output_dir=output_dir,
            api_key=None,
        )

    assert not output_dir.exists()


def test_mock_rejects_credential_before_output_mutation(tmp_path: Path) -> None:
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ApiContractError, match="credential"):
        run_single_pass(
            config=load_api_config(MOCK_CONFIG),
            output_dir=output_dir,
            api_key=SecretValue("not-for-mock"),
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "config",
    [
        replace(load_api_config(MOCK_CONFIG), enabled=False),
        replace(load_api_config(MOCK_CONFIG), prompt_version="wrong"),
        replace(load_api_config(MOCK_CONFIG), response_schema_version="wrong"),
        replace(load_api_config(MOCK_CONFIG), requested_model_identifier="wrong"),
        replace(load_api_config(MOCK_CONFIG), synthetic_input_required=False),
        replace(load_api_config(MOCK_CONFIG), cache_required=False),
        replace(load_api_config(MOCK_CONFIG), data_upload_authorized=True),
        replace(load_api_config(MOCK_CONFIG), max_causal_frames=2),
        replace(
            load_api_config(MOCK_CONFIG),
            generation_parameters={"max_output_tokens": 4096},
        ),
        replace(
            load_api_config(MOCK_CONFIG),
            generation_parameters={"max_output_tokens": 4096, "temperature": 0.0},
        ),
        replace(
            load_api_config(MOCK_CONFIG),
            provider_options={"timeout_seconds": 120.0},
        ),
    ],
)
def test_nonexact_config_is_rejected_before_output_mutation(
    tmp_path: Path,
    config: object,
) -> None:
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ApiContractError):
        run_single_pass(
            config=config,  # type: ignore[arg-type]
            output_dir=output_dir,
            api_key=None,
        )

    assert not output_dir.exists()


def test_mock_cli_prints_only_safe_status_and_artifact_path(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_api_single_pass.py",
            "--config",
            str(MOCK_CONFIG),
            "--output-root",
            str(tmp_path),
            "--run-id",
            "cli-mock",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    artifact_path = (tmp_path / "cli-mock/single_pass_artifact.json").resolve()
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout == (
        f"JOINT_SINGLE_PASS_MOCK_COMPLETE artifact={artifact_path}\n"
    )
    assert artifact_path.is_file()


def test_real_cli_failure_emits_only_safe_category(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_api_single_pass.py",
            "--real",
            "--config",
            str(REAL_CONFIG),
            "--output-root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr == "JOINT_SINGLE_PASS_FAILED category=contract_error\n"
    assert "Traceback" not in completed.stderr
    assert not list(tmp_path.iterdir())
