"""Run one gold-free synthetic state through the canonical perception pipeline."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import ApiRequest, ProviderResponse, ProviderTransport
from surgical_agent.api.credentials import (
    SecretValue,
    assert_secret_absent,
    resolve_api_key,
)
from surgical_agent.api.errors import (
    ApiCallFailure,
    ApiContractError,
    ApiError,
    ApiTransportError,
)
from surgical_agent.api.registry import build_transport, build_validator
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.config.loader import load_api_config
from surgical_agent.config.schema import ApiConfig
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.frame_result_writer import FrameResultWriter
from surgical_agent.inference.writer import ArtifactWriteError
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.joint_api_vlm import (
    JointApiVlm,
    JointPerceptionRequestBuilder,
)
from surgical_agent.research.signals.frame_evidence import (
    FrameEvidenceSignalExtractor,
)
from surgical_agent.systems.pipeline import (
    CanonicalStreamingPipeline,
    DisabledCandidateGenerator,
    DisabledSpecialistRegistry,
    NeverVerify,
    NoOpCausalStore,
    NoOpCoordinator,
    PipelineComponents,
    PredictionFinalizer,
)

_JOINT_VERSION = "joint_perception_frame_v1"
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_REAL_MODEL = "openai/gpt-5.6-sol"
_MOCK_MODEL = "mock-joint-perception-v1"
_MOCK_ENDPOINT = "mock://local/p3"
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SAFE_CALL_FIELDS = (
    "request_hash",
    "cache_hit",
    "provider_call_count",
    "retry_count",
    "provider_cost",
    "origin_provider_cost",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "latency_ms",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the mock/real CLI without credential-bearing defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/perception/joint_mock.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/joint_perception",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--real", action="store_true")
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--api-key-file", type=Path)
    credentials.add_argument("--api-key")
    return parser


def _require_exact_config(config: ApiConfig, *, has_credential: bool) -> None:
    """Fail closed on every identity and policy field before any output exists."""

    if not isinstance(config, ApiConfig):
        raise TypeError("config must be an ApiConfig")
    if type(has_credential) is not bool:
        raise TypeError("has_credential must be boolean")
    config.require_enabled()
    config.validate()
    if config.prompt_version != _JOINT_VERSION:
        raise ApiContractError("single pass requires the exact prompt version")
    if config.response_schema_version != _JOINT_VERSION:
        raise ApiContractError("single pass requires the exact response schema")
    generation = dict(config.generation_parameters)
    if (
        set(generation) != {"max_output_tokens", "reasoning"}
        or type(generation["max_output_tokens"]) is not int
        or generation["max_output_tokens"] != 4096
        or not isinstance(generation["reasoning"], Mapping)
        or dict(generation["reasoning"]) != {"effort": "low"}
    ):
        raise ApiContractError("single pass requires the exact generation settings")
    if config.synthetic_input_required is not True:
        raise ApiContractError("single pass requires synthetic input")
    if config.cache_required is not True:
        raise ApiContractError("single pass requires the request cache")
    if config.data_upload_authorized is not False:
        raise ApiContractError("single pass forbids dataset upload authorization")
    if config.max_causal_frames != 3:
        raise ApiContractError("single pass requires exactly three causal frames")

    if config.mode == "real" and config.provider == "openrouter":
        if config.endpoint_identifier != _OPENROUTER_ENDPOINT:
            raise ApiContractError("real single pass requires the OpenRouter endpoint")
        if config.requested_model_identifier != _REAL_MODEL:
            raise ApiContractError(
                "real single pass requires the exact requested model"
            )
        options = dict(config.provider_options)
        if (
            set(options) != {"timeout_seconds"}
            or type(options["timeout_seconds"]) is not float
            or options["timeout_seconds"] != 120.0
        ):
            raise ApiContractError("real single pass requires the exact timeout")
        if not has_credential:
            raise ApiContractError("real single pass requires exactly one credential")
        return

    if config.mode == "mock" and config.provider == "mock":
        if config.endpoint_identifier != _MOCK_ENDPOINT:
            raise ApiContractError("mock single pass requires the exact endpoint")
        if config.requested_model_identifier != _MOCK_MODEL:
            raise ApiContractError(
                "mock single pass requires the exact requested model"
            )
        if dict(config.provider_options):
            raise ApiContractError("mock single pass rejects provider options")
        if has_credential:
            raise ApiContractError("mock single pass rejects credential inputs")
        return

    raise ApiContractError("single pass requires an exact mock or real configuration")


def _require_transport_identity(
    config: ApiConfig,
    transport: ProviderTransport,
) -> None:
    if not hasattr(transport, "send") or not callable(transport.send):
        raise ApiContractError("transport must provide send(request)")
    if transport.provider != config.provider:
        raise ApiContractError("transport provider does not match config")
    if transport.endpoint_identifier != config.endpoint_identifier:
        raise ApiContractError("transport endpoint does not match config")


def _require_complete_real_accounting(response: ProviderResponse) -> None:
    token_counts = (
        response.input_tokens,
        response.output_tokens,
        response.total_tokens,
    )
    if any(type(value) is not int or value < 0 for value in token_counts):
        raise ApiTransportError(
            "real single-pass accounting is incomplete",
            code="response_usage_invalid",
            retryable=False,
        )
    cost = response.provider_cost
    if (
        not isinstance(cost, (int, float))
        or isinstance(cost, bool)
        or not math.isfinite(float(cost))
        or cost < 0
    ):
        raise ApiTransportError(
            "real single-pass accounting is incomplete",
            code="response_usage_invalid",
            retryable=False,
        )


class _RealAccountingTransport:
    """Fail before parsing/persistence when an origin response lacks accounting."""

    def __init__(self, transport: ProviderTransport) -> None:
        self._transport = transport
        self.provider = transport.provider
        self.endpoint_identifier = transport.endpoint_identifier

    def send(self, request: ApiRequest) -> ProviderResponse:
        response = self._transport.send(request)
        _require_complete_real_accounting(response)
        return response


def _synthetic_input() -> tuple[InferenceSample, torch.Tensor]:
    """Return three distinct deterministic 32x32 RGB frames and safe references."""

    frames = torch.zeros((3, 3, 32, 32), dtype=torch.float32)
    frames[0, 0] = 1.0
    frames[1, 1] = 1.0
    frames[2, 2] = 1.0
    sample = InferenceSample(
        video_id="SYNTHETIC01",
        target_frame_id=2,
        causal_frame_ids=(0, 1, 2),
        media_refs=(
            "synthetic:joint:0",
            "synthetic:joint:1",
            "synthetic:joint:2",
        ),
        source_split=DatasetSplit.TESTING,
        alignment_version="synthetic_joint_single_pass_v1",
    )
    return sample, frames


def _safe_call_summary(record: Mapping[str, Any]) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise ApiContractError("usage record is unavailable")
    try:
        return {field: record[field] for field in _SAFE_CALL_FIELDS}
    except KeyError:
        raise ApiContractError("usage record is incomplete") from None


def _assert_call_contract(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> None:
    if first["cache_hit"] is not False or first["provider_call_count"] != 1:
        raise ApiContractError("first single-pass request must be one cache miss")
    if second["cache_hit"] is not True:
        raise ApiContractError("single-pass cache probe must be a cache hit")
    if second["provider_call_count"] != 0:
        raise ApiContractError("cache probe must not call the provider")
    if second["provider_cost"] != 0.0:
        raise ApiContractError("cache probe must have zero current provider cost")
    if first["request_hash"] != second["request_hash"]:
        raise ApiContractError("cache probe changed the canonical request hash")


def _assert_secret_absent_from_runtime(
    api_key: SecretValue,
    output_dir: Path,
) -> None:
    """Scan tracked source and this run, excluding ignored credential files."""

    tracked = (
        subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
        .stdout.decode("utf-8")
        .split("\0")
    )
    paths = [
        *(PROJECT_ROOT / item for item in tracked if item),
        *(path for path in output_dir.rglob("*") if path.is_file()),
    ]
    try:
        assert_secret_absent(api_key, paths)
    except RuntimeError:
        raise ApiContractError(
            "credential scan detected persisted credential"
        ) from None


def run_single_pass(
    *,
    config: ApiConfig,
    output_dir: Path,
    api_key: SecretValue | None,
    transport: ProviderTransport | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    """Run the canonical pipeline once, then replay its original request from cache."""

    if api_key is not None and not isinstance(api_key, SecretValue):
        raise TypeError("api_key must be SecretValue or None")
    _require_exact_config(config, has_credential=api_key is not None)
    effective_run_id = "joint-single-pass" if run_id is None else run_id
    if not isinstance(effective_run_id, str) or not _SAFE_RUN_ID.fullmatch(
        effective_run_id
    ):
        raise ApiContractError("run_id is not a safe artifact identifier")
    selected_transport = transport or build_transport(config, api_key=api_key)
    _require_transport_identity(config, selected_transport)
    validator = build_validator(config)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ApiContractError("single-pass output directory must be fresh")
    destination.mkdir(parents=True, exist_ok=False)

    usage = UsageLedger(destination / "api_usage.jsonl")
    writer = FrameResultWriter(destination, run_id=effective_run_id)
    context_builder = CausalPerceptionContextBuilder(
        max_frames=config.max_causal_frames
    )
    request_builder = JointPerceptionRequestBuilder(config=config)
    sample, frames = _synthetic_input()

    with tempfile.TemporaryDirectory(prefix="joint-perception-cache-") as cache_root:
        client_transport = (
            _RealAccountingTransport(selected_transport)
            if config.mode == "real"
            else selected_transport
        )
        client = CachedMultimodalApiClient(
            transport=client_transport,
            cache=FileApiCache(cache_root),
            usage=usage,
            validator=validator,
            retry_policy=RetryPolicy(
                max_attempts=1,
                base_delay_seconds=0.0,
                max_delay_seconds=0.0,
            ),
            sleep=lambda _seconds: None,
        )
        pipeline = CanonicalStreamingPipeline(
            PipelineComponents(
                context_builder=context_builder,
                perception=JointApiVlm(
                    client=client,
                    request_builder=request_builder,
                    data_upload_authorized=config.data_upload_authorized,
                ),
                candidate_generator=DisabledCandidateGenerator(),
                signal_extractor=FrameEvidenceSignalExtractor(),
                gate_policy=NeverVerify(),
                specialist_registry=DisabledSpecialistRegistry(),
                coordinator=NoOpCoordinator(),
                finalizer=PredictionFinalizer(),
                workflow_store=NoOpCausalStore("workflow"),
                event_memory=NoOpCausalStore("memory"),
                result_sink=writer,
            )
        )
        pipeline_result = pipeline.run(sample, frames, run_id=effective_run_id)
        first_rows = usage.records()
        if len(first_rows) != 1:
            raise ApiContractError("pipeline must produce exactly one logical API call")

        probe_context = context_builder.build(
            sample,
            frames,
            workflow_snapshot={},
            memory_snapshot={},
            prior_finalized_prediction=None,
        )
        probe_request = request_builder.build(probe_context)
        probe_metadata = canonical_request_metadata(probe_request)
        if first_rows[0]["request_hash"] != probe_metadata.request_hash:
            raise ApiContractError(
                "rebuilt no-prior request differs from pipeline request"
            )
        client.call(probe_request)

    rows = usage.records()
    if len(rows) != 2:
        raise ApiContractError("single pass requires one call and one cache probe")
    first, second = rows
    _assert_call_contract(first, second)
    image_rows = [
        {"identifier": image.identifier, "sha256": image.sha256}
        for image in probe_metadata.images
    ]
    if [row["identifier"] for row in image_rows] != list(sample.media_refs) or len(
        {row["sha256"] for row in image_rows}
    ) != 3:
        raise ApiContractError("synthetic image provenance is incomplete or unordered")
    prediction = pipeline_result.prediction
    if (
        prediction.gate_action != "ACCEPT"
        or prediction.verification_status != "NOT_REQUESTED"
        or "10_keep_initial_prediction" not in prediction.trace
    ):
        raise ApiContractError("single pass did not preserve NeverVerify KEEP")

    manifest_path = writer.finalize({"paper_metric_eligible": False})
    artifact_file = "single_pass_artifact.json"
    artifact: dict[str, object] = {
        "schema_version": "joint_perception_single_pass_artifact_v1",
        "status": (
            "REAL_RESPONSE_RECEIVED" if config.mode == "real" else "MOCK_COMPLETE"
        ),
        "run_id": effective_run_id,
        "provider": config.provider,
        "endpoint_identifier": config.endpoint_identifier,
        "prompt_version": config.prompt_version,
        "response_schema_version": config.response_schema_version,
        "model_requested": first["requested_model_identifier"],
        "model_returned": first["returned_model_identifier"],
        "response_id": first["provider_request_id"],
        "request_hash": first["request_hash"],
        "causal_frame_ids": list(sample.causal_frame_ids),
        "frame_tensor_shape": list(frames.shape),
        "images": image_rows,
        "first": _safe_call_summary(first),
        "second": _safe_call_summary(second),
        "usage": usage.summarize(),
        "pipeline": {
            "run_count": 1,
            "gate_policy": "NeverVerify",
            "gate_action": prediction.gate_action,
            "coordination_action": "KEEP",
            "verification_status": prediction.verification_status,
        },
        "paper_metric_eligible": False,
        "track20_image_uploaded": False,
        "paired_artifacts": {
            "predictions_file": "predictions/SYNTHETIC01.jsonl",
            "evidence_file": "evidence/SYNTHETIC01.jsonl",
            "manifest_file": manifest_path.relative_to(destination).as_posix(),
            "api_usage_file": "api_usage.jsonl",
        },
        "artifact_file": artifact_file,
    }
    atomic_write_json(destination / artifact_file, artifact)
    if config.mode == "real":
        assert api_key is not None
        _assert_secret_absent_from_runtime(api_key, destination)
    return artifact


def run(args: argparse.Namespace) -> Path:
    """Resolve explicit mode and credentials, then execute one fresh run."""

    config = load_api_config(args.config)
    has_credential = args.api_key is not None or args.api_key_file is not None
    if args.real:
        if config.mode != "real" or config.provider != "openrouter":
            raise ApiContractError("--real requires the exact real configuration")
        if not has_credential:
            raise ApiContractError("real single pass requires exactly one credential")
        api_key = resolve_api_key(
            api_key=args.api_key,
            api_key_file=args.api_key_file,
        )
        default_prefix = "joint_real"
    else:
        if config.mode != "mock" or config.provider != "mock":
            raise ApiContractError("mock mode requires the exact mock configuration")
        if has_credential:
            raise ApiContractError("mock single pass rejects credential inputs")
        api_key = None
        default_prefix = "joint_mock"
    run_id = args.run_id or datetime.now(UTC).strftime(
        f"{default_prefix}_%Y%m%dT%H%M%SZ"
    )
    output_dir = args.output_root.expanduser().resolve() / run_id
    artifact = run_single_pass(
        config=config,
        output_dir=output_dir,
        api_key=api_key,
        run_id=run_id,
    )
    artifact_path = output_dir / str(artifact["artifact_file"])
    print(f"JOINT_SINGLE_PASS_{artifact['status']} artifact={artifact_path}")
    return artifact_path


def _safe_error_category(exc: BaseException) -> str:
    if isinstance(exc, ApiCallFailure):
        return exc.cause.code
    if isinstance(exc, ApiError):
        return exc.code
    if isinstance(exc, ArtifactWriteError):
        return "artifact_error"
    if isinstance(exc, UnicodeError):
        return "invalid_text"
    if isinstance(exc, OSError):
        return "io_error"
    if isinstance(exc, (TypeError, ValueError)):
        return "invalid_input"
    return "internal_error"


def main() -> None:
    try:
        run(build_parser().parse_args())
    except Exception as exc:  # noqa: BLE001 - CLI must never echo provider details
        print(
            f"JOINT_SINGLE_PASS_FAILED category={_safe_error_category(exc)}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
