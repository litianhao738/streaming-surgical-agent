"""Run a leak-proof two-call P3 API smoke using only a synthetic image."""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from PIL import Image

from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import (
    ApiImageInput,
    ApiRequest,
    ApiResponseRecord,
    ProviderTransport,
    thaw_json,
)
from surgical_agent.api.credentials import (
    SecretValue,
    assert_secret_absent,
    resolve_api_key,
)
from surgical_agent.api.errors import ApiContractError, ApiError
from surgical_agent.api.registry import (
    build_transport,
    build_validator,
    determine_p3_status,
)
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_source_tree
from surgical_agent.config.loader import load_api_config, load_yaml
from surgical_agent.config.schema import ApiConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit mock/real CLI without credential-bearing defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/api/mock.yaml"
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "artifacts/p3"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--real", action="store_true")
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--api-key-file", type=Path)
    credentials.add_argument("--api-key")
    return parser


def _synthetic_png() -> bytes:
    """Create a non-sensitive deterministic 8x8 RGB transport probe."""

    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color=(23, 91, 177)).save(stream, format="PNG")
    return stream.getvalue()


def build_smoke_request(config: ApiConfig) -> ApiRequest:
    """Return the single canonical synthetic request used for both calls."""

    image = ApiImageInput(
        identifier="synthetic:p3:blue-square-v1",
        mime_type="image/png",
        content=_synthetic_png(),
    )
    return ApiRequest(
        provider=config.provider,
        model_identifier=config.requested_model_identifier,
        endpoint_identifier=config.endpoint_identifier,
        prompt_version=config.prompt_version,
        response_schema_version=config.response_schema_version,
        payload={
            "input_text": (
                "Inspect the synthetic blue square. Return the strict P3 "
                "transport-probe object and no additional fields."
            ),
            "probe": "multimodal_transport_and_structured_response_only",
            "p4_surgical_schema_frozen": False,
        },
        images=(image,),
        generation_parameters=config.generation_parameters,
    )


def _git_info() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit_sha": commit, "dirty": bool(status), "status_entries": status}


def safe_call_summary(record: ApiResponseRecord) -> dict[str, object]:
    """Allowlist accounting only; payloads and provider bodies never escape."""

    return {
        "request_hash": record.request_hash,
        "cache_hit": record.cache_hit,
        "provider_call_count": record.provider_call_count,
        "retry_count": record.retry_count,
        "provider_cost": record.provider_cost,
        "origin_provider_cost": record.origin_provider_cost,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "total_tokens": record.total_tokens,
        "latency_ms": record.latency_ms,
    }


def _validate_mode(config: ApiConfig, *, real: bool, has_credential: bool) -> None:
    """Reject inconsistent operation before constructing a provider transport."""

    config.require_enabled()
    config.validate()
    if real:
        if config.mode != "real" or config.provider != "requesty":
            raise ApiContractError(
                "real smoke requires the Requesty real configuration"
            )
        if not has_credential:
            raise ApiContractError("real smoke requires exactly one credential input")
        return
    if config.mode != "mock" or config.provider != "mock":
        raise ApiContractError("mock smoke requires an explicit mock configuration")
    if has_credential:
        raise ApiContractError("mock smoke rejects credential inputs")


def _assert_secret_absent_from_persisted_files(
    api_key: SecretValue,
    output_dir: Path,
) -> None:
    """Scan only git-tracked source and current runtime files, never ignored keys."""

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
    scan_paths = [
        *(PROJECT_ROOT / item for item in tracked if item),
        *(path for path in output_dir.rglob("*") if path.is_file()),
    ]
    try:
        assert_secret_absent(api_key, scan_paths)
    except RuntimeError:
        raise ApiContractError(
            "credential scan detected persisted credential"
        ) from None


def run_smoke(
    config: ApiConfig,
    *,
    output_dir: Path,
    api_key: SecretValue | None,
    transport: ProviderTransport | None = None,
) -> Path:
    """Execute the canonical request twice and write only the v2 safe artifact."""

    real = config.mode == "real"
    _validate_mode(config, real=real, has_credential=api_key is not None)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_transport = transport or build_transport(config, api_key=api_key)
    usage = UsageLedger(output_dir / "api_usage.jsonl")
    client = CachedMultimodalApiClient(
        transport=selected_transport,
        cache=FileApiCache(output_dir / "api_cache"),
        usage=usage,
        validator=build_validator(config),
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
        ),
        sleep=lambda _: None,
    )
    request = build_smoke_request(config)
    first = client.call(request)
    second = client.call(request)
    if first.cache_hit:
        raise ApiContractError("first P3 smoke call must be a cache miss")
    if not second.cache_hit:
        raise ApiContractError("second P3 smoke call must be a cache hit")
    if second.provider_call_count != 0:
        raise ApiContractError("cache replay must not call the provider")
    if second.provider_cost != 0.0:
        raise ApiContractError("cache replay must have zero provider cost")
    if first.request_hash != second.request_hash:
        raise ApiContractError("cache replay changed the canonical request hash")

    artifact = {
        "schema_version": "p3_api_smoke_artifact_v2",
        "status": determine_p3_status(first, transport_gates_passed=True),
        "synthetic_non_sensitive_image": True,
        "track20_image_uploaded": False,
        "structured_schema_version": config.response_schema_version,
        "request_hash": first.request_hash,
        "response_id": first.provider_request_id,
        "requested_model_identifier": first.requested_model_identifier,
        "returned_model_identifier": first.returned_model_identifier,
        "exact_backend_model_identifier": first.exact_backend_model_identifier,
        "exact_identity_evidence_source": first.exact_identity_evidence_source,
        "safe_provider_metadata": thaw_json(first.safe_metadata),
        "first": safe_call_summary(first),
        "second": safe_call_summary(second),
        "usage": usage.summarize(),
        "source_tree_sha256": sha256_source_tree(PROJECT_ROOT),
        "git": _git_info(),
        "timestamp": datetime.now(UTC).isoformat(),
        "p4_prediction_granularity": "DEFERRED_NOT_FROZEN_BY_P3",
    }
    artifact_name = (
        "p3_mock_smoke.json" if config.provider == "mock" else "p3_api_smoke.json"
    )
    artifact_path = atomic_write_json(output_dir / artifact_name, artifact)
    if real:
        assert api_key is not None
        _assert_secret_absent_from_persisted_files(api_key, output_dir)
    return artifact_path


def run(args: argparse.Namespace) -> Path:
    """Validate config before resolving credentials and select a safe mode."""

    config = load_api_config(args.config)
    if args.real:
        _validate_mode(config, real=True, has_credential=True)
        api_key = resolve_api_key(api_key=args.api_key, api_key_file=args.api_key_file)
        run_id = args.run_id or datetime.now(UTC).strftime("p3_real_%Y%m%dT%H%M%SZ")
    else:
        if args.api_key is not None or args.api_key_file is not None:
            raise ApiContractError("mock smoke rejects credential inputs")
        _validate_mode(config, real=False, has_credential=False)
        api_key = None
        run_id = args.run_id or datetime.now(UTC).strftime("p3_mock_%Y%m%dT%H%M%SZ")
    output_dir = args.output_root.expanduser().resolve() / run_id
    legacy_mock_transport = (
        build_transport(load_yaml(args.config)) if config.provider == "mock" else None
    )
    artifact = run_smoke(
        config,
        output_dir=output_dir,
        api_key=api_key,
        transport=legacy_mock_transport,
    )
    if config.provider == "mock":
        print(f"P3 MOCK PASS / REAL API BLOCKED: {artifact}")
    else:
        print(f"P3 REAL SMOKE COMPLETE: {artifact}")
    return artifact


def _safe_error_category(exc: BaseException) -> str:
    if isinstance(exc, ApiError):
        return exc.code
    if isinstance(exc, UnicodeError):
        return "invalid_text"
    if isinstance(exc, OSError):
        return "io_error"
    return "invalid_input"


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (ApiError, ValueError, UnicodeError, OSError) as exc:
        print(f"P3 smoke failed: {_safe_error_category(exc)}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
