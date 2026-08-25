"""Run the P3 mock transport smoke or fail closed for unapproved real API use."""

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
from surgical_agent.api.contracts import ApiImageInput, ApiRequest
from surgical_agent.api.errors import ApiContractError
from surgical_agent.api.registry import build_transport
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.schema import P3_SMOKE_SCHEMA_VERSION, validate_p3_smoke_payload
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import (
    atomic_write_json,
    sha256_source_tree,
)
from surgical_agent.config.loader import load_yaml


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/api/mock.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/p3",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--real", action="store_true")
    return parser


def _synthetic_png() -> bytes:
    """Create a non-sensitive deterministic 8x8 RGB transport probe."""

    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color=(23, 91, 177)).save(stream, format="PNG")
    return stream.getvalue()


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


def run_mock(config: dict[str, Any], output_dir: Path) -> Path:
    if config.get("mode") != "mock" or config.get("provider") != "mock":
        raise ApiContractError("Mock smoke requires an explicit mock configuration")
    if config.get("synthetic_input_required") is not True:
        raise ApiContractError("P3 mock smoke must require synthetic input")
    output_dir.mkdir(parents=True, exist_ok=False)
    transport = build_transport(config)
    usage = UsageLedger(output_dir / "api_usage.jsonl")
    client = CachedMultimodalApiClient(
        transport=transport,
        cache=FileApiCache(output_dir / "api_cache"),
        usage=usage,
        validator=validate_p3_smoke_payload,
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
        ),
        sleep=lambda _: None,
    )
    image = ApiImageInput(
        identifier="synthetic:p3:blue-square-v1",
        mime_type="image/png",
        content=_synthetic_png(),
    )
    request = ApiRequest(
        provider=str(config["provider"]),
        model_identifier=str(config["requested_model_identifier"]),
        endpoint_identifier=str(config["endpoint_identifier"]),
        prompt_version=str(config["prompt_version"]),
        response_schema_version=str(config["response_schema_version"]),
        payload={
            "probe": "multimodal_transport_and_structured_response_only",
            "p4_surgical_schema_frozen": False,
        },
        images=(image,),
        generation_parameters={"deterministic_mock": True},
    )
    first = client.call(request)
    second = client.call(request)
    if first.cache_hit or not second.cache_hit:
        raise AssertionError("P3 cache miss/hit sequence is invalid")
    if first.request_hash != second.request_hash:
        raise AssertionError("P3 cache replay changed the request hash")
    summary = usage.summarize()
    artifact = {
        "schema_version": "p3_mock_smoke_artifact_v1",
        "status": "P3_PARTIAL_REAL_API_SMOKE_BLOCKED",
        "synthetic_non_sensitive_image": True,
        "track20_image_uploaded": False,
        "request_hash": first.request_hash,
        "requested_model_identifier": first.requested_model_identifier,
        "returned_model_identifier": first.model_identifier,
        "returned_model_is_mock_only": True,
        "structured_schema_version": P3_SMOKE_SCHEMA_VERSION,
        "first_cache_hit": first.cache_hit,
        "second_cache_hit": second.cache_hit,
        "first_retry_count": first.retry_count,
        "usage": summary,
        "source_tree_sha256": sha256_source_tree(PROJECT_ROOT),
        "git": _git_info(),
        "timestamp": datetime.now(UTC).isoformat(),
        "real_api_smoke": "BLOCKED_PROVIDER_ENDPOINT_MODEL_CREDENTIAL_UNVERIFIED",
        "p4_prediction_granularity": "DEFERRED_NOT_FROZEN_BY_P3",
    }
    return atomic_write_json(output_dir / "p3_mock_smoke.json", artifact)


def run(args: argparse.Namespace) -> Path:
    config = load_yaml(args.config)
    if args.real:
        raise ApiContractError(
            "REAL_API_SMOKE_BLOCKED: provider, endpoint, credential, and returned "
            "model identifier are not approved or verified"
        )
    run_id = args.run_id or datetime.now(UTC).strftime("p3_mock_%Y%m%dT%H%M%SZ")
    output_dir = args.output_root.expanduser().resolve() / run_id
    artifact = run_mock(config, output_dir)
    print(f"P3 MOCK PASS / REAL API BLOCKED: {artifact}")
    return artifact


def main() -> None:
    try:
        run(_parser().parse_args())
    except ApiContractError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
