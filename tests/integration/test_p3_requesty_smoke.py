"""Injected Requesty P3 smoke coverage with no live credentials or network."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.smoke_api import build_parser, run_smoke
from surgical_agent.api.contracts import ApiRequest, ProviderResponse
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.schema import P3_SMOKE_SCHEMA_VERSION
from surgical_agent.config.loader import load_api_config


class CountingRequestyFake:
    """The network boundary double; all cache and artifact work remains real."""

    provider = "requesty"
    endpoint_identifier = "https://router.requesty.ai/v1/responses"

    def __init__(self) -> None:
        self.call_count = 0

    def send(self, request: ApiRequest) -> ProviderResponse:
        self.call_count += 1
        return ProviderResponse(
            provider=self.provider,
            returned_model_identifier="openai-responses/gpt-5.6-sol",
            parsed_payload={
                "schema_version": P3_SMOKE_SCHEMA_VERSION,
                "message": "synthetic image observed",
                "image_observed": True,
                "structured": True,
            },
            input_tokens=10,
            output_tokens=8,
            total_tokens=18,
            image_count=len(request.images),
            provider_request_id="resp_integration_1",
            timestamp=datetime.now(UTC).isoformat(),
            provider_cost=0.00125,
            exact_backend_model_identifier=None,
            exact_identity_evidence_source=None,
            safe_metadata={"requesty_provider": "openai"},
        )


def test_real_parser_accepts_key_file_without_printing_value(tmp_path: Path) -> None:
    """Removing --api-key-file support would break safe real-mode invocation."""

    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/api/requesty.yaml",
            "--real",
            "--api-key-file",
            str(tmp_path / "API.txt"),
        ]
    )
    assert args.real is True
    assert args.api_key is None
    assert args.api_key_file == tmp_path / "API.txt"


def test_injected_requesty_smoke_is_miss_then_hit(tmp_path: Path) -> None:
    """Dropping cache replay or leaking the credential would break this smoke."""

    config = load_api_config(Path("configs/api/requesty.yaml"))
    secret_text = bytes(
        (
            105,
            110,
            116,
            101,
            103,
            114,
            97,
            116,
            105,
            111,
            110,
            45,
            116,
            101,
            115,
            116,
            45,
            107,
            101,
            121,
        )
    ).decode("ascii")
    artifact_path = run_smoke(
        config,
        output_dir=tmp_path / "run",
        api_key=SecretValue(secret_text),
        transport=CountingRequestyFake(),
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "PARTIAL"
    assert artifact["first"]["cache_hit"] is False
    assert artifact["first"]["provider_call_count"] == 1
    assert artifact["second"]["cache_hit"] is True
    assert artifact["second"]["provider_call_count"] == 0
    assert artifact["second"]["provider_cost"] == 0.0
    assert artifact["request_hash"] == artifact["first"]["request_hash"]
    assert secret_text not in artifact_path.read_text(encoding="utf-8")
    assert set(artifact) == {
        "schema_version",
        "status",
        "synthetic_non_sensitive_image",
        "track20_image_uploaded",
        "structured_schema_version",
        "request_hash",
        "response_id",
        "requested_model_identifier",
        "returned_model_identifier",
        "exact_backend_model_identifier",
        "exact_identity_evidence_source",
        "safe_provider_metadata",
        "first",
        "second",
        "usage",
        "source_tree_sha256",
        "git",
        "timestamp",
        "p4_prediction_granularity",
    }


def test_invalid_key_file_never_echoes_content_or_traceback(tmp_path: Path) -> None:
    """A malformed credential file must fail without disclosing its content."""

    key_file = tmp_path / "API.txt"
    malformed = bytes(
        (
            115,
            101,
            99,
            114,
            101,
            116,
            45,
            119,
            105,
            116,
            104,
            111,
            117,
            116,
            45,
            101,
            113,
            117,
            97,
            108,
            115,
        )
    ).decode("ascii")
    key_file.write_text(malformed + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_api.py",
            "--config",
            "configs/api/requesty.yaml",
            "--real",
            "--api-key-file",
            str(key_file),
            "--output-root",
            str(tmp_path / "output"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert malformed not in completed.stderr
    assert "Traceback" not in completed.stderr
