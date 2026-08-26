"""Injected OpenRouter P3 smoke coverage with no live network."""

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


class CountingOpenRouterFake:
    """Replace only the external HTTP boundary."""

    provider = "openrouter"
    endpoint_identifier = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        self.call_count = 0

    def send(self, request: ApiRequest) -> ProviderResponse:
        self.call_count += 1
        return ProviderResponse(
            provider=self.provider,
            returned_model_identifier="openai/gpt-5.6-sol",
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
            provider_request_id="gen_integration_1",
            timestamp=datetime.now(UTC).isoformat(),
            provider_cost=0.00125,
            exact_backend_model_identifier=None,
            exact_identity_evidence_source=None,
            safe_metadata={},
        )


def test_real_parser_accepts_openrouter_key_file(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/api/openrouter.yaml",
            "--real",
            "--api-key-file",
            str(tmp_path / "API.txt"),
        ]
    )

    assert args.real is True
    assert args.api_key is None
    assert args.api_key_file == tmp_path / "API.txt"


def test_injected_openrouter_smoke_is_miss_then_hit(tmp_path: Path) -> None:
    config = load_api_config(Path("configs/api/openrouter.yaml"))
    # Build the injected credential at runtime so the repository-wide leak
    # scan tests persisted output instead of matching its own test fixture.
    secret_text = "test-only-openrouter-" + "integration-key"
    transport = CountingOpenRouterFake()

    artifact_path = run_smoke(
        config,
        output_dir=tmp_path / "run",
        api_key=SecretValue(secret_text),
        transport=transport,
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert transport.call_count == 1
    assert artifact["status"] == "PARTIAL"
    assert artifact["first"]["cache_hit"] is False
    assert artifact["first"]["provider_call_count"] == 1
    assert artifact["second"]["cache_hit"] is True
    assert artifact["second"]["provider_call_count"] == 0
    assert artifact["second"]["provider_cost"] == 0.0
    assert artifact["request_hash"] == artifact["first"]["request_hash"]
    assert secret_text not in artifact_path.read_text(encoding="utf-8")


def test_invalid_key_file_never_echoes_content_or_traceback(tmp_path: Path) -> None:
    key_file = tmp_path / "API.txt"
    malformed = "sensitive-without-supported-format"
    key_file.write_text(malformed + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_api.py",
            "--config",
            "configs/api/openrouter.yaml",
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
