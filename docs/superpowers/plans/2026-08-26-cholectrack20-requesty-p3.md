# CholecTrack20 Portability and Requesty P3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Prove CholecTrack20 runs from one relocatable data root, complete the provider-neutral Requesty Responses P3 path, run a real synthetic multimodal structured smoke, and report an evidence-based P3 PASS or PARTIAL verdict without entering P4.

**Architecture:** Extend the existing P3 request/client/cache/retry/usage boundaries instead of adding a Requesty bypass. Add a read-only portability verifier beside the existing dataset adapter, keep credentials in a redacted transport-only wrapper, and persist only whitelisted request/response provenance. The real smoke executes one canonical request twice so the first call proves transport and the second proves local cache replay.

**Tech Stack:** Python 3.11, standard-library urllib, dataclasses, PyYAML, Pillow, pytest, Ruff, PowerShell for local evidence commands.

**Spec:** docs/superpowers/specs/2026-08-26-cholectrack20-requesty-p3-design.md

## Global Constraints

- Work in D:\PythonProject7 and preserve all pre-existing user changes.
- Use .venv-p2\Scripts\python.exe (Python 3.11) for authoritative tests.
- Do not copy, rebuild, rewrite, or mutate D:\cholec_dataset.
- CHOLECTRACK20_ROOT is the only runtime/training data-root variable.
- Preserve the official 10/2/8 split and all existing VID30/VID31 sidecar, provenance, mask, and label semantics.
- Do not introduce a P4 instance-aware schema, prediction granularity, or matching rule.
- Keep P9 gate-error semantics deferred.
- Add docs/API.txt to .gitignore and prove it is ignored and untracked before its first read.
- Never print or persist the Requesty key in hashes, cache, usage, artifacts, reports, tracebacks, commands, or Git.
- Use the real API only with a synthetic non-sensitive image.
- Do not intentionally trigger real retry failures; test error paths with injected fakes.
- Retain historical test/report evidence and append new dated evidence.
- Do not commit datasets, docs/API.txt, cache, checkpoints, temporary files, complete provider responses, or runtime artifacts.
- Use apply_patch for source/document edits and make one focused commit per task.

## File Responsibility Map

**Create**

- configs/data/cholectrack20_autodl_bundle.yaml — declarative single-root bundle contract and frozen repair-manifest digest.
- src/surgical_agent/data/portability.py — read-only path enumeration, containment checks, hash verification, and deterministic report.
- scripts/verify_autodl_bundle.py — thin CLI over the portability verifier.
- tests/unit/test_p2_single_root_portability.py — isolated bundle and path-boundary tests.
- src/surgical_agent/api/credentials.py — redacted secret wrapper, CLI/file resolution, and exact-value leak scan.
- tests/unit/test_p3_credentials.py — credential parsing, representation, CLI, and persistence-boundary tests.
- tests/unit/test_p3_api_config_schema.py — effective config and strict P3 schema tests.
- src/surgical_agent/api/providers/requesty.py — Requesty Responses HTTP construction, parsing, safe metadata, and error mapping.
- tests/unit/test_p3_requesty.py — injected-HTTP Requesty transport tests.
- tests/integration/test_p3_requesty_smoke.py — provider-neutral miss/hit orchestration with an injected Requesty fake.
- configs/api/requesty.yaml — non-secret approved Requesty candidate configuration.

**Modify**

- .gitignore — ignore docs/API.txt exactly.
- src/surgical_agent/config/schema.py — make ApiConfig the effective normalized runtime contract.
- src/surgical_agent/config/loader.py — parse and validate API YAML into ApiConfig.
- configs/api/default.yaml and configs/api/mock.yaml — use one field vocabulary.
- src/surgical_agent/api/schema.py — strict JSON Schema, exact runtime validator, and schema registry.
- src/surgical_agent/api/contracts.py — canonical metadata and sanitized provider/logical response records.
- src/surgical_agent/api/request_hash.py — construct canonical request metadata and its hash.
- src/surgical_agent/api/cache.py — versioned request-bound cache envelope.
- src/surgical_agent/api/errors.py — safe transport taxonomy and counted terminal failure.
- src/surgical_agent/api/retry.py — return/raise complete attempt accounting.
- src/surgical_agent/api/usage.py — versioned whitelisted ledger records.
- src/surgical_agent/api/client.py — integrate metadata, cache, retry, validation, usage, and replay cost semantics.
- src/surgical_agent/api/registry.py — effective config/provider/schema routing.
- src/surgical_agent/api/providers/mock.py — emit the sanitized normalized response contract.
- src/surgical_agent/api/providers/__init__.py and src/surgical_agent/api/__init__.py — export only supported public interfaces.
- scripts/smoke_api.py — mock/real CLI, synthetic request, two-call assertions, safe artifact, and identity verdict.
- tests/unit/test_p3_api_request_hash.py and tests/unit/test_p3_api_client.py — new contracts and regression coverage.
- tests/integration/test_p3_mock_smoke.py — normalized CLI behavior and safe artifacts.
- tests/integration/test_vid30_vid31_derived_local.py — explicit single-root/no-upstream integration evidence.
- docs/AUTODL_QUICKSTART.md — current upload, verification, and key instructions.
- reports/P3_REPORT.md and docs/V3_1_API_IMPLEMENTATION_AUDIT.md — append sanitized evidence and final verdict.

## Scope Rationale

Portability verification and Requesty transport are implemented as separate, independently reviewable task commits. They remain in one plan because the objective has one coupled release gate: the final AutoDL/report evidence must prove both the single-root data boundary and the real P3 credential/transport boundary at the same implementation revision. No task introduces a reusable subsystem beyond that gate.

---

### Task 1: Seal the credential boundary before any key read

**Files:**
- Modify: .gitignore
- Create: src/surgical_agent/api/credentials.py
- Create: tests/unit/test_p3_credentials.py

**Interfaces:**
- Produces: SecretValue, load_api_key_file(path), resolve_api_key(api_key, api_key_file), assert_secret_absent(secret, paths).
- Consumes: pathlib.Path only; it must not import request, cache, usage, or provider modules.

- [ ] **Step 1: Write failing credential tests**

~~~python
from pathlib import Path

import pytest

from surgical_agent.api.credentials import (
    SecretValue,
    assert_secret_absent,
    load_api_key_file,
    resolve_api_key,
)


def test_api_key_file_splits_only_the_first_equals(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    path.write_text("REQUESTY_API_KEY=alpha=beta\n", encoding="utf-8")
    secret = load_api_key_file(path)
    assert secret.reveal() == "alpha=beta"
    assert str(secret) == "<redacted>"
    assert repr(secret) == "SecretValue(<redacted>)"


def test_api_key_file_errors_never_include_file_content(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    path.write_text("sensitive-without-separator\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_api_key_file(path)
    assert "sensitive-without-separator" not in str(caught.value)


def test_api_key_inputs_are_mutually_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    path.write_text("REQUESTY_API_KEY=value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_api_key(api_key="value", api_key_file=path)


def test_exact_secret_scan_reports_only_the_path(tmp_path: Path) -> None:
    secret = SecretValue("needle-value")
    safe = tmp_path / "safe.json"
    safe.write_text('{"value":"safe"}', encoding="utf-8")
    assert_secret_absent(secret, (safe,))
    leaked = tmp_path / "leaked.json"
    leaked.write_text('{"value":"needle-value"}', encoding="utf-8")
    with pytest.raises(RuntimeError) as caught:
        assert_secret_absent(secret, (leaked,))
    assert str(leaked) in str(caught.value)
    assert "needle-value" not in str(caught.value)
~~~

- [ ] **Step 2: Run the new tests and verify import failure**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_credentials.py -q
~~~

Expected: FAIL because surgical_agent.api.credentials does not exist.

- [ ] **Step 3: Ignore the real key path without opening it**

Add this exact line next to the existing scripts/API.txt rule:

~~~gitignore
docs/API.txt
~~~

Verify metadata only:

~~~powershell
git ls-files --error-unmatch -- docs/API.txt
git check-ignore -v -- docs/API.txt
~~~

Expected: the first command exits nonzero; the second identifies .gitignore. Do not run Get-Content, type, cat, or any content search against docs/API.txt.

- [ ] **Step 4: Implement the redacted credential interfaces**

~~~python
"""Credential loading and exact-value leak checks without serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class SecretValue:
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("API key must be non-empty")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __str__(self) -> str:
        return "<redacted>"

    def __repr__(self) -> str:
        return "SecretValue(<redacted>)"


def load_api_key_file(path: str | Path) -> SecretValue:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) != 1 or "=" not in lines[0]:
        raise ValueError("API key file must contain one NAME=value line")
    name, value = lines[0].split("=", 1)
    if not name.strip() or not value:
        raise ValueError("API key file must contain one non-empty NAME=value line")
    return SecretValue(value)


def resolve_api_key(
    *,
    api_key: str | None,
    api_key_file: str | Path | None,
) -> SecretValue:
    if api_key is not None and api_key_file is not None:
        raise ValueError("--api-key and --api-key-file are mutually exclusive")
    if api_key_file is not None:
        return load_api_key_file(api_key_file)
    if api_key is not None:
        return SecretValue(api_key)
    raise ValueError("real API mode requires --api-key or --api-key-file")


def assert_secret_absent(secret: SecretValue, paths: Iterable[Path]) -> None:
    needle = secret.reveal().encode("utf-8")
    leaked: list[str] = []
    for path in paths:
        if path.is_file() and needle in path.read_bytes():
            leaked.append(str(path))
    if leaked:
        raise RuntimeError(f"credential value found in persisted files: {sorted(leaked)}")
~~~

- [ ] **Step 5: Run credential and repository safety tests**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_credentials.py tests/unit/test_p0_repository.py -q
git status --short
~~~

Expected: tests PASS; docs/API.txt no longer appears in status; only intended source/test files appear.

- [ ] **Step 6: Commit the credential boundary**

~~~powershell
git add -- .gitignore src/surgical_agent/api/credentials.py tests/unit/test_p3_credentials.py
git diff --cached --check
git commit -m "security: seal P3 credential boundary"
~~~

### Task 2: Add the single-root bundle contract and read-only verifier

**Files:**
- Create: configs/data/cholectrack20_autodl_bundle.yaml
- Create: src/surgical_agent/data/portability.py
- Create: scripts/verify_autodl_bundle.py
- Create: tests/unit/test_p2_single_root_portability.py
- Modify: tests/integration/test_vid30_vid31_derived_local.py

**Interfaces:**
- Produces: PortabilityError, verify_single_root(dataset_root, bundle_path) -> dict[str, object].
- Consumes: discover_official_split_manifest, load_derived_supervision_manifest, sha256_file, repair_manifest.json.

- [ ] **Step 1: Write failing containment and manifest tests**

~~~python
import json
import os
from pathlib import Path

import pytest
import yaml

from surgical_agent.data.portability import (
    PortabilityError,
    assert_within_root,
    verify_single_root,
)


@pytest.fixture
def cholectrack20_root() -> Path:
    value = os.environ.get("CHOLECTRACK20_ROOT")
    if value is None or not Path(value).is_dir():
        pytest.skip("CholecTrack20 local-data integration fixture is unavailable")
    return Path(value).resolve()


def test_resolved_path_containment_rejects_sibling_prefix(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sibling = tmp_path / "data-escape"
    root.mkdir()
    sibling.mkdir()
    with pytest.raises(PortabilityError, match="outside"):
        assert_within_root(sibling, root, role="media")


def test_bundle_manifest_contains_only_relative_runtime_paths() -> None:
    raw = yaml.safe_load(
        Path("configs/data/cholectrack20_autodl_bundle.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert raw["schema_version"] == "cholectrack20_autodl_bundle_v1"
    assert raw["root_env"] == "CHOLECTRACK20_ROOT"
    assert raw["official_split_video_counts"] == {
        "Training": 10,
        "Validation": 2,
        "Testing": 8,
    }
    for relative in raw["required_relative_paths"].values():
        assert not Path(relative).is_absolute()


def test_single_root_report_never_serializes_absolute_runtime_paths(
    cholectrack20_root: Path,
) -> None:
    report = verify_single_root(
        cholectrack20_root,
        Path("configs/data/cholectrack20_autodl_bundle.yaml"),
    )
    rendered = json.dumps(report, sort_keys=True)
    assert report["status"] == "PASS"
    assert str(cholectrack20_root) not in rendered
    assert report["runtime_data_root_env"] == "CHOLECTRACK20_ROOT"
~~~

Add a local-data fixture in this test module that reads CHOLECTRACK20_ROOT and skips only when it is absent. Do not add a Cholec80/CholecT50 fixture.

- [ ] **Step 2: Run unit tests and verify the missing module/config failures**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p2_single_root_portability.py -q
~~~

Expected: FAIL because the bundle config and portability module do not exist.

- [ ] **Step 3: Add the declarative bundle file**

~~~yaml
schema_version: cholectrack20_autodl_bundle_v1
dataset: CholecTrack20
root_env: CHOLECTRACK20_ROOT
official_split_video_counts:
  Training: 10
  Validation: 2
  Testing: 8
required_relative_paths:
  repair_manifest: repair_manifest.json
  vid30_repaired: Validation/VID30/vid30_repaired.json
  vid31_phase_repaired: Training/VID31/vid31_phase_repaired.json
  vid31_frame_ivt_repaired: Training/VID31/vid31_frame_ivt_repaired.json
repair_manifest_sha256: c3ebb7e0db734be5f8c8418ac1e85bd54a21281664f623c2dacdfba5842d8d04
hash_authority: repair_manifest.json
provenance_or_regeneration_only:
  - Cholec80
  - CholecT50
optional_upstream_environment_variables:
  - CHOLEC80_30_31_ROOT
runtime_external_data_roots_allowed: false
dataset_mutation_allowed: false
~~~

- [ ] **Step 4: Implement deterministic path and hash verification**

Implement these public functions and report keys:

~~~python
class PortabilityError(RuntimeError):
    """Raised when the AutoDL bundle is incomplete or escapes its root."""


def assert_within_root(path: Path, root: Path, *, role: str) -> Path:
    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PortabilityError(f"{role} resolves outside CHOLECTRACK20_ROOT") from exc
    return resolved


def relative_runtime_path(path: Path, root: Path, *, role: str) -> str:
    resolved = assert_within_root(path, root, role=role)
    return resolved.relative_to(root.expanduser().resolve()).as_posix()


def verify_single_root(
    dataset_root: str | Path,
    bundle_path: str | Path,
) -> dict[str, object]:
    root = Path(dataset_root).expanduser().resolve()
    bundle = load_yaml(bundle_path)
    entries = discover_official_split_manifest(root)
    repair_path = assert_within_root(
        root / str(bundle["required_relative_paths"]["repair_manifest"]),
        root,
        role="repair_manifest",
    )
    derived = load_derived_supervision_manifest(
        repair_path,
        dataset_root_override=root,
    )
    adapter = CholecTrack20DatasetAdapter(
        root,
        derived_manifest_path=repair_path,
    )
    split_counts = Counter(entry.split.value for entry in entries)
    expected_counts = {
        name.casefold(): int(count)
        for name, count in bundle["official_split_video_counts"].items()
    }
    if dict(sorted(split_counts.items())) != dict(sorted(expected_counts.items())):
        raise PortabilityError("official split counts do not match the bundle")
    if len(entries) != 20 or len({entry.video_id for entry in entries}) != 20:
        raise PortabilityError("official bundle must contain 20 unique videos")
    runtime_paths: dict[str, Path] = {"repair_manifest": repair_path}
    for entry in entries:
        runtime_paths[f"{entry.video_id}:annotation"] = Path(entry.annotation_file)
        runtime_paths[f"{entry.video_id}:media"] = Path(entry.media_source)
    vid30 = derived.video("VID30")
    vid31 = derived.video("VID31")
    runtime_paths["VID30:repaired_annotation"] = require_path(
        vid30.annotation_source, "VID30 annotation"
    )
    runtime_paths["VID31:phase"] = require_path(vid31.phase_source, "VID31 phase")
    runtime_paths["VID31:frame_ivt"] = require_path(
        vid31.frame_level_action_source, "VID31 frame IVT"
    )
    for role, path in runtime_paths.items():
        assert_within_root(path, root, role=role)
        if not path.exists():
            raise PortabilityError(f"required runtime path is missing: {role}")
    expected_manifest_hash = str(bundle["repair_manifest_sha256"])
    if sha256_file(repair_path) != expected_manifest_hash:
        raise PortabilityError("repair_manifest SHA-256 mismatch")
    required_relative = {
        str(role): str(value)
        for role, value in bundle["required_relative_paths"].items()
    }
    observed_materialized = {
        "repair_manifest": relative_runtime_path(repair_path, root, role="repair_manifest"),
        "vid30_repaired": relative_runtime_path(
            require_path(vid30.annotation_source, "VID30 annotation"),
            root,
            role="VID30 annotation",
        ),
        "vid31_phase_repaired": relative_runtime_path(
            require_path(vid31.phase_source, "VID31 phase"),
            root,
            role="VID31 phase",
        ),
        "vid31_frame_ivt_repaired": relative_runtime_path(
            require_path(vid31.frame_level_action_source, "VID31 frame IVT"),
            root,
            role="VID31 frame IVT",
        ),
    }
    if observed_materialized != required_relative:
        raise PortabilityError("materialized paths do not match the bundle contract")
    adapter.collect(("VID02", "VID31"), samples_per_video=1)
    tuple(adapter.iter_video("VID30", max_samples=1))
    return {
        "schema_version": "cholectrack20_single_root_report_v1",
        "status": "PASS",
        "runtime_data_root_env": "CHOLECTRACK20_ROOT",
        "official_video_count": len(entries),
        "split_counts": dict(sorted(split_counts.items())),
        "runtime_paths": {
            role: relative_runtime_path(path, root, role=role)
            for role, path in sorted(runtime_paths.items())
        },
        "repair_manifest_sha256": expected_manifest_hash,
        "external_runtime_dependencies": [],
        "provenance_or_regeneration_only": ["Cholec80", "CholecT50"],
        "dataset_modified": False,
        "p4_contract_frozen": False,
    }
~~~

Define the helper without fallback guessing:

~~~python
def require_path(value: Path | None, role: str) -> Path:
    if value is None:
        raise PortabilityError(f"required runtime path is missing: {role}")
    return value
~~~

Load the YAML with surgical_agent.config.loader.load_yaml. Instantiate the adapter so its existing source and materialized hash checks remain authoritative.

- [ ] **Step 5: Add the thin verifier CLI**

~~~python
def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path)
    result.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "configs/data/cholectrack20_autodl_bundle.yaml",
    )
    result.add_argument("--output", type=Path)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    data_config = load_yaml(PROJECT_ROOT / "configs/data/cholectrack20.yaml")
    root = resolve_dataset_root(data_config, cli_root=args.dataset_root)
    report = verify_single_root(root, args.bundle)
    if args.output is not None:
        atomic_write_json(args.output.expanduser().resolve(), report)
    print("CholecTrack20 single-root portability: PASS")
    return report
~~~

The CLI prints only the status. The optional JSON report contains relative paths only.

- [ ] **Step 6: Add no-upstream integration assertions**

Extend tests/integration/test_vid30_vid31_derived_local.py with:

~~~python
def test_runtime_routes_need_only_cholectrack20_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHOLEC80_30_31_ROOT", raising=False)
    adapter = CholecTrack20DatasetAdapter(DATASET_ROOT)
    vid30 = next(adapter.iter_video("VID30", max_samples=1))
    vid31 = next(adapter.iter_video("VID31", max_samples=1))
    root = DATASET_ROOT.resolve()
    for sample in (vid30, vid31):
        for media_ref in sample.inference.media_refs:
            Path(media_ref).resolve().relative_to(root)
    Path(vid30.provenance.annotation_source).resolve().relative_to(root)
    Path(vid31.provenance.phase_source).resolve().relative_to(root)
    Path(vid31.provenance.frame_action_source).resolve().relative_to(root)
~~~

- [ ] **Step 7: Run unit and real-root portability tests**

Run:

~~~powershell
$env:CHOLECTRACK20_ROOT = "D:\cholec_dataset"
Remove-Item Env:CHOLEC80_30_31_ROOT -ErrorAction SilentlyContinue
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p2_single_root_portability.py tests/integration/test_vid30_vid31_derived_local.py -q -rs
.venv-p2\Scripts\python.exe scripts/verify_autodl_bundle.py --dataset-root D:\cholec_dataset
~~~

Expected: all required tests PASS; only the optional upstream source-hash test may skip; CLI reports PASS.

- [ ] **Step 8: Commit the portability contract**

~~~powershell
git add -- configs/data/cholectrack20_autodl_bundle.yaml src/surgical_agent/data/portability.py scripts/verify_autodl_bundle.py tests/unit/test_p2_single_root_portability.py tests/integration/test_vid30_vid31_derived_local.py
git diff --cached --check
git commit -m "feat: verify CholecTrack20 single-root bundle"
~~~

### Task 3: Make API configuration effective and the P3 schema strict

**Files:**
- Modify: src/surgical_agent/config/schema.py
- Modify: src/surgical_agent/config/loader.py
- Modify: src/surgical_agent/api/schema.py
- Modify: configs/api/default.yaml
- Modify: configs/api/mock.yaml
- Create: configs/api/requesty.yaml
- Create: tests/unit/test_p3_api_config_schema.py

**Interfaces:**
- Produces: ApiConfig.from_mapping(raw), load_api_config(path), schema_for(version), validator_for(version).
- Consumes: no credentials; all committed configs remain non-secret.

- [ ] **Step 1: Write failing config and schema tests**

~~~python
from pathlib import Path

import pytest

from surgical_agent.api.errors import ApiContractError, ApiSchemaError
from surgical_agent.api.schema import (
    P3_SMOKE_SCHEMA_VERSION,
    schema_for,
    validate_p3_smoke_payload,
    validator_for,
)
from surgical_agent.config.loader import load_api_config
from surgical_agent.config.schema import ApiConfig


def test_requesty_config_is_effective_and_non_secret() -> None:
    config = load_api_config(Path("configs/api/requesty.yaml"))
    assert config.enabled is True
    assert config.mode == "real"
    assert config.provider == "requesty"
    assert config.endpoint_identifier == "https://router.requesty.ai/v1/responses"
    assert config.requested_model_identifier == "openai-responses/gpt-5.6-sol"
    assert not hasattr(config, "api_key")


def test_disabled_config_fails_before_transport_construction() -> None:
    with pytest.raises(ApiContractError, match="disabled"):
        ApiConfig.from_mapping(
            {
                "enabled": False,
                "mode": "real",
                "provider": "requesty",
            }
        ).require_enabled()


def test_unknown_schema_fails_closed() -> None:
    with pytest.raises(ApiContractError, match="schema"):
        validator_for("unknown")


def test_p3_validator_rejects_extra_and_p4_fields() -> None:
    payload = {
        "schema_version": P3_SMOKE_SCHEMA_VERSION,
        "message": "ok",
        "image_observed": True,
        "structured": True,
        "instances": [],
    }
    with pytest.raises(ApiSchemaError, match="exact fields"):
        validate_p3_smoke_payload(payload)
    assert schema_for(P3_SMOKE_SCHEMA_VERSION)["additionalProperties"] is False
~~~

- [ ] **Step 2: Run tests and verify contract failures**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_api_config_schema.py -q
~~~

Expected: FAIL because load_api_config, strict schema registry, and requesty.yaml do not exist.

- [ ] **Step 3: Normalize ApiConfig and YAML fields**

Implement ApiConfig with these exact public fields:

~~~python
@dataclass(frozen=True)
class ApiConfig:
    enabled: bool
    mode: str
    provider: str
    endpoint_identifier: str
    requested_model_identifier: str
    prompt_version: str
    response_schema_version: str
    generation_parameters: Mapping[str, object] = field(default_factory=dict)
    provider_options: Mapping[str, object] = field(default_factory=dict)
    synthetic_input_required: bool = True
    cache_required: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "ApiConfig":
        required = (
            "enabled",
            "mode",
            "provider",
            "endpoint_identifier",
            "requested_model_identifier",
            "prompt_version",
            "response_schema_version",
        )
        missing = [name for name in required if name not in raw]
        if missing:
            raise ApiContractError(f"API config is missing fields: {missing}")
        config = cls(
            enabled=raw["enabled"],
            mode=str(raw["mode"]),
            provider=str(raw["provider"]),
            endpoint_identifier=str(raw["endpoint_identifier"]),
            requested_model_identifier=str(raw["requested_model_identifier"]),
            prompt_version=str(raw["prompt_version"]),
            response_schema_version=str(raw["response_schema_version"]),
            generation_parameters=MappingProxyType(
                copy.deepcopy(dict(raw.get("generation_parameters", {})))
            ),
            provider_options=MappingProxyType(
                copy.deepcopy(dict(raw.get("provider_options", {})))
            ),
            synthetic_input_required=bool(raw.get("synthetic_input_required", True)),
            cache_required=bool(raw.get("cache_required", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ApiContractError("API enabled must be boolean")
        for name in (
            "mode",
            "provider",
            "endpoint_identifier",
            "requested_model_identifier",
            "prompt_version",
            "response_schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ApiContractError(f"API {name} must be non-empty text")
        if self.mode not in {"mock", "real"}:
            raise ApiContractError("API mode must be mock or real")
        if self.provider == "requesty":
            if self.mode != "real":
                raise ApiContractError("Requesty requires real mode")
            if self.endpoint_identifier != "https://router.requesty.ai/v1/responses":
                raise ApiContractError("Requesty endpoint is not approved")
        if self.provider == "mock" and self.mode != "mock":
            raise ApiContractError("mock provider requires mock mode")

    def require_enabled(self) -> None:
        if not self.enabled:
            raise ApiContractError("API config is disabled")
~~~

Make load_api_config(path) call load_yaml(path), ApiConfig.from_mapping, and validator_for(config.response_schema_version). Normalize default.yaml and mock.yaml to the same names. requesty.yaml must contain the approved endpoint/model, max_output_tokens 128, and no credential fields.

Use this exact Requesty configuration:

~~~yaml
enabled: true
mode: real
provider: requesty
endpoint_identifier: https://router.requesty.ai/v1/responses
requested_model_identifier: openai-responses/gpt-5.6-sol
prompt_version: p3_transport_probe_v1
response_schema_version: p3_multimodal_smoke_v1
generation_parameters:
  max_output_tokens: 128
provider_options:
  timeout_seconds: 60.0
synthetic_input_required: true
cache_required: true
status: P3_REAL_API_CANDIDATE_NOT_EXACT_IDENTITY
~~~

- [ ] **Step 4: Implement one strict schema and registry**

~~~python
P3_SMOKE_SCHEMA_VERSION = "p3_multimodal_smoke_v1"
P3_SMOKE_ALLOWED_KEYS = {
    "schema_version",
    "message",
    "image_observed",
    "structured",
}
P3_SMOKE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": P3_SMOKE_SCHEMA_VERSION},
        "message": {"type": "string", "minLength": 1},
        "image_observed": {"type": "boolean"},
        "structured": {"const": True},
    },
    "required": sorted(P3_SMOKE_ALLOWED_KEYS),
}


def validate_p3_smoke_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != P3_SMOKE_ALLOWED_KEYS:
        raise ApiSchemaError("P3 smoke response must contain exact fields")
    if payload["schema_version"] != P3_SMOKE_SCHEMA_VERSION:
        raise ApiSchemaError("P3 smoke response has an unsupported schema_version")
    if not isinstance(payload["message"], str) or not payload["message"].strip():
        raise ApiSchemaError("P3 smoke response requires a non-empty message")
    if type(payload["image_observed"]) is not bool:
        raise ApiSchemaError("P3 smoke response requires boolean image_observed")
    if payload["structured"] is not True:
        raise ApiSchemaError("P3 smoke response must declare structured=true")


SCHEMAS = {
    P3_SMOKE_SCHEMA_VERSION: (
        P3_SMOKE_JSON_SCHEMA,
        validate_p3_smoke_payload,
    )
}
~~~

schema_for and validator_for must return only entries in SCHEMAS and otherwise raise ApiContractError. Do not add any surgical prediction fields.

- [ ] **Step 5: Run config/schema and existing P3 tests**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_api_config_schema.py tests/unit/test_p3_api_request_hash.py tests/integration/test_p3_mock_smoke.py -q
~~~

Expected: PASS after adapting existing config consumers to normalized field names.

- [ ] **Step 6: Commit effective configuration and schema**

~~~powershell
git add -- src/surgical_agent/config/schema.py src/surgical_agent/config/loader.py src/surgical_agent/api/schema.py configs/api/default.yaml configs/api/mock.yaml configs/api/requesty.yaml tests/unit/test_p3_api_config_schema.py
git diff --cached --check
git commit -m "feat: enforce effective P3 API configuration"
~~~

### Task 4: Bind canonical request provenance and sanitize response contracts

**Files:**
- Modify: src/surgical_agent/api/contracts.py
- Modify: src/surgical_agent/api/request_hash.py
- Modify: src/surgical_agent/api/providers/mock.py
- Modify: tests/unit/test_p3_api_request_hash.py
- Modify: tests/unit/test_p3_api_client.py

**Interfaces:**
- Produces: ImageProvenance, CanonicalRequestMetadata, canonical_request_metadata(request).
- Produces: ProviderResponse and ApiResponseRecord without raw_response fields.
- Consumes: ApiRequest and ApiImageInput.

- [ ] **Step 1: Write failing metadata and response-safety tests**

~~~python
from dataclasses import fields

from surgical_agent.api.contracts import (
    ApiResponseRecord,
    ProviderResponse,
)
from surgical_agent.api.request_hash import canonical_request_metadata


def test_canonical_metadata_reconstructs_the_safe_request() -> None:
    request = _request()
    metadata = canonical_request_metadata(request)
    assert metadata.provider == request.provider
    assert metadata.endpoint_identifier == request.endpoint_identifier
    assert metadata.requested_model_identifier == request.model_identifier
    assert metadata.prompt_version == request.prompt_version
    assert metadata.response_schema_version == request.response_schema_version
    assert metadata.images[0].sha256 == request.images[0].sha256
    assert metadata.images[0].size_bytes == len(request.images[0].content)
    assert len(metadata.payload_sha256) == 64
    assert len(metadata.request_hash) == 64


def test_persistable_response_contracts_have_no_raw_response_field() -> None:
    assert "raw_response" not in {item.name for item in fields(ProviderResponse)}
    assert "raw_response" not in {item.name for item in fields(ApiResponseRecord)}
~~~

- [ ] **Step 2: Run targeted tests and verify failures**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_api_request_hash.py tests/unit/test_p3_api_client.py -q
~~~

Expected: FAIL because canonical metadata and sanitized response shapes are absent.

- [ ] **Step 3: Add immutable provenance dataclasses**

~~~python
@dataclass(frozen=True)
class ImageProvenance:
    identifier: str
    mime_type: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CanonicalRequestMetadata:
    schema_version: str
    provider: str
    endpoint_identifier: str
    requested_model_identifier: str
    prompt_version: str
    response_schema_version: str
    generation_parameters: Mapping[str, Any]
    payload_sha256: str
    images: tuple[ImageProvenance, ...]
    request_hash: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "endpoint_identifier": self.endpoint_identifier,
            "requested_model_identifier": self.requested_model_identifier,
            "prompt_version": self.prompt_version,
            "response_schema_version": self.response_schema_version,
            "generation_parameters": thaw_json(self.generation_parameters),
            "payload_sha256": self.payload_sha256,
            "images": [asdict(image) for image in self.images],
            "request_hash": self.request_hash,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "CanonicalRequestMetadata":
        expected = {
            "schema_version",
            "provider",
            "endpoint_identifier",
            "requested_model_identifier",
            "prompt_version",
            "response_schema_version",
            "generation_parameters",
            "payload_sha256",
            "images",
            "request_hash",
        }
        if set(value) != expected:
            raise ValueError("canonical request metadata has invalid fields")
        images = value["images"]
        if not isinstance(images, list):
            raise TypeError("canonical request images must be a list")
        return cls(
            schema_version=value["schema_version"],
            provider=value["provider"],
            endpoint_identifier=value["endpoint_identifier"],
            requested_model_identifier=value["requested_model_identifier"],
            prompt_version=value["prompt_version"],
            response_schema_version=value["response_schema_version"],
            generation_parameters=_freeze_json(
                value["generation_parameters"],
                path="generation_parameters",
            ),
            payload_sha256=value["payload_sha256"],
            images=tuple(ImageProvenance(**dict(item)) for item in images),
            request_hash=value["request_hash"],
        )
~~~

The dataclass __post_init__ methods enforce exact schema_version, non-empty scalar strings, SHA-256 hex length/content, nonnegative image sizes, and immutable generation parameters. Keep ApiRequest credential-free. Define the JSON helpers exactly once in contracts.py:

~~~python
def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
~~~

- [ ] **Step 4: Build metadata and hash from one canonical body**

~~~python
def canonical_request_metadata(request: ApiRequest) -> CanonicalRequestMetadata:
    payload = thaw_json(request.payload)
    generation = thaw_json(request.generation_parameters)
    image_rows = tuple(
        ImageProvenance(
            identifier=image.identifier,
            mime_type=image.mime_type,
            size_bytes=len(image.content),
            sha256=image.sha256,
        )
        for image in request.images
    )
    payload_bytes = canonical_json_bytes(payload)
    hash_body = {
        "provider": request.provider,
        "endpoint_identifier": request.endpoint_identifier,
        "requested_model_identifier": request.model_identifier,
        "prompt_version": request.prompt_version,
        "response_schema_version": request.response_schema_version,
        "generation_parameters": generation,
        "payload": payload,
        "images": [asdict(image) for image in image_rows],
    }
    request_hash = hashlib.sha256(canonical_json_bytes(hash_body)).hexdigest()
    return CanonicalRequestMetadata(
        schema_version="api_request_metadata_v1",
        provider=request.provider,
        endpoint_identifier=request.endpoint_identifier,
        requested_model_identifier=request.model_identifier,
        prompt_version=request.prompt_version,
        response_schema_version=request.response_schema_version,
        generation_parameters=_freeze_json(generation, path="generation_parameters"),
        payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        images=image_rows,
        request_hash=request_hash,
    )
~~~

Make canonical_request_hash(request) return canonical_request_metadata(request).request_hash so only one definition exists.

- [ ] **Step 5: Replace raw response fields with whitelisted normalized fields**

Use these exact persistable fields:

~~~python
@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    returned_model_identifier: str
    parsed_payload: Mapping[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    image_count: int | None = None
    provider_request_id: str | None = None
    timestamp: str | None = None
    provider_cost: float | None = None
    exact_backend_model_identifier: str | None = None
    exact_identity_evidence_source: str | None = None
    safe_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ApiResponseRecord:
    provider: str
    endpoint_identifier: str
    request_hash: str
    requested_model_identifier: str
    returned_model_identifier: str
    parsed_payload: Mapping[str, Any]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    image_count: int | None = None
    latency_ms: float | None = None
    retry_count: int = 0
    provider_call_count: int = 1
    timestamp: str | None = None
    cache_hit: bool = False
    provider_request_id: str | None = None
    provider_cost: float | None = None
    origin_provider_cost: float | None = None
    exact_backend_model_identifier: str | None = None
    exact_identity_evidence_source: str | None = None
    safe_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_identifier(self) -> str:
        return self.returned_model_identifier

    @property
    def provider_call(self) -> bool:
        return self.provider_call_count > 0

    def to_persisted_mapping(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint_identifier": self.endpoint_identifier,
            "request_hash": self.request_hash,
            "requested_model_identifier": self.requested_model_identifier,
            "returned_model_identifier": self.returned_model_identifier,
            "parsed_payload": thaw_json(self.parsed_payload),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "image_count": self.image_count,
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count,
            "provider_call_count": self.provider_call_count,
            "timestamp": self.timestamp,
            "cache_hit": self.cache_hit,
            "provider_request_id": self.provider_request_id,
            "provider_cost": self.provider_cost,
            "origin_provider_cost": self.origin_provider_cost,
            "exact_backend_model_identifier": self.exact_backend_model_identifier,
            "exact_identity_evidence_source": self.exact_identity_evidence_source,
            "safe_metadata": thaw_json(self.safe_metadata),
        }

    @classmethod
    def from_persisted_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ApiResponseRecord":
        expected = {
            "provider",
            "endpoint_identifier",
            "request_hash",
            "requested_model_identifier",
            "returned_model_identifier",
            "parsed_payload",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "image_count",
            "latency_ms",
            "retry_count",
            "provider_call_count",
            "timestamp",
            "cache_hit",
            "provider_request_id",
            "provider_cost",
            "origin_provider_cost",
            "exact_backend_model_identifier",
            "exact_identity_evidence_source",
            "safe_metadata",
        }
        if set(value) != expected:
            raise ValueError("persisted API response has invalid fields")
        return cls(**{name: value[name] for name in sorted(expected)})
~~~

Both __post_init__ methods must copy/freeze parsed_payload and safe_metadata, require non-empty identity strings where non-null, validate all token/count fields as nonnegative integers, and validate latency/cost fields as finite nonnegative numbers. from_persisted_mapping rejects unknown/missing persisted keys before construction.

Update MockProviderTransport in this task to accept provider_cost: float | None = 0.0 and return the new ProviderResponse fields. This gives Task 5 a nonzero-cost fake without waiting for registry work.

- [ ] **Step 6: Run request and client tests**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_api_request_hash.py tests/unit/test_p3_api_client.py -q
~~~

Expected: PASS with no raw response persistence fields.

- [ ] **Step 7: Commit canonical provenance contracts**

~~~powershell
git add -- src/surgical_agent/api/contracts.py src/surgical_agent/api/request_hash.py src/surgical_agent/api/providers/mock.py tests/unit/test_p3_api_request_hash.py tests/unit/test_p3_api_client.py
git diff --cached --check
git commit -m "refactor: bind safe canonical API provenance"
~~~

### Task 5: Version and validate the cache envelope

**Files:**
- Modify: src/surgical_agent/api/cache.py
- Modify: src/surgical_agent/api/client.py
- Modify: tests/unit/test_p3_api_client.py

**Interfaces:**
- Produces: FileApiCache.get(metadata) and FileApiCache.put(metadata, response).
- Consumes: CanonicalRequestMetadata and ApiResponseRecord.

- [ ] **Step 1: Add failing tamper and replay-cost tests**

~~~python
@pytest.mark.parametrize(
    "field,replacement",
    [
        ("provider", "other"),
        ("endpoint_identifier", "https://invalid.example/responses"),
        ("requested_model_identifier", "other/model"),
        ("prompt_version", "other_prompt"),
        ("response_schema_version", "other_schema"),
        ("generation_parameters", {"temperature": 1}),
        ("payload_sha256", "0" * 64),
    ],
)
def test_cache_rejects_tampered_request_metadata(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    transport = MockProviderTransport()
    client, _ = _client(tmp_path, transport)
    request = _request()
    first = client.call(request)
    cache_path = tmp_path / "cache" / f"{first.request_hash}.json"
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    raw["request"][field] = replacement
    cache_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ApiCacheError, match="request metadata"):
        client.call(request)


def test_cache_hit_has_zero_current_provider_cost(tmp_path: Path) -> None:
    transport = MockProviderTransport(provider_cost=0.25)
    client, _ = _client(tmp_path, transport)
    request = _request()
    first = client.call(request)
    second = client.call(request)
    assert first.provider_cost == 0.25
    assert second.cache_hit is True
    assert second.provider_call_count == 0
    assert second.retry_count == 0
    assert second.provider_cost == 0.0
    assert second.origin_provider_cost == 0.25
~~~

- [ ] **Step 2: Run the cache tests and verify failure**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_api_client.py -q
~~~

Expected: FAIL because cache entries do not bind request metadata and replay cost is retained.

- [ ] **Step 3: Implement cache envelope v2**

Persist only this envelope:

~~~python
envelope = {
    "schema_version": "api_cache_entry_v2",
    "request": metadata.to_mapping(),
    "response": response.to_persisted_mapping(),
}
~~~

FileApiCache.get must:

1. derive the filename from metadata.request_hash;
2. return None only when the file is absent;
3. require exact top-level keys;
4. parse CanonicalRequestMetadata.from_mapping;
5. compare the parsed mapping to the current metadata mapping;
6. parse ApiResponseRecord.from_persisted_mapping; and
7. verify response.request_hash equals metadata.request_hash.

Any malformed or mismatched present entry raises ApiCacheError. Do not silently convert corruption into a miss.

- [ ] **Step 4: Make client replay a zero-cost logical call**

~~~python
cached = self.cache.get(metadata)
if cached is not None:
    self.validator(cached.parsed_payload)
    replay = replace(
        cached,
        cache_hit=True,
        provider_call_count=0,
        retry_count=0,
        latency_ms=0.0,
        provider_cost=0.0,
        origin_provider_cost=(
            cached.origin_provider_cost
            if cached.origin_provider_cost is not None
            else cached.provider_cost
        ),
    )
    self.usage.log_success(request, metadata, replay)
    return replay
~~~

- [ ] **Step 5: Run client/cache regressions**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_api_client.py tests/integration/test_p3_mock_smoke.py -q
~~~

Expected: PASS, including every metadata tamper case and zero replay cost.

- [ ] **Step 6: Commit cache envelope v2**

~~~powershell
git add -- src/surgical_agent/api/cache.py src/surgical_agent/api/client.py tests/unit/test_p3_api_client.py
git diff --cached --check
git commit -m "fix: bind P3 cache entries to requests"
~~~

### Task 6: Preserve terminal retry counts and sanitize the usage ledger

**Files:**
- Modify: src/surgical_agent/api/errors.py
- Modify: src/surgical_agent/api/retry.py
- Modify: src/surgical_agent/api/usage.py
- Modify: src/surgical_agent/api/client.py
- Modify: tests/unit/test_p3_api_client.py

**Interfaces:**
- Produces: ApiTransportError(code, retryable, status_code), ApiCallFailure(cause, attempt_count, retry_count), RetryResult(value, attempt_count, retry_count).
- Produces: UsageRecord schema api_usage_record_v2 with no raw bodies.

- [ ] **Step 1: Write failing mixed-failure and ledger-safety tests**

~~~python
from surgical_agent.api.contracts import ProviderResponse
from surgical_agent.api.errors import ApiCallFailure, ApiTransportError


class SequenceTransport:
    provider = "mock"
    endpoint_identifier = "mock://local/p3"

    def __init__(self, outcomes: list[ApiTransportError | ProviderResponse]) -> None:
        self.outcomes = list(outcomes)
        self.provider_call_count = 0

    def send(self, request: ApiRequest) -> ProviderResponse:
        self.provider_call_count += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, ApiTransportError):
            raise outcome
        return outcome


def test_retryable_then_nonretryable_preserves_all_counts(tmp_path: Path) -> None:
    transport = SequenceTransport(
        [
            ApiTransportError("safe", code="rate_limit", retryable=True),
            ApiTransportError("safe", code="authentication", retryable=False),
        ]
    )
    client, _ = _client(tmp_path, transport)
    request = _request()
    usage_path = tmp_path / "api_usage.jsonl"
    with pytest.raises(ApiCallFailure) as caught:
        client.call(request)
    assert caught.value.attempt_count == 2
    assert caught.value.retry_count == 1
    row = json.loads(usage_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["provider_call_count"] == 2
    assert row["retry_count"] == 1
    assert row["error"]["code"] == "authentication"


def test_failure_ledger_has_no_raw_response_or_payload(tmp_path: Path) -> None:
    transport = MockProviderTransport(malformed_payload=True)
    client, _ = _client(tmp_path, transport)
    usage_path = tmp_path / "api_usage.jsonl"
    with pytest.raises(ApiSchemaError):
        client.call(_request())
    rendered = usage_path.read_text(encoding="utf-8")
    assert "raw_response" not in rendered
    assert "parsed_payload" not in rendered
    assert "authorization" not in rendered.lower()
~~~

- [ ] **Step 2: Run mixed retry tests and verify incorrect counts**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_api_client.py -q
~~~

Expected: FAIL because the second non-retryable error loses the first attempt.

- [ ] **Step 3: Implement safe terminal retry outcomes**

~~~python
class ApiTransportError(ApiError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code


class ApiCallFailure(ApiError):
    code = "api_call_failed"

    def __init__(
        self,
        cause: ApiTransportError,
        *,
        attempt_count: int,
        retry_count: int,
    ) -> None:
        super().__init__(f"API call failed with category {cause.code}")
        self.cause = cause
        self.attempt_count = attempt_count
        self.provider_call_count = attempt_count
        self.retry_count = retry_count


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    value: T
    attempt_count: int
    retry_count: int


def execute(
    self,
    operation: Callable[[], T],
    *,
    sleep: SleepFunction = time.sleep,
) -> RetryResult[T]:
    attempts = 0
    retries = 0
    while True:
        attempts += 1
        try:
            return RetryResult(
                value=operation(),
                attempt_count=attempts,
                retry_count=retries,
            )
        except ApiTransportError as exc:
            if not exc.retryable or attempts >= self.max_attempts:
                raise ApiCallFailure(
                    exc,
                    attempt_count=attempts,
                    retry_count=retries,
                ) from exc
            delay = min(
                self.base_delay_seconds * (2**retries),
                self.max_delay_seconds,
            )
            sleep(delay)
            retries += 1
~~~

Use execute as RetryPolicy.execute. It always wraps a terminal ApiTransportError in ApiCallFailure. The wrapper message contains only the safe category.

- [ ] **Step 4: Implement usage schema v2 as an allowlist**

UsageRecord must include:

~~~python
@dataclass(frozen=True)
class UsageRecord:
    schema_version: str
    request: Mapping[str, Any]
    request_hash: str
    provider: str
    endpoint_identifier: str
    requested_model_identifier: str
    returned_model_identifier: str | None
    exact_backend_model_identifier: str | None
    exact_identity_evidence_source: str | None
    provider_request_id: str | None
    safe_provider_metadata: Mapping[str, Any]
    cache_hit: bool
    logical_call_count: int
    provider_call_count: int
    retry_count: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: float | None
    provider_cost: float | None
    origin_provider_cost: float | None
    timestamp: str
    error: Mapping[str, Any] | None
~~~

Failure error mappings have exact keys code, retryable, status_code. Success records do not include parsed payload or raw output text. summarize() sums provider_cost, logical calls, provider calls, retries, and cache hits from v2 rows.

- [ ] **Step 5: Integrate counted failure logging in the client**

Catch ApiCallFailure, log its cause code/status and its exact counts, then re-raise it unchanged. Wrap unexpected validator exceptions as ApiSchemaError before logging; do not persist exception text.

- [ ] **Step 6: Run all client, retry, cache, and usage tests**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_api_client.py tests/unit/test_p3_api_request_hash.py -q
~~~

Expected: PASS with provider_call_count 2 and retry_count 1 for the mixed failure.

- [ ] **Step 7: Commit retry and ledger contracts**

~~~powershell
git add -- src/surgical_agent/api/errors.py src/surgical_agent/api/retry.py src/surgical_agent/api/usage.py src/surgical_agent/api/client.py tests/unit/test_p3_api_client.py
git diff --cached --check
git commit -m "fix: preserve P3 retry and usage evidence"
~~~

### Task 7: Implement the Requesty Responses transport

**Files:**
- Create: src/surgical_agent/api/providers/requesty.py
- Modify: src/surgical_agent/api/providers/__init__.py
- Create: tests/unit/test_p3_requesty.py

**Interfaces:**
- Produces: HttpResponse, HttpSender, urllib_send_json, RequestyTransport.
- Consumes: SecretValue, ApiRequest, ProviderResponse, schema_for.

- [ ] **Step 1: Write failing request-construction and response-parsing tests**

~~~python
import json
import urllib.error
from copy import deepcopy

import pytest

from surgical_agent.api.contracts import ApiImageInput, ApiRequest
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiTransportError
from surgical_agent.api.providers.requesty import HttpResponse, RequestyTransport
from surgical_agent.api.schema import P3_SMOKE_SCHEMA_VERSION


def sample_request_value() -> ApiRequest:
    return ApiRequest(
        provider="requesty",
        model_identifier="openai-responses/gpt-5.6-sol",
        endpoint_identifier="https://router.requesty.ai/v1/responses",
        prompt_version="p3_transport_probe_v1",
        response_schema_version=P3_SMOKE_SCHEMA_VERSION,
        payload={"input_text": "Inspect the synthetic blue square."},
        images=(ApiImageInput("synthetic:test", "image/png", b"png-bytes"),),
        generation_parameters={"max_output_tokens": 128},
    )


def success_response() -> dict[str, object]:
    payload = {
        "schema_version": P3_SMOKE_SCHEMA_VERSION,
        "message": "synthetic image observed",
        "image_observed": True,
        "structured": True,
    }
    return {
        "id": "resp_1",
        "object": "response",
        "model": "openai-responses/gpt-5.6-sol",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(payload, sort_keys=True),
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8,
            "total_tokens": 18,
            "cost": 0.00125,
        },
    }


def requesty_transport_returning(status: int, body: bytes) -> RequestyTransport:
    def sender(url, headers, request_body, timeout_seconds):
        return HttpResponse(status_code=status, headers={}, body=body)

    return RequestyTransport(
        api_key=SecretValue("test-only-key"),
        endpoint_identifier="https://router.requesty.ai/v1/responses",
        sender=sender,
    )


def test_requesty_sends_responses_multimodal_json() -> None:
    sample_request = sample_request_value()
    captured: dict[str, object] = {}

    def sender(url, headers, body, timeout_seconds):
        captured.update(url=url, headers=headers, body=body, timeout=timeout_seconds)
        return HttpResponse(
            status_code=200,
            headers={
                "x-requesty-provider": "openai",
                "x-requesty-request-id": "req_gateway_1",
                "x-requesty-latency-ms": "45",
                "x-requesty-cache": "MISS",
            },
            body=json.dumps(success_response()).encode("utf-8"),
        )

    transport = RequestyTransport(
        api_key=SecretValue("test-only-key"),
        endpoint_identifier="https://router.requesty.ai/v1/responses",
        sender=sender,
    )
    response = transport.send(sample_request)
    sent = json.loads(captured["body"])
    content = sent["input"][0]["content"]
    assert captured["url"] == "https://router.requesty.ai/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer test-only-key"
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert sent["text"]["format"]["type"] == "json_schema"
    assert sent["text"]["format"]["strict"] is True
    assert response.returned_model_identifier == "openai-responses/gpt-5.6-sol"
    assert response.provider_request_id == "resp_1"
    assert response.provider_cost == 0.00125
    assert response.safe_metadata["requesty_provider"] == "openai"
    assert response.exact_backend_model_identifier is None
~~~

- [ ] **Step 2: Write failing taxonomy parameter tests**

~~~python
@pytest.mark.parametrize(
    "status,code,retryable",
    [
        (401, "authentication", False),
        (403, "authentication", False),
        (429, "rate_limit", True),
        (500, "provider_5xx", True),
        (503, "provider_5xx", True),
        (400, "provider_4xx", False),
        (404, "provider_4xx", False),
    ],
)
def test_requesty_http_taxonomy(status: int, code: str, retryable: bool) -> None:
    transport = requesty_transport_returning(status, b'{"error":{"message":"hidden"}}')
    with pytest.raises(ApiTransportError) as caught:
        transport.send(sample_request_value())
    assert caught.value.code == code
    assert caught.value.retryable is retryable
    assert caught.value.status_code == status
    assert "hidden" not in str(caught.value)
~~~

Add these exact edge tests:

~~~python
def test_requesty_timeout_is_retryable() -> None:
    def sender(url, headers, body, timeout_seconds):
        raise TimeoutError("provider detail must not escape")

    transport = RequestyTransport(
        api_key=SecretValue("test-only-key"),
        endpoint_identifier="https://router.requesty.ai/v1/responses",
        sender=sender,
    )
    with pytest.raises(ApiTransportError) as caught:
        transport.send(sample_request_value())
    assert (caught.value.code, caught.value.retryable) == ("timeout", True)
    assert "provider detail" not in str(caught.value)


def test_requesty_connection_failure_is_retryable() -> None:
    def sender(url, headers, body, timeout_seconds):
        raise urllib.error.URLError("provider detail must not escape")

    transport = RequestyTransport(
        api_key=SecretValue("test-only-key"),
        endpoint_identifier="https://router.requesty.ai/v1/responses",
        sender=sender,
    )
    with pytest.raises(ApiTransportError) as caught:
        transport.send(sample_request_value())
    assert (caught.value.code, caught.value.retryable) == ("connection", True)
    assert "provider detail" not in str(caught.value)


@pytest.mark.parametrize(
    "mutation",
    ["malformed_json", "incomplete", "missing_output", "invalid_output_json", "negative_tokens"],
)
def test_requesty_parse_failures_are_typed_and_nonretryable(mutation: str) -> None:
    response = deepcopy(success_response())
    if mutation == "malformed_json":
        body = b"{"
    else:
        if mutation == "incomplete":
            response["status"] = "in_progress"
        elif mutation == "missing_output":
            response["output"] = []
        elif mutation == "invalid_output_json":
            response["output"][0]["content"][0]["text"] = "not-json"
        elif mutation == "negative_tokens":
            response["usage"]["input_tokens"] = -1
        body = json.dumps(response).encode("utf-8")
    transport = requesty_transport_returning(200, body)
    with pytest.raises(ApiTransportError) as caught:
        transport.send(sample_request_value())
    assert caught.value.code == "parse_failure"
    assert caught.value.retryable is False


def test_requesty_persists_only_whitelisted_headers() -> None:
    def sender(url, headers, body, timeout_seconds):
        return HttpResponse(
            status_code=200,
            headers={
                "x-requesty-provider": "openai",
                "authorization": "must-not-persist",
                "x-unexpected": "must-not-persist",
            },
            body=json.dumps(success_response()).encode("utf-8"),
        )

    response = RequestyTransport(
        api_key=SecretValue("test-only-key"),
        endpoint_identifier="https://router.requesty.ai/v1/responses",
        sender=sender,
    ).send(sample_request_value())
    assert response.safe_metadata == {"requesty_provider": "openai"}
~~~

- [ ] **Step 3: Run Requesty unit tests and verify import failure**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_requesty.py -q
~~~

Expected: FAIL because the Requesty transport does not exist.

- [ ] **Step 4: Implement the injectable standard-library HTTP boundary**

~~~python
@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes = field(repr=False)


class HttpSender(Protocol):
    def __call__(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse:
        """Return a response or raise timeout/connection exceptions."""


def urllib_send_json(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout_seconds: float,
) -> HttpResponse:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return HttpResponse(
                status_code=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status_code=exc.code,
            headers={key.lower(): value for key, value in exc.headers.items()},
            body=exc.read(),
        )
~~~

RequestyTransport catches TimeoutError and socket.timeout as timeout, and urllib.error.URLError/OSError as connection. Error messages contain fixed safe text only.

- [ ] **Step 5: Implement Requesty request and response normalization**

RequestyTransport.send must:

1. require request.provider == requesty and endpoint equality;
2. require exactly one synthetic image and a non-empty payload input_text;
3. build a base64 data URL without storing it after the HTTP call;
4. use schema_for(request.response_schema_version);
5. permit only max_output_tokens, temperature, top_p, and reasoning from generation parameters;
6. send Authorization, Content-Type, and X-Title headers;
7. classify non-2xx status before parsing;
8. require object response, status completed, id, model, output_text, and usage;
9. parse output_text JSON to a mapping;
10. whitelist only documented Requesty headers; and
11. return ProviderResponse with exact identity fields set to None unless a dedicated documented exact-identity field is implemented.

Use this body shape:

~~~python
body = {
    "model": request.model_identifier,
    "instructions": (
        "Return only the requested P3 transport-probe JSON. "
        "Do not emit surgical predictions or P4 instance fields."
    ),
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": request.payload["input_text"]},
                {
                    "type": "input_image",
                    "image_url": (
                        f"data:{image.mime_type};base64,"
                        f"{base64.b64encode(image.content).decode('ascii')}"
                    ),
                },
            ],
        }
    ],
    "text": {
        "format": {
            "type": "json_schema",
            "name": "p3_multimodal_smoke",
            "strict": True,
            "schema": schema_for(request.response_schema_version),
        }
    },
    **thaw_json(request.generation_parameters),
}
~~~

Use fixed HTTP classification and output extraction helpers:

~~~python
def http_failure(status_code: int) -> ApiTransportError:
    if status_code in {401, 403}:
        return ApiTransportError(
            "Requesty authentication failed",
            code="authentication",
            retryable=False,
            status_code=status_code,
        )
    if status_code == 429:
        return ApiTransportError(
            "Requesty rate limit",
            code="rate_limit",
            retryable=True,
            status_code=status_code,
        )
    if 500 <= status_code <= 599:
        return ApiTransportError(
            "Requesty provider failure",
            code="provider_5xx",
            retryable=True,
            status_code=status_code,
        )
    if 400 <= status_code <= 499:
        return ApiTransportError(
            "Requesty request failed",
            code="provider_4xx",
            retryable=False,
            status_code=status_code,
        )
    return ApiTransportError(
        "Requesty returned an unexpected HTTP status",
        code="unexpected_http_status",
        retryable=False,
        status_code=status_code,
    )


def extract_output_text(response: Mapping[str, Any]) -> str:
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("output must be a list")
    texts = [
        item["text"]
        for message in output
        if isinstance(message, Mapping) and message.get("type") == "message"
        for item in message.get("content", [])
        if isinstance(item, Mapping)
        and item.get("type") == "output_text"
        and isinstance(item.get("text"), str)
    ]
    if len(texts) != 1:
        raise ValueError("response must contain one output_text")
    return texts[0]
~~~

Wrap JSON decoding, response-shape extraction, usage conversion, and ProviderResponse validation in one parse_failure boundary whose raised message is exactly "Requesty response parsing failed". Do not include the caught exception or response body in that message.

- [ ] **Step 6: Run Requesty and schema tests**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_requesty.py tests/unit/test_p3_api_config_schema.py -q
~~~

Expected: PASS for construction, parsing, safe headers, and every error category.

- [ ] **Step 7: Commit the Requesty transport**

~~~powershell
git add -- src/surgical_agent/api/providers/requesty.py src/surgical_agent/api/providers/__init__.py tests/unit/test_p3_requesty.py
git diff --cached --check
git commit -m "feat: add Requesty Responses transport"
~~~

### Task 8: Route providers and identity verdicts through effective config

**Files:**
- Modify: src/surgical_agent/api/registry.py
- Modify: src/surgical_agent/api/providers/mock.py
- Modify: src/surgical_agent/api/__init__.py
- Modify: tests/unit/test_p3_api_config_schema.py
- Modify: tests/unit/test_p3_api_client.py

**Interfaces:**
- Produces: build_transport(config, api_key, mock_options), build_validator(config), determine_p3_status(record, transport_gates_passed).
- Consumes: ApiConfig, SecretValue, MockProviderTransport, RequestyTransport.

- [ ] **Step 1: Write failing routing and verdict tests**

~~~python
from dataclasses import replace

from surgical_agent.api.contracts import ApiResponseRecord


def enabled_mock_config() -> ApiConfig:
    return load_api_config(Path("configs/api/mock.yaml"))


def success_record() -> ApiResponseRecord:
    return ApiResponseRecord(
        provider="requesty",
        endpoint_identifier="https://router.requesty.ai/v1/responses",
        request_hash="a" * 64,
        requested_model_identifier="openai-responses/gpt-5.6-sol",
        returned_model_identifier="openai-responses/gpt-5.6-sol",
        parsed_payload={
            "schema_version": "p3_multimodal_smoke_v1",
            "message": "ok",
            "image_observed": True,
            "structured": True,
        },
        input_tokens=10,
        output_tokens=8,
        total_tokens=18,
        image_count=1,
        latency_ms=50.0,
        retry_count=0,
        provider_call_count=1,
        timestamp="2026-08-26T00:00:00+00:00",
        cache_hit=False,
        provider_request_id="resp_1",
        provider_cost=0.00125,
        origin_provider_cost=None,
        exact_backend_model_identifier=None,
        exact_identity_evidence_source=None,
        safe_metadata={"requesty_provider": "openai"},
    )


def test_registry_rejects_disabled_and_unknown_provider() -> None:
    disabled = load_api_config(Path("configs/api/default.yaml"))
    with pytest.raises(ApiContractError, match="disabled"):
        build_transport(disabled, api_key=None)
    unknown = replace(enabled_mock_config(), provider="unknown")
    with pytest.raises(ApiContractError, match="unknown provider"):
        build_transport(unknown, api_key=None)


def test_requesty_registry_requires_a_secret() -> None:
    config = load_api_config(Path("configs/api/requesty.yaml"))
    with pytest.raises(ApiContractError, match="credential"):
        build_transport(config, api_key=None)


def test_alias_only_identity_remains_partial() -> None:
    record = replace(
        success_record(),
        requested_model_identifier="openai-responses/gpt-5.6-sol",
        returned_model_identifier="openai-responses/gpt-5.6-sol",
        exact_backend_model_identifier=None,
        exact_identity_evidence_source=None,
    )
    assert determine_p3_status(record, transport_gates_passed=True) == "PARTIAL"


def test_explicit_exact_identity_can_pass() -> None:
    record = replace(
        success_record(),
        exact_backend_model_identifier="immutable-backend-id",
        exact_identity_evidence_source="documented_response_field",
    )
    assert determine_p3_status(record, transport_gates_passed=True) == "PASS"
~~~

- [ ] **Step 2: Run registry tests and verify failures**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_api_config_schema.py tests/unit/test_p3_api_client.py -q
~~~

Expected: FAIL because the registry ignores effective config and no verdict function exists.

- [ ] **Step 3: Implement fail-closed registry routing**

~~~python
def build_transport(
    config: ApiConfig,
    *,
    api_key: SecretValue | None,
    mock_options: Mapping[str, object] | None = None,
) -> ProviderTransport:
    config.require_enabled()
    if config.provider == "mock":
        if api_key is not None:
            raise ApiContractError("mock provider must not receive a credential")
        return MockProviderTransport.from_config(config, mock_options or {})
    if config.provider == "requesty":
        if api_key is None:
            raise ApiContractError("Requesty provider requires a credential")
        return RequestyTransport(
            api_key=api_key,
            endpoint_identifier=config.endpoint_identifier,
            timeout_seconds=float(config.provider_options.get("timeout_seconds", 60.0)),
        )
    raise ApiContractError(f"unknown provider: {config.provider}")


def build_validator(config: ApiConfig) -> ResponseValidator:
    config.require_enabled()
    return validator_for(config.response_schema_version)
~~~

MockProviderTransport.from_config must use normalized provider_options and return a ProviderResponse with no raw body:

~~~python
@classmethod
def from_config(
    cls,
    config: ApiConfig,
    overrides: Mapping[str, object],
) -> "MockProviderTransport":
    values = {**dict(config.provider_options), **dict(overrides)}
    return cls(
        returned_model_identifier=str(
            values.get("returned_model_identifier", "mock-model-returned-v1")
        ),
        retryable_failures_before_success=int(
            values.get("retryable_failures_before_success", 0)
        ),
        malformed_payload=bool(values.get("malformed_payload", False)),
        provider_cost=(
            None
            if values.get("provider_cost") is None
            else float(values["provider_cost"])
        ),
    )
~~~

- [ ] **Step 4: Implement explicit verdict logic**

~~~python
def determine_p3_status(
    record: ApiResponseRecord,
    *,
    transport_gates_passed: bool,
) -> str:
    exact = record.exact_backend_model_identifier
    source = record.exact_identity_evidence_source
    if transport_gates_passed and exact and source:
        return "PASS"
    return "PARTIAL"
~~~

Do not infer exactness from requested/returned string equality or naming patterns.

- [ ] **Step 5: Run registry, mock, and client tests**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_api_config_schema.py tests/unit/test_p3_api_client.py tests/integration/test_p3_mock_smoke.py -q
~~~

Expected: PASS with unknown/disabled/missing-secret fail-closed behavior.

- [ ] **Step 6: Commit provider routing and verdict logic**

~~~powershell
git add -- src/surgical_agent/api/registry.py src/surgical_agent/api/providers/mock.py src/surgical_agent/api/__init__.py tests/unit/test_p3_api_config_schema.py tests/unit/test_p3_api_client.py
git diff --cached --check
git commit -m "feat: route effective P3 providers and identity"
~~~

### Task 9: Build the two-call mock/real smoke and leak-proof artifacts

**Files:**
- Modify: scripts/smoke_api.py
- Modify: tests/integration/test_p3_mock_smoke.py
- Create: tests/integration/test_p3_requesty_smoke.py
- Modify: tests/unit/test_p3_credentials.py

**Interfaces:**
- Produces: build_smoke_request(config), run_smoke(config, output_dir, api_key, transport), safe artifact schema p3_api_smoke_artifact_v2.
- Consumes: effective config, provider registry, CachedMultimodalApiClient, credential scan.

- [ ] **Step 1: Write failing parser and two-call integration tests**

~~~python
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from surgical_agent.api.contracts import ApiRequest, ProviderResponse
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.schema import P3_SMOKE_SCHEMA_VERSION
from surgical_agent.config.schema import ApiConfig
from scripts.smoke_api import build_parser, run_smoke


class CountingRequestyFake:
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
    config = load_api_config(Path("configs/api/requesty.yaml"))
    secret_text = "-".join(("integration", "test", "key"))
    secret = SecretValue(secret_text)
    transport = CountingRequestyFake()
    artifact_path = run_smoke(
        config,
        output_dir=tmp_path / "run",
        api_key=secret,
        transport=transport,
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert transport.call_count == 1
    assert artifact["first"]["cache_hit"] is False
    assert artifact["first"]["provider_call_count"] == 1
    assert artifact["second"]["cache_hit"] is True
    assert artifact["second"]["provider_call_count"] == 0
    assert artifact["second"]["provider_cost"] == 0.0
    assert artifact["request_hash"] == artifact["first"]["request_hash"]
    assert secret_text not in artifact_path.read_text(encoding="utf-8")


def test_invalid_key_file_never_echoes_content_or_traceback(tmp_path: Path) -> None:
    key_file = tmp_path / "API.txt"
    key_file.write_text("secret-without-equals\n", encoding="utf-8")
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
    assert "secret-without-equals" not in completed.stderr
    assert "Traceback" not in completed.stderr
~~~

- [ ] **Step 2: Run smoke tests and verify real path failure**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/integration/test_p3_mock_smoke.py tests/integration/test_p3_requesty_smoke.py tests/unit/test_p3_credentials.py -q
~~~

Expected: FAIL because --real remains blocked and no shared smoke orchestration exists.

- [ ] **Step 3: Replace parser with explicit safe modes**

~~~python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/p3")
    parser.add_argument("--run-id")
    parser.add_argument("--real", action="store_true")
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--api-key-file", type=Path)
    credentials.add_argument("--api-key")
    return parser
~~~

Mock mode rejects credential flags. Real mode resolves the credential only after config validation. Never print args or Namespace.

main catches ApiError, ValueError, UnicodeError, and OSError, prints only the already-sanitized category/message, and exits with SystemExit(2) from None so no traceback or chained provider exception is rendered.

- [ ] **Step 4: Build one canonical synthetic request**

~~~python
def build_smoke_request(config: ApiConfig) -> ApiRequest:
    image = ApiImageInput(
        identifier="synthetic:p3:blue-square-v1",
        mime_type="image/png",
        content=synthetic_png(),
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
~~~

- [ ] **Step 5: Implement two-call assertions and sanitized artifact**

run_smoke must create a new output directory, one FileApiCache, one UsageLedger, and one CachedMultimodalApiClient. It calls the exact same ApiRequest twice and asserts:

~~~python
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
~~~

The artifact contains schema/status, synthetic-image booleans, request hash, response ID, requested/returned/exact identity fields, safe Requesty metadata, compact first/second accounting, usage summary, P4 deferred flag, timestamp, source-tree hash, and Git provenance. It excludes parsed message text, raw responses, image data, credential data, and full commands.

Build call summaries and the artifact with this allowlist:

~~~python
def safe_call_summary(record: ApiResponseRecord) -> dict[str, object]:
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
    "git": git_info(),
    "timestamp": datetime.now(UTC).isoformat(),
    "p4_prediction_granularity": "DEFERRED_NOT_FROZEN_BY_P3",
}
~~~

- [ ] **Step 6: Scan tracked text and runtime output for the exact credential**

After writing cache, usage, and artifact files, collect:

~~~python
tracked = subprocess.run(
    ["git", "ls-files", "-z"],
    cwd=PROJECT_ROOT,
    check=True,
    capture_output=True,
).stdout.decode("utf-8").split("\0")
scan_paths = [
    *(PROJECT_ROOT / item for item in tracked if item),
    *(path for path in output_dir.rglob("*") if path.is_file()),
]
assert_secret_absent(api_key, scan_paths)
~~~

Run this only in real mode. docs/API.txt is ignored and therefore absent from git ls-files.

- [ ] **Step 7: Run integration and complete targeted P3 tests**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_credentials.py tests/unit/test_p3_api_config_schema.py tests/unit/test_p3_api_request_hash.py tests/unit/test_p3_api_client.py tests/unit/test_p3_requesty.py tests/integration/test_p3_mock_smoke.py tests/integration/test_p3_requesty_smoke.py -q
~~~

Expected: all targeted tests PASS without a network call.

- [ ] **Step 8: Run mock CLI regression**

Run:

~~~powershell
.venv-p2\Scripts\python.exe scripts/smoke_api.py --config configs/api/mock.yaml --run-id p3_mock_plan_verification
~~~

Expected: one safe status/path line, miss/hit assertions pass, and no real provider call.

- [ ] **Step 9: Commit smoke orchestration**

~~~powershell
git add -- scripts/smoke_api.py tests/integration/test_p3_mock_smoke.py tests/integration/test_p3_requesty_smoke.py tests/unit/test_p3_credentials.py
git diff --cached --check
git commit -m "feat: add leak-proof Requesty P3 smoke"
~~~

### Task 10: Verify implementation before spending a real API call

**Files:**
- Modify only if a test exposes a defect in Tasks 1-9.

**Interfaces:**
- Consumes every implementation task.
- Produces a clean implementation commit SHA used as the real-evidence revision.

- [ ] **Step 1: Read the verification-before-completion skill**

Read C:\Users\litia\.codex\skills\verification-before-completion\SKILL.md completely before running final gates.

- [ ] **Step 2: Run Ruff and compile checks**

Run:

~~~powershell
.venv-p2\Scripts\python.exe -m ruff check .
.venv-p2\Scripts\python.exe -m compileall -q src scripts tools tests
~~~

Expected: both exit 0 with no diagnostics.

- [ ] **Step 3: Run the complete suite with the real data root and no upstream root**

Run:

~~~powershell
$env:CHOLECTRACK20_ROOT = "D:\cholec_dataset"
Remove-Item Env:CHOLEC80_30_31_ROOT -ErrorAction SilentlyContinue
.venv-p2\Scripts\python.exe -m pytest -q -rs
~~~

Expected: all non-optional tests PASS; at most the explicit optional Cholec80 source-hash test skips.

- [ ] **Step 4: Run the single-root verifier and P2 smoke**

Run:

~~~powershell
$env:CHOLECTRACK20_ROOT = "D:\cholec_dataset"
Remove-Item Env:CHOLEC80_30_31_ROOT -ErrorAction SilentlyContinue
.venv-p2\Scripts\python.exe scripts/verify_autodl_bundle.py --dataset-root D:\cholec_dataset --output artifacts/portability/cholectrack20_single_root.json
.venv-p2\Scripts\python.exe scripts/run_local_smoke.py --config configs/experiments/local_smoke.yaml --dataset-root D:\cholec_dataset --output-root artifacts/p2 --run-id p2_single_root_20260826 --device cpu
~~~

Expected: portability PASS and P2 PASS. Inspect only sanitized relative-path/report fields.

- [ ] **Step 5: Audit the pre-real Git boundary**

Run:

~~~powershell
git diff --check
git status --short --branch
git ls-files --error-unmatch -- docs/API.txt
git check-ignore -v -- docs/API.txt
git rev-parse HEAD
~~~

Expected: implementation files are committed; runtime artifacts and docs/API.txt are absent from status; the tracked-file command fails; the ignore command succeeds. Record the HEAD SHA as the evidence revision.

### Task 11: Execute the real Requesty miss/hit smoke safely

**Files:**
- Read in process only: docs/API.txt through --api-key-file.
- Write ignored runtime evidence under artifacts/p3.

**Interfaces:**
- Consumes: Requesty config, redacted credential loader, two-call smoke.
- Produces: one sanitized artifact, cache envelope, usage ledger, and an evidence-based verdict.

- [ ] **Step 1: Confirm the credential boundary immediately before first read**

Run:

~~~powershell
git ls-files --error-unmatch -- docs/API.txt
git check-ignore -v -- docs/API.txt
git status --short --branch
~~~

Expected: docs/API.txt is untracked, ignored, and absent from status. Do not inspect its contents manually.

- [ ] **Step 2: Run the real smoke using only the file path**

Run:

~~~powershell
.venv-p2\Scripts\python.exe scripts/smoke_api.py --config configs/api/requesty.yaml --real --api-key-file docs/API.txt --run-id p3_requesty_real_20260826
~~~

Expected: the CLI prints only a safe verdict and artifact path. First call is a real Requesty cache miss; second is a local cache hit. The command itself contains no key.

- [ ] **Step 3: Validate sanitized evidence without printing response content**

Run a short read-only assertion:

~~~powershell
.venv-p2\Scripts\python.exe -c "import json, pathlib; p=pathlib.Path('artifacts/p3/p3_requesty_real_20260826/p3_api_smoke.json'); d=json.loads(p.read_text(encoding='utf-8')); assert d['first']['cache_hit'] is False; assert d['second']['cache_hit'] is True; assert d['second']['provider_call_count']==0; assert d['second']['provider_cost']==0.0; assert d['request_hash']==d['first']['request_hash']; assert d['response_id']; assert d['returned_model_identifier']; print(d['status'])"
~~~

Expected: prints PASS or PARTIAL only. Do not print the model response message or complete artifact.

- [ ] **Step 4: Re-run the exact-value leak scan and targeted tests**

The smoke already scans using the in-memory key. Then run:

~~~powershell
.venv-p2\Scripts\python.exe -m pytest tests/unit/test_p3_credentials.py tests/unit/test_p3_requesty.py tests/integration/test_p3_requesty_smoke.py -q
git status --short --branch
~~~

Expected: tests PASS; ignored runtime evidence and docs/API.txt do not appear in status.

- [ ] **Step 5: Apply the identity verdict without inference**

Read only these sanitized artifact fields in process: requested_model_identifier, returned_model_identifier, exact_backend_model_identifier, exact_identity_evidence_source, and status. If exact identity and evidence source are both non-empty and all transport gates passed, status is PASS. Otherwise status remains PARTIAL. Do not promote the returned alias or x-requesty-provider to exact identity.

### Task 12: Append reports, update AutoDL guidance, and commit final evidence

**Files:**
- Modify: docs/AUTODL_QUICKSTART.md
- Modify: reports/P3_REPORT.md
- Modify: docs/V3_1_API_IMPLEMENTATION_AUDIT.md

**Interfaces:**
- Consumes: sanitized portability, P2, test, Git, and P3 smoke facts.
- Produces: append-only human-readable evidence and exact AutoDL deployment instructions.

- [ ] **Step 1: Update AutoDL instructions with current reality**

Replace the stale claim that P3 is unimplemented. Document these exact operational facts:

- upload the repository and one complete CholecTrack20 root;
- the root must contain the official split, repair_manifest.json, and three existing sidecars;
- set CHOLECTRACK20_ROOT only;
- do not upload Cholec80/CholecT50 unless auditing or regenerating provenance;
- run scripts/verify_autodl_bundle.py before P2/P3;
- provide the Requesty key separately as one NAME=value line;
- run real smoke with --api-key-file, never a literal key in shared shell history;
- P2 remains engineering-only and P4 remains deferred.

- [ ] **Step 2: Append a dated P3 evidence section**

Append, without deleting historical results:

- date and supported Python environment;
- evidence revision SHA from Task 10;
- CHOLECTRACK20_ROOT set and CHOLEC80_30_31_ROOT absent;
- prior 79 passed, 78 passed/1 skipped, and 95 passed/1 skipped records retained;
- new targeted/full test results and runtime;
- portability verifier and P2 smoke result;
- Requesty endpoint and requested model;
- sanitized returned model exactly as observed;
- response ID presence, tokens/cost availability, and safe provider header evidence;
- first miss/second hit/provider_call_count/cost evidence;
- exact backend identity evidence or its absence;
- resulting P3 PASS or PARTIAL verdict;
- explicit statement that formal performance/paper experiments did not run.

Do not include complete raw responses, parsed message text, key-file contents, or a command containing a key.

- [ ] **Step 3: Append implementation-audit evidence**

Update the P3 row and append reproducibility/security evidence. Preserve all P1/P2 historical scope statements as historical evidence. State that the sidecars and research semantics were unchanged and that no P4/P9 contract was frozen.

- [ ] **Step 4: Run report consistency and secret-pattern checks**

Run:

~~~powershell
$env:CHOLECTRACK20_ROOT = "D:\cholec_dataset"
Remove-Item Env:CHOLEC80_30_31_ROOT -ErrorAction SilentlyContinue
.venv-p2\Scripts\python.exe -m pytest -q
.venv-p2\Scripts\python.exe -m ruff check .
git diff --check
git diff -- docs/AUTODL_QUICKSTART.md reports/P3_REPORT.md docs/V3_1_API_IMPLEMENTATION_AUDIT.md
git grep -n -I -E "(Bearer[[:space:]]+[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}|PRIVATE KEY)" -- . ":(exclude)docs/API.txt"
~~~

Expected: tests/Ruff/diff checks pass; no realistic secret pattern appears. Treat short lexical sk- fragments as candidates requiring length-aware review, not automatic leaks.

- [ ] **Step 5: Commit documentation and reports**

~~~powershell
git add -- docs/AUTODL_QUICKSTART.md reports/P3_REPORT.md docs/V3_1_API_IMPLEMENTATION_AUDIT.md
git diff --cached --check
git commit -m "docs: record portable Requesty P3 evidence"
~~~

### Task 13: Final requirement-by-requirement completion audit

**Files:**
- Inspect all changed files and committed evidence.
- Do not modify unless the audit finds a concrete gap.

**Interfaces:**
- Consumes: objective, spec, plan, code, tests, ignored runtime evidence, reports, and Git.
- Produces: final commit SHA and a defensible completion decision.

- [ ] **Step 1: Re-read the objective completion gates**

Read C:\Users\litia\.codex\attachments\780c896f-82b9-433a-9911-1caa57387dfa\goal-objective.md completely and map every numbered requirement to current evidence.

- [ ] **Step 2: Verify the final tree and commit**

Run:

~~~powershell
git status --short --branch
git diff --check
git log -8 --oneline --decorate
git rev-parse HEAD
git ls-files -- docs/API.txt
git check-ignore -v -- docs/API.txt
~~~

Expected: tracked tree clean; docs/API.txt is absent from git ls-files and confirmed ignored; final SHA is recorded for handoff.

- [ ] **Step 3: Run the final authoritative gates once more**

Run:

~~~powershell
$env:CHOLECTRACK20_ROOT = "D:\cholec_dataset"
Remove-Item Env:CHOLEC80_30_31_ROOT -ErrorAction SilentlyContinue
.venv-p2\Scripts\python.exe -m ruff check .
.venv-p2\Scripts\python.exe -m compileall -q src scripts tools tests
.venv-p2\Scripts\python.exe -m pytest -q -rs
.venv-p2\Scripts\python.exe scripts/verify_autodl_bundle.py --dataset-root D:\cholec_dataset
~~~

Expected: all non-optional gates pass. Do not re-run the paid real call merely to prove it again; validate its sanitized artifact and committed report.

- [ ] **Step 4: Confirm every frozen boundary**

Use diffs and tests to confirm no dataset content changed, no external runtime path was introduced, VID30/VID31 semantics stayed fixed, no P4/P9 schema was added, no raw response is persisted, and historical results remain present.

- [ ] **Step 5: Mark the goal complete only if all evidence is direct**

If every completion gate is proven, call update_goal with status complete and report the final SHA, exact returned model, PASS/PARTIAL verdict, what AutoDL needs uploaded, and the evidence for single-root containment. If any gate lacks evidence, keep the goal active and repair the gap rather than narrowing the completion claim.
