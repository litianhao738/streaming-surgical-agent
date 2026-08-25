"""P0 repository and phase-gate tests."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from surgical_agent.api import ApiRequest
from surgical_agent.config import load_yaml
from surgical_agent.data.schemas import EvaluationTarget, InferenceSample
from surgical_agent.runtime.phase import (
    Phase,
    PhaseGateError,
    PhaseStatus,
    require_prior_phases_passed,
)
from surgical_agent.workflow import WorkflowStateStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_current_v31_api_documents_are_present() -> None:
    assert (
        PROJECT_ROOT
        / "docs"
        / "architecture"
        / "Streaming_SurgicalAgent_V3_1_API_Codex_Implementation_Spec.md"
    ).is_file()
    assert (
        PROJECT_ROOT
        / "docs"
        / "architecture"
        / "Streaming_SurgicalAgent_V3_1_API_学术修订版完整项目与代码架构说明.md"
    ).is_file()
    assert (PROJECT_ROOT / "docs" / "V3_1_API_IMPLEMENTATION_AUDIT.md").is_file()


@pytest.mark.parametrize(
    "module_name",
    [
        "surgical_agent",
        "surgical_agent.api",
        "surgical_agent.config",
        "surgical_agent.data.schemas",
        "surgical_agent.tracking.contracts",
        "surgical_agent.models.perception.model",
        "surgical_agent.research.knowledge.prior_store",
        "surgical_agent.research.gate.policy",
        "surgical_agent.research.verification.verifier",
        "surgical_agent.research.reliability.profile",
        "surgical_agent.research.memory.store",
        "surgical_agent.systems.v3_streaming",
        "surgical_agent.training.rollout",
        "surgical_agent.inference.engine",
        "surgical_agent.evaluation.evaluator",
        "surgical_agent.workflow",
        "streaming_surgical_agent",
    ],
)
def test_v31_framework_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)


def test_base_config_has_no_workstation_dataset_default() -> None:
    config = load_yaml(PROJECT_ROOT / "configs" / "base.yaml")
    assert config["data"]["root"] is None
    assert config["data"]["read_only"] is True
    assert config["project"]["current_phase"] == "P2"
    assert config["project"]["phase_status"] == "PASS"
    assert config["project"]["next_phase"] == "P3"
    assert config["project"]["method_implementation"] == "p2_local_smoke_only"
    assert config["data"]["current_phase_blockers"] == []


def test_inference_schema_is_structurally_gold_free() -> None:
    runtime_fields = set(InferenceSample.__dataclass_fields__)
    evaluation_fields = set(EvaluationTarget.__dataclass_fields__)
    forbidden = {"gt", "labels", "label_mask", "oracle_track", "gold_phase"}
    assert runtime_fields.isdisjoint(forbidden)
    assert "instances" in evaluation_fields


def test_api_request_has_no_credential_fields() -> None:
    request_fields = set(ApiRequest.__dataclass_fields__)
    forbidden = {"api_key", "token", "password", "secret", "authorization"}
    assert request_fields.isdisjoint(forbidden)


def test_compatibility_namespace_matches_canonical_version() -> None:
    canonical = importlib.import_module("surgical_agent")
    compatibility = importlib.import_module("streaming_surgical_agent")
    assert compatibility.__version__ == canonical.__version__


def test_p0_contract_config_files_exist() -> None:
    for relative_path in (
        "configs/api/default.yaml",
        "configs/workflow/default.yaml",
        "configs/memory/default.yaml",
        "configs/gate/default.yaml",
    ):
        assert (PROJECT_ROOT / relative_path).is_file()


def test_declared_local_data_and_backbone_profiles_are_explicit() -> None:
    data_config = load_yaml(PROJECT_ROOT / "configs" / "data" / "cholectrack20.yaml")
    api_config = load_yaml(PROJECT_ROOT / "configs" / "api" / "default.yaml")
    assert data_config["root"] is None
    assert data_config["root_env"] == "CHOLECTRACK20_ROOT"
    assert data_config["root_portability"] == "portable_external_dataset"
    assert api_config["declared_model_name"] == "GPT-5.6 Terra"
    assert api_config["model_identifier"] is None
    assert api_config["model_identifier_status"].startswith("BLOCKED_")


def test_workflow_contract_is_not_an_event_retriever() -> None:
    assert hasattr(WorkflowStateStore, "snapshot")
    assert hasattr(WorkflowStateStore, "update")
    assert not hasattr(WorkflowStateStore, "retrieve")


def test_phase_gate_rejects_skipping_p1() -> None:
    statuses = {Phase.P0: PhaseStatus.PASS}
    with pytest.raises(PhaseGateError, match="P1 is NOT_STARTED"):
        require_prior_phases_passed(Phase.P2, statuses)


def test_phase_gate_allows_next_phase_only() -> None:
    statuses = {Phase.P0: PhaseStatus.PASS}
    require_prior_phases_passed(Phase.P1, statuses)


def test_phase_gate_allows_p2_after_p1_pass() -> None:
    statuses = {Phase.P0: PhaseStatus.PASS, Phase.P1: PhaseStatus.PASS}
    require_prior_phases_passed(Phase.P2, statuses)
