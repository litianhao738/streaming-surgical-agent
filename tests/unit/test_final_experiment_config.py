from __future__ import annotations

from pathlib import Path

import pytest

from surgical_agent.config.final_experiment import (
    load_tracker_gate_cell,
    validate_tracker_gate_matrix,
)
from surgical_agent.config.loader import load_api_config
from surgical_agent.config.schema import ApiConfig
from surgical_agent.research.gate.formal_policy import RuleBenefitGate
from surgical_agent.systems.final_pipeline_factory import (
    build_final_api_pipeline,
    validate_heterogeneous_verifier_pair,
    validate_verifier_pair,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_formal_cells():
    names = ("a_base", "b_tracker", "c_gate", "d_full")
    return [
        load_tracker_gate_cell(PROJECT_ROOT / "configs" / "ablations" / f"{name}.yaml")
        for name in names
    ]


def test_formal_tracker_gate_matrix_changes_only_two_factors() -> None:
    cells = _load_formal_cells()

    validate_tracker_gate_matrix(cells)

    by_name = {cell.cell: cell for cell in cells}
    assert (by_name["A_base"].tracker_enabled, by_name["A_base"].gate_mode) == (
        False,
        "RULE",
    )
    assert (by_name["B_tracker"].tracker_enabled, by_name["B_tracker"].gate_mode) == (
        True,
        "RULE",
    )
    assert (by_name["C_gate"].tracker_enabled, by_name["C_gate"].gate_mode) == (
        False,
        "LEARNED",
    )
    assert (by_name["D_full"].tracker_enabled, by_name["D_full"].gate_mode) == (
        True,
        "LEARNED",
    )


def test_matrix_rejects_missing_cell() -> None:
    with pytest.raises(ValueError, match="exactly"):
        validate_tracker_gate_matrix(_load_formal_cells()[:-1])


class _UnusedClient:
    def call(self, request):  # pragma: no cover - construction test only
        raise AssertionError(request)


def _api_config(*, max_images: int = 6) -> ApiConfig:
    return ApiConfig.from_mapping(
        {
            "enabled": True,
            "mode": "mock",
            "provider": "mock",
            "endpoint_identifier": "mock://final-pipeline",
            "requested_model_identifier": "mock-final-pipeline-v1",
            "prompt_version": "joint_perception_v1",
            "response_schema_version": "joint_perception_v1",
            "max_causal_frames": 6,
            "max_api_images": max_images,
            "provider_options": {
                "frame_selection_strategy": "fixed_all",
                "history_image_detail": "low",
                "target_image_detail": "auto",
                "initial_prompt_profile": "fixed_visual_only",
                "verification_prompt_profile": "fixed_visual_only",
            },
        }
    )


def _verification_api_config(
    *,
    model: str = "mock-heterogeneous-verifier-v1",
) -> ApiConfig:
    return ApiConfig.from_mapping(
        {
            "enabled": True,
            "mode": "mock",
            "provider": "mock",
            "endpoint_identifier": "mock://final-pipeline",
            "requested_model_identifier": model,
            "prompt_version": "targeted_verification_prompt_v6",
            "response_schema_version": "targeted_verification_v1",
            "max_causal_frames": 6,
            "max_api_images": 6,
            "provider_options": {
                "frame_selection_strategy": "fixed_all",
                "history_image_detail": "low",
                "target_image_detail": "auto",
                "initial_prompt_profile": "fixed_visual_only",
                "verification_prompt_profile": "fixed_visual_only",
            },
        }
    )


def test_factory_assembles_rule_cell_without_legacy_coordinator(tmp_path: Path) -> None:
    cell = _load_formal_cells()[0]

    pipeline = build_final_api_pipeline(
        client=_UnusedClient(),  # type: ignore[arg-type]
        api_config=_api_config(),
        verification_api_config=_verification_api_config(),
        cell=cell,
        state_dir=tmp_path,
    )

    assert isinstance(pipeline.components.benefit_gate, RuleBenefitGate)
    assert pipeline.components.track_provider is None
    assert not hasattr(pipeline.components, "coordinator")
    assert pipeline.components.context_builder.selection_strategy == "fixed_all"


def test_factory_accepts_an_explicit_heterogeneous_verifier(tmp_path: Path) -> None:
    pipeline = build_final_api_pipeline(
        client=_UnusedClient(),  # type: ignore[arg-type]
        api_config=_api_config(),
        verification_api_config=_verification_api_config(),
        cell=_load_formal_cells()[0],
        state_dir=tmp_path,
    )

    assert pipeline.components.max_verify_attempts == 1
    assert pipeline.components.initial_model_requested == "mock-final-pipeline-v1"
    assert (
        pipeline.components.verification_model_requested
        == "mock-heterogeneous-verifier-v1"
    )


def test_heterogeneous_verifier_rejects_the_initial_model() -> None:
    with pytest.raises(ValueError, match="different model"):
        validate_heterogeneous_verifier_pair(
            _api_config(),
            _verification_api_config(model="mock-final-pipeline-v1"),
        )


def test_conservative_v7_accepts_the_same_initial_model() -> None:
    initial = load_api_config(
        PROJECT_ROOT / "configs/perception/joint_openrouter_final_fixed6.yaml"
    )
    verification = load_api_config(
        PROJECT_ROOT
        / "configs/perception/targeted_openrouter_gpt56sol_constrained_fixed6.yaml"
    )

    validate_verifier_pair(initial, verification)

    assert initial.requested_model_identifier == verification.requested_model_identifier
    assert verification.prompt_version == "targeted_verification_prompt_v7"


def test_cross_provider_verifier_requires_an_independent_client(tmp_path: Path) -> None:
    verification = ApiConfig.from_mapping(
        {
            "enabled": True,
            "mode": "real",
            "provider": "openai_compatible",
            "endpoint_identifier": "https://api.x.ai/v1/chat/completions",
            "requested_model_identifier": "grok-4.6",
            "prompt_version": "targeted_verification_prompt_v6",
            "response_schema_version": "targeted_verification_v1",
            "max_causal_frames": 6,
            "max_api_images": 6,
            "provider_options": {
                "frame_selection_strategy": "fixed_all",
                "history_image_detail": "low",
                "target_image_detail": "auto",
                "initial_prompt_profile": "fixed_visual_only",
                "verification_prompt_profile": "fixed_visual_only",
            },
        }
    )
    initial = load_api_config(
        PROJECT_ROOT / "configs/perception/joint_openrouter_final_fixed6.yaml"
    )

    validate_heterogeneous_verifier_pair(initial, verification)
    with pytest.raises(ValueError, match="verification_client"):
        build_final_api_pipeline(
            client=_UnusedClient(),  # type: ignore[arg-type]
            api_config=initial,
            verification_api_config=verification,
            cell=_load_formal_cells()[0],
            state_dir=tmp_path,
        )

    pipeline = build_final_api_pipeline(
        client=_UnusedClient(),  # type: ignore[arg-type]
        verification_client=_UnusedClient(),  # type: ignore[arg-type]
        api_config=initial,
        verification_api_config=verification,
        cell=_load_formal_cells()[0],
        state_dir=tmp_path,
    )
    assert pipeline.components.verification_model_requested == "grok-4.6"


@pytest.mark.parametrize(
    ("config_name", "expected_model"),
    [
        (
            "targeted_openrouter_gemini31pro_fixed6.yaml",
            "google/gemini-3.1-pro-preview",
        ),
        (
            "targeted_openrouter_claude46_fixed6.yaml",
            "anthropic/claude-sonnet-4.6",
        ),
        ("targeted_xai_grok46_fixed6.yaml", "grok-4.6"),
    ],
)
def test_checked_in_capability_candidates_are_cross_family_and_blind(
    config_name: str,
    expected_model: str,
) -> None:
    initial = load_api_config(
        PROJECT_ROOT / "configs/perception/joint_openrouter_final_fixed6.yaml"
    )
    verification = load_api_config(
        PROJECT_ROOT / "configs/perception" / config_name
    )

    validate_heterogeneous_verifier_pair(initial, verification)

    assert initial.requested_model_identifier == "openai/gpt-5.6-sol"
    assert verification.requested_model_identifier == expected_model
    assert verification.provider_options["verification_prompt_profile"] == (
        "fixed_visual_only"
    )


def test_factory_rejects_an_api_config_that_hides_causal_images() -> None:
    with pytest.raises(ValueError, match="all six"):
        build_final_api_pipeline(
            client=_UnusedClient(),  # type: ignore[arg-type]
            api_config=_api_config(max_images=3),
            verification_api_config=_verification_api_config(),
            cell=_load_formal_cells()[0],
        )
