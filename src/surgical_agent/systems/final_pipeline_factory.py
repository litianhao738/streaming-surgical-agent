"""Strict assembly boundary for the uploaded final Pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.schema import TARGETED_VERIFICATION_SCHEMA_VERSION
from surgical_agent.config.final_experiment import TrackerGateCell
from surgical_agent.config.schema import ApiConfig
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION
from surgical_agent.perception.joint_api_vlm import (
    JointApiVlm,
    JointPerceptionRequestBuilder,
)
from surgical_agent.research.gate.benefit_model import FrozenLinearBenefitArtifact
from surgical_agent.research.gate.budget import VerificationBudgetManager
from surgical_agent.research.gate.formal_policy import (
    LearnedBenefitGate,
    RuleBenefitGate,
)
from surgical_agent.research.memory.pending import (
    BoundedPendingResolver,
    PendingResolutionBackend,
)
from surgical_agent.research.outcome import OutcomeFinalizer
from surgical_agent.research.safety import (
    DecisionSupportBuilder,
    MandatorySafetyGuard,
    SafetyValidator,
)
from surgical_agent.research.signals.contracts import PhaseTransitionGraph
from surgical_agent.research.signals.frame_evidence import FrameEvidenceSignalExtractor
from surgical_agent.research.verification.hypotheses import FactorizedCandidateGenerator
from surgical_agent.research.verification.targeted_api import (
    TARGETED_VERIFICATION_PROMPT_V6,
    TARGETED_VERIFICATION_PROMPT_V7,
    TARGETED_VERIFICATION_PROMPT_V8,
    TARGETED_VERIFICATION_PROMPT_V9,
    TargetedApiVerifier,
    TargetedVerificationRequestBuilder,
)
from surgical_agent.research.verification.verifier import BoundTargetedSpecialist
from surgical_agent.runtime.finalization import AtomicFinalizationStore
from surgical_agent.systems.final_pipeline import (
    FinalPipelineComponents,
    FinalStreamingPipeline,
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"shared config section {name} must be a mapping")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"shared config field {name} must be an integer")
    return value


def _openrouter_model_owner(model_identifier: str) -> str:
    """Return the explicit OpenRouter model owner used as a family boundary."""

    owner, separator, _model = model_identifier.partition("/")
    if not separator or not owner:
        raise ValueError(
            "formal OpenRouter model identifiers must include an owner prefix"
        )
    return owner.casefold()


def _model_family(config: ApiConfig) -> str:
    if config.provider == "openrouter":
        return _openrouter_model_owner(config.requested_model_identifier)
    if config.provider == "openai":
        return "openai"
    if config.provider == "openai_compatible":
        hostname = urlsplit(config.endpoint_identifier).hostname
        if not hostname:
            raise ValueError("compatible Verifier endpoint must have a hostname")
        return hostname.casefold()
    return f"{config.provider}:{config.requested_model_identifier.casefold()}"


def validate_heterogeneous_verifier_pair(
    initial_config: ApiConfig,
    verification_config: ApiConfig,
) -> None:
    """Reject verifier configurations that are not an auditable model-family split."""

    if not isinstance(initial_config, ApiConfig):
        raise TypeError("initial_config must be ApiConfig")
    if not isinstance(verification_config, ApiConfig):
        raise TypeError("verification_config must be ApiConfig")
    initial_config.validate()
    verification_config.validate()
    if (
        initial_config.requested_model_identifier
        == verification_config.requested_model_identifier
    ):
        raise ValueError("heterogeneous Verifier must use a different model")
    if _model_family(initial_config) == _model_family(verification_config):
        raise ValueError(
            "heterogeneous Verifier must use a different auditable model family"
        )
    if verification_config.prompt_version != TARGETED_VERIFICATION_PROMPT_V6:
        raise ValueError("Verifier config must declare the active V6 prompt")
    if (
        verification_config.response_schema_version
        != TARGETED_VERIFICATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "Verifier config must declare the targeted verification schema"
        )


def validate_verifier_pair(
    initial_config: ApiConfig,
    verification_config: ApiConfig,
) -> None:
    """Validate legacy heterogeneous or conservative same-model verification."""

    if not isinstance(initial_config, ApiConfig):
        raise TypeError("initial_config must be ApiConfig")
    if not isinstance(verification_config, ApiConfig):
        raise TypeError("verification_config must be ApiConfig")
    initial_config.validate()
    verification_config.validate()
    if (
        verification_config.response_schema_version
        != TARGETED_VERIFICATION_SCHEMA_VERSION
    ):
        raise ValueError(
            "Verifier config must declare the targeted verification schema"
        )
    same_model = (
        initial_config.provider == verification_config.provider
        and initial_config.endpoint_identifier
        == verification_config.endpoint_identifier
        and initial_config.requested_model_identifier
        == verification_config.requested_model_identifier
    )
    if same_model:
        if verification_config.prompt_version not in {
            TARGETED_VERIFICATION_PROMPT_V7,
            TARGETED_VERIFICATION_PROMPT_V8,
            TARGETED_VERIFICATION_PROMPT_V9,
        }:
            raise ValueError(
                "same-model Verifier must use V7, V8, or V9 review prompts"
            )
        return
    validate_heterogeneous_verifier_pair(initial_config, verification_config)


def verifier_pair_is_heterogeneous(
    initial_config: ApiConfig,
    verification_config: ApiConfig,
) -> bool:
    """Return the architecture flag after validating the configured pair."""

    validate_verifier_pair(initial_config, verification_config)
    return not (
        initial_config.provider == verification_config.provider
        and initial_config.endpoint_identifier
        == verification_config.endpoint_identifier
        and initial_config.requested_model_identifier
        == verification_config.requested_model_identifier
    )


def build_final_api_pipeline(
    *,
    client: CachedMultimodalApiClient,
    verification_client: CachedMultimodalApiClient | None = None,
    api_config: ApiConfig,
    verification_api_config: ApiConfig,
    cell: TrackerGateCell,
    state_dir: str | Path | None = None,
    track_provider: object | None = None,
    phase_transition_graph: PhaseTransitionGraph | None = None,
    phase_instrument_ivt_prior: Mapping[tuple[int, int], tuple[int, ...]] | None = None,
    strict_phase_allowed_ivt: Mapping[int, tuple[int, ...]] | None = None,
    phase_ivt_support: Mapping[int, tuple[int, ...]] | None = None,
    pending_resolution_backend: PendingResolutionBackend | None = None,
) -> FinalStreamingPipeline:
    """Assemble one executable cell; no legacy coordinator or Gate may enter."""

    if not isinstance(api_config, ApiConfig):
        raise TypeError("api_config must be ApiConfig")
    api_config.validate()
    if api_config.response_schema_version == FINAL_ONLY_SCHEMA_VERSION:
        raise ValueError(
            "final-only H0 has no confidence rankings and cannot use the legacy "
            "Tracker/Gate/Verifier pipeline; run scripts/run_dataset_api_pipeline.py "
            "with the main H0 single_pass experiment, or select an explicit ranked "
            "research configuration for this factory"
        )
    validate_verifier_pair(api_config, verification_api_config)
    if verification_client is None and (
        api_config.provider != verification_api_config.provider
        or api_config.endpoint_identifier != verification_api_config.endpoint_identifier
    ):
        raise ValueError("cross-provider verification requires verification_client")
    effective_verification_client = verification_client or client
    if api_config.max_causal_frames != 3 or api_config.max_api_images != 3:
        raise ValueError(
            "the final Pipeline API config must expose all three causal images"
        )
    provider_options = api_config.provider_options
    if provider_options.get("frame_selection_strategy") != "fixed_all":
        raise ValueError("the final Pipeline forbids adaptive frame selection")
    if provider_options.get("history_image_detail") != "low":
        raise ValueError("the final Pipeline requires low-detail historical frames")
    if provider_options.get("target_image_detail") != "auto":
        raise ValueError("the final Pipeline requires auto-detail current frames")
    if provider_options.get("initial_prompt_profile") != "fixed_visual_only":
        raise ValueError(
            "formal JointPerception must keep H0 independent of the Tracker ablation"
        )
    if provider_options.get("verification_prompt_profile") != "fixed_visual_only":
        raise ValueError(
            "formal Specialist input must remain identical across Tracker cells"
        )
    verification_provider_options = verification_api_config.provider_options
    if (
        verification_api_config.max_causal_frames != 3
        or verification_api_config.max_api_images != 3
    ):
        raise ValueError("the Verifier must expose all three causal images")
    if verification_provider_options.get("frame_selection_strategy") != "fixed_all":
        raise ValueError("the Verifier forbids adaptive frame selection")
    if verification_provider_options.get("history_image_detail") != "low":
        raise ValueError("the Verifier requires low-detail history")
    if verification_provider_options.get("target_image_detail") != "auto":
        raise ValueError("the Verifier requires auto-detail target")
    if (
        verification_provider_options.get("verification_prompt_profile")
        != "fixed_visual_only"
    ):
        raise ValueError("Specialist input must remain identical across Tracker cells")
    if cell.tracker_enabled and track_provider is None:
        raise ValueError("Tracker-enabled cells require one predicted-track provider")
    if not cell.tracker_enabled and track_provider is not None:
        raise ValueError("Tracker-disabled cells forbid a track provider")

    shared = cell.shared
    coverage_review = (
        verification_api_config.prompt_version == "targeted_verification_prompt_v9"
    )
    if coverage_review and cell.gate_mode == "LEARNED":
        raise ValueError(
            "V9 coverage requires freshly collected Gate supervision; use RULE for the capability pilot"
        )
    context = _mapping(shared["context"], "context")
    safety = _mapping(shared["safety"], "safety")
    verification = _mapping(shared["verification"], "verification")
    budget = _mapping(shared["budget"], "budget")
    memory = _mapping(shared["memory"], "memory")
    expected_scopes = ("instrument_presence", "interaction", "workflow")
    if tuple(verification["scopes"]) != expected_scopes:  # type: ignore[arg-type]
        raise ValueError("formal verification scopes were changed")
    if safety["mandatory_guard_enabled"] is not True:
        raise ValueError("MandatorySafetyGuard is fixed on in every formal cell")

    validator = SafetyValidator(
        strict_phase_allowed_ivt=strict_phase_allowed_ivt,
        require_component_coverage=bool(safety["require_component_coverage"]),
    )
    if cell.gate_mode == "RULE":
        gate = RuleBenefitGate(
            risk_threshold=float(shared.get("rule_gate_threshold", 0.5))
        )
    else:
        assert cell.gate_artifact is not None
        gate = LearnedBenefitGate(
            FrozenLinearBenefitArtifact.from_json(cell.gate_artifact)
        )

    verifier = TargetedApiVerifier(
        client=effective_verification_client,
        request_builder=TargetedVerificationRequestBuilder(
            config=verification_api_config
        ),
        data_upload_authorized=verification_api_config.data_upload_authorized,
    )

    def specialist_factory(context_value, candidates, evidence):
        return BoundTargetedSpecialist(
            verifier=verifier,
            context=context_value,
            candidates=candidates,
            evidence=evidence,
            phase_transition_graph=phase_transition_graph,
        )

    return FinalStreamingPipeline(
        FinalPipelineComponents(
            context_builder=CausalPerceptionContextBuilder(
                max_frames=_integer(context["max_frames"], "context.max_frames"),
                max_images=_integer(context["max_images"], "context.max_images"),
                selection_strategy=str(context["selection_strategy"]),
                history_image_detail=str(context["history_image_detail"]),
                target_image_detail=str(context["target_image_detail"]),
            ),
            perception=JointApiVlm(
                client=client,
                request_builder=JointPerceptionRequestBuilder(config=api_config),
                data_upload_authorized=api_config.data_upload_authorized,
            ),
            candidate_generator=FactorizedCandidateGenerator(
                max_candidates_per_task=20,
                phase_instrument_ivt_prior=phase_instrument_ivt_prior,
                complete_ontology=coverage_review,
            ),
            signal_extractor=FrameEvidenceSignalExtractor(
                phase_transition_graph=phase_transition_graph
            ),
            safety_validator=validator,
            support_builder=DecisionSupportBuilder(
                phase_ivt_support=phase_ivt_support if coverage_review else None
            ),
            mandatory_guard=MandatorySafetyGuard(),
            benefit_gate=gate,
            budget=VerificationBudgetManager(
                safety_reserve=_integer(
                    budget["safety_reserve"], "budget.safety_reserve"
                ),
                optional_capacity=_integer(
                    budget["optional_capacity"], "budget.optional_capacity"
                ),
                shared_capacity=_integer(
                    budget["shared_capacity"], "budget.shared_capacity"
                ),
            ),
            specialist_factory=specialist_factory,
            outcome_finalizer=OutcomeFinalizer(),
            finalization_store=AtomicFinalizationStore(
                state_dir=state_dir,
                max_verified=_integer(memory["max_verified"], "memory.max_verified"),
                max_accepted=_integer(memory["max_accepted"], "memory.max_accepted"),
                max_resolution_attempts=_integer(
                    memory["max_resolution_attempts"],
                    "memory.max_resolution_attempts",
                ),
            ),
            initial_model_requested=api_config.requested_model_identifier,
            verification_model_requested=(
                verification_api_config.requested_model_identifier
            ),
            track_provider=track_provider,  # type: ignore[arg-type]
            pending_resolver=(
                None
                if pending_resolution_backend is None
                else BoundedPendingResolver(
                    backend=pending_resolution_backend,
                    validator=validator,
                )
            ),
            max_verify_attempts=_integer(
                verification["max_attempts"], "verification.max_attempts"
            ),
            max_coverage_scopes=3 if coverage_review else 1,
        )
    )


__all__ = [
    "build_final_api_pipeline",
    "validate_heterogeneous_verifier_pair",
    "validate_verifier_pair",
    "verifier_pair_is_heterogeneous",
]
