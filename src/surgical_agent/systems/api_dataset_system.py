"""Dataset API rollout application built around the canonical pipeline."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import thaw_json
from surgical_agent.artifacts.manifest import atomic_write_text
from surgical_agent.cli.progress import progress_bar
from surgical_agent.config.schema import ApiConfig
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.api_rollout_selection import RolloutSelection
from surgical_agent.inference.frame_result_writer import FrameResultWriter
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION
from surgical_agent.perception.joint_api_vlm import (
    JointApiVlm,
    JointPerceptionRequestBuilder,
)
from surgical_agent.research.gate.policy import (
    AlwaysVerifyPolicy,
    EvidenceThresholdPolicy,
    ForcedTargetedVerificationPolicy,
    FrozenLinearBenefitGate,
)
from surgical_agent.research.gate.reliability import ReliabilityGatePolicy
from surgical_agent.research.memory.store import BoundedEventMemory
from surgical_agent.research.reporting import (
    EventReportGenerator,
    EventReportManager,
    EventReportWriter,
)
from surgical_agent.research.signals.contracts import PhaseTransitionGraph
from surgical_agent.research.signals.frame_evidence import (
    FrameEvidenceSignalExtractor,
)
from surgical_agent.research.verification.coordinator import DeterministicCoordinator
from surgical_agent.research.verification.hypotheses import (
    FactorizedCandidateGenerator,
)
from surgical_agent.research.verification.targeted_api import (
    TargetedApiVerifier,
    TargetedVerificationRequestBuilder,
)
from surgical_agent.systems.pipeline import (
    CanonicalStreamingPipeline,
    DisabledCandidateGenerator,
    DisabledSpecialistRegistry,
    GateObserver,
    NeverVerify,
    NoOpCausalStore,
    NoOpCoordinator,
    PipelineComponents,
    PipelineContractError,
    PredictionFinalizer,
)
from surgical_agent.tracking.predicted_provider import (
    PrecomputedPredictedTrackProvider,
    UnavailablePredictedTrackProvider,
)
from surgical_agent.workflow.state_store import FinalizedWorkflowStateStore


@dataclass(frozen=True)
class DatasetApiRunResult:
    predictions: tuple[PredictionRecord, ...]
    manifest_path: Path
    frame_counts: Mapping[str, int]
    usage_summary: Mapping[str, object]
    verification_summary: Mapping[str, object]
    report_manifest_path: Path
    causal_window_audit_path: Path
    report_count: int
    report_mode: str
    backbone_policy: str
    initial_model_requested: str
    verification_model_requested: str
    main_profile_backbone_match: bool


class DatasetApiPipelineSystem:
    """Run an ordered dataset selection through one causal API pipeline."""

    def __init__(
        self,
        *,
        client: CachedMultimodalApiClient,
        config: ApiConfig,
        verification_config: ApiConfig | None = None,
        writer: FrameResultWriter,
        media_loader: CausalApiMediaLoader,
        pipeline_profile: str = "single_pass",
        evidence_threshold: float = 0.75,
        max_candidates_per_task: int = 5,
        max_memory_events: int = 64,
        gate_artifact: str | Path | None = None,
        context_profile: str | None = None,
        phase_transition_graph: PhaseTransitionGraph | None = None,
        phase_allowed_ivt: Mapping[int, tuple[int, ...]] | None = None,
        track_provider: PrecomputedPredictedTrackProvider | None = None,
        event_memory_enabled: bool | None = None,
        report_mode: str = "template_report",
        report_window_size: int = 30,
        report_generator: EventReportGenerator | None = None,
        gate_observer: GateObserver | None = None,
    ) -> None:
        if pipeline_profile not in {
            "single_pass",
            "always_verify",
            "rule_gate",
            "selective_verify",
            "cascade_verify",
            "learned_gate",
            "counterfactual_verify",
        }:
            raise ValueError("unsupported API pipeline profile")
        if not isinstance(config, ApiConfig):
            raise TypeError("config must be an ApiConfig")
        final_only = config.response_schema_version == FINAL_ONLY_SCHEMA_VERSION
        if final_only:
            if pipeline_profile != "single_pass":
                raise ValueError(
                    "final-only H0 requires pipeline_profile=single_pass; "
                    "legacy Gate and Verifier profiles require ranked predictions"
                )
            if context_profile not in {None, "frames_only"}:
                raise ValueError("final-only H0 requires context_profile=frames_only")
            if event_memory_enabled is True:
                raise ValueError("final-only H0 requires event_memory_enabled=False")
        if verification_config is not None and not isinstance(
            verification_config, ApiConfig
        ):
            raise TypeError("verification_config must be an ApiConfig or None")
        if pipeline_profile == "cascade_verify":
            if verification_config is None:
                raise ValueError("cascade_verify requires verification_config")
            if (
                config.provider != verification_config.provider
                or config.endpoint_identifier != verification_config.endpoint_identifier
            ):
                raise ValueError(
                    "cascade_verify requires the same provider and endpoint"
                )
            if (
                config.requested_model_identifier != "openai/gpt-5.6-luna"
                or verification_config.requested_model_identifier
                != "openai/gpt-5.6-sol"
            ):
                raise ValueError(
                    "cascade_verify requires the approved Luna initial and Sol verification pair"
                )
            backbone_policy = "cascade_efficiency"
        else:
            if verification_config is not None and verification_config != config:
                raise ValueError("shared profiles require one shared ApiConfig")
            verification_config = config
            backbone_policy = "shared"
        if pipeline_profile == "learned_gate" and gate_artifact is None:
            raise ValueError("learned_gate requires gate_artifact")
        if pipeline_profile != "learned_gate" and gate_artifact is not None:
            raise ValueError("gate_artifact is only valid for learned_gate")
        if event_memory_enabled is not None and type(event_memory_enabled) is not bool:
            raise TypeError("event_memory_enabled must be boolean or None")
        resolved_event_memory_enabled = (
            not final_only if event_memory_enabled is None else event_memory_enabled
        )
        resolved_context_profile = context_profile
        if resolved_context_profile is None:
            resolved_context_profile = "frames_only" if final_only else "workflow"
        if resolved_context_profile not in {
            "frames_only",
            "track_only",
            "workflow",
            "track_workflow",
        }:
            raise ValueError("unsupported context profile")
        if (
            resolved_context_profile in {"frames_only", "track_only"}
            and phase_transition_graph is not None
        ):
            raise ValueError(
                f"{resolved_context_profile} context forbids a phase transition graph"
            )
        if resolved_context_profile in {"track_only", "track_workflow"}:
            if not isinstance(track_provider, PrecomputedPredictedTrackProvider):
                raise ValueError(
                    f"{resolved_context_profile} requires a predicted-track artifact"
                )
        elif track_provider is not None:
            raise ValueError("predicted-track artifacts require a track context")
        self.client = client
        self.config = config
        self.writer = writer
        self.media_loader = media_loader
        self.pipeline_profile = pipeline_profile
        self.backbone_policy = backbone_policy
        self.initial_model_requested = config.requested_model_identifier
        self.verification_model_requested = (
            verification_config.requested_model_identifier
        )
        self.context_profile = resolved_context_profile
        self.event_memory_enabled = resolved_event_memory_enabled
        self.phase_transition_graph = phase_transition_graph
        self.report_manager = EventReportManager(
            window_size=report_window_size,
            mode=report_mode,
            generator=report_generator,
        )
        self.report_writer = EventReportWriter(writer.output_dir)
        perception = JointApiVlm(
            client=client,
            request_builder=JointPerceptionRequestBuilder(config=config),
            data_upload_authorized=config.data_upload_authorized,
        )
        if pipeline_profile == "single_pass":
            candidate_generator = DisabledCandidateGenerator()
            gate_policy = NeverVerify()
            specialist_registry = DisabledSpecialistRegistry()
            coordinator = NoOpCoordinator()
            event_memory = (
                BoundedEventMemory(max_events_per_video=max_memory_events)
                if resolved_event_memory_enabled
                else NoOpCausalStore("memory")
            )
        else:
            candidate_generator = FactorizedCandidateGenerator(
                max_candidates_per_task=max_candidates_per_task
            )
            if pipeline_profile == "always_verify":
                gate_policy = AlwaysVerifyPolicy(scope="joint")
            elif pipeline_profile == "counterfactual_verify":
                gate_policy = ForcedTargetedVerificationPolicy(
                    ReliabilityGatePolicy(
                        confidence_threshold=evidence_threshold,
                        phase_transition_graph=phase_transition_graph,
                        phase_allowed_ivt=phase_allowed_ivt,
                        use_tracker=(
                            resolved_context_profile in {"track_only", "track_workflow"}
                        ),
                    )
                )
            elif pipeline_profile == "rule_gate":
                gate_policy = EvidenceThresholdPolicy(
                    threshold=evidence_threshold,
                    scope="joint",
                )
            elif pipeline_profile in {"selective_verify", "cascade_verify"}:
                gate_policy = ReliabilityGatePolicy(
                    confidence_threshold=evidence_threshold,
                    phase_transition_graph=phase_transition_graph,
                    phase_allowed_ivt=phase_allowed_ivt,
                    use_tracker=(
                        resolved_context_profile in {"track_only", "track_workflow"}
                    ),
                )
            else:
                assert gate_artifact is not None
                gate_policy = FrozenLinearBenefitGate.from_json(
                    gate_artifact,
                    finding_policy=ReliabilityGatePolicy(
                        confidence_threshold=evidence_threshold,
                        phase_transition_graph=phase_transition_graph,
                        phase_allowed_ivt=phase_allowed_ivt,
                        use_tracker=(
                            resolved_context_profile in {"track_only", "track_workflow"}
                        ),
                    ),
                    phase_allowed_ivt=phase_allowed_ivt,
                )
                if gate_policy.artifact.scope_order != ("joint",):
                    raise ValueError(
                        "learned_gate artifact scopes must match the joint verifier"
                    )
            specialist_registry = TargetedApiVerifier(
                client=client,
                request_builder=TargetedVerificationRequestBuilder(
                    config=verification_config
                ),
                data_upload_authorized=verification_config.data_upload_authorized,
            )
            coordinator = DeterministicCoordinator()
            event_memory = (
                BoundedEventMemory(max_events_per_video=max_memory_events)
                if resolved_event_memory_enabled
                else NoOpCausalStore("memory")
            )
        workflow_store = (
            NoOpCausalStore("workflow")
            if resolved_context_profile in {"frames_only", "track_only"}
            else FinalizedWorkflowStateStore()
        )
        effective_track_provider = (
            track_provider
            if resolved_context_profile in {"track_only", "track_workflow"}
            else UnavailablePredictedTrackProvider()
        )
        self.pipeline = CanonicalStreamingPipeline(
            PipelineComponents(
                context_builder=CausalPerceptionContextBuilder(
                    max_frames=config.max_causal_frames,
                    max_images=config.max_api_images,
                    selection_strategy=str(
                        config.provider_options.get(
                            "frame_selection_strategy", "adaptive"
                        )
                    ),
                    history_image_detail=str(
                        config.provider_options.get("history_image_detail", "auto")
                    ),
                    target_image_detail=str(
                        config.provider_options.get("target_image_detail", "auto")
                    ),
                ),
                perception=perception,
                candidate_generator=candidate_generator,
                signal_extractor=FrameEvidenceSignalExtractor(
                    phase_transition_graph=phase_transition_graph
                ),
                gate_policy=gate_policy,
                specialist_registry=specialist_registry,
                coordinator=coordinator,
                finalizer=PredictionFinalizer(),
                workflow_store=workflow_store,
                event_memory=event_memory,
                result_sink=writer,
                track_provider=effective_track_provider,
                backbone_policy=backbone_policy,
                gate_observer=gate_observer,
            )
        )

    def run(
        self,
        selection: RolloutSelection,
        *,
        run_id: str,
        defer_completion: bool = False,
        progress_enabled: bool = False,
        progress_file: IO[str] | None = None,
    ) -> DatasetApiRunResult:
        if not isinstance(selection, RolloutSelection):
            raise TypeError("selection must be a RolloutSelection")
        predictions: list[PredictionRecord] = []
        causal_window_audits: list[dict[str, object]] = []
        with progress_bar(
            total=len(selection.samples),
            description="API inference",
            unit="frame",
            enabled=progress_enabled,
            file=progress_file,
        ) as progress:
            for sample in selection.samples:
                first_content_latency_ms: float | None = None
                progress.set_postfix_str(
                    f"{sample.video_id}:{sample.target_frame_id} waiting",
                    refresh=True,
                )
                set_first_content_callback = getattr(
                    self.client.transport,
                    "set_first_content_callback",
                    None,
                )
                if callable(set_first_content_callback):

                    def show_first_content(
                        latency_ms: float,
                        *,
                        video_id: str = sample.video_id,
                        frame_id: int = sample.target_frame_id,
                    ) -> None:
                        nonlocal first_content_latency_ms
                        first_content_latency_ms = latency_ms
                        progress.set_postfix_str(
                            f"{video_id}:{frame_id} first-token={latency_ms / 1000.0:.1f}s",
                            refresh=True,
                        )

                    set_first_content_callback(show_first_content)
                try:
                    loaded = self.media_loader.load(sample)
                    result = self.pipeline.run(
                        loaded.runtime_sample,
                        loaded.frames,
                        run_id=run_id,
                    )
                finally:
                    if callable(set_first_content_callback):
                        set_first_content_callback(None)
                predictions.append(result.prediction)
                causal_window_audits.append(
                    {
                        "video_id": sample.video_id,
                        "target_frame_id": sample.target_frame_id,
                        "causal_frame_ids": list(sample.causal_frame_ids),
                        "selected_image_frame_ids": list(
                            result.selected_image_frame_ids
                        ),
                        "temporal_evidence": thaw_json(result.temporal_evidence),
                    }
                )
                self.report_writer.write_many(self.report_manager.observe(result.event))
                completed_status = (
                    f"{sample.video_id}:{sample.target_frame_id} complete"
                )
                if first_content_latency_ms is not None:
                    completed_status += (
                        f" first-token={first_content_latency_ms / 1000.0:.1f}s"
                    )
                progress.set_postfix_str(completed_status, refresh=False)
                progress.update()

        self._assert_selection_matches(selection, predictions)
        causal_window_audit_path = self.writer.output_dir / "causal_window_audit.jsonl"
        atomic_write_text(
            causal_window_audit_path,
            "".join(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
                for row in causal_window_audits
            ),
        )
        self.report_writer.write_many(self.report_manager.finalize())
        report_manifest_path = self.report_writer.finalize()
        manifest_path = self.writer.finalize(
            {"paper_metric_eligible": False},
            defer_completion=defer_completion,
        )
        return DatasetApiRunResult(
            predictions=tuple(predictions),
            manifest_path=manifest_path,
            frame_counts=dict(selection.frame_counts),
            usage_summary=self.client.usage.summarize(),
            verification_summary=self._summarize_verification(predictions),
            report_manifest_path=report_manifest_path,
            causal_window_audit_path=causal_window_audit_path,
            report_count=len(self.report_writer.records),
            report_mode=self.report_manager.mode,
            backbone_policy=self.backbone_policy,
            initial_model_requested=self.initial_model_requested,
            verification_model_requested=self.verification_model_requested,
            main_profile_backbone_match=(self.backbone_policy == "shared"),
        )

    @staticmethod
    def _summarize_verification(
        predictions: list[PredictionRecord],
    ) -> Mapping[str, object]:
        gate_counts = Counter(record.gate_action for record in predictions)
        outcome_counts = Counter(record.verification_status for record in predictions)
        requested = gate_counts["VERIFY"]
        successful = outcome_counts["VERIFIED_KEEP"] + outcome_counts["VERIFIED_REPAIR"]
        return {
            "gate_accept_count": gate_counts["ACCEPT"],
            "gate_verify_count": requested,
            "not_requested_count": outcome_counts["NOT_REQUESTED"],
            "verified_keep_count": outcome_counts["VERIFIED_KEEP"],
            "verified_repair_count": outcome_counts["VERIFIED_REPAIR"],
            "fallback_keep_count": outcome_counts["FALLBACK_KEEP"],
            "verification_success_rate": (
                None if requested == 0 else successful / requested
            ),
            "verification_contract_satisfied": outcome_counts["FALLBACK_KEEP"] == 0,
        }

    @staticmethod
    def _assert_selection_matches(
        selection: RolloutSelection,
        predictions: list[PredictionRecord],
    ) -> None:
        expected_ids = tuple(
            (sample.video_id, sample.target_frame_id) for sample in selection.samples
        )
        actual_ids = tuple(
            (prediction.video_id, prediction.frame_id) for prediction in predictions
        )
        if actual_ids != expected_ids:
            raise PipelineContractError(
                "dataset result identities do not match selection"
            )
        actual_counts = Counter(prediction.video_id for prediction in predictions)
        if dict(actual_counts) != dict(selection.frame_counts):
            raise PipelineContractError(
                "dataset result frame counts do not match selection"
            )
