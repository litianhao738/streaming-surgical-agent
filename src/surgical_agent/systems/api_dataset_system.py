"""Dataset API rollout application built around the canonical pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.config.schema import ApiConfig
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.api_rollout_selection import RolloutSelection
from surgical_agent.inference.frame_result_writer import FrameResultWriter
from surgical_agent.inference.schemas import PredictionRecord
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
    PipelineContractError,
    PredictionFinalizer,
)


@dataclass(frozen=True)
class DatasetApiRunResult:
    predictions: tuple[PredictionRecord, ...]
    manifest_path: Path
    frame_counts: Mapping[str, int]
    usage_summary: Mapping[str, object]


class DatasetApiPipelineSystem:
    """Run an ordered dataset selection through one causal API pipeline."""

    def __init__(
        self,
        *,
        client: CachedMultimodalApiClient,
        config: ApiConfig,
        writer: FrameResultWriter,
        media_loader: CausalApiMediaLoader,
    ) -> None:
        self.client = client
        self.config = config
        self.writer = writer
        self.media_loader = media_loader
        self.pipeline = CanonicalStreamingPipeline(
            PipelineComponents(
                context_builder=CausalPerceptionContextBuilder(
                    max_frames=config.max_causal_frames
                ),
                perception=JointApiVlm(
                    client=client,
                    request_builder=JointPerceptionRequestBuilder(config=config),
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

    def run(
        self,
        selection: RolloutSelection,
        *,
        run_id: str,
        defer_completion: bool = False,
    ) -> DatasetApiRunResult:
        if not isinstance(selection, RolloutSelection):
            raise TypeError("selection must be a RolloutSelection")
        predictions: list[PredictionRecord] = []
        for sample in selection.samples:
            loaded = self.media_loader.load(sample)
            result = self.pipeline.run(
                loaded.runtime_sample,
                loaded.frames,
                run_id=run_id,
            )
            predictions.append(result.prediction)

        self._assert_selection_matches(selection, predictions)
        manifest_path = self.writer.finalize(
            {"paper_metric_eligible": False},
            defer_completion=defer_completion,
        )
        return DatasetApiRunResult(
            predictions=tuple(predictions),
            manifest_path=manifest_path,
            frame_counts=dict(selection.frame_counts),
            usage_summary=self.client.usage.summarize(),
        )

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
            raise PipelineContractError("dataset result identities do not match selection")
        actual_counts = Counter(prediction.video_id for prediction in predictions)
        if dict(actual_counts) != dict(selection.frame_counts):
            raise PipelineContractError("dataset result frame counts do not match selection")
