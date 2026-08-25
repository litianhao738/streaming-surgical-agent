"""P2 assembly around the single canonical pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from surgical_agent.data.dataset import (
    CurrentFrameTensorDataset,
    ResolvedSample,
)
from surgical_agent.evaluation.evaluator import EvaluationEngine, SampleSmokeMetrics
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.inference.writer import PredictionWriter
from surgical_agent.models.baseline import LocalSmokeModel
from surgical_agent.systems.pipeline import (
    CanonicalStreamingPipeline,
    DisabledCandidateGenerator,
    DisabledSpecialistRegistry,
    FramesOnlyContextBuilder,
    LocalSmokePerception,
    NeverVerify,
    NoOpCausalStore,
    NoOpCoordinator,
    NoOpSignalExtractor,
    PipelineComponents,
    PredictionFinalizer,
)


@dataclass(frozen=True)
class BaselineRunResult:
    predictions: tuple[PredictionRecord, ...]
    sample_metrics: tuple[SampleSmokeMetrics, ...]
    metric_summary: dict[str, object]
    manifest_path: Path


class P2BaselineSystem:
    """Application layer that keeps runtime prediction and offline GT evaluation split."""

    def __init__(
        self,
        model: LocalSmokeModel,
        *,
        device: torch.device,
        writer: PredictionWriter,
        image_size: int = 96,
    ) -> None:
        self.writer = writer
        self.image_size = image_size
        self.evaluator = EvaluationEngine()
        self.pipeline = CanonicalStreamingPipeline(
            PipelineComponents(
                context_builder=FramesOnlyContextBuilder(),
                perception=LocalSmokePerception(model, device=device),
                candidate_generator=DisabledCandidateGenerator(),
                signal_extractor=NoOpSignalExtractor(),
                gate_policy=NeverVerify(),
                specialist_registry=DisabledSpecialistRegistry(),
                coordinator=NoOpCoordinator(),
                finalizer=PredictionFinalizer(),
                workflow_store=NoOpCausalStore("workflow"),
                event_memory=NoOpCausalStore("memory"),
                prediction_writer=writer,
            )
        )

    def run(
        self,
        records: tuple[ResolvedSample, ...],
        *,
        run_id: str,
        manifest_metadata: dict[str, object],
    ) -> BaselineRunResult:
        dataset = CurrentFrameTensorDataset(records, image_size=self.image_size)
        predictions: list[PredictionRecord] = []
        sample_metrics: list[SampleSmokeMetrics] = []
        for index, resolved in enumerate(records):
            tensor_sample = dataset[index]
            runtime_result = self.pipeline.run(
                resolved.inference,
                tensor_sample.image.unsqueeze(0),
                run_id=run_id,
            )
            predictions.append(runtime_result.prediction)
            if resolved.evaluation is not None:
                sample_metrics.append(
                    self.evaluator.evaluate(
                        runtime_result.prediction,
                        resolved.evaluation,
                        resolved.frame_supervision,
                    )
                )
        metric_summary = self.evaluator.summarize(tuple(sample_metrics))
        manifest_path = self.writer.finalize(manifest_metadata)
        return BaselineRunResult(
            predictions=tuple(predictions),
            sample_metrics=tuple(sample_metrics),
            metric_summary=metric_summary,
            manifest_path=manifest_path,
        )
