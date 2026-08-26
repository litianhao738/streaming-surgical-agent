"""P2 assembly around the single canonical pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from surgical_agent.data.dataset import ResolvedSample
from surgical_agent.data.schemas import InferenceSample
from surgical_agent.evaluation.evaluator import EvaluationEngine, SampleSmokeMetrics
from surgical_agent.inference.frame_result_writer import FrameResultWriter
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.models.baseline import LocalSmokeModel
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.research.signals.frame_evidence import (
    FrameEvidenceSignalExtractor,
)
from surgical_agent.systems.pipeline import (
    CanonicalStreamingPipeline,
    DisabledCandidateGenerator,
    DisabledSpecialistRegistry,
    LocalSmokePerception,
    NeverVerify,
    NoOpCausalStore,
    NoOpCoordinator,
    PipelineComponents,
    PredictionFinalizer,
)


@dataclass(frozen=True)
class BaselineRunResult:
    predictions: tuple[PredictionRecord, ...]
    sample_metrics: tuple[SampleSmokeMetrics, ...]
    metric_summary: dict[str, object]
    manifest_path: Path


def _load_causal_frames(sample: InferenceSample, *, image_size: int) -> torch.Tensor:
    import numpy as np
    from PIL import Image

    frames: list[torch.Tensor] = []
    for media_ref in sample.media_refs:
        with Image.open(Path(media_ref)) as image:
            rgb = image.convert("RGB").resize((image_size, image_size))
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    return torch.stack(frames)


class P2BaselineSystem:
    """Application layer that keeps runtime prediction and offline GT evaluation split."""

    def __init__(
        self,
        model: LocalSmokeModel,
        *,
        device: torch.device,
        writer: FrameResultWriter,
        image_size: int = 96,
    ) -> None:
        self.writer = writer
        self.image_size = image_size
        self.evaluator = EvaluationEngine()
        self.pipeline = CanonicalStreamingPipeline(
            PipelineComponents(
                context_builder=CausalPerceptionContextBuilder(),
                perception=LocalSmokePerception(model, device=device),
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
        records: tuple[ResolvedSample, ...],
        *,
        run_id: str,
        manifest_metadata: dict[str, object],
    ) -> BaselineRunResult:
        predictions: list[PredictionRecord] = []
        sample_metrics: list[SampleSmokeMetrics] = []
        for resolved in records:
            runtime_result = self.pipeline.run(
                resolved.inference,
                _load_causal_frames(
                    resolved.inference,
                    image_size=self.image_size,
                ),
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
        paired_manifest_metadata = (
            {
                "paper_metric_eligible": manifest_metadata[
                    "paper_metric_eligible"
                ]
            }
            if "paper_metric_eligible" in manifest_metadata
            else {}
        )
        manifest_path = self.writer.finalize(paired_manifest_metadata)
        return BaselineRunResult(
            predictions=tuple(predictions),
            sample_metrics=tuple(sample_metrics),
            metric_summary=metric_summary,
            manifest_path=manifest_path,
        )
