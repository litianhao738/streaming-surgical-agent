"""Dataset runner for the uploaded final Pipeline and its four formal cells."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.api_rollout_selection import RolloutSelection
from surgical_agent.research.outcome import ExecutionOutcome
from surgical_agent.runtime.final_artifacts import FinalPipelineArtifactWriter
from surgical_agent.runtime.state import ObservationIdentity
from surgical_agent.systems.final_pipeline import FinalStreamingPipeline


@dataclass(frozen=True)
class FinalDatasetRunResult:
    manifest_path: Path
    outcome_counts: dict[str, int]
    processed_count: int
    resumed_count: int


class FinalDatasetPipelineSystem:
    """Run an ordered selection with crash-safe, per-video resume semantics."""

    def __init__(
        self,
        *,
        pipeline: FinalStreamingPipeline,
        media_loader: CausalApiMediaLoader,
        writer: FinalPipelineArtifactWriter,
    ) -> None:
        if not isinstance(pipeline, FinalStreamingPipeline):
            raise TypeError("pipeline must be FinalStreamingPipeline")
        if not isinstance(media_loader, CausalApiMediaLoader):
            raise TypeError("media_loader must be CausalApiMediaLoader")
        if not isinstance(writer, FinalPipelineArtifactWriter):
            raise TypeError("writer must be FinalPipelineArtifactWriter")
        self.pipeline = pipeline
        self.media_loader = media_loader
        self.writer = writer

    def run(self, selection: RolloutSelection) -> FinalDatasetRunResult:
        if not isinstance(selection, RolloutSelection):
            raise TypeError("selection must be RolloutSelection")
        self.writer.begin()
        current_video: str | None = None
        records_by_key = {}
        processed = resumed = 0
        for sample in selection.samples:
            if current_video != sample.video_id:
                self.pipeline.reset(sample.video_id, recover=True)
                current_video = sample.video_id
            observation = ObservationIdentity.from_sample(sample)
            existing = self.pipeline.components.finalization_store.record_for(
                observation.key
            )
            if existing is not None:
                record = existing
                resumed += 1
            else:
                loaded = self.media_loader.load(sample)
                result = self.pipeline.run(loaded.runtime_sample, loaded.frames)
                record = result.finalization
                for resolved_record in result.resolved_pending:
                    self.writer.write(resolved_record)
                    records_by_key[resolved_record.observation.key] = resolved_record
                processed += 1
            self.writer.write(record)
            records_by_key[record.observation.key] = record
        manifest = self.writer.finalize()
        if processed + resumed != len(selection.samples):
            raise RuntimeError("formal dataset runner lost an observation")
        counts: dict[str, int] = {}
        for record in records_by_key.values():
            outcome = record.outcome
            key = outcome.status if isinstance(outcome, ExecutionOutcome) else outcome.state
            counts[key] = counts.get(key, 0) + 1
        return FinalDatasetRunResult(
            manifest_path=manifest,
            outcome_counts=counts,
            processed_count=processed,
            resumed_count=resumed,
        )


__all__ = ["FinalDatasetPipelineSystem", "FinalDatasetRunResult"]
