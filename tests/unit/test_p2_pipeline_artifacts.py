"""Canonical runner, Gold-free prediction, evaluator, and writer tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.schemas import InitialPrediction, PredictionRecord
from surgical_agent.inference.writer import ArtifactWriteError, PredictionWriter
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


def _sample(video_id: str, frame_id: int) -> InferenceSample:
    return InferenceSample(
        video_id=video_id,
        target_frame_id=frame_id,
        causal_frame_ids=(frame_id,),
        media_refs=(f"{frame_id}.png",),
        source_split=DatasetSplit.TRAINING,
        alignment_version="unit-test",
    )


def _pipeline(writer: PredictionWriter) -> CanonicalStreamingPipeline:
    return CanonicalStreamingPipeline(
        PipelineComponents(
            context_builder=FramesOnlyContextBuilder(),
            perception=LocalSmokePerception(
                LocalSmokeModel(),
                device=torch.device("cpu"),
            ),
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


def test_prediction_record_schema_is_gold_free() -> None:
    names = {field.name for field in fields(PredictionRecord)}
    assert not names & {
        "evaluation_target",
        "frame_supervision",
        "ground_truth",
        "label_mask",
    }


def test_prediction_schema_rejects_wrong_probability_shape() -> None:
    probabilities = {
        task: (0.5,) * class_count
        for task, class_count in TASK_CLASS_COUNTS.items()
    }
    probabilities["phase"] = (0.5,)
    with pytest.raises(ValueError, match="phase probabilities"):
        InitialPrediction(
            instrument_ids=(0,),
            verb_ids=(1,),
            target_ids=(2,),
            triplet_ids=(3,),
            phase_id=4,
            probabilities=probabilities,
        )


def test_pipeline_resets_both_state_slots_at_video_boundary(tmp_path: Path) -> None:
    writer = PredictionWriter(tmp_path, run_id="reset-test")
    pipeline = _pipeline(writer)
    frame = torch.rand(1, 3, 32, 32)

    first = pipeline.run(_sample("VID02", 1), frame, run_id="reset-test")
    second = pipeline.run(_sample("VID02", 2), frame, run_id="reset-test")
    third = pipeline.run(_sample("VID30", 1), frame, run_id="reset-test")

    assert first.runtime_trace[0] == "01_video_boundary_reset"
    assert second.runtime_trace[0] == "01_video_boundary_continue"
    assert third.runtime_trace[0] == "01_video_boundary_reset"
    assert pipeline.components.workflow_store.reset_history == ["VID02", "VID30"]
    assert pipeline.components.event_memory.reset_history == ["VID02", "VID30"]
    assert all(record.gate_action == "ACCEPT" for record in writer.records)


def test_prediction_writer_materializes_hash_verified_completion_last(
    tmp_path: Path,
) -> None:
    writer = PredictionWriter(tmp_path, run_id="writer-test")
    pipeline = _pipeline(writer)
    pipeline.run(
        _sample("VID02", 1),
        torch.rand(1, 3, 32, 32),
        run_id="writer-test",
    )
    manifest_path = writer.finalize({"paper_metric_eligible": False})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert manifest["record_count"] == 1
    assert manifest["predictions_sha256"] == hashlib.sha256(
        writer.prediction_path.read_bytes()
    ).hexdigest()
    assert not tuple(tmp_path.glob("*.tmp"))
    with pytest.raises(ArtifactWriteError, match="after writer finalization"):
        writer.write(writer.records[0])
