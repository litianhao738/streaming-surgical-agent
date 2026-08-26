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
from surgical_agent.inference.frame_result_writer import FrameResultWriter
from surgical_agent.inference.schemas import InitialPrediction, PredictionRecord
from surgical_agent.inference.writer import ArtifactWriteError
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


def _sample(video_id: str, frame_id: int) -> InferenceSample:
    return InferenceSample(
        video_id=video_id,
        target_frame_id=frame_id,
        causal_frame_ids=(frame_id,),
        media_refs=(f"{frame_id}.png",),
        source_split=DatasetSplit.TRAINING,
        alignment_version="unit-test",
    )


def _pipeline(writer: FrameResultWriter) -> CanonicalStreamingPipeline:
    return CanonicalStreamingPipeline(
        PipelineComponents(
            context_builder=CausalPerceptionContextBuilder(),
            perception=LocalSmokePerception(
                LocalSmokeModel(),
                device=torch.device("cpu"),
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
    writer = FrameResultWriter(tmp_path, run_id="reset-test")
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
    assert all(record.gate_action == "ACCEPT" for record in writer.predictions)


def test_frame_result_writer_materializes_both_hash_verified_outputs(
    tmp_path: Path,
) -> None:
    writer = FrameResultWriter(tmp_path, run_id="writer-test")
    pipeline = _pipeline(writer)
    runtime_result = pipeline.run(
        _sample("VID02", 1),
        torch.rand(1, 3, 32, 32),
        run_id="writer-test",
    )
    manifest_path = writer.finalize({"paper_metric_eligible": False})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert manifest["record_count"] == 1
    video_manifest = manifest["videos"]["VID02"]
    prediction_path = tmp_path / video_manifest["predictions_file"]
    evidence_path = tmp_path / video_manifest["evidence_file"]
    assert video_manifest["predictions_sha256"] == hashlib.sha256(
        prediction_path.read_bytes()
    ).hexdigest()
    assert video_manifest["evidence_sha256"] == hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    assert prediction_path.is_file()
    assert evidence_path.is_file()
    assert not tuple(tmp_path.glob("*.tmp"))
    with pytest.raises(ArtifactWriteError, match="after writer finalization"):
        writer.write(runtime_result.prediction, runtime_result.evidence)
