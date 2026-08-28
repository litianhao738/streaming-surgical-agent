"""Offline prediction/GT alignment and report persistence tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import (
    DatasetSplit,
    FrameSupervisionTarget,
    FrameTaskMask,
)
from surgical_agent.evaluation.frame_ground_truth import (
    EvaluationData,
    GroundTruthSource,
)
from surgical_agent.evaluation.offline_artifacts import (
    ArtifactFile,
    OfflineEvaluationError,
)
from surgical_agent.evaluation.offline_frame import (
    align_scored_pairs,
    evaluate_and_write,
    resolve_evaluation_output,
)
from surgical_agent.inference.schemas import PredictionRecord


def _prediction(frame_id: int) -> PredictionRecord:
    return PredictionRecord(
        run_id="eval-unit",
        video_id="VID110",
        frame_id=frame_id,
        source_split=DatasetSplit.VALIDATION,
        causal_frame_ids=(frame_id,),
        instrument_ids=(0,),
        verb_ids=(1,),
        target_ids=(2,),
        triplet_ids=(3,),
        phase_id=1,
        granularity="frame_multilabel",
        backend="mock",
        gate_action="ACCEPT",
        verification_status="NOT_REQUESTED",
        alignment_version="ct20_exact_png_stem_v1",
        probabilities={
            task: tuple(0.5 for _ in range(count))
            for task, count in TASK_CLASS_COUNTS.items()
        },
        score_semantics="uncalibrated_rank_v1",
    )


def _target(frame_id: int) -> FrameSupervisionTarget:
    return FrameSupervisionTarget(
        video_id="VID110",
        frame_id=frame_id,
        instrument_ids=(0,),
        verb_ids=(1,),
        target_ids=(2,),
        triplet_ids=(3,),
        phase_id=1,
        mask=FrameTaskMask(True, True, True, True, True),
        source_granularity="frame_multilabel",
        source="Validation/VID110/vid110.json",
    )


def _run(tmp_path: Path) -> SimpleNamespace:
    predictions = (_prediction(1), _prediction(2))
    artifact = ArtifactFile("predictions/VID110.jsonl", "a" * 64)
    evidence = ArtifactFile("evidence/VID110.jsonl", "b" * 64)
    return SimpleNamespace(
        run_dir=tmp_path / "run",
        run_id="eval-unit",
        mode="engineering",
        effective_split=DatasetSplit.VALIDATION,
        video_ids=("VID110",),
        frame_counts={"VID110": 2},
        predictions=predictions,
        prediction_identities=(("VID110", 1), ("VID110", 2)),
        prediction_files={"VID110": artifact},
        evidence_files={"VID110": evidence},
        input_hashes={
            "run_status": "c" * 64,
            "frame_manifest": "d" * 64,
            "rollout_artifact": "e" * 64,
        },
        provider="mock",
        model_requested="mock",
        models_returned=("mock",),
        prompt_version="joint_perception_frame_v1",
        response_schema_version="joint_perception_frame_v1",
        alignment_versions=("ct20_exact_png_stem_v1",),
        paper_metric_eligible=False,
    )


def _data() -> EvaluationData:
    return EvaluationData(
        split=DatasetSplit.VALIDATION,
        runtime_identities=(("VID110", 1), ("VID110", 2)),
        targets=MappingProxyType({("VID110", 1): _target(1)}),
        sources=MappingProxyType(
            {
                "VID110": GroundTruthSource(
                    "Validation/VID110/vid110.json", "f" * 64, "official_raw"
                )
            }
        ),
        provenance_by_video=MappingProxyType({"VID110": "official_raw"}),
        repair_manifest_sha256="0" * 64,
    )


def test_evaluator_scores_gt_and_reports_media_only_prediction(tmp_path: Path) -> None:
    destination = tmp_path / "evaluation"

    result = evaluate_and_write(
        _run(tmp_path), _data(), destination, test_gt_authorized=False
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    status = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert report["scored_frame_counts"] == {"VID110": 1}
    assert report["unscored_prediction_counts"] == {"VID110": 1}
    assert report["metrics"]["schema_version"] == "frame_recognition_metrics_v1"
    assert report["paper_metric_eligible"] is False
    assert status["status"] == "COMPLETE"


def test_alignment_rejects_missing_gt_bearing_prediction(tmp_path: Path) -> None:
    data = _data()
    missing = EvaluationData(
        split=data.split,
        runtime_identities=(("VID110", 1),),
        targets=MappingProxyType(
            {("VID110", 1): _target(1), ("VID110", 2): _target(2)}
        ),
        sources=data.sources,
        provenance_by_video=data.provenance_by_video,
        repair_manifest_sha256=data.repair_manifest_sha256,
    )
    run = _run(tmp_path)
    run.predictions = (_prediction(1),)
    run.prediction_identities = (("VID110", 1),)

    with pytest.raises(OfflineEvaluationError, match="GT-bearing"):
        align_scored_pairs(run, missing)


def test_output_paths_must_not_overlap_inputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    dataset = tmp_path / "dataset"

    with pytest.raises(OfflineEvaluationError, match="overlap"):
        resolve_evaluation_output(run_dir, dataset, run_dir / "evaluation")
