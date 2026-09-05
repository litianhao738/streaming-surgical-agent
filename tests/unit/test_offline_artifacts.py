"""Strict loading of completed frame-result runs for offline evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import (
    DatasetSplit,
    FrameSupervisionTarget,
    FrameTaskMask,
)
from surgical_agent.evaluation.frame_ground_truth import EvaluationData
from surgical_agent.evaluation.offline_artifacts import (
    OfflineEvaluationError,
    identity_sha256,
    load_completed_run,
)
from surgical_agent.evaluation.offline_frame import evaluate_and_write
from surgical_agent.inference.frame_result_writer import FrameResultWriter
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.research.signals.contracts import EvidenceProfile, EvidenceValue


def _prediction(
    *, frame_id: int = 1, score_semantics: str = "uncalibrated_rank_v1"
) -> PredictionRecord:
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
        backend="mock-joint-perception-v1",
        gate_action="ACCEPT",
        verification_status="NOT_REQUESTED",
        alignment_version="ct20_exact_png_stem_v1",
        probabilities={
            task: tuple(
                float(index == {"instrument": 0, "verb": 1, "target": 2, "ivt": 3, "phase": 1}[task])
                if score_semantics == "hard_label_v1"
                else 0.5
                for index in range(count)
            )
            for task, count in TASK_CLASS_COUNTS.items()
        },
        trace=("perception_validated",),
        score_semantics=score_semantics,
    )


def _value(name: str, frame_id: int) -> EvidenceValue:
    sources = {
        "candidate_ambiguity": "joint_rank_margin",
        "ivt_internal_conflict": "ivt_component_map_v1",
        "phase_change_anomaly": "frozen_phase_transition_graph",
        "self_reported_uncertainty": "joint_self_reported_confidence",
        "temporal_set_change": "finalized_prior_jaccard",
    }
    return EvidenceValue(0.25, True, sources[name], frame_id)


def _evidence(*, frame_id: int = 1) -> EvidenceProfile:
    task_values = {}
    for task in TASK_CLASS_COUNTS:
        names = (
            (
                "candidate_ambiguity",
                "ivt_internal_conflict",
                "self_reported_uncertainty",
                "temporal_set_change",
            )
            if task != "phase"
            else (
                "candidate_ambiguity",
                "phase_change_anomaly",
                "self_reported_uncertainty",
            )
        )
        task_values[task] = {name: _value(name, frame_id) for name in names}
    return EvidenceProfile(
        video_id="VID110",
        frame_id=frame_id,
        task_values=task_values,
        global_values={
            "ivt_internal_conflict": _value("ivt_internal_conflict", frame_id)
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_completed_run(
    root: Path,
    *,
    frame_ids: tuple[int, ...] = (1,),
    score_semantics: str = "uncalibrated_rank_v1",
) -> Path:
    run_dir = root / "eval-unit"
    writer = FrameResultWriter(run_dir, run_id="eval-unit")
    writer.begin()
    for frame_id in frame_ids:
        evidence = _evidence(frame_id=frame_id)
        if score_semantics == "hard_label_v1":
            evidence = replace(
                evidence,
                task_values={
                    task: {
                        name: EvidenceValue(None, False, value.source, frame_id)
                        if name in {"candidate_ambiguity", "self_reported_uncertainty"}
                        else value
                        for name, value in values.items()
                    }
                    for task, values in evidence.task_values.items()
                },
            )
        writer.write(
            _prediction(frame_id=frame_id, score_semantics=score_semantics), evidence
        )
    writer.finalize({"paper_metric_eligible": False})
    _write_json(
        run_dir / "dataset_rollout_artifact.json",
        {
            "schema_version": "cholectrack20_api_rollout_v1",
            "status": "MOCK_COMPLETE",
            "run_id": "eval-unit",
            "mode": "engineering",
            "split": None,
            "video_ids": ["VID110"],
            "expected_frame_counts": {"VID110": len(frame_ids)},
            "completed_frame_counts": {"VID110": len(frame_ids)},
            "provider": "mock",
            "model_requested": "mock-joint-perception-v1",
            "models_returned": ["mock-joint-perception-v1"],
            "prompt_version": "joint_perception_frame_v1",
            "response_schema_version": "joint_perception_frame_v1",
            "repair_manifest_sha256": "a" * 64,
            "alignment_versions": ["ct20_exact_png_stem_v1"],
            "usage": {},
            "cache_entry_count": 1,
            "track20_image_uploaded": False,
            "paper_metric_eligible": False,
            "manifest_file": "manifest.json",
        },
    )
    return run_dir


def _rehash_prediction_file(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_path = run_dir / manifest["videos"]["VID110"]["predictions_file"]
    manifest["videos"]["VID110"]["predictions_sha256"] = hashlib.sha256(
        prediction_path.read_bytes()
    ).hexdigest()
    _write_json(manifest_path, manifest)


def test_load_completed_run_reconstructs_verified_predictions(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path)

    loaded = load_completed_run(run_dir)

    assert loaded.run_id == "eval-unit"
    assert loaded.mode == "engineering"
    assert loaded.effective_split is DatasetSplit.VALIDATION
    assert loaded.predictions[0].source_split is DatasetSplit.VALIDATION
    assert loaded.prediction_identities == (("VID110", 1),)
    assert loaded.frame_counts == {"VID110": 1}
    assert loaded.paper_metric_eligible is False
    assert identity_sha256(loaded.prediction_identities) == hashlib.sha256(
        b'[["VID110",1]]'
    ).hexdigest()


def test_hard_label_artifacts_remain_label_only_through_offline_evaluation(tmp_path):
    run_dir = _write_completed_run(tmp_path, score_semantics="hard_label_v1")
    loaded = load_completed_run(run_dir)
    assert loaded.predictions[0].score_semantics == "hard_label_v1"
    target = FrameSupervisionTarget(
        video_id="VID110",
        frame_id=1,
        instrument_ids=(0,),
        verb_ids=(1,),
        target_ids=(2,),
        triplet_ids=(3,),
        phase_id=1,
        mask=FrameTaskMask(True, True, True, True, True),
        source_granularity="frame_multilabel",
        source="unit-test",
    )
    data = EvaluationData(
        split=DatasetSplit.VALIDATION,
        runtime_identities=loaded.prediction_identities,
        targets={("VID110", 1): target},
        sources={},
        provenance_by_video={"VID110": "unit-test"},
        repair_manifest_sha256=loaded.repair_manifest_sha256,
    )

    result = evaluate_and_write(
        loaded, data, tmp_path / "evaluation", test_gt_authorized=False
    )

    report = json.loads(result.report_path.read_text())
    manifest = json.loads(result.manifest_path.read_text())
    metrics = report["metrics"]
    assert metrics["schema_version"] == "frame_recognition_metrics_v2"
    assert manifest["metric_schema_version"] == "frame_recognition_metrics_v2"
    assert manifest["score_semantics"] == ["hard_label_v1"]
    for task in ("instrument", "verb", "target", "ivt"):
        assert metrics["tasks"][task]["video_wise_map"] is None
        assert metrics["tasks"][task]["status"] == "unsupported"
        assert metrics["tasks"][task]["reason"] == "hard_label_v1_does_not_provide_ranking_scores"
    for task in TASK_CLASS_COUNTS:
        assert metrics["label_metrics"][task]["micro_f1"] == 1.0
        assert metrics["label_metrics"][task]["exact_set_accuracy"] == 1.0
    assert metrics["phase"]["video_wise_accuracy"] == 1.0


def test_load_completed_run_accepts_current_runtime_rollout_fields(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path)
    rollout_path = run_dir / "dataset_rollout_artifact.json"
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    rollout.update(
        {
            "provider_routing_profile": None,
            "causal_window": {},
            "pipeline_profile": "single_pass",
            "backbone_policy": "shared",
            "initial_model_requested": "mock-joint-perception-v1",
            "verification_model_requested": "mock-joint-perception-v1",
            "main_profile_backbone_match": True,
            "context_profile": "frames_only",
            "event_memory_enabled": False,
            "context_experiment_sha256": None,
            "phase_transition_graph": None,
            "predicted_track_artifact_sha256": None,
            "predicted_track_provenance": None,
            "evidence_threshold": None,
            "gate_artifact_sha256": None,
            "verification_summary": {},
            "final_status_counts": {"Accepted": 1},
            "memory_action_counts": {"SKIP": 1},
            "report_manifest_path": "event_report_manifest.json",
            "causal_window_audit_path": "causal_window_audit.jsonl",
            "report_count": 1,
            "report_mode": "template_report",
            "telemetry_summary": {},
        }
    )
    _write_json(rollout_path, rollout)

    assert load_completed_run(run_dir).run_id == "eval-unit"


def test_load_completed_run_rejects_boolean_frame_id(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path)
    prediction_path = run_dir / "predictions/VID110.jsonl"
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    payload["frame_id"] = True
    prediction_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _rehash_prediction_file(run_dir)

    with pytest.raises(OfflineEvaluationError, match="frame_id"):
        load_completed_run(run_dir)


def test_load_completed_run_rejects_duplicate_json_key(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path)
    status_path = run_dir / "run_status.json"
    text = status_path.read_text(encoding="utf-8")
    status_path.write_text(
        text.replace(
            '"status": "COMPLETE"',
            '"status": "COMPLETE",\n  "status": "COMPLETE"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(OfflineEvaluationError, match="duplicate JSON key"):
        load_completed_run(run_dir)


def test_load_completed_run_rejects_tampered_prediction_file(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path)
    prediction_path = run_dir / "predictions/VID110.jsonl"
    prediction_path.write_text(
        prediction_path.read_text(encoding="utf-8").replace('"phase_id": 1', '"phase_id": 2'),
        encoding="utf-8",
    )

    with pytest.raises(OfflineEvaluationError, match="SHA-256"):
        load_completed_run(run_dir)


def test_load_completed_run_rejects_eligibility_disagreement(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path)
    rollout_path = run_dir / "dataset_rollout_artifact.json"
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    rollout["paper_metric_eligible"] = True
    _write_json(rollout_path, rollout)

    with pytest.raises(OfflineEvaluationError, match="paper_metric_eligible"):
        load_completed_run(run_dir)


def test_load_completed_run_rejects_mixed_score_semantics(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path, frame_ids=(1, 2))
    prediction_path = run_dir / "predictions/VID110.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["score_semantics"] = "probability_v1"
    prediction_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rehash_prediction_file(run_dir)

    with pytest.raises(OfflineEvaluationError, match="score semantics"):
        load_completed_run(run_dir)


def test_paper_validation_requires_canonical_two_video_membership(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path)
    rollout_path = run_dir / "dataset_rollout_artifact.json"
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    rollout["mode"] = "paper"
    rollout["split"] = "validation"
    _write_json(rollout_path, rollout)

    with pytest.raises(OfflineEvaluationError, match="canonical split membership"):
        load_completed_run(run_dir)


def test_completed_run_requires_positive_frame_counts(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path)
    rollout_path = run_dir / "dataset_rollout_artifact.json"
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    rollout["expected_frame_counts"]["VID110"] = 0
    rollout["completed_frame_counts"]["VID110"] = 0
    _write_json(rollout_path, rollout)

    with pytest.raises(OfflineEvaluationError, match="positive integer"):
        load_completed_run(run_dir)
