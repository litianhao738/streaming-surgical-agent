"""Behavioral tests for logically atomic prediction/evidence persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.inference import frame_result_writer as frame_writer_module
from surgical_agent.inference.frame_result_writer import (
    FrameResultWriter,
    prediction_record_sha256,
)
from surgical_agent.inference.schemas import PredictionRecord
from surgical_agent.inference.writer import ArtifactWriteError
from surgical_agent.research.signals.contracts import EvidenceProfile, EvidenceValue


def _prediction(
    *,
    run_id: str = "run",
    video_id: str = "VID02",
    frame_id: int = 12,
    score_semantics: str = "uncalibrated_rank_v1",
) -> PredictionRecord:
    return PredictionRecord(
        run_id=run_id,
        video_id=video_id,
        frame_id=frame_id,
        source_split=DatasetSplit.TESTING,
        causal_frame_ids=(frame_id - 1, frame_id),
        instrument_ids=(0,),
        verb_ids=(2,),
        target_ids=(1,),
        triplet_ids=(0,),
        phase_id=1,
        granularity="frame_multilabel",
        backend="unit-test",
        gate_action="ACCEPT",
        verification_status="SKIPPED",
        alignment_version="unit-test",
        probabilities={
            task: (0.0,) * class_count
            for task, class_count in TASK_CLASS_COUNTS.items()
        },
        trace=("perception_validated",),
        score_semantics=score_semantics,
    )


def _evidence_value(signal_name: str, frame_id: int) -> EvidenceValue:
    sources = {
        "candidate_ambiguity": "joint_rank_margin",
        "ivt_internal_conflict": "ivt_component_map_v1",
        "phase_change_anomaly": "frozen_phase_transition_graph",
        "self_reported_uncertainty": "joint_self_reported_confidence",
        "temporal_set_change": "finalized_prior_jaccard",
    }
    return EvidenceValue(0.25, True, sources[signal_name], frame_id)


def _evidence(*, video_id: str = "VID02", frame_id: int = 12) -> EvidenceProfile:
    shared = (
        "candidate_ambiguity",
        "self_reported_uncertainty",
    )
    temporal = (
        "ivt_internal_conflict",
        "temporal_set_change",
    )
    task_values = {
        task: {
            signal_name: _evidence_value(signal_name, frame_id)
            for signal_name in (
                (*shared, *temporal)
                if task in {"instrument", "verb", "target", "ivt"}
                else (*shared, "phase_change_anomaly")
            )
        }
        for task in TASK_CLASS_COUNTS
    }
    return EvidenceProfile(
        video_id=video_id,
        frame_id=frame_id,
        task_values=task_values,
        global_values={
            "ivt_internal_conflict": _evidence_value(
                "ivt_internal_conflict", frame_id
            )
        },
    )


def _json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writer_persists_matching_per_video_records_and_hash(tmp_path: Path) -> None:
    writer = FrameResultWriter(tmp_path, run_id="run")

    evidence_record = writer.write(_prediction(), _evidence())

    assert evidence_record.prediction_sha256 == prediction_record_sha256(
        writer.predictions[0]
    )
    assert _json_lines(tmp_path / "predictions/VID02.jsonl")[0]["frame_id"] == 12
    persisted_evidence = _json_lines(tmp_path / "evidence/VID02.jsonl")[0]
    assert persisted_evidence["frame_id"] == 12
    assert persisted_evidence["prediction_sha256"] == evidence_record.prediction_sha256


def test_prediction_hash_includes_score_semantics() -> None:
    ranked = _prediction(score_semantics="uncalibrated_rank_v1")
    probability = replace(ranked, score_semantics="probability_v1")

    assert prediction_record_sha256(ranked) != prediction_record_sha256(probability)


@pytest.mark.parametrize(
    ("prediction", "evidence", "message"),
    [
        (_prediction(video_id="VID30"), _evidence(), "identity"),
        (_prediction(frame_id=13), _evidence(), "identity"),
        (_prediction(run_id="other"), _evidence(), "run_id"),
    ],
)
def test_writer_rejects_mismatched_identity_or_run(
    tmp_path: Path,
    prediction: PredictionRecord,
    evidence: EvidenceProfile,
    message: str,
) -> None:
    writer = FrameResultWriter(tmp_path, run_id="run")

    with pytest.raises(ArtifactWriteError, match=message):
        writer.write(prediction, evidence)

    assert not (tmp_path / "run_status.json").exists()
    assert not (tmp_path / "predictions").exists()
    assert not (tmp_path / "evidence").exists()


def test_duplicate_frame_is_rejected_without_changing_durable_records(
    tmp_path: Path,
) -> None:
    writer = FrameResultWriter(tmp_path, run_id="run")
    writer.write(_prediction(), _evidence())
    prediction_bytes = (tmp_path / "predictions/VID02.jsonl").read_bytes()
    evidence_bytes = (tmp_path / "evidence/VID02.jsonl").read_bytes()

    with pytest.raises(ArtifactWriteError, match="Duplicate"):
        writer.write(_prediction(), _evidence())

    assert (tmp_path / "predictions/VID02.jsonl").read_bytes() == prediction_bytes
    assert (tmp_path / "evidence/VID02.jsonl").read_bytes() == evidence_bytes
    assert len(writer.predictions) == 1


@pytest.mark.parametrize(
    ("signal_name", "source"),
    [
        ("raw_response", "joint_rank_margin"),
        ("candidate_ambiguity", "api_key=do-not-persist"),
    ],
)
def test_evidence_profile_rejects_non_allowlisted_names_and_free_form_sources(
    signal_name: str,
    source: str,
) -> None:
    profile = _evidence()
    task_values = {
        task: dict(values) for task, values in profile.task_values.items()
    }
    task_values["instrument"][signal_name] = EvidenceValue(
        0.5, True, source, profile.frame_id
    )

    with pytest.raises(ValueError, match="allowlisted"):
        EvidenceProfile(
            video_id=profile.video_id,
            frame_id=profile.frame_id,
            task_values=task_values,
            global_values=profile.global_values,
        )


def test_evidence_write_failure_keeps_run_incomplete_and_refuses_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = FrameResultWriter(tmp_path, run_id="run")
    real_atomic_write = frame_writer_module.atomic_write_text

    def fail_evidence_write(path: Path, content: str) -> None:
        if path.parent.name == "evidence":
            raise ArtifactWriteError("injected evidence write failure")
        real_atomic_write(path, content)

    monkeypatch.setattr(frame_writer_module, "atomic_write_text", fail_evidence_write)

    with pytest.raises(ArtifactWriteError, match="injected evidence write failure"):
        writer.write(_prediction(), _evidence())

    status = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    assert status == {
        "current_sample": {"frame_id": 12, "video_id": "VID02"},
        "run_id": "run",
        "schema_version": "frame_result_run_status_v1",
        "status": "INCOMPLETE",
    }
    assert (tmp_path / "predictions/VID02.jsonl").is_file()
    assert not (tmp_path / "evidence/VID02.jsonl").exists()
    assert writer.predictions == ()
    assert writer.evidence_records == ()
    with pytest.raises(ArtifactWriteError, match="incomplete"):
        writer.finalize({"paper_metric_eligible": False})


def test_finalize_rejects_an_empty_run(tmp_path: Path) -> None:
    writer = FrameResultWriter(tmp_path, run_id="run")

    with pytest.raises(ArtifactWriteError, match="empty"):
        writer.finalize({})


@pytest.mark.parametrize("tamper", ["sample", "hash"])
def test_finalize_verifies_persisted_sample_sets_and_prediction_hashes(
    tmp_path: Path, tamper: str
) -> None:
    writer = FrameResultWriter(tmp_path, run_id="run")
    writer.write(_prediction(), _evidence())
    evidence_path = tmp_path / "evidence/VID02.jsonl"
    persisted = _json_lines(evidence_path)[0]
    if tamper == "sample":
        persisted["frame_id"] = 13
    else:
        persisted["prediction_sha256"] = "0" * 64
    evidence_path.write_text(json.dumps(persisted) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactWriteError, match="sample sets|prediction hash"):
        writer.finalize({})

    assert json.loads((tmp_path / "run_status.json").read_text())["status"] == (
        "INCOMPLETE"
    )


def test_finalize_writes_per_video_hashes_and_complete_status_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = FrameResultWriter(tmp_path, run_id="run")
    writer.write(_prediction(frame_id=12), _evidence(frame_id=12))
    writer.write(
        _prediction(video_id="VID30", frame_id=4),
        _evidence(video_id="VID30", frame_id=4),
    )
    real_atomic_write = frame_writer_module.atomic_write_text
    finalize_writes: list[Path] = []

    def record_atomic_write(path: Path, content: str) -> None:
        finalize_writes.append(path)
        real_atomic_write(path, content)

    monkeypatch.setattr(frame_writer_module, "atomic_write_text", record_atomic_write)

    manifest_path = writer.finalize({"paper_metric_eligible": False})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETE"
    assert manifest["record_count"] == 2
    assert set(manifest["videos"]) == {"VID02", "VID30"}
    for video_id in manifest["videos"]:
        video_manifest = manifest["videos"][video_id]
        prediction_path = tmp_path / video_manifest["predictions_file"]
        evidence_path = tmp_path / video_manifest["evidence_file"]
        assert video_manifest["predictions_sha256"] == hashlib.sha256(
            prediction_path.read_bytes()
        ).hexdigest()
        assert video_manifest["evidence_sha256"] == hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest()
    assert [path.name for path in finalize_writes] == [
        "manifest.json",
        "run_status.json",
    ]
    assert json.loads((tmp_path / "run_status.json").read_text()) == {
        "manifest_file": "manifest.json",
        "run_id": "run",
        "schema_version": "frame_result_run_status_v1",
        "status": "COMPLETE",
    }
    assert not tuple(tmp_path.rglob("*.tmp"))
    with pytest.raises(ArtifactWriteError, match="after writer finalization"):
        writer.write(_prediction(frame_id=13), _evidence(frame_id=13))
