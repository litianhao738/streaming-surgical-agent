"""Behavioral tests for event-boundary reporting artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from surgical_agent.inference.writer import ArtifactWriteError
from surgical_agent.research.reporting import (
    EventReportManager,
    EventReportWriter,
    ReportRecord,
    TemplateReportGenerator,
)
from surgical_agent.systems.pipeline import FinalizedEvent


def _event(
    frame_id: int,
    *,
    video_id: str = "VID01",
    phase_id: int = 1,
    final_status: str = "Verified",
) -> FinalizedEvent:
    actions = {
        "Candidate": "SKIP",
        "Accepted": "WRITE_SHORT_TERM",
        "Verified": "WRITE_RELIABLE",
        "Pending": "BUFFER_PENDING",
        "Rejected": "SKIP",
    }
    return FinalizedEvent(
        video_id=video_id,
        frame_id=frame_id,
        instrument_ids=(2,),
        verb_ids=(3,),
        target_ids=(4,),
        triplet_ids=(5,),
        phase_id=phase_id,
        backend="mock",
        gate_action="VERIFY",
        verification_status="VERIFIED_KEEP",
        score_semantics="topk_only",
        final_status=final_status,
        memory_action=actions[final_status],
    )


def test_template_report_uses_stable_sha_id_and_separate_status_clauses() -> None:
    manager = EventReportManager(window_size=10)
    for event in (
        _event(1, final_status="Verified"),
        _event(2, final_status="Accepted"),
        _event(3, final_status="Pending"),
        _event(4, final_status="Candidate"),
        _event(5, final_status="Rejected"),
    ):
        assert manager.observe(event) == ()

    (report,) = manager.finalize()
    repeated = EventReportManager(window_size=10)
    for event in (
        _event(1, final_status="Verified"),
        _event(2, final_status="Accepted"),
        _event(3, final_status="Pending"),
        _event(4, final_status="Candidate"),
        _event(5, final_status="Rejected"),
    ):
        repeated.observe(event)
    (same_report,) = repeated.finalize()

    assert len(report.report_id) == 64
    assert int(report.report_id, 16) >= 0
    assert report.report_id == same_report.report_id
    assert report.included_final_statuses == ("Verified", "Accepted", "Pending")
    sentences = [part.strip() for part in report.text.split(".") if part.strip()]
    assert any(sentence.startswith("Confirmed") for sentence in sentences)
    assert any(
        sentence.startswith("High-confidence observation") for sentence in sentences
    )
    pending = [sentence for sentence in sentences if sentence.startswith("Unresolved")]
    assert len(pending) == 1
    assert any(word in pending[0].lower() for word in ("may", "possible", "unresolved"))
    assert "Candidate" not in report.text
    assert "Rejected" not in report.text


def test_phase_change_flushes_old_video_segment_with_both_reasons() -> None:
    manager = EventReportManager(window_size=30)
    manager.observe(_event(1, phase_id=1))
    manager.observe(_event(2, phase_id=1))

    (report,) = manager.observe(_event(3, phase_id=2))

    assert (report.start_frame_id, report.end_frame_id) == (1, 2)
    assert report.phase_ids == (1,)
    assert report.trigger_reasons == ("phase_change", "event_segment_end")
    assert manager.finalize()[0].start_frame_id == 3


def test_video_switch_is_only_video_end_and_enforces_per_video_frame_order() -> None:
    manager = EventReportManager(window_size=30)
    manager.observe(_event(9, video_id="VID01", phase_id=4))

    (report,) = manager.observe(_event(1, video_id="VID02", phase_id=1))

    assert report.video_id == "VID01"
    assert report.trigger_reasons == ("video_end",)
    with pytest.raises(ValueError, match="strictly increasing"):
        manager.observe(_event(1, video_id="VID02", phase_id=1))


def test_fixed_window_counts_excluded_events_without_emitting_empty_report() -> None:
    manager = EventReportManager(window_size=2)

    assert manager.observe(_event(1, final_status="Rejected")) == ()
    assert manager.observe(_event(2, final_status="Candidate")) == ()
    assert manager.observe(_event(3, final_status="Verified")) == ()
    (report,) = manager.observe(_event(4, final_status="Rejected"))

    assert report.trigger_reasons == ("fixed_window",)
    assert (report.start_frame_id, report.end_frame_id) == (3, 3)


class _FakeLlmGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def generate(
        self,
        events: tuple[FinalizedEvent, ...],
        *,
        trigger_reasons: tuple[str, ...],
    ) -> str:
        del trigger_reasons
        self.calls.append(tuple(event.frame_id for event in events))
        return f"Confirmed facts: LLM segment {len(self.calls)}."


def test_injected_llm_generator_runs_once_per_event_segment_not_per_frame() -> None:
    fake = _FakeLlmGenerator()
    manager = EventReportManager(
        window_size=30,
        mode="llm_report",
        generator=fake,
    )
    manager.observe(_event(1, phase_id=1))
    manager.observe(_event(2, phase_id=1))
    manager.observe(_event(3, phase_id=2))
    manager.finalize()

    assert fake.calls == [(1, 2), (3,)]
    assert [report.text for report in manager.records] == [
        "Confirmed facts: LLM segment 1.",
        "Confirmed facts: LLM segment 2.",
    ]


def test_template_generator_is_local_and_deterministic() -> None:
    generator = TemplateReportGenerator()
    events = (_event(1), _event(2))

    first = generator.generate(events, trigger_reasons=("video_end",))
    second = generator.generate(events, trigger_reasons=("video_end",))

    assert first == second


def test_report_record_rejects_ineligible_statuses_and_certain_pending_text() -> None:
    common = {
        "video_id": "VID01",
        "start_frame_id": 1,
        "end_frame_id": 1,
        "phase_ids": (1,),
        "trigger_reasons": ("video_end",),
        "mode": "template_report",
    }
    with pytest.raises(ValueError, match="eligible final statuses"):
        ReportRecord.create(
            **common,
            included_final_statuses=("Rejected",),
            text="Rejected candidate.",
        )
    with pytest.raises(ValueError, match="uncertainty language"):
        ReportRecord.create(
            **common,
            included_final_statuses=("Pending",),
            text="Confirmed fact.",
        )


@pytest.mark.parametrize(
    "included_statuses",
    [("Pending",), ("Verified", "Pending")],
)
def test_report_record_rejects_uncertainty_token_appended_to_confirmed_claim(
    included_statuses: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="status-aware clauses"):
        ReportRecord.create(
            video_id="VID01",
            start_frame_id=1,
            end_frame_id=1,
            phase_ids=(1,),
            trigger_reasons=("video_end",),
            included_final_statuses=included_statuses,
            mode="llm_report",
            text=(
                "Confirmed fact: hemostasis is complete; "
                "a possible issue remains."
            ),
        )


def test_writer_atomically_rewrites_jsonl_and_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    manager = EventReportManager(window_size=1)
    (report,) = manager.observe(_event(1))
    writer = EventReportWriter(tmp_path)

    writer.write(report)
    with pytest.raises(ArtifactWriteError, match="Duplicate report ID"):
        writer.write(report)
    manifest_path = writer.finalize()

    content = (tmp_path / "event_reports.jsonl").read_bytes()
    persisted = json.loads(content.decode("utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["report_id"] == report.report_id
    assert manifest["record_count"] == 1
    assert manifest["event_reports_sha256"] == hashlib.sha256(content).hexdigest()


def test_writer_finalizes_complete_empty_report_set(tmp_path: Path) -> None:
    (tmp_path / "event_reports.jsonl").write_text("stale\n", encoding="utf-8")
    writer = EventReportWriter(tmp_path)

    manifest_path = writer.finalize()

    report_path = tmp_path / "event_reports.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report_path.read_bytes() == b""
    assert manifest == {
        "schema_version": "event_report_manifest_v1",
        "status": "COMPLETE",
        "record_count": 0,
        "event_reports_file": "event_reports.jsonl",
        "event_reports_sha256": hashlib.sha256(b"").hexdigest(),
    }
