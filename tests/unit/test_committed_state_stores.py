"""Finalized-only workflow and event-memory state behavior."""

from __future__ import annotations

import pytest

from surgical_agent.research.memory.store import BoundedEventMemory
from surgical_agent.systems.pipeline import FinalizedEvent, PipelineContractError
from surgical_agent.workflow.state_store import FinalizedWorkflowStateStore

_INVALID_STATE_ACTION_PAIRS = (
    ("Candidate", "WRITE_RELIABLE"),
    ("Candidate", "WRITE_SHORT_TERM"),
    ("Candidate", "BUFFER_PENDING"),
    ("Rejected", "WRITE_RELIABLE"),
    ("Rejected", "WRITE_SHORT_TERM"),
    ("Rejected", "BUFFER_PENDING"),
    ("Pending", "WRITE_RELIABLE"),
    ("Pending", "WRITE_SHORT_TERM"),
    ("Pending", "SKIP"),
    ("Verified", "WRITE_SHORT_TERM"),
    ("Verified", "BUFFER_PENDING"),
    ("Verified", "SKIP"),
    ("Accepted", "BUFFER_PENDING"),
    ("Accepted", "SKIP"),
)


def _event(
    frame_id: int,
    *,
    video_id: str = "VID02",
    phase_id: int = 1,
    final_status: str = "Verified",
    memory_action: str = "WRITE_RELIABLE",
) -> FinalizedEvent:
    return FinalizedEvent(
        video_id=video_id,
        frame_id=frame_id,
        instrument_ids=(0,),
        verb_ids=(1,),
        target_ids=(2,),
        triplet_ids=(3,),
        phase_id=phase_id,
        backend="joint_mock",
        gate_action="VERIFY",
        verification_status="VERIFIED_KEEP",
        score_semantics="uncalibrated_rank_v1",
        initial_state="Candidate",
        final_status=final_status,
        memory_action=memory_action,
    )


def test_event_memory_is_bounded_finalized_only_and_snapshot_is_immutable() -> None:
    memory = BoundedEventMemory(max_events_per_video=2)
    memory.reset("VID02")

    before = memory.snapshot()
    memory.update(_event(10, phase_id=0))
    memory.update(_event(11, phase_id=1))
    memory.update(_event(12, phase_id=1))
    after = memory.snapshot()

    assert before["events"] == ()
    assert [event["frame_id"] for event in after["events"]] == [11, 12]
    assert after["source_max_frame_id"] == 12
    with pytest.raises(TypeError):
        after["events"][0]["phase_id"] = 6  # type: ignore[index]


def test_event_memory_rejects_cross_video_and_non_increasing_commits() -> None:
    memory = BoundedEventMemory(max_events_per_video=2)
    memory.reset("VID02")
    memory.update(_event(10))

    with pytest.raises(PipelineContractError, match="video boundary"):
        memory.update(_event(11, video_id="VID30"))
    with pytest.raises(PipelineContractError, match="increasing"):
        memory.update(_event(10))


@pytest.mark.parametrize(
    ("final_status", "memory_action"),
    _INVALID_STATE_ACTION_PAIRS,
)
def test_finalized_event_rejects_incoherent_reliability_state_action(
    final_status: str,
    memory_action: str,
) -> None:
    with pytest.raises(ValueError, match="memory_action.*final_status"):
        _event(
            10,
            final_status=final_status,
            memory_action=memory_action,
        )


@pytest.mark.parametrize(
    ("final_status", "memory_action"),
    _INVALID_STATE_ACTION_PAIRS,
)
def test_event_memory_rejects_tampered_state_action_before_advancing_watermark(
    final_status: str,
    memory_action: str,
) -> None:
    memory = BoundedEventMemory(max_events_per_video=2)
    memory.reset("VID02")
    memory.update(_event(10))
    before = memory.snapshot()
    tampered = _event(11)
    object.__setattr__(tampered, "final_status", final_status)
    object.__setattr__(tampered, "memory_action", memory_action)

    with pytest.raises(PipelineContractError, match="reliability state"):
        memory.update(tampered)

    assert memory.snapshot() == before


def test_workflow_store_exposes_only_prior_finalized_phase_history() -> None:
    workflow = FinalizedWorkflowStateStore(max_recent_phases=3)
    workflow.reset("VID02")
    initial = workflow.snapshot()
    workflow.update(_event(10, phase_id=0))
    first = workflow.snapshot()
    workflow.update(_event(11, phase_id=1))
    second = workflow.snapshot()

    assert initial["source_max_frame_id"] is None
    assert first["recent_finalized_phases"] == ("0",)
    assert first["source_max_frame_id"] == 10
    assert second["recent_finalized_phases"] == ("0", "1")
    assert second["observed_transitions"] == (("0", "1"),)
    assert second["source_max_frame_id"] == 11


def test_event_memory_routes_statuses_and_keeps_untrimmed_last_seen_watermark() -> None:
    memory = BoundedEventMemory(max_events_per_video=2)
    memory.reset("VID02")

    memory.update(_event(10, memory_action="WRITE_RELIABLE"))
    memory.update(
        _event(
            11,
            final_status="Accepted",
            memory_action="WRITE_SHORT_TERM",
        )
    )
    memory.update(
        _event(12, final_status="Pending", memory_action="BUFFER_PENDING")
    )
    memory.update(_event(13, final_status="Rejected", memory_action="SKIP"))
    memory.update(_event(14, final_status="Candidate", memory_action="SKIP"))

    snapshot = memory.snapshot()

    assert [item["frame_id"] for item in snapshot["reliable_events"]] == [10]
    assert [item["frame_id"] for item in snapshot["short_term_events"]] == [11]
    assert [item["frame_id"] for item in snapshot["pending_events"]] == [12]
    assert [item["frame_id"] for item in snapshot["events"]] == [10, 11]
    assert snapshot["source_max_frame_id"] == 14


def test_workflow_uses_only_accepted_verified_but_advances_every_frame() -> None:
    workflow = FinalizedWorkflowStateStore(max_recent_phases=3)
    workflow.reset("VID02")

    workflow.update(
        _event(
            10,
            phase_id=0,
            final_status="Accepted",
            memory_action="WRITE_SHORT_TERM",
        )
    )
    workflow.update(
        _event(11, phase_id=1, final_status="Pending", memory_action="BUFFER_PENDING")
    )
    workflow.update(
        _event(12, phase_id=2, final_status="Rejected", memory_action="SKIP")
    )
    workflow.update(_event(13, phase_id=3))

    snapshot = workflow.snapshot()

    assert snapshot["recent_finalized_phases"] == ("0", "3")
    assert snapshot["observed_transitions"] == (("0", "3"),)
    assert snapshot["source_max_frame_id"] == 13
    with pytest.raises(PipelineContractError, match="increasing"):
        workflow.update(_event(12))
