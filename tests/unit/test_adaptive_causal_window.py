"""Adaptive six-frame causal-window behavior and auditability."""

from __future__ import annotations

import torch

from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder


def _track(frame_id: int, x: float) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "tracks": (
            {
                "track_id": "tool-1",
                "instrument_id": 0,
                "bbox_tlwh": (x, 0.2, 0.2, 0.2),
                "score": 0.9,
                "age": frame_id,
            },
        ),
    }


def test_six_frames_feed_evidence_but_only_three_images_are_encoded() -> None:
    frame_ids = (10, 11, 12, 13, 14, 15)
    sample = InferenceSample(
        video_id="VID30",
        target_frame_id=15,
        causal_frame_ids=frame_ids,
        media_refs=tuple(f"synthetic:{frame_id}" for frame_id in frame_ids),
        source_split=DatasetSplit.VALIDATION,
        alignment_version="unit-test",
    )
    frames = torch.zeros((6, 3, 32, 32))
    frames[2:] = 0.6
    frames[4:] = 1.0
    context = CausalPerceptionContextBuilder(max_frames=6, max_images=3).build(
        sample,
        frames,
        workflow_snapshot={},
        memory_snapshot={},
        prior_finalized_prediction=None,
        track_snapshot={
            "status": "AVAILABLE",
            "source_max_frame_id": 15,
            "frames": tuple(
                _track(frame_id, 0.05 * index)
                for index, frame_id in enumerate(frame_ids)
            ),
        },
    )

    assert context.frames.shape[0] == 6
    assert len(context.images) == 3
    assert len(context.selected_image_frame_ids) == 3
    assert context.selected_image_frame_ids[-1] == 15
    assert context.selected_image_frame_ids == tuple(
        sorted(context.selected_image_frame_ids)
    )
    assert context.temporal_evidence["candidate_frame_ids"] == frame_ids
    assert context.temporal_evidence["selected_image_frame_ids"] == (
        context.selected_image_frame_ids
    )
    assert context.temporal_evidence["summary"]["window_frame_count"] == 6
    assert context.temporal_evidence["summary"]["uploaded_image_count"] == 3
    assert context.temporal_evidence["summary"]["tracker_available"] is True


def test_target_is_always_uploaded_when_history_has_larger_change() -> None:
    frame_ids = (1, 2, 3, 4, 5, 6)
    frames = torch.zeros((6, 3, 16, 16))
    frames[1] = 1.0
    frames[2:] = 0.0
    sample = InferenceSample(
        video_id="VID30",
        target_frame_id=6,
        causal_frame_ids=frame_ids,
        media_refs=tuple(f"synthetic:{frame_id}" for frame_id in frame_ids),
        source_split=DatasetSplit.VALIDATION,
        alignment_version="unit-test",
    )

    context = CausalPerceptionContextBuilder(max_frames=6, max_images=2).build(
        sample,
        frames,
        workflow_snapshot={},
        memory_snapshot={},
        prior_finalized_prediction=None,
    )

    assert context.selected_image_frame_ids[-1] == 6
    assert len(context.images) == 2
