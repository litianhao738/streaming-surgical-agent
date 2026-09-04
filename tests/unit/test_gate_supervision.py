from __future__ import annotations

from types import SimpleNamespace

from surgical_agent.data.masks import canonical_label_mask
from surgical_agent.data.schemas import (
    DatasetSplit,
    EvaluationTarget,
    FrameSupervisionTarget,
    FrameTaskMask,
)
from surgical_agent.research.gate.supervision import (
    deterministic_timeline_sample,
    gate_training_target,
)


def test_timeline_sampler_spreads_across_complete_sequence() -> None:
    assert deterministic_timeline_sample(tuple(range(10)), 3) == (0, 4, 9)
    assert deterministic_timeline_sample(tuple(range(10)), 1) == (4,)


def test_gate_target_prefers_conservative_instance_supervision() -> None:
    instance = SimpleNamespace(
        instrument_id=2,
        verb_id=3,
        target_id=4,
        triplet_id=5,
        phase_id=6,
        mask=canonical_label_mask(
            instrument_id=2,
            verb_id=3,
            target_id=4,
            triplet_id=5,
            phase_id=6,
        ),
    )
    phase_only = FrameSupervisionTarget(
        video_id="VID02",
        frame_id=1,
        instrument_ids=(),
        verb_ids=(),
        target_ids=(),
        triplet_ids=(),
        phase_id=6,
        mask=FrameTaskMask(False, False, False, False, True),
        source_granularity="phase_only",
        source="annotation.json",
    )
    sample = SimpleNamespace(
        inference=SimpleNamespace(source_split=DatasetSplit.TRAINING),
        evaluation=EvaluationTarget("VID02", 1, (instance,)),
        frame_supervision=phase_only,
        provenance=SimpleNamespace(annotation_source="annotation.json"),
    )

    target = gate_training_target(sample)

    assert target.instrument_ids == (2,)
    assert target.verb_ids == (3,)
    assert target.target_ids == (4,)
    assert target.triplet_ids == (5,)
    assert all(
        getattr(target.mask, task)
        for task in ("instrument", "verb", "target", "ivt", "phase")
    )
