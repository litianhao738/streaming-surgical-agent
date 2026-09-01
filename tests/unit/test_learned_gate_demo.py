from __future__ import annotations

from pathlib import Path

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import FrameSupervisionTarget, FrameTaskMask
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.research.gate.demo import (
    DemoObservation,
    evaluate_demo_group,
    load_demo_observations,
    oracle_benefit_labels,
    write_demo_observations,
)


def _prediction(class_id: int) -> InitialPrediction:
    return InitialPrediction(
        instrument_ids=(class_id,),
        verb_ids=(class_id,),
        target_ids=(class_id,),
        triplet_ids=(class_id,),
        phase_id=class_id,
        probabilities={
            task: tuple(0.0 for _ in range(count))
            for task, count in TASK_CLASS_COUNTS.items()
        },
    )


def _target(frame_id: int) -> FrameSupervisionTarget:
    return FrameSupervisionTarget(
        video_id="VID30",
        frame_id=frame_id,
        instrument_ids=(0,),
        verb_ids=(0,),
        target_ids=(0,),
        triplet_ids=(0,),
        phase_id=0,
        mask=FrameTaskMask(True, True, True, True, True),
        source_granularity="frame_multilabel",
        source="unit",
    )


def test_demo_observations_round_trip_and_evaluate(tmp_path: Path) -> None:
    values = (
        DemoObservation(
            video_id="VID30",
            frame_id=1,
            initial_prediction=_prediction(1),
            final_prediction=_prediction(0),
            gate_action="VERIFY",
            verification_status="VERIFIED_REPAIR",
            flagged_fields=("instrument", "verb", "target", "ivt", "phase"),
            benefit_probability=0.9,
        ),
        DemoObservation(
            video_id="VID30",
            frame_id=2,
            initial_prediction=_prediction(0),
            final_prediction=_prediction(0),
            gate_action="ACCEPT",
            verification_status="NOT_REQUESTED",
            flagged_fields=(),
            benefit_probability=0.1,
        ),
    )
    path = write_demo_observations(tmp_path / "observations.jsonl", values)
    loaded = load_demo_observations(path)
    assert loaded == values
    targets = {("VID30", frame_id): _target(frame_id) for frame_id in (1, 2)}
    labels = oracle_benefit_labels(loaded, targets)
    assert labels == {("VID30", 1): 1, ("VID30", 2): 0}
    metrics = evaluate_demo_group(loaded, targets=targets, benefit_labels=labels)
    assert metrics["recognition"]["ivt_f1"] == 1.0
    assert metrics["recognition"]["phase_accuracy"] == 1.0
    assert metrics["gate"]["f1"] == 1.0
    assert metrics["verification_rate"] == 0.5
    assert metrics["logical_calls_per_frame"] == 1.5
    assert metrics["repair_success_rate"] == 1.0
