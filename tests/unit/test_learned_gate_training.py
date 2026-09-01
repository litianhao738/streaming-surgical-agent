from __future__ import annotations

import json
from pathlib import Path

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import FrameSupervisionTarget, FrameTaskMask
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.research.gate.benefit_model import FrozenLinearBenefitArtifact
from surgical_agent.research.gate.features import (
    NO_TRACKER_FEATURE_ORDER,
    WITH_TRACKER_FEATURE_ORDER,
    _phase_triplet_conflict,
)
from surgical_agent.research.gate.learned import (
    LearnedGateExample,
    fit_learned_gate_pair,
    load_examples,
    prediction_utility,
    write_examples,
)


def _prediction(*, correct: bool) -> InitialPrediction:
    return InitialPrediction(
        instrument_ids=(0,) if correct else (1,),
        verb_ids=(0,) if correct else (1,),
        target_ids=(0,) if correct else (1,),
        triplet_ids=(0,) if correct else (1,),
        phase_id=0 if correct else 1,
        probabilities={
            task: tuple(0.0 for _ in range(count))
            for task, count in TASK_CLASS_COUNTS.items()
        },
    )


def _target() -> FrameSupervisionTarget:
    return FrameSupervisionTarget(
        video_id="VID31",
        frame_id=100,
        instrument_ids=(0,),
        verb_ids=(0,),
        target_ids=(0,),
        triplet_ids=(0,),
        phase_id=0,
        mask=FrameTaskMask(True, True, True, True, True),
        source_granularity="frame_multilabel",
        source="unit",
    )


def _example(index: int, *, partition: str, label: int) -> LearnedGateExample:
    base = 0.8 if label else 0.2
    no_tracker = {
        name: base + (position % 3) * 0.01
        for position, name in enumerate(NO_TRACKER_FEATURE_ORDER)
    }
    with_tracker = {
        name: base + (position % 3) * 0.01
        for position, name in enumerate(WITH_TRACKER_FEATURE_ORDER)
    }
    return LearnedGateExample(
        sample_id=f"VID31:{index}",
        video_id="VID31",
        frame_id=index,
        partition=partition,
        no_tracker_features=no_tracker,
        with_tracker_features=with_tracker,
        initial_utility=0.0 if label else 1.0,
        verified_utility=1.0,
        benefit_delta=1.0 if label else 0.0,
        benefit_label=label,
        valid_tasks=("instrument", "verb", "target", "ivt", "phase"),
    )


def test_prediction_utility_obeys_task_masks() -> None:
    utility, tasks = prediction_utility(_prediction(correct=True), _target())
    assert utility == 1.0
    assert set(tasks) == {"instrument", "verb", "target", "ivt", "phase"}
    wrong, _ = prediction_utility(_prediction(correct=False), _target())
    assert wrong == 0.0


def test_phase_triplet_feature_uses_frozen_training_support() -> None:
    prediction = _prediction(correct=True)

    assert not _phase_triplet_conflict(prediction, {0: (0, 1), 1: (1,)})
    assert _phase_triplet_conflict(prediction, {0: (1,), 1: (0,)})


def test_examples_round_trip_and_train_both_runtime_artifacts(tmp_path: Path) -> None:
    examples = tuple(
        _example(index, partition="train" if index < 8 else "dev", label=index % 2)
        for index in range(12)
    )
    path = write_examples(tmp_path / "examples.jsonl", examples)
    loaded = load_examples(path)
    assert loaded == examples

    summary = fit_learned_gate_pair(loaded, output_dir=tmp_path / "models", seed=7)
    assert summary["source_split"] == "Training"
    for variant, expected_order in (
        ("gate_no_tracker", NO_TRACKER_FEATURE_ORDER),
        ("gate_with_tracker", WITH_TRACKER_FEATURE_ORDER),
    ):
        artifact_path = tmp_path / "models" / f"{variant}.json"
        artifact = FrozenLinearBenefitArtifact.from_json(artifact_path)
        assert artifact.feature_order == expected_order
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert payload["source_split"] == "training"
        suffix = variant.removeprefix("gate_")
        assert (tmp_path / "models" / f"{variant}.pkl").is_file()
        assert (tmp_path / "models" / f"scaler_{suffix}.pkl").is_file()
