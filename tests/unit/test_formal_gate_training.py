from __future__ import annotations

from pathlib import Path

import pytest

from surgical_agent.research.gate.benefit_model import FrozenLinearBenefitArtifact
from surgical_agent.research.gate.calibration import calibrate_formal_gate
from surgical_agent.research.gate.contracts import (
    FORMAL_GATE_FEATURE_ORDER,
    REPAIR_SCOPE_ORDER,
)
from surgical_agent.research.gate.oof_dataset import (
    CounterfactualCollectionStore,
    CounterfactualGateRecord,
    build_paired_oof_examples,
    load_oof_examples,
    partition_oof_examples,
    write_oof_examples,
)
from surgical_agent.training.gate_trainer import (
    fit_cross_fitted_bootstrap_gates,
    fit_formal_gate,
)


def _records() -> tuple[CounterfactualGateRecord, ...]:
    result = []
    for index in range(8):
        positive = index % 2
        features = {
            name: float(positive) + feature_index * 0.001
            for feature_index, name in enumerate(FORMAL_GATE_FEATURE_ORDER)
        }
        features["tracker_available"] = 1.0
        features["tracker_conflict"] = float(positive)
        features["tracker_instrument_count"] = float(index % 3)
        result.append(
            CounterfactualGateRecord(
                sample_id=f"sample-{index}",
                video_id="VID02" if index < 4 else "VID04",
                frame_id=index,
                features=features,
                benefit_by_scope={scope: positive for scope in REPAIR_SCOPE_ORDER},
                tracker_artifact_sha256="a" * 64,
            )
        )
    return tuple(result)


def test_formal_gate_build_train_load_and_calibrate(tmp_path) -> None:
    examples = build_paired_oof_examples(_records(), fold_count=2)
    assert len(examples) == 16
    assert {item.tracker_view for item in examples} == {"T0", "T1"}
    assert {
        item.fold for item in examples if item.video_id == "VID02"
    } != {item.fold for item in examples if item.video_id == "VID04"}
    for item in examples:
        if item.tracker_view == "T0":
            values = dict(zip(item.feature_order, item.feature_values, strict=True))
            assert values["tracker_available"] == 0.0
            assert values["tracker_conflict"] == 0.0
            assert values["tracker_instrument_count"] == 0.0

    examples_path = write_oof_examples(tmp_path / "examples.jsonl", examples)
    loaded = load_oof_examples(examples_path)
    assert loaded == examples

    fitting, held_out = partition_oof_examples(examples, calibration_fold=0)
    assert {item.video_id for item in fitting}.isdisjoint(
        {item.video_id for item in held_out}
    )

    training = fit_formal_gate(fitting, output_dir=tmp_path / "training")
    assert training["example_count"] == 8
    artifact_path = tmp_path / "training" / "bootstrap_gate.json"
    with pytest.raises(ValueError, match="not deployable"):
        FrozenLinearBenefitArtifact.from_json(artifact_path)
    artifact = FrozenLinearBenefitArtifact.from_json(
        artifact_path, allow_bootstrap=True
    )
    assert artifact.feature_order == FORMAL_GATE_FEATURE_ORDER
    assert artifact.scope_order == REPAIR_SCOPE_ORDER

    calibration = calibrate_formal_gate(
        held_out,
        artifact_path=artifact_path,
        output_dir=tmp_path / "calibration",
    )
    assert calibration["source_split"] == "Training"
    assert calibration["partition"] == "training_internal_diagnostic"
    assert calibration["paper_final"] is False
    assert calibration["deployable"] is False
    assert calibration["test_data_seen"] is False
    calibrated = FrozenLinearBenefitArtifact.from_json(
        tmp_path / "calibration" / "bootstrap_gate_diagnostic.json",
        allow_bootstrap=True,
    )
    assert calibrated.threshold == calibration["threshold"]
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_formal_gate(
            fitting,
            artifact_path=artifact_path,
            output_dir=tmp_path / "invalid_calibration",
        )


def test_bootstrap_training_is_cross_fitted_by_video(tmp_path: Path) -> None:
    examples = build_paired_oof_examples(_records(), fold_count=2)

    result = fit_cross_fitted_bootstrap_gates(
        examples,
        output_dir=tmp_path / "bootstrap",
    )

    assert result["paper_final"] is False
    assert result["deployable"] is False
    assert len(result["folds"]) == 2
    for row in result["folds"]:
        assert set(row["fit_video_ids"]).isdisjoint(row["held_out_video_ids"])
        artifact = FrozenLinearBenefitArtifact.from_json(
            tmp_path / "bootstrap" / row["artifact"],
            allow_bootstrap=True,
        )
        assert artifact.gate_stage == "bootstrap_g0"


def test_counterfactual_collection_store_is_atomic_idempotent_and_resumable(
    tmp_path: Path,
) -> None:
    store = CounterfactualCollectionStore(tmp_path / "observations")
    first = _records()[0]

    store.write_observation(
        video_id=first.video_id,
        frame_id=first.frame_id,
        safety_class="HARD_VALID",
        record=first,
    )
    store.write_observation(
        video_id=first.video_id,
        frame_id=first.frame_id,
        safety_class="HARD_VALID",
        record=first,
    )
    store.write_observation(
        video_id="VID04",
        frame_id=99,
        safety_class="HARD_INVALID",
        record=None,
    )

    assert CounterfactualCollectionStore(tmp_path / "observations").records() == (
        first,
    )
    assert len(store.observations()) == 2
    with pytest.raises(ValueError, match="conflicts"):
        store.write_observation(
            video_id=first.video_id,
            frame_id=first.frame_id,
            safety_class="HARD_INVALID",
            record=None,
        )
