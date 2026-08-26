"""Behavioral tests for formal video-wise frame recognition metrics."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import asdict

import numpy as np
import pytest

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import (
    DatasetSplit,
    FrameSupervisionTarget,
    FrameTaskMask,
)
from surgical_agent.evaluation.evaluator import EvaluationEngine
from surgical_agent.evaluation.frame_metrics import (
    FrameMetricAccumulator,
    compute_frame_metric_report,
    framework_gain,
    paired_video_bootstrap,
)
from surgical_agent.inference.schemas import PredictionRecord


def _scores(**overrides: tuple[float, ...]) -> dict[str, tuple[float, ...]]:
    values = {
        task: (0.0,) * class_count for task, class_count in TASK_CLASS_COUNTS.items()
    }
    values.update(overrides)
    return values


def _class_scores(task: str, **values: float) -> tuple[float, ...]:
    scores = [0.0] * TASK_CLASS_COUNTS[task]
    for class_id, score in values.items():
        scores[int(class_id)] = score
    return tuple(scores)


def _prediction(
    video_id: str,
    frame_id: int,
    *,
    instrument_ids: tuple[int, ...] = (),
    verb_ids: tuple[int, ...] = (),
    target_ids: tuple[int, ...] = (),
    triplet_ids: tuple[int, ...] = (),
    phase_id: int = 0,
    probabilities: dict[str, tuple[float, ...]] | None = None,
    score_semantics: str = "uncalibrated_rank_v1",
) -> PredictionRecord:
    return PredictionRecord(
        run_id="formal-metrics-test",
        video_id=video_id,
        frame_id=frame_id,
        source_split=DatasetSplit.TESTING,
        causal_frame_ids=(frame_id,),
        instrument_ids=instrument_ids,
        verb_ids=verb_ids,
        target_ids=target_ids,
        triplet_ids=triplet_ids,
        phase_id=phase_id,
        granularity="frame_multilabel",
        backend="unit-test",
        gate_action="ACCEPT",
        verification_status="SKIPPED",
        alignment_version="unit-test",
        probabilities=_scores() if probabilities is None else probabilities,
        score_semantics=score_semantics,
    )


def _target(
    video_id: str,
    frame_id: int,
    *,
    instrument_ids: tuple[int, ...] = (),
    verb_ids: tuple[int, ...] = (),
    target_ids: tuple[int, ...] = (),
    triplet_ids: tuple[int, ...] = (),
    phase_id: int | None = 0,
    mask: FrameTaskMask | None = None,
) -> FrameSupervisionTarget:
    return FrameSupervisionTarget(
        video_id=video_id,
        frame_id=frame_id,
        instrument_ids=instrument_ids,
        verb_ids=verb_ids,
        target_ids=target_ids,
        triplet_ids=triplet_ids,
        phase_id=phase_id,
        mask=FrameTaskMask(True, True, True, True, True) if mask is None else mask,
        source_granularity="frame_multilabel",
        source="unit-test",
    )


def _accumulator(
    pairs: tuple[tuple[PredictionRecord, FrameSupervisionTarget], ...],
) -> FrameMetricAccumulator:
    accumulator = FrameMetricAccumulator()
    for prediction, target in pairs:
        accumulator.update(prediction, target)
    return accumulator


def _duplicate_sensitive_pairs() -> tuple[
    tuple[PredictionRecord, FrameSupervisionTarget], ...
]:
    return (
        (
            _prediction(
                "VID01",
                0,
                phase_id=0,
                probabilities=_scores(
                    instrument=_class_scores("instrument", **{"0": 0.9})
                ),
            ),
            _target("VID01", 0, instrument_ids=(0,), phase_id=0),
        ),
        (
            _prediction(
                "VID01",
                1,
                phase_id=0,
                probabilities=_scores(
                    instrument=_class_scores("instrument", **{"0": 0.8})
                ),
            ),
            _target("VID01", 1, phase_id=1),
        ),
    )


def test_video_wise_map_is_not_pooled_frame_ap() -> None:
    pairs: list[tuple[PredictionRecord, FrameSupervisionTarget]] = []
    fixtures = (
        ("VID01", (1,), (0.9,)),
        ("VID02", (1, 0, 0, 0), (0.1, 0.8, 0.7, 0.6)),
    )
    for video_id, labels, scores in fixtures:
        for frame_id, (label, score) in enumerate(zip(labels, scores, strict=True)):
            probabilities = _scores(
                instrument=_class_scores("instrument", **{"0": score})
            )
            pairs.append(
                (
                    _prediction(video_id, frame_id, probabilities=probabilities),
                    _target(
                        video_id,
                        frame_id,
                        instrument_ids=(0,) if label else (),
                    ),
                )
            )

    report = _accumulator(tuple(pairs)).compute()
    instrument = report.tasks["instrument"]
    pooled_ap = 0.7

    assert instrument.class_ap[0] == pytest.approx(0.625)
    assert instrument.class_video_support[0] == 2
    assert instrument.video_wise_map == pytest.approx(0.625)
    assert instrument.video_wise_map != pytest.approx(pooled_ap)


def test_task_masks_and_no_positive_classes_control_ap_eligibility() -> None:
    eligible = (
        _prediction(
            "VID01",
            0,
            probabilities=_scores(
                instrument=_class_scores("instrument", **{"0": 0.8, "1": 0.9})
            ),
        ),
        _target(
            "VID01",
            0,
            instrument_ids=(0,),
            phase_id=None,
            mask=FrameTaskMask(True, False, False, False, False),
        ),
    )
    masked = (
        _prediction(
            "VID02",
            0,
            probabilities=_scores(instrument=_class_scores("instrument", **{"1": 1.0})),
        ),
        _target(
            "VID02",
            0,
            instrument_ids=(1,),
            phase_id=None,
            mask=FrameTaskMask(False, False, False, False, False),
        ),
    )

    report = _accumulator((eligible, masked)).compute()
    instrument = report.tasks["instrument"]

    assert instrument.class_ap[0] == 1.0
    assert instrument.class_ap[1] is None
    assert instrument.class_video_support[0] == 1
    assert instrument.class_video_support[1] == 0
    assert instrument.video_wise_map == 1.0
    assert report.phase.per_video == {}
    assert report.phase.video_wise_accuracy is None


def test_ivt_null_classes_are_excluded_but_support_is_retained() -> None:
    pairs = (
        (
            _prediction(
                "VID01",
                0,
                probabilities=_scores(
                    ivt=_class_scores("ivt", **{"0": 0.8, "94": 0.9})
                ),
            ),
            _target("VID01", 0, triplet_ids=(0, 94)),
        ),
        (
            _prediction(
                "VID01",
                1,
                probabilities=_scores(
                    ivt=_class_scores("ivt", **{"0": 0.2, "94": 0.1})
                ),
            ),
            _target("VID01", 1),
        ),
    )

    ivt = _accumulator(pairs).compute().tasks["ivt"]

    assert ivt.excluded_classes == (94, 95, 96, 97, 98, 99)
    assert ivt.class_video_support[94] == 1
    assert ivt.class_ap[94] is None
    assert ivt.class_ap[0] == 1.0
    assert ivt.video_wise_map == 1.0


def test_dense_zero_ivt_tail_scores_remain_ranking_scores() -> None:
    pairs = (
        (_prediction("VID01", 0), _target("VID01", 0, triplet_ids=(93,))),
        (_prediction("VID01", 1), _target("VID01", 1)),
    )

    report = _accumulator(pairs).compute()

    assert report.tasks["ivt"].class_ap[93] == 0.5
    assert report.tasks["ivt"].class_video_support[93] == 1
    assert report.score_semantics == ("uncalibrated_rank_v1",)


def test_phase_macro_f1_excludes_absent_and_scores_missed_present_class_zero() -> None:
    pairs = (
        (_prediction("VID02", 0, phase_id=0), _target("VID02", 0, phase_id=0)),
        (_prediction("VID02", 1, phase_id=0), _target("VID02", 1, phase_id=1)),
        (_prediction("VID02", 2, phase_id=0), _target("VID02", 2, phase_id=1)),
    )

    phase = _accumulator(pairs).compute().phase
    video = phase.per_video["VID02"]

    assert video.accuracy == pytest.approx(1.0 / 3.0)
    assert video.class_support == {0: 1, 1: 2, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    assert video.class_f1[0] == 0.5
    assert video.class_f1[1] == 0.0
    assert video.excluded_classes == (2, 3, 4, 5, 6)
    assert video.macro_f1 == 0.25
    assert phase.video_wise_accuracy == pytest.approx(1.0 / 3.0)
    assert phase.video_wise_macro_f1 == 0.25


def test_phase_metrics_average_complete_videos_equally() -> None:
    pairs = (
        (_prediction("VID01", 0, phase_id=0), _target("VID01", 0, phase_id=0)),
        (_prediction("VID02", 0, phase_id=1), _target("VID02", 0, phase_id=0)),
        (_prediction("VID02", 1, phase_id=0), _target("VID02", 1, phase_id=0)),
        (_prediction("VID02", 2, phase_id=0), _target("VID02", 2, phase_id=0)),
    )

    phase = _accumulator(pairs).compute().phase

    assert phase.per_video["VID01"].accuracy == 1.0
    assert phase.per_video["VID02"].accuracy == pytest.approx(2.0 / 3.0)
    assert phase.video_wise_accuracy == pytest.approx(5.0 / 6.0)
    assert phase.video_wise_macro_f1 == pytest.approx(0.9)


def test_report_preserves_all_score_semantics_and_is_json_serializable() -> None:
    first = _prediction("VID01", 0, score_semantics="probability_v1")
    second = _prediction("VID01", 1, score_semantics="uncalibrated_rank_v1")

    report = compute_frame_metric_report(
        ((first, _target("VID01", 0)), (second, _target("VID01", 1)))
    )

    assert report.score_semantics == (
        "probability_v1",
        "uncalibrated_rank_v1",
    )
    assert json.loads(json.dumps(asdict(report)))["schema_version"] == (
        "frame_recognition_metrics_v1"
    )


def test_accumulator_and_evaluator_require_exact_prediction_target_identity() -> None:
    prediction = _prediction("VID01", 0)

    with pytest.raises(ValueError, match="prediction and target identity mismatch"):
        FrameMetricAccumulator().update(prediction, _target("VID02", 0))

    with pytest.raises(ValueError, match="same length"):
        EvaluationEngine().summarize_frame_recognition((prediction,), ())


def test_accumulator_rejects_duplicate_frames_without_biasing_any_metric() -> None:
    pairs = _duplicate_sensitive_pairs()
    accumulator = _accumulator(pairs)

    with pytest.raises(ValueError, match="duplicate prediction and target identity"):
        accumulator.update(*pairs[0])

    report = accumulator.compute()
    assert report.tasks["instrument"].video_wise_map == 1.0
    assert report.phase.video_wise_accuracy == 0.5
    assert report.phase.video_wise_macro_f1 == pytest.approx(1.0 / 3.0)


def test_direct_report_builder_rejects_duplicate_frames() -> None:
    pairs = _duplicate_sensitive_pairs()

    with pytest.raises(ValueError, match="duplicate prediction and target identity"):
        compute_frame_metric_report((*pairs, pairs[0]))


@pytest.mark.parametrize(
    "mutated_scores",
    [
        (0.0,),
        (math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_accumulator_snapshots_mutable_probability_mappings(
    mutated_scores: tuple[float, ...],
) -> None:
    probabilities = _scores(instrument=_class_scores("instrument", **{"0": 0.9}))
    accumulator = FrameMetricAccumulator()
    accumulator.update(
        _prediction("VID01", 0, probabilities=probabilities),
        _target("VID01", 0, instrument_ids=(0,)),
    )
    probabilities["instrument"] = mutated_scores
    accumulator.update(
        _prediction(
            "VID01",
            1,
            probabilities=_scores(instrument=_class_scores("instrument", **{"0": 0.8})),
        ),
        _target("VID01", 1),
    )

    report = accumulator.compute()

    assert report.tasks["instrument"].class_ap[0] == 1.0


def test_accumulator_snapshots_prediction_target_and_mask_values() -> None:
    prediction = _prediction("VID01", 0, phase_id=0)
    target = _target("VID01", 0, instrument_ids=(0,), phase_id=0)
    accumulator = FrameMetricAccumulator()
    accumulator.update(prediction, target)

    object.__setattr__(prediction, "phase_id", 1)
    object.__setattr__(target, "instrument_ids", ())
    object.__setattr__(target.mask, "instrument", False)

    report = accumulator.compute()

    assert report.tasks["instrument"].class_ap[0] == 1.0
    assert report.phase.video_wise_accuracy == 1.0


def test_direct_report_builder_snapshots_each_pair_while_ingesting() -> None:
    probabilities = _scores(instrument=_class_scores("instrument", **{"0": 0.9}))
    first = (
        _prediction("VID01", 0, probabilities=probabilities),
        _target("VID01", 0, instrument_ids=(0,)),
    )
    second = (
        _prediction(
            "VID01",
            1,
            probabilities=_scores(instrument=_class_scores("instrument", **{"0": 0.8})),
        ),
        _target("VID01", 1),
    )

    def records() -> Iterator[tuple[PredictionRecord, FrameSupervisionTarget]]:
        yield first
        probabilities["instrument"] = (math.nan,)
        yield second

    report = compute_frame_metric_report(records())

    assert report.tasks["instrument"].class_ap[0] == 1.0


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("instrument", "true"),
        ("verb", 1),
        ("target", 0.5),
        ("ivt", np.bool_(True)),
        ("phase", 1),
    ],
)
def test_report_rejects_non_bool_task_masks(
    field_name: str,
    invalid_value: object,
) -> None:
    mask_values: dict[str, object] = {
        "instrument": True,
        "verb": True,
        "target": True,
        "ivt": True,
        "phase": True,
    }
    mask_values[field_name] = invalid_value
    mask = FrameTaskMask(**mask_values)  # type: ignore[arg-type]
    pair = (
        _prediction("VID01", 0),
        _target("VID01", 0, mask=mask),
    )

    with pytest.raises(TypeError, match="task masks must contain exact bool values"):
        compute_frame_metric_report((pair,))


@pytest.mark.parametrize("owner", ["prediction", "target"])
@pytest.mark.parametrize("invalid_phase_id", ["1", 1.5, True])
def test_report_rejects_non_integer_or_boolean_phase_ids(
    owner: str,
    invalid_phase_id: object,
) -> None:
    prediction = _prediction("VID01", 0)
    target = _target("VID01", 0)
    object.__setattr__(
        prediction if owner == "prediction" else target,
        "phase_id",
        invalid_phase_id,
    )

    with pytest.raises(TypeError, match="phase_id must be a non-boolean integer"):
        compute_frame_metric_report(((prediction, target),))


def test_evaluation_engine_exposes_formal_metrics_and_preserves_smoke_names() -> None:
    prediction = _prediction("VID01", 0, phase_id=0)
    target = _target("VID01", 0, phase_id=0)
    engine = EvaluationEngine()

    report = engine.summarize_frame_recognition((prediction,), (target,))

    assert report == compute_frame_metric_report(((prediction, target),))
    assert engine.metric_names == (
        "instrument_frame_exact",
        "verb_frame_exact",
        "target_frame_exact",
        "ivt_frame_exact",
        "phase_accuracy",
    )


def test_ivt_video_ap_matches_official_ivtmetrics_reference() -> None:
    from ivtmetrics import Recognition

    labels = np.zeros((4, 100), dtype=np.float64)
    scores = np.zeros((4, 100), dtype=np.float64)
    labels[0, :95] = 1.0
    labels[1, (0,)] = 1.0
    labels[2, (2,)] = 1.0
    scores[:, 0] = (0.9, 0.8, 0.7, 0.1)
    scores[:, 2] = (0.2, 0.1, 0.9, 0.8)
    scores[:, 94] = (0.95, 0.3, 0.2, 0.1)
    pairs = tuple(
        (
            _prediction(
                "VID01",
                frame_id,
                probabilities=_scores(ivt=tuple(scores[frame_id])),
            ),
            _target(
                "VID01",
                frame_id,
                triplet_ids=tuple(np.flatnonzero(labels[frame_id]).tolist()),
            ),
        )
        for frame_id in range(4)
    )
    official = Recognition(num_class=100)
    official.update(labels, scores)
    official.video_end()

    expected = official.compute_video_AP("ivt", ignore_null=True)
    actual = _accumulator(pairs).compute().tasks["ivt"]

    assert actual.video_wise_map == pytest.approx(expected["mAP"])
    for class_id, class_ap in enumerate(expected["AP"]):
        if not math.isnan(class_ap):
            assert actual.class_ap[class_id] == pytest.approx(class_ap)


def test_framework_gain_is_full_minus_baseline_and_rejects_invalid_values() -> None:
    assert framework_gain(0.81, 0.72) == pytest.approx(0.09)

    for full, baseline in ((True, 0.0), (0.0, False), (math.inf, 0.0)):
        with pytest.raises((TypeError, ValueError)):
            framework_gain(full, baseline)


def test_paired_bootstrap_uses_named_whole_videos_and_is_deterministic() -> None:
    full = {"VID30": 0.9, "VID02": 0.8, "VID31": 0.4}
    baseline = {"VID31": 0.1, "VID30": 0.5, "VID02": 0.7}

    first = paired_video_bootstrap(full, baseline, seed=1234, resamples=2000)
    second = paired_video_bootstrap(full, baseline, seed=1234, resamples=2000)

    assert first == second
    assert first.estimate == pytest.approx((0.4 + 0.1 + 0.3) / 3.0)
    assert first.lower == pytest.approx(0.1)
    assert first.upper == pytest.approx(0.4)
    assert first.confidence == 0.95
    assert first.resampling_unit == "video"


def test_paired_bootstrap_uses_linear_non_grid_percentiles() -> None:
    interval = paired_video_bootstrap(
        {"VID01": 0.0, "VID02": 0.5, "VID03": 1.0},
        {"VID01": 0.0, "VID02": 0.0, "VID03": 0.0},
        seed=6,
        resamples=7,
    )

    assert interval.lower == pytest.approx(43.0 / 120.0)
    assert interval.upper == pytest.approx(97.0 / 120.0)


def test_empty_accumulator_returns_complete_undefined_report() -> None:
    report = FrameMetricAccumulator().compute()

    assert report.schema_version == "frame_recognition_metrics_v1"
    assert tuple(report.tasks) == ("instrument", "verb", "target", "ivt")
    for task, task_report in report.tasks.items():
        assert task_report.class_ap == {
            class_id: None for class_id in range(TASK_CLASS_COUNTS[task])
        }
        assert task_report.class_video_support == {
            class_id: 0 for class_id in range(TASK_CLASS_COUNTS[task])
        }
        assert task_report.video_wise_map is None
        assert task_report.excluded_classes == (
            (94, 95, 96, 97, 98, 99) if task == "ivt" else ()
        )
    assert report.tasks["ivt"].excluded_classes == (94, 95, 96, 97, 98, 99)
    assert report.phase.video_wise_accuracy is None
    assert report.phase.video_wise_macro_f1 is None
    assert report.phase.per_video == {}
    assert report.score_semantics == ()


@pytest.mark.parametrize(
    ("full", "baseline", "seed", "resamples"),
    [
        ({"VID01": 0.2}, {"VID02": 0.1}, 0, 100),
        ({}, {}, 0, 100),
        ({"VID01": math.nan}, {"VID01": 0.1}, 0, 100),
        ({"VID01": 0.2}, {"VID01": True}, 0, 100),
        ({"VID01": 0.2}, {"VID01": 0.1}, True, 100),
        ({"VID01": 0.2}, {"VID01": 0.1}, 1.5, 100),
        ({"VID01": 0.2}, {"VID01": 0.1}, 0, False),
        ({"VID01": 0.2}, {"VID01": 0.1}, 0, 0),
    ],
)
def test_paired_bootstrap_rejects_mismatched_or_invalid_inputs(
    full: dict[str, object],
    baseline: dict[str, object],
    seed: object,
    resamples: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        paired_video_bootstrap(  # type: ignore[arg-type]
            full,
            baseline,
            seed=seed,
            resamples=resamples,
        )
