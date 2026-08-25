"""Sole offline component allowed to compare predictions with GT."""

from __future__ import annotations

from dataclasses import dataclass

from surgical_agent.data.schemas import EvaluationTarget, FrameSupervisionTarget
from surgical_agent.inference.schemas import PredictionRecord


@dataclass(frozen=True)
class SampleSmokeMetrics:
    video_id: str
    frame_id: int
    values: dict[str, float]
    eligibility: dict[str, bool]
    semantics: str = "ENGINEERING_SMOKE_NOT_PAPER_METRICS"


class EvaluationEngine:
    """Granularity-aware P2 evaluator; paper metrics remain deferred to P4."""

    metric_names = (
        "instrument_frame_exact",
        "verb_frame_exact",
        "target_frame_exact",
        "ivt_frame_exact",
        "phase_accuracy",
    )

    def evaluate(
        self,
        prediction: PredictionRecord,
        evaluation: EvaluationTarget,
        frame_target: FrameSupervisionTarget | None,
    ) -> SampleSmokeMetrics:
        if (prediction.video_id, prediction.frame_id) != (
            evaluation.video_id,
            evaluation.frame_id,
        ):
            raise ValueError("Prediction and EvaluationTarget identity mismatch")
        if frame_target is not None and (prediction.video_id, prediction.frame_id) != (
            frame_target.video_id,
            frame_target.frame_id,
        ):
            raise ValueError("Prediction and FrameSupervisionTarget identity mismatch")
        values: dict[str, float] = {}
        eligibility = {name: False for name in self.metric_names}
        if frame_target is not None:
            comparisons = {
                "instrument_frame_exact": (
                    prediction.instrument_ids,
                    frame_target.instrument_ids,
                    frame_target.mask.instrument,
                ),
                "verb_frame_exact": (
                    prediction.verb_ids,
                    frame_target.verb_ids,
                    frame_target.mask.verb,
                ),
                "target_frame_exact": (
                    prediction.target_ids,
                    frame_target.target_ids,
                    frame_target.mask.target,
                ),
                "ivt_frame_exact": (
                    prediction.triplet_ids,
                    frame_target.triplet_ids,
                    frame_target.mask.ivt,
                ),
            }
            for name, (predicted, expected, enabled) in comparisons.items():
                eligibility[name] = enabled
                if enabled:
                    values[name] = float(predicted == expected)
            eligibility["phase_accuracy"] = frame_target.mask.phase
            if frame_target.mask.phase:
                values["phase_accuracy"] = float(
                    prediction.phase_id == frame_target.phase_id
                )
        return SampleSmokeMetrics(
            video_id=prediction.video_id,
            frame_id=prediction.frame_id,
            values=values,
            eligibility=eligibility,
        )

    def summarize(self, samples: tuple[SampleSmokeMetrics, ...]) -> dict[str, object]:
        metrics: dict[str, dict[str, float | int | None]] = {}
        for name in self.metric_names:
            values = [sample.values[name] for sample in samples if name in sample.values]
            metrics[name] = {
                "value": sum(values) / len(values) if values else None,
                "support": len(values),
            }
        return {
            "schema_version": "p2_engineering_smoke_metrics_v1",
            "semantics": "ENGINEERING_SMOKE_NOT_PAPER_METRICS",
            "sample_count": len(samples),
            "metrics": metrics,
        }
