"""Evaluation contracts for the fixed 200-point Learned Gate demo."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from surgical_agent.data.schemas import (
    EvaluationTarget,
    FrameSupervisionTarget,
    FrameTaskMask,
)
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.contracts import TASK_NAMES
from surgical_agent.research.gate.learned import prediction_utility
from surgical_agent.research.reliability.state import final_status_for

DEMO_OBSERVATION_SCHEMA_VERSION = "learned_gate_demo_observation_v1"


def _prediction_payload(prediction: InitialPrediction) -> dict[str, object]:
    return {
        "instrument_ids": list(prediction.instrument_ids),
        "verb_ids": list(prediction.verb_ids),
        "target_ids": list(prediction.target_ids),
        "triplet_ids": list(prediction.triplet_ids),
        "phase_id": prediction.phase_id,
        "probabilities": {
            task: list(prediction.probabilities[task]) for task in TASK_NAMES
        },
        "granularity": prediction.granularity,
        "backend": prediction.backend,
        "score_semantics": prediction.score_semantics,
    }


def _prediction_from_payload(value: Mapping[str, object]) -> InitialPrediction:
    expected = {
        "instrument_ids",
        "verb_ids",
        "target_ids",
        "triplet_ids",
        "phase_id",
        "probabilities",
        "granularity",
        "backend",
        "score_semantics",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("demo prediction fields do not match the schema")
    probabilities = value["probabilities"]
    if not isinstance(probabilities, Mapping):
        raise TypeError("demo prediction probabilities must be a mapping")
    return InitialPrediction(
        instrument_ids=tuple(value["instrument_ids"]),  # type: ignore[arg-type]
        verb_ids=tuple(value["verb_ids"]),  # type: ignore[arg-type]
        target_ids=tuple(value["target_ids"]),  # type: ignore[arg-type]
        triplet_ids=tuple(value["triplet_ids"]),  # type: ignore[arg-type]
        phase_id=value["phase_id"],  # type: ignore[arg-type]
        probabilities={
            task: tuple(probabilities[task]) for task in TASK_NAMES  # type: ignore[arg-type]
        },
        granularity=value["granularity"],  # type: ignore[arg-type]
        backend=value["backend"],  # type: ignore[arg-type]
        score_semantics=value["score_semantics"],  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class DemoObservation:
    video_id: str
    frame_id: int
    initial_prediction: InitialPrediction
    final_prediction: InitialPrediction
    gate_action: str
    verification_status: str
    flagged_fields: tuple[str, ...]
    benefit_probability: float | None
    schema_version: str = DEMO_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.video_id or self.frame_id < 0:
            raise ValueError("demo observation identity is invalid")
        if not isinstance(self.initial_prediction, InitialPrediction) or not isinstance(
            self.final_prediction, InitialPrediction
        ):
            raise TypeError("demo observations require InitialPrediction values")
        if self.gate_action not in {"ACCEPT", "VERIFY"}:
            raise ValueError("demo gate_action is unsupported")
        if self.schema_version != DEMO_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported demo observation schema")
        fields = tuple(self.flagged_fields)
        if any(field not in TASK_NAMES for field in fields) or len(set(fields)) != len(fields):
            raise ValueError("demo flagged_fields are invalid")
        probability = self.benefit_probability
        if probability is not None and (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("benefit_probability must lie in [0, 1]")
        object.__setattr__(self, "flagged_fields", fields)
        if probability is not None:
            object.__setattr__(self, "benefit_probability", float(probability))

    @property
    def identity(self) -> tuple[str, int]:
        return self.video_id, self.frame_id

    @property
    def final_status(self) -> str:
        return final_status_for(self.verification_status)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
            "initial_prediction": _prediction_payload(self.initial_prediction),
            "final_prediction": _prediction_payload(self.final_prediction),
            "gate_action": self.gate_action,
            "verification_status": self.verification_status,
            "flagged_fields": list(self.flagged_fields),
            "benefit_probability": self.benefit_probability,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DemoObservation:
        expected = {
            "schema_version",
            "video_id",
            "frame_id",
            "initial_prediction",
            "final_prediction",
            "gate_action",
            "verification_status",
            "flagged_fields",
            "benefit_probability",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("demo observation fields do not match the schema")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            video_id=value["video_id"],  # type: ignore[arg-type]
            frame_id=value["frame_id"],  # type: ignore[arg-type]
            initial_prediction=_prediction_from_payload(value["initial_prediction"]),  # type: ignore[arg-type]
            final_prediction=_prediction_from_payload(value["final_prediction"]),  # type: ignore[arg-type]
            gate_action=value["gate_action"],  # type: ignore[arg-type]
            verification_status=value["verification_status"],  # type: ignore[arg-type]
            flagged_fields=tuple(value["flagged_fields"]),  # type: ignore[arg-type]
            benefit_probability=value["benefit_probability"],  # type: ignore[arg-type]
        )


def write_demo_observations(
    path: str | Path,
    observations: Sequence[DemoObservation],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = tuple(observations)
    if not values:
        raise ValueError("demo observations must not be empty")
    identities = tuple(item.identity for item in values)
    if len(set(identities)) != len(identities):
        raise ValueError("demo observation identities must be unique")
    destination.write_text(
        "".join(
            json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in values
        ),
        encoding="utf-8",
    )
    return destination


def load_demo_observations(path: str | Path) -> tuple[DemoObservation, ...]:
    values: list[DemoObservation] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid demo JSONL line {line_number}") from error
            values.append(DemoObservation.from_dict(payload))
    if not values:
        raise ValueError("demo observation file is empty")
    identities = tuple(item.identity for item in values)
    if len(set(identities)) != len(identities):
        raise ValueError("demo observation identities must be unique")
    return tuple(values)


def frame_target_from_evaluation(evaluation: EvaluationTarget) -> FrameSupervisionTarget:
    """Aggregate instance labels into one mask-aware frame recognition target."""

    if not isinstance(evaluation, EvaluationTarget):
        raise TypeError("evaluation must be an EvaluationTarget")
    instances = tuple(evaluation.instances)
    availability = {
        task: bool(instances) and all(getattr(instance.mask, task) for instance in instances)
        for task in TASK_NAMES
    }

    def ids(task: str, attribute: str) -> tuple[int, ...]:
        if not availability[task]:
            return ()
        return tuple(sorted({getattr(instance, attribute) for instance in instances}))

    phase_id: int | None = None
    if availability["phase"]:
        phases = {instance.phase_id for instance in instances}
        if len(phases) != 1:
            raise ValueError("evaluation instances contain conflicting phases")
        phase_id = next(iter(phases))
    return FrameSupervisionTarget(
        video_id=evaluation.video_id,
        frame_id=evaluation.frame_id,
        instrument_ids=ids("instrument", "instrument_id"),
        verb_ids=ids("verb", "verb_id"),
        target_ids=ids("target", "target_id"),
        triplet_ids=ids("ivt", "triplet_id"),
        phase_id=phase_id,
        mask=FrameTaskMask(
            availability["instrument"],
            availability["verb"],
            availability["target"],
            availability["ivt"],
            availability["phase"],
        ),
        source_granularity="frame_multilabel",
        source="validation_instance_aggregation",
    )


def _set_f1(predicted: Sequence[int], expected: Sequence[int]) -> float:
    first, second = set(predicted), set(expected)
    if not first and not second:
        return 1.0
    return 2.0 * len(first & second) / (len(first) + len(second))


def _task_ids(prediction: InitialPrediction, task: str) -> tuple[int, ...]:
    return {
        "instrument": prediction.instrument_ids,
        "verb": prediction.verb_ids,
        "target": prediction.target_ids,
        "ivt": prediction.triplet_ids,
    }[task]


def oracle_benefit_labels(
    observations: Sequence[DemoObservation],
    targets: Mapping[tuple[str, int], FrameSupervisionTarget],
) -> Mapping[tuple[str, int], int]:
    labels: dict[tuple[str, int], int] = {}
    for item in observations:
        target = targets[item.identity]
        initial, _ = prediction_utility(item.initial_prediction, target)
        verified, _ = prediction_utility(item.final_prediction, target)
        labels[item.identity] = int(verified > initial)
    return labels


def evaluate_demo_group(
    observations: Sequence[DemoObservation],
    *,
    targets: Mapping[tuple[str, int], FrameSupervisionTarget],
    benefit_labels: Mapping[tuple[str, int], int],
) -> dict[str, object]:
    values = tuple(observations)
    if not values:
        raise ValueError("demo group observations must not be empty")
    identities = tuple(item.identity for item in values)
    if set(identities) != set(targets) or set(identities) != set(benefit_labels):
        raise ValueError("demo group, GT and benefit label identities must match")
    recognition: dict[str, float] = {}
    for task in ("instrument", "verb", "target", "ivt"):
        scores = [
            _set_f1(_task_ids(item.final_prediction, task), _task_ids_target(targets[item.identity], task))
            for item in values
            if getattr(targets[item.identity].mask, task)
        ]
        recognition[f"{task}_f1"] = sum(scores) / len(scores) if scores else 0.0
    phase_scores = [
        float(item.final_prediction.phase_id == targets[item.identity].phase_id)
        for item in values
        if targets[item.identity].mask.phase
    ]
    recognition["phase_accuracy"] = (
        sum(phase_scores) / len(phase_scores) if phase_scores else 0.0
    )

    truth = [int(benefit_labels[item.identity]) for item in values]
    predicted = [int(item.gate_action == "VERIFY") for item in values]
    tp = sum(a == 1 and b == 1 for a, b in zip(truth, predicted, strict=True))
    fp = sum(a == 0 and b == 1 for a, b in zip(truth, predicted, strict=True))
    fn = sum(a == 1 and b == 0 for a, b in zip(truth, predicted, strict=True))
    tn = sum(a == 0 and b == 0 for a, b in zip(truth, predicted, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    gate_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    verified_repairs = [
        item for item in values if item.verification_status == "VERIFIED_REPAIR"
    ]
    successful_repairs = 0
    for item in verified_repairs:
        target = targets[item.identity]
        initial, _ = prediction_utility(item.initial_prediction, target)
        final, _ = prediction_utility(item.final_prediction, target)
        successful_repairs += int(final > initial)
    verify_count = sum(predicted)
    return {
        "sample_count": len(values),
        "recognition": recognition,
        "gate": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": gate_f1,
            "error_capture_rate": recall,
        },
        "verification_rate": verify_count / len(values),
        "logical_api_calls": len(values) + verify_count,
        "logical_calls_per_frame": (len(values) + verify_count) / len(values),
        "repair_success_rate": (
            successful_repairs / len(verified_repairs) if verified_repairs else None
        ),
        "state_counts": dict(Counter(item.final_status for item in values)),
        "verification_status_counts": dict(
            Counter(item.verification_status for item in values)
        ),
    }


def _task_ids_target(target: FrameSupervisionTarget, task: str) -> tuple[int, ...]:
    return {
        "instrument": target.instrument_ids,
        "verb": target.verb_ids,
        "target": target.target_ids,
        "ivt": target.triplet_ids,
    }[task]


__all__ = [
    "DemoObservation",
    "evaluate_demo_group",
    "frame_target_from_evaluation",
    "load_demo_observations",
    "oracle_benefit_labels",
    "write_demo_observations",
]
