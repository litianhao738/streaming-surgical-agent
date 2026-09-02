"""Full Joint Perception optimization and Validation threshold calibration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.optim import Optimizer

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.schemas import FrameSupervisionTarget
from surgical_agent.models.baseline import FrameLogits
from surgical_agent.training.perception_data import PerceptionTrainingBatch

_MULTILABEL = {
    "instrument": ("instrument_ids", "instrument"),
    "verb": ("verb_ids", "verb"),
    "target": ("target_ids", "target"),
    "ivt": ("triplet_ids", "ivt"),
}


@dataclass(frozen=True)
class JointLossResult:
    total: Tensor
    task_losses: Mapping[str, Tensor]
    valid_counts: Mapping[str, int]


@dataclass(frozen=True)
class EpochResult:
    mean_loss: float
    task_losses: Mapping[str, float]
    optimizer_steps: int


def _task_logits(logits: FrameLogits, task: str) -> Tensor:
    return getattr(logits, task)


def _multihot(
    targets: Sequence[FrameSupervisionTarget],
    *,
    task: str,
    attribute: str,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    labels = torch.zeros(
        (len(targets), TASK_CLASS_COUNTS[task]),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros(len(targets), dtype=torch.bool, device=device)
    for index, target in enumerate(targets):
        if not getattr(target.mask, task):
            continue
        mask[index] = True
        for class_id in getattr(target, attribute):
            labels[index, class_id] = 1.0
    return labels, mask


def weighted_joint_loss(
    logits: FrameLogits,
    targets: Sequence[FrameSupervisionTarget],
    *,
    class_weights: Mapping[str, Tensor],
    task_weights: Mapping[str, float] | None = None,
) -> JointLossResult:
    """Apply task-masked class-balanced BCE plus phase cross entropy."""

    if logits.batch_size != len(targets):
        raise ValueError("logit batch size and target count differ")
    if set(class_weights) != set(TASK_CLASS_COUNTS):
        raise ValueError("class_weights must contain the five tasks")
    resolved_task_weights = {
        task: float((task_weights or {}).get(task, 1.0))
        for task in TASK_CLASS_COUNTS
    }
    if any(value <= 0 for value in resolved_task_weights.values()):
        raise ValueError("task weights must be positive")
    device = logits.instrument.device
    losses: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    for task, (attribute, mask_name) in _MULTILABEL.items():
        task_logits = _task_logits(logits, task)
        labels, mask = _multihot(
            targets,
            task=mask_name,
            attribute=attribute,
            device=device,
        )
        count = int(mask.sum().item())
        counts[task] = count
        if count:
            losses[task] = functional.binary_cross_entropy_with_logits(
                task_logits[mask],
                labels[mask],
                pos_weight=class_weights[task].to(device=device),
            )
        else:
            losses[task] = task_logits.sum() * 0.0

    phase_labels = torch.zeros(len(targets), dtype=torch.long, device=device)
    phase_mask = torch.zeros(len(targets), dtype=torch.bool, device=device)
    for index, target in enumerate(targets):
        if target.mask.phase and target.phase_id is not None:
            phase_mask[index] = True
            phase_labels[index] = target.phase_id
    counts["phase"] = int(phase_mask.sum().item())
    if counts["phase"]:
        losses["phase"] = functional.cross_entropy(
            logits.phase[phase_mask],
            phase_labels[phase_mask],
            weight=class_weights["phase"].to(device=device),
        )
    else:
        losses["phase"] = logits.phase.sum() * 0.0
    total = sum(
        losses[task] * resolved_task_weights[task]
        for task in TASK_CLASS_COUNTS
    )
    if not torch.isfinite(total):
        raise FloatingPointError("Joint Perception loss is non-finite")
    return JointLossResult(total=total, task_losses=losses, valid_counts=counts)


class ValidationAccumulator:
    """Collect CPU probabilities and select one threshold per multi-label task."""

    def __init__(self) -> None:
        self.probabilities = {task: [] for task in TASK_CLASS_COUNTS}
        self.targets: list[FrameSupervisionTarget] = []

    def update(
        self,
        logits: FrameLogits,
        targets: Sequence[FrameSupervisionTarget],
    ) -> None:
        if logits.batch_size != len(targets):
            raise ValueError("validation batch size mismatch")
        for task in _MULTILABEL:
            self.probabilities[task].extend(
                torch.sigmoid(_task_logits(logits, task)).detach().cpu()
            )
        self.probabilities["phase"].extend(
            torch.softmax(logits.phase, dim=-1).detach().cpu()
        )
        self.targets.extend(targets)

    @staticmethod
    def _set_f1(predicted: set[int], expected: set[int]) -> float:
        if not predicted and not expected:
            return 1.0
        return 2.0 * len(predicted & expected) / (len(predicted) + len(expected))

    def compute(
        self,
        *,
        threshold_grid: Sequence[float],
    ) -> dict[str, object]:
        if not self.targets:
            raise ValueError("validation accumulator is empty")
        thresholds: dict[str, float] = {}
        metrics: dict[str, float] = {}
        support: dict[str, int] = {}
        for task, (attribute, mask_name) in _MULTILABEL.items():
            eligible = [
                index
                for index, target in enumerate(self.targets)
                if getattr(target.mask, mask_name)
            ]
            support[task] = len(eligible)
            best_threshold = 0.5
            best_score = -1.0
            for threshold in threshold_grid:
                scores = []
                for index in eligible:
                    predicted = {
                        class_id
                        for class_id, probability in enumerate(
                            self.probabilities[task][index]
                        )
                        if float(probability) >= threshold
                    }
                    expected = set(getattr(self.targets[index], attribute))
                    scores.append(self._set_f1(predicted, expected))
                score = sum(scores) / len(scores) if scores else 0.0
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)
            thresholds[task] = best_threshold
            metrics[f"{task}_f1"] = best_score

        phase_eligible = [
            index
            for index, target in enumerate(self.targets)
            if target.mask.phase and target.phase_id is not None
        ]
        support["phase"] = len(phase_eligible)
        phase_correct = sum(
            int(
                int(torch.argmax(self.probabilities["phase"][index]).item())
                == self.targets[index].phase_id
            )
            for index in phase_eligible
        )
        metrics["phase_accuracy"] = (
            phase_correct / len(phase_eligible) if phase_eligible else 0.0
        )
        metrics["five_task_macro"] = sum(
            metrics[name]
            for name in (
                "instrument_f1",
                "verb_f1",
                "target_f1",
                "ivt_f1",
                "phase_accuracy",
            )
        ) / 5.0
        return {
            "schema_version": "joint_perception_validation_v1",
            "thresholds": thresholds,
            "metrics": metrics,
            "valid_frame_counts": support,
        }


class JointPerceptionTrainer:
    """Train one epoch with finite-gradient and progress callback hooks."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        *,
        device: torch.device,
        class_weights: Mapping[str, Tensor],
        task_weights: Mapping[str, float] | None = None,
        gradient_clip_norm: float = 5.0,
    ) -> None:
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.class_weights = class_weights
        self.task_weights = task_weights
        self.gradient_clip_norm = gradient_clip_norm

    def train_epoch(
        self,
        batches: Iterable[PerceptionTrainingBatch],
        *,
        max_batches: int | None = None,
        progress: object | None = None,
    ) -> EpochResult:
        self.model.train()
        totals: list[float] = []
        by_task = {task: [] for task in TASK_CLASS_COUNTS}
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            frames = batch.frames.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(frames)
            loss = weighted_joint_loss(
                logits,
                batch.targets,
                class_weights=self.class_weights,
                task_weights=self.task_weights,
            )
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Joint Perception gradients are non-finite")
            self.optimizer.step()
            totals.append(float(loss.total.detach().cpu()))
            for task, value in loss.task_losses.items():
                by_task[task].append(float(value.detach().cpu()))
            if progress is not None:
                progress.set_postfix(loss=f"{totals[-1]:.4f}", refresh=False)
                progress.update()
        if not totals:
            raise RuntimeError("Joint Perception completed no optimizer steps")
        return EpochResult(
            mean_loss=sum(totals) / len(totals),
            task_losses={task: sum(values) / len(values) for task, values in by_task.items()},
            optimizer_steps=len(totals),
        )


__all__ = [
    "EpochResult",
    "JointLossResult",
    "JointPerceptionTrainer",
    "ValidationAccumulator",
    "weighted_joint_loss",
]
