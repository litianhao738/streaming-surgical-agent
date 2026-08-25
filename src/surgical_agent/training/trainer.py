"""Minimal P2 trainer with strict split and finite-gradient checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import Optimizer

from surgical_agent.data.dataset import SmokeBatch
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.models.baseline import LocalSmokeModel
from surgical_agent.training.losses import masked_multitask_loss


@dataclass(frozen=True)
class TrainStepResult:
    """Serializable summary of one real optimizer step."""

    total_loss: float
    task_losses: dict[str, float]
    valid_counts: dict[str, int]
    batch_sample_ids: tuple[str, ...]
    gradients_finite: bool


class LocalSmokeTrainer:
    """One-stage trainer used to validate wiring, not scientific performance."""

    def __init__(
        self,
        model: LocalSmokeModel,
        optimizer: Optimizer,
        *,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch: SmokeBatch) -> TrainStepResult:
        if any(
            provenance.split is not DatasetSplit.TRAINING
            for provenance in batch.provenance
        ):
            raise ValueError("Optimizer input must contain training samples only")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(batch.images.to(self.device))
        result = masked_multitask_loss(logits, batch.frame_targets)
        result.total.backward()
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        gradients_finite = bool(gradients) and all(
            bool(torch.isfinite(gradient).all().item()) for gradient in gradients
        )
        if not gradients_finite:
            raise FloatingPointError("P2 smoke step produced missing or non-finite gradients")
        self.optimizer.step()
        sample_ids = tuple(
            f"{sample.video_id}:{sample.target_frame_id}"
            for sample in batch.inference_samples
        )
        return TrainStepResult(
            total_loss=float(result.total.detach().cpu().item()),
            task_losses={
                task: float(loss.detach().cpu().item())
                for task, loss in result.task_losses.items()
            },
            valid_counts=result.valid_counts,
            batch_sample_ids=sample_ids,
            gradients_finite=gradients_finite,
        )
