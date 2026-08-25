"""Small offline model used only to validate the P2 engineering path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.inference.schemas import InitialPrediction


@dataclass(frozen=True)
class FrameLogits:
    """Fixed-shape logits with an explicit frame-level prediction contract."""

    instrument: Tensor
    verb: Tensor
    target: Tensor
    ivt: Tensor
    phase: Tensor
    granularity: str = "frame_multilabel"

    @property
    def batch_size(self) -> int:
        return int(self.instrument.shape[0])


class LocalSmokeModel(nn.Module):
    """Compact CNN with independent frame-level heads and no remote weights."""

    def __init__(self, *, hidden_dim: int = 64, pretrained: bool = False) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("P2 forbids pretrained or runtime-downloaded weights")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(hidden_dim, class_count)
                for task, class_count in TASK_CLASS_COUNTS.items()
            }
        )

    def forward(self, frames: Tensor) -> FrameLogits:
        """Predict from the current frame; a causal tensor may include history."""

        if frames.ndim == 5:
            frames = frames[:, -1]
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("frames must have shape [B,3,H,W] or [B,T,3,H,W]")
        if not frames.is_floating_point():
            raise TypeError("frames must be floating-point tensors")
        features = self.encoder(frames)
        outputs = {task: head(features) for task, head in self.heads.items()}
        return FrameLogits(
            instrument=outputs["instrument"],
            verb=outputs["verb"],
            target=outputs["target"],
            ivt=outputs["ivt"],
            phase=outputs["phase"],
        )


def _selected_ids(probabilities: Tensor, threshold: float) -> tuple[int, ...]:
    return tuple(
        int(index)
        for index in torch.nonzero(probabilities >= threshold, as_tuple=False)
        .flatten()
        .tolist()
    )


def decode_frame_logits(
    logits: FrameLogits,
    *,
    threshold: float = 0.5,
) -> tuple[InitialPrediction, ...]:
    """Convert logits to schema-valid predictions without consulting targets."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    probability_tensors = {
        "instrument": torch.sigmoid(logits.instrument),
        "verb": torch.sigmoid(logits.verb),
        "target": torch.sigmoid(logits.target),
        "ivt": torch.sigmoid(logits.ivt),
        "phase": torch.softmax(logits.phase, dim=-1),
    }
    predictions: list[InitialPrediction] = []
    for batch_index in range(logits.batch_size):
        probabilities = {
            task: tuple(float(value) for value in tensor[batch_index].detach().cpu())
            for task, tensor in probability_tensors.items()
        }
        predictions.append(
            InitialPrediction(
                instrument_ids=_selected_ids(
                    probability_tensors["instrument"][batch_index], threshold
                ),
                verb_ids=_selected_ids(
                    probability_tensors["verb"][batch_index], threshold
                ),
                target_ids=_selected_ids(
                    probability_tensors["target"][batch_index], threshold
                ),
                triplet_ids=_selected_ids(
                    probability_tensors["ivt"][batch_index], threshold
                ),
                phase_id=int(probability_tensors["phase"][batch_index].argmax().item()),
                probabilities=probabilities,
            )
        )
    return tuple(predictions)
