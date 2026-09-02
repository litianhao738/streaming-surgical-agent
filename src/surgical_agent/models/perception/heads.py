"""Five task heads for frame-level surgical scene understanding."""

from __future__ import annotations

from torch import Tensor, nn

from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.models.baseline import FrameLogits


class JointTaskHeads(nn.Module):
    """Emit four multi-label heads and one single-label phase head."""

    def __init__(self, *, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                task: nn.Linear(input_dim, class_count)
                for task, class_count in TASK_CLASS_COUNTS.items()
            }
        )

    def forward(self, fused: Tensor) -> FrameLogits:
        if fused.ndim != 2:
            raise ValueError("fused features must have shape [B,D]")
        hidden = self.dropout(fused)
        outputs = {task: head(hidden) for task, head in self.heads.items()}
        return FrameLogits(
            instrument=outputs["instrument"],
            verb=outputs["verb"],
            target=outputs["target"],
            ivt=outputs["ivt"],
            phase=outputs["phase"],
        )


__all__ = ["JointTaskHeads"]
