"""Trainable causal Joint Perception model."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from surgical_agent.models.baseline import FrameLogits
from surgical_agent.models.perception.fusion import CausalTemporalFusion
from surgical_agent.models.perception.heads import JointTaskHeads
from surgical_agent.models.perception.visual_encoder import MobileNetV3VisualEncoder


@dataclass(frozen=True)
class JointPerceptionModelConfig:
    architecture: str = "mobilenet_v3_small"
    visual_feature_dim: int = 256
    temporal_hidden_dim: int = 384
    temporal_layers: int = 1
    dropout: float = 0.2
    pretrained: bool = True

    def __post_init__(self) -> None:
        if self.architecture not in {"mobilenet_v3_small", "mobilenet_v3_large"}:
            raise ValueError("unsupported Joint Perception architecture")
        if min(
            self.visual_feature_dim,
            self.temporal_hidden_dim,
            self.temporal_layers,
        ) <= 0:
            raise ValueError("Joint Perception dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")


class CausalJointPerceptionModel(nn.Module):
    """Encode a causal frame window and jointly predict all five tasks."""

    def __init__(self, config: JointPerceptionModelConfig) -> None:
        super().__init__()
        if not isinstance(config, JointPerceptionModelConfig):
            raise TypeError("config must be JointPerceptionModelConfig")
        self.config = config
        self.visual_encoder = MobileNetV3VisualEncoder(
            architecture=config.architecture,
            feature_dim=config.visual_feature_dim,
            pretrained=config.pretrained,
        )
        self.temporal_fusion = CausalTemporalFusion(
            input_dim=self.visual_encoder.output_dim,
            hidden_dim=config.temporal_hidden_dim,
            layers=config.temporal_layers,
            dropout=config.dropout,
        )
        self.task_heads = JointTaskHeads(
            input_dim=self.temporal_fusion.output_dim,
            dropout=config.dropout,
        )

    def forward(self, frames: Tensor) -> FrameLogits:
        return self.task_heads(self.temporal_fusion(self.visual_encoder(frames)))


__all__ = [
    "CausalJointPerceptionModel",
    "JointPerceptionModelConfig",
]
