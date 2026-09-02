"""ImageNet-initialized frame encoder for causal joint perception."""

from __future__ import annotations

from torch import Tensor, nn


class MobileNetV3VisualEncoder(nn.Module):
    """Encode every frame independently before causal temporal fusion."""

    def __init__(
        self,
        *,
        architecture: str = "mobilenet_v3_small",
        feature_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if architecture not in {"mobilenet_v3_small", "mobilenet_v3_large"}:
            raise ValueError("unsupported visual encoder architecture")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        from torchvision.models import (
            MobileNet_V3_Large_Weights,
            MobileNet_V3_Small_Weights,
            mobilenet_v3_large,
            mobilenet_v3_small,
        )

        if architecture == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
        else:
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v3_large(weights=weights)
        backbone_dim = int(backbone.classifier[0].in_features)
        self.architecture = architecture
        self.pretrained = pretrained
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.projection = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(backbone_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.output_dim = feature_dim

    def forward(self, frames: Tensor) -> Tensor:
        """Return ``[B,T,D]`` features from ``[B,T,3,H,W]`` frames."""

        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        if not frames.is_floating_point():
            raise TypeError("frames must be floating-point tensors")
        batch_size, frame_count = frames.shape[:2]
        flattened = frames.reshape(batch_size * frame_count, *frames.shape[2:])
        encoded = self.projection(self.pool(self.features(flattened)))
        return encoded.reshape(batch_size, frame_count, self.output_dim)


__all__ = ["MobileNetV3VisualEncoder"]
