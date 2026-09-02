"""Strictly causal temporal fusion for ordered frame features."""

from __future__ import annotations

from torch import Tensor, nn


class CausalTemporalFusion(nn.Module):
    """Fuse an ordered past-to-present sequence with a unidirectional GRU."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 384,
        layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or layers <= 0:
            raise ValueError("fusion dimensions and layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,T,D]")
        if features.shape[1] <= 0:
            raise ValueError("causal feature sequence must not be empty")
        sequence, _state = self.gru(features)
        return self.output(sequence[:, -1])


__all__ = ["CausalTemporalFusion"]
