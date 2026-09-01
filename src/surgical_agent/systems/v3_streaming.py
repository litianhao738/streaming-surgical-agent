"""Profile selection for the canonical V3 dataset API runtime."""

from __future__ import annotations

from pathlib import Path

NO_TRAINING_V3_PROFILE = "rule_gate"
TRAINED_V3_PROFILE = "learned_gate"


def select_v3_profile(gate_artifact: str | Path | None) -> str:
    """Use the learned Gate only when its frozen artifact is explicit."""

    return (
        NO_TRAINING_V3_PROFILE
        if gate_artifact is None
        else TRAINED_V3_PROFILE
    )
