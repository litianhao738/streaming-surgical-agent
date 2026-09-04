"""Strict loader for the four Tracker x Gate ablation cells."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from surgical_agent.config.loader import load_yaml

_EXPECTED = {
    "A_base": (False, "RULE"),
    "B_tracker": (True, "RULE"),
    "C_gate": (False, "LEARNED"),
    "D_full": (True, "LEARNED"),
}


@dataclass(frozen=True)
class TrackerGateCell:
    cell: Literal["A_base", "B_tracker", "C_gate", "D_full"]
    tracker_enabled: bool
    gate_mode: Literal["RULE", "LEARNED"]
    gate_artifact: Path | None
    shared_config: Path
    shared: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.cell not in _EXPECTED:
            raise ValueError("unknown Tracker x Gate ablation cell")
        if (self.tracker_enabled, self.gate_mode) != _EXPECTED[self.cell]:
            raise ValueError("ablation cell factors do not match its canonical definition")
        if self.gate_mode == "LEARNED" and self.gate_artifact is None:
            raise ValueError("LEARNED cells require one shared Gate artifact path")
        if self.gate_mode == "RULE" and self.gate_artifact is not None:
            raise ValueError("RULE cells cannot load a learned Gate artifact")
        object.__setattr__(self, "shared", MappingProxyType(dict(self.shared)))


def load_tracker_gate_cell(path: str | Path) -> TrackerGateCell:
    source = Path(path).expanduser().resolve()
    raw = load_yaml(source)
    expected_fields = {
        "schema_version",
        "cell",
        "shared_config",
        "tracker_enabled",
        "gate_mode",
        "gate_artifact",
    }
    if set(raw) != expected_fields or raw["schema_version"] != "tracker_gate_ablation_cell_v1":
        raise ValueError("ablation cell fields do not match the formal schema")
    project_root = Path(__file__).resolve().parents[3]
    shared_path = project_root / str(raw["shared_config"])
    shared = load_yaml(shared_path)
    if shared.get("schema_version") != "final_pipeline_experiment_v1":
        raise ValueError("unsupported shared final Pipeline config")
    artifact = raw["gate_artifact"]
    return TrackerGateCell(
        cell=raw["cell"],
        tracker_enabled=raw["tracker_enabled"],
        gate_mode=raw["gate_mode"],
        gate_artifact=None if artifact is None else project_root / str(artifact),
        shared_config=shared_path,
        shared=shared,
    )


def validate_tracker_gate_matrix(cells: Sequence[TrackerGateCell]) -> None:
    values = tuple(cells)
    if {item.cell for item in values} != set(_EXPECTED) or len(values) != 4:
        raise ValueError("the formal matrix requires exactly A_base/B_tracker/C_gate/D_full")
    shared_paths = {item.shared_config.resolve() for item in values}
    shared_values = {repr(dict(item.shared)) for item in values}
    if len(shared_paths) != 1 or len(shared_values) != 1:
        raise ValueError("all four cells must share identical support infrastructure")
    learned_artifacts = {
        item.gate_artifact.resolve()
        for item in values
        if item.gate_artifact is not None
    }
    if len(learned_artifacts) != 1:
        raise ValueError("C_gate and D_full must load the same learned Gate artifact")


__all__ = [
    "TrackerGateCell",
    "load_tracker_gate_cell",
    "validate_tracker_gate_matrix",
]
