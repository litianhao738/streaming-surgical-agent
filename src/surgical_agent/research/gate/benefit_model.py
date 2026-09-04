"""Strict deploy-time contract for a small frozen linear Benefit Gate."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from surgical_agent.research.gate.contracts import FORMAL_GATE_FEATURE_ORDER
from surgical_agent.research.gate.features import (
    ALL_DEMO_GATE_FEATURE_NAMES,
    NO_TRACKER_FEATURE_ORDER,
    WITH_TRACKER_FEATURE_ORDER,
)
from surgical_agent.research.signals.contracts import GATE_FEATURE_NAMES

_DEPLOYABLE_FEATURE_NAMES = (
    GATE_FEATURE_NAMES | ALL_DEMO_GATE_FEATURE_NAMES | set(FORMAL_GATE_FEATURE_ORDER)
)

_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "gate_stage",
        "source_split",
        "feature_order",
        "weights_by_scope",
        "bias_by_scope",
        "scope_order",
        "threshold",
        "training_dataset_ids",
        "training_recipe_id",
        "normalization_version",
        "rollout_policy_id",
    }
)


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _finite(value: object, *, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class FrozenLinearBenefitArtifact:
    feature_order: tuple[str, ...]
    weights_by_scope: Mapping[str, tuple[float, ...]]
    bias_by_scope: Mapping[str, float]
    scope_order: tuple[str, ...]
    threshold: float
    training_dataset_ids: tuple[str, ...]
    training_recipe_id: str
    normalization_version: str
    rollout_policy_id: str
    schema_version: str = "benefit_gate_linear_v1"
    gate_stage: str = "final_g1"
    source_split: str = "training"

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        allow_bootstrap: bool = False,
    ) -> FrozenLinearBenefitArtifact:
        source = Path(path)
        if not source.is_file():
            raise ValueError("Benefit Gate artifact file is missing")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError("Benefit Gate artifact is not valid JSON") from None
        if not isinstance(payload, Mapping) or set(payload) != _REQUIRED_KEYS:
            raise ValueError("Benefit Gate artifact fields do not match the schema")
        if payload["schema_version"] != "benefit_gate_linear_v1":
            raise ValueError("unsupported Benefit Gate artifact schema")
        gate_stage = payload["gate_stage"]
        if gate_stage not in {"bootstrap_g0", "final_g1"}:
            raise ValueError("unsupported Benefit Gate stage; only final_g1 is deployable")
        if gate_stage != "final_g1" and not allow_bootstrap:
            raise ValueError("bootstrap_g0 is not deployable; final_g1 is required")
        if payload["source_split"] != "training":
            raise ValueError("Benefit Gate artifact must be train-derived")
        raw_feature_order = payload["feature_order"]
        if not isinstance(raw_feature_order, (list, tuple)):
            raise TypeError("feature_order must be an array")
        feature_order = tuple(raw_feature_order)
        if not feature_order or any(
            not isinstance(name, str) or name not in _DEPLOYABLE_FEATURE_NAMES
            for name in feature_order
        ) or len(set(feature_order)) != len(feature_order):
            raise ValueError("feature_order contains unknown or duplicate features")
        is_evidence_schema = set(feature_order) <= GATE_FEATURE_NAMES
        is_demo_schema = feature_order in {
            NO_TRACKER_FEATURE_ORDER,
            WITH_TRACKER_FEATURE_ORDER,
        }
        is_formal_schema = feature_order == FORMAL_GATE_FEATURE_ORDER
        if not (is_evidence_schema or is_demo_schema or is_formal_schema):
            raise ValueError("feature_order mixes incompatible Gate schemas")
        raw_scope_order = payload["scope_order"]
        if not isinstance(raw_scope_order, (list, tuple)):
            raise TypeError("scope_order must be an array")
        scope_order = tuple(raw_scope_order)
        if not scope_order or any(
            not isinstance(scope, str) or not scope for scope in scope_order
        ) or len(set(scope_order)) != len(scope_order):
            raise ValueError("scope_order must contain unique non-empty scopes")
        raw_weights = payload["weights_by_scope"]
        raw_bias = payload["bias_by_scope"]
        if not isinstance(raw_weights, Mapping) or not isinstance(raw_bias, Mapping):
            raise TypeError("Gate weights and bias must be mappings")
        if set(raw_weights) != set(scope_order) or set(raw_bias) != set(scope_order):
            raise ValueError("Gate parameter scopes must match scope_order")
        weights: dict[str, tuple[float, ...]] = {}
        bias: dict[str, float] = {}
        for scope in scope_order:
            raw_vector = raw_weights[scope]
            if not isinstance(raw_vector, (list, tuple)):
                raise TypeError("Gate weight vectors must be arrays")
            vector = tuple(
                _finite(value, name=f"weights_by_scope.{scope}")
                for value in raw_vector
            )
            if len(vector) != len(feature_order):
                raise ValueError("Gate weight dimension must match feature_order")
            weights[scope] = vector
            bias[scope] = _finite(raw_bias[scope], name=f"bias_by_scope.{scope}")
        threshold = _finite(payload["threshold"], name="threshold")
        raw_training_ids = payload["training_dataset_ids"]
        if not isinstance(raw_training_ids, (list, tuple)):
            raise TypeError("training_dataset_ids must be an array")
        training_ids = tuple(raw_training_ids)
        if not training_ids or any(
            not isinstance(value, str) or not value for value in training_ids
        ):
            raise ValueError("training_dataset_ids must not be empty")
        normalization_version = _nonempty_text(
            payload["normalization_version"], name="normalization_version"
        )
        if normalization_version not in {
            "evidence_frame_v1_raw",
            "demo_gate_features_v1_raw",
            "formal_gate_features_v1_raw",
        }:
            raise ValueError(
                "only an allowlisted raw normalization is deployable"
            )
        expected_normalization = "evidence_frame_v1_raw"
        if is_demo_schema:
            expected_normalization = "demo_gate_features_v1_raw"
        elif is_formal_schema:
            expected_normalization = "formal_gate_features_v1_raw"
        if normalization_version != expected_normalization:
            raise ValueError("normalization_version does not match feature_order")
        return cls(
            feature_order=feature_order,
            weights_by_scope=MappingProxyType(weights),
            bias_by_scope=MappingProxyType(bias),
            scope_order=scope_order,
            threshold=threshold,
            training_dataset_ids=training_ids,
            training_recipe_id=_nonempty_text(
                payload["training_recipe_id"], name="training_recipe_id"
            ),
            normalization_version=normalization_version,
            rollout_policy_id=_nonempty_text(
                payload["rollout_policy_id"], name="rollout_policy_id"
            ),
            gate_stage=gate_stage,
        )
