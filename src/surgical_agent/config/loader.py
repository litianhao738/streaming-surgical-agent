"""Portable, deterministic configuration composition."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from surgical_agent.api.schema import validator_for
from surgical_agent.config.schema import ApiConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping without resolving references or touching data."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a YAML mapping in {config_path}")
    return loaded


def load_api_config(path: str | Path) -> ApiConfig:
    """Load a validated API configuration with a known response schema."""

    config = ApiConfig.from_mapping(load_yaml(path))
    validator_for(config.response_schema_version)
    return config


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result = copy.deepcopy(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_dataset_root(
    data_config: dict[str, Any],
    *,
    cli_root: str | Path | None = None,
    environment: dict[str, str] | None = None,
) -> Path:
    """Resolve the external dataset path with an auditable priority order."""

    env = os.environ if environment is None else environment
    env_name = data_config.get("root_env", "CHOLECTRACK20_ROOT")
    if not isinstance(env_name, str) or not env_name:
        raise ValueError("data.root_env must be non-empty text")
    configured = data_config.get("root")
    candidate = cli_root or env.get(env_name) or configured
    if candidate is None or not str(candidate).strip():
        raise ValueError(
            "CholecTrack20 root is required via --dataset-root, "
            f"{env_name}, or data.root"
        )
    root = Path(candidate).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CholecTrack20 root does not exist: {root}")
    return root


def load_experiment_config(
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compose one experiment and resolve its external dataset location."""

    experiment_path = Path(path).expanduser().resolve()
    experiment = load_yaml(experiment_path)
    references = {
        "base_config": None,
        "data_config": "data",
        "train_config": "train",
        "eval_config": "eval",
    }
    resolved: dict[str, Any] = {}
    for reference_key, destination in references.items():
        reference = experiment.get(reference_key)
        if reference is None:
            continue
        if not isinstance(reference, str) or not reference:
            raise TypeError(f"{reference_key} must be a non-empty path")
        loaded = load_yaml(experiment_path.parent / reference)
        if destination is None:
            resolved = deep_merge(resolved, loaded)
        else:
            current = resolved.get(destination, {})
            if not isinstance(current, dict):
                raise TypeError(f"Resolved {destination} config must be a mapping")
            resolved[destination] = deep_merge(current, loaded)

    local = {
        key: value
        for key, value in experiment.items()
        if key not in references
    }
    resolved = deep_merge(resolved, local)
    data = resolved.get("data")
    if not isinstance(data, dict):
        raise TypeError("Resolved experiment must contain a data mapping")
    root = resolve_dataset_root(
        data,
        cli_root=dataset_root,
        environment=environment,
    )
    data["root"] = str(root)
    env = os.environ if environment is None else environment
    data["root_resolved_from"] = (
        "cli"
        if dataset_root is not None
        else "environment"
        if env.get(str(data.get("root_env", "CHOLECTRACK20_ROOT")))
        else "config"
    )
    resolved["config_source"] = str(experiment_path)
    return resolved
