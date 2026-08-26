"""Small typed configuration contracts used during repository bootstrap."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from surgical_agent.api.errors import ApiContractError


def _freeze_json_value(value: object, *, field_name: str) -> object:
    """Validate JSON-compatible option data and recursively make it immutable."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ApiContractError(f"API {field_name} must contain finite JSON data")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ApiContractError(f"API {field_name} must use text mapping keys")
            frozen[key] = _freeze_json_value(nested_value, field_name=field_name)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, field_name=field_name) for item in value
        )
    raise ApiContractError(f"API {field_name} must contain JSON-compatible data")


def _freeze_option_mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ApiContractError(f"API {field_name} must be a mapping")
    frozen = _freeze_json_value(value, field_name=field_name)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset location and read-only policy.

    A missing root is intentional until a CLI option or resolved config supplies
    it. Production source code must not embed a workstation-specific path.
    """

    root: Path | None = None
    read_only: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Bootstrap-level experiment identity and safety switches."""

    name: str
    seed: int = 42
    research_enabled: bool = False


@dataclass(frozen=True)
class ApiConfig:
    """Effective P3 API configuration without credential material."""

    enabled: bool
    mode: str
    provider: str
    endpoint_identifier: str
    requested_model_identifier: str
    prompt_version: str
    response_schema_version: str
    generation_parameters: Mapping[str, object] = field(default_factory=dict)
    provider_options: Mapping[str, object] = field(default_factory=dict)
    synthetic_input_required: bool = True
    cache_required: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ApiConfig:
        if not isinstance(raw, Mapping):
            raise ApiContractError("API config must be a mapping")
        required = (
            "enabled",
            "mode",
            "provider",
            "endpoint_identifier",
            "requested_model_identifier",
            "prompt_version",
            "response_schema_version",
        )
        missing = [name for name in required if name not in raw]
        if missing:
            raise ApiContractError(f"API config is missing fields: {missing}")
        if type(raw["enabled"]) is not bool:
            raise ApiContractError("API enabled must be boolean")
        text_values: dict[str, str] = {}
        for name in required[1:]:
            value = raw[name]
            if not isinstance(value, str) or not value.strip():
                raise ApiContractError(f"API {name} must be non-empty text")
            text_values[name] = value
        synthetic_input_required = raw.get("synthetic_input_required", True)
        cache_required = raw.get("cache_required", True)
        if type(synthetic_input_required) is not bool:
            raise ApiContractError("API synthetic_input_required must be boolean")
        if type(cache_required) is not bool:
            raise ApiContractError("API cache_required must be boolean")
        config = cls(
            enabled=raw["enabled"],
            mode=text_values["mode"],
            provider=text_values["provider"],
            endpoint_identifier=text_values["endpoint_identifier"],
            requested_model_identifier=text_values["requested_model_identifier"],
            prompt_version=text_values["prompt_version"],
            response_schema_version=text_values["response_schema_version"],
            generation_parameters=_freeze_option_mapping(
                raw.get("generation_parameters", {}),
                field_name="generation_parameters",
            ),
            provider_options=_freeze_option_mapping(
                raw.get("provider_options", {}),
                field_name="provider_options",
            ),
            synthetic_input_required=synthetic_input_required,
            cache_required=cache_required,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ApiContractError("API enabled must be boolean")
        if type(self.synthetic_input_required) is not bool:
            raise ApiContractError("API synthetic_input_required must be boolean")
        if type(self.cache_required) is not bool:
            raise ApiContractError("API cache_required must be boolean")
        for name in (
            "mode",
            "provider",
            "endpoint_identifier",
            "requested_model_identifier",
            "prompt_version",
            "response_schema_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ApiContractError(f"API {name} must be non-empty text")
        _freeze_option_mapping(
            self.generation_parameters,
            field_name="generation_parameters",
        )
        _freeze_option_mapping(
            self.provider_options,
            field_name="provider_options",
        )
        if self.mode not in {"mock", "real"}:
            raise ApiContractError("API mode must be mock or real")
        if self.provider == "requesty":
            if self.mode != "real":
                raise ApiContractError("Requesty requires real mode")
            if self.endpoint_identifier != "https://router.requesty.ai/v1/responses":
                raise ApiContractError("Requesty endpoint is not approved")
        if self.provider == "mock" and self.mode != "mock":
            raise ApiContractError("mock provider requires mock mode")

    def require_enabled(self) -> None:
        if not self.enabled:
            raise ApiContractError("API config is disabled")


@dataclass(frozen=True)
class WorkflowConfig:
    """Independent compact workflow-state switch."""

    enabled: bool = True
    finalized_only: bool = True


@dataclass(frozen=True)
class EventMemoryConfig:
    """Independent episodic-memory switch and per-video bound."""

    enabled: bool = True
    finalized_only: bool = True
    max_events_per_video: int | None = None
