"""Small typed configuration contracts used during repository bootstrap."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from surgical_agent.api.errors import ApiContractError


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
    def from_mapping(cls, raw: Mapping[str, object]) -> "ApiConfig":
        if raw.get("enabled") is False:
            config = cls(
                enabled=False,
                mode=str(raw.get("mode", "mock")),
                provider=str(raw.get("provider", "disabled")),
                endpoint_identifier=str(raw.get("endpoint_identifier", "disabled")),
                requested_model_identifier=str(
                    raw.get("requested_model_identifier", "disabled")
                ),
                prompt_version=str(raw.get("prompt_version", "disabled")),
                response_schema_version=str(
                    raw.get("response_schema_version", "disabled")
                ),
                generation_parameters=MappingProxyType(
                    copy.deepcopy(dict(raw.get("generation_parameters", {})))
                ),
                provider_options=MappingProxyType(
                    copy.deepcopy(dict(raw.get("provider_options", {})))
                ),
                synthetic_input_required=bool(raw.get("synthetic_input_required", True)),
                cache_required=bool(raw.get("cache_required", True)),
            )
            config.validate()
            return config
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
        config = cls(
            enabled=raw["enabled"],
            mode=str(raw["mode"]),
            provider=str(raw["provider"]),
            endpoint_identifier=str(raw["endpoint_identifier"]),
            requested_model_identifier=str(raw["requested_model_identifier"]),
            prompt_version=str(raw["prompt_version"]),
            response_schema_version=str(raw["response_schema_version"]),
            generation_parameters=MappingProxyType(
                copy.deepcopy(dict(raw.get("generation_parameters", {})))
            ),
            provider_options=MappingProxyType(
                copy.deepcopy(dict(raw.get("provider_options", {})))
            ),
            synthetic_input_required=bool(raw.get("synthetic_input_required", True)),
            cache_required=bool(raw.get("cache_required", True)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ApiContractError("API enabled must be boolean")
        if not self.enabled:
            return
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
