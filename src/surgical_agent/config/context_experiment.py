"""Strict runtime context switches loaded from an experiment YAML."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from surgical_agent.config.loader import load_yaml

_CONTEXT_PROFILES = frozenset(
    {"frames_only", "track_only", "workflow", "track_workflow"}
)
_RUNTIME_FIELDS = frozenset(
    {
        "context_profile",
        "event_memory_enabled",
        "phase_transition_graph",
        "predicted_track_artifact",
        "allow_oracle_track_provider",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _artifact_path(
    value: object,
    *,
    name: str,
    source_path: Path,
) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be null or non-empty text")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{name} must be relative to the experiment YAML")
    for parent in source_path.parents:
        if (parent / "pyproject.toml").is_file():
            return (parent / candidate).resolve()
    return (source_path.parent / candidate).resolve()


@dataclass(frozen=True)
class ContextExperimentConfig:
    """The context/event-memory choices that are safe to pass to runtime."""

    context_profile: str
    event_memory_enabled: bool
    phase_transition_graph: Path | None
    predicted_track_artifact: Path | None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
        *,
        source_path: Path,
    ) -> ContextExperimentConfig:
        runtime = _mapping(raw.get("runtime"), name="runtime")
        unknown_runtime_fields = set(runtime) - _RUNTIME_FIELDS
        if unknown_runtime_fields:
            raise ValueError(
                "runtime contains unsupported context fields: "
                f"{sorted(unknown_runtime_fields)!r}"
            )
        required_runtime_fields = {
            "context_profile",
            "event_memory_enabled",
            "phase_transition_graph",
            "predicted_track_artifact",
        }
        missing_runtime_fields = required_runtime_fields - set(runtime)
        if missing_runtime_fields:
            raise ValueError(
                "runtime is missing context fields: "
                f"{sorted(missing_runtime_fields)!r}"
            )

        context_profile = runtime["context_profile"]
        if context_profile not in _CONTEXT_PROFILES:
            raise ValueError("runtime.context_profile is unsupported")
        workflow = _mapping(raw.get("workflow"), name="workflow")
        memory = _mapping(raw.get("memory"), name="memory")
        workflow_enabled = _boolean(
            workflow.get("enabled"), name="workflow.enabled"
        )
        memory_enabled = _boolean(memory.get("enabled"), name="memory.enabled")
        event_memory_enabled = _boolean(
            runtime["event_memory_enabled"], name="runtime.event_memory_enabled"
        )
        if memory_enabled != event_memory_enabled:
            raise ValueError("memory.enabled must equal runtime.event_memory_enabled")

        phase_transition_graph = _artifact_path(
            runtime["phase_transition_graph"],
            name="runtime.phase_transition_graph",
            source_path=source_path,
        )
        predicted_track_artifact = _artifact_path(
            runtime["predicted_track_artifact"],
            name="runtime.predicted_track_artifact",
            source_path=source_path,
        )
        if context_profile == "frames_only":
            if workflow_enabled:
                raise ValueError("frames_only requires workflow.enabled=false")
            if phase_transition_graph is not None:
                raise ValueError("frames_only forbids phase_transition_graph")
            if predicted_track_artifact is not None:
                raise ValueError("frames_only forbids predicted_track_artifact")
        elif context_profile == "track_only":
            if workflow_enabled:
                raise ValueError("track_only requires workflow.enabled=false")
            if phase_transition_graph is not None:
                raise ValueError("track_only forbids phase_transition_graph")
            if predicted_track_artifact is None:
                raise ValueError("track_only requires predicted_track_artifact")
        else:
            if not workflow_enabled:
                raise ValueError(f"{context_profile} requires workflow.enabled=true")
            if phase_transition_graph is None:
                raise ValueError(f"{context_profile} requires phase_transition_graph")
            if (
                context_profile == "workflow"
                and predicted_track_artifact is not None
            ):
                raise ValueError("workflow forbids predicted_track_artifact")
            if (
                context_profile == "track_workflow"
                and predicted_track_artifact is None
            ):
                raise ValueError("track_workflow requires predicted_track_artifact")
        return cls(
            context_profile=context_profile,
            event_memory_enabled=event_memory_enabled,
            phase_transition_graph=phase_transition_graph,
            predicted_track_artifact=predicted_track_artifact,
        )


def load_context_experiment(path: str | Path) -> ContextExperimentConfig:
    """Load one experiment's strict context/event-memory runtime subset."""

    source_path = Path(path).expanduser().resolve()
    return ContextExperimentConfig.from_mapping(
        load_yaml(source_path),
        source_path=source_path,
    )
