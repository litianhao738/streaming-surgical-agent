"""Strict context-experiment configuration contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from surgical_agent.config.context_experiment import load_context_experiment

PROJECT_ROOT = Path(__file__).parents[2]


def _write_config(
    path: Path,
    *,
    context_profile: str = "workflow",
    workflow_enabled: bool = True,
    event_memory_enabled: bool = False,
    memory_enabled: bool = False,
    phase_transition_graph: str | None = "artifacts/phase_graph.json",
    predicted_track_artifact: str | None = None,
) -> Path:
    path.write_text(
        "\n".join(
            (
                "runtime:",
                f"  context_profile: {context_profile}",
                f"  event_memory_enabled: {str(event_memory_enabled).lower()}",
                "  phase_transition_graph:"
                + (" null" if phase_transition_graph is None else f" {phase_transition_graph}"),
                "  predicted_track_artifact:"
                + (
                    " null"
                    if predicted_track_artifact is None
                    else f" {predicted_track_artifact}"
                ),
                "workflow:",
                f"  enabled: {str(workflow_enabled).lower()}",
                "memory:",
                f"  enabled: {str(memory_enabled).lower()}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def test_loader_resolves_context_artifact_paths_relative_to_yaml(tmp_path: Path) -> None:
    config = _write_config(tmp_path / "workflow.yaml")

    loaded = load_context_experiment(config)

    assert loaded.context_profile == "workflow"
    assert loaded.event_memory_enabled is False
    assert loaded.phase_transition_graph == tmp_path / "artifacts/phase_graph.json"
    assert loaded.predicted_track_artifact is None


@pytest.mark.parametrize(
    ("context_profile", "workflow_enabled", "phase_transition_graph", "message"),
    [
        ("frames_only", True, None, "workflow.enabled"),
        ("track_only", False, "artifacts/graph.json", "phase_transition_graph"),
        ("workflow", True, None, "phase_transition_graph"),
        ("track_workflow", True, "artifacts/graph.json", "predicted_track_artifact"),
    ],
)
def test_loader_rejects_profile_inconsistent_context_fields(
    tmp_path: Path,
    context_profile: str,
    workflow_enabled: bool,
    phase_transition_graph: str | None,
    message: str,
) -> None:
    config = _write_config(
        tmp_path / "invalid.yaml",
        context_profile=context_profile,
        workflow_enabled=workflow_enabled,
        phase_transition_graph=phase_transition_graph,
    )

    with pytest.raises(ValueError, match=message):
        load_context_experiment(config)


def test_loader_rejects_disagreement_between_memory_and_runtime_switches(
    tmp_path: Path,
) -> None:
    config = _write_config(
        tmp_path / "memory.yaml",
        event_memory_enabled=True,
        memory_enabled=False,
    )

    with pytest.raises(ValueError, match="memory.enabled"):
        load_context_experiment(config)


@pytest.mark.parametrize(
    ("relative_path", "profile", "event_memory_enabled"),
    [
        ("configs/ablations/track_only.yaml", "track_only", False),
        ("configs/ablations/no_event_memory.yaml", "track_workflow", False),
        ("configs/ablations/no_workflow.yaml", "frames_only", False),
        ("configs/ablations/workflow_only.yaml", "workflow", False),
        ("configs/experiments/v3_track_workflow.yaml", "track_workflow", True),
    ],
)
def test_standard_context_experiments_expose_one_runtime_memory_switch(
    relative_path: str,
    profile: str,
    event_memory_enabled: bool,
) -> None:
    loaded = load_context_experiment(PROJECT_ROOT / relative_path)

    assert loaded.context_profile == profile
    assert loaded.event_memory_enabled is event_memory_enabled
    if profile in {"workflow", "track_workflow"}:
        assert loaded.phase_transition_graph == (
            PROJECT_ROOT / "artifacts/training/phase_transition_graph.json"
        )
    if profile == "track_only":
        assert loaded.phase_transition_graph is None
    if profile in {"track_only", "track_workflow"}:
        assert loaded.predicted_track_artifact == (
            PROJECT_ROOT
            / "artifacts/training/tracker_clip_v2_oof5_20260906/predicted_tracks.json"
        )
