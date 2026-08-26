"""Behavioral tests for frozen train-only phase transition artifacts."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.signals.contracts import PhaseTransitionGraph
from surgical_agent.research.signals.phase_graph import (
    PhaseObservation,
    build_phase_transition_graph,
    load_phase_transition_graph,
    write_phase_transition_graph,
)


def obs(
    video_id: str,
    frame_id: int,
    phase_id: int,
    split: DatasetSplit = DatasetSplit.TRAINING,
) -> PhaseObservation:
    return PhaseObservation(video_id, frame_id, phase_id, split)


def _script_module():
    script_path = Path(__file__).parents[2] / "scripts" / "build_phase_transition_graph.py"
    spec = importlib.util.spec_from_file_location("build_phase_transition_graph", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_graph_contains_self_edges_and_observed_directed_edges_only() -> None:
    graph = build_phase_transition_graph(
        (obs("VID01", 1, 0), obs("VID01", 2, 0), obs("VID01", 3, 2))
    )

    assert graph.transitions == ((0, 0), (0, 2), (2, 2))
    assert graph.source_video_ids == ("VID01",)
    assert graph.version == "phase_transition_train_v1"


def test_graph_rejects_validation_observation() -> None:
    with pytest.raises(ValueError, match="training"):
        build_phase_transition_graph((obs("VID30", 1, 0, DatasetSplit.VALIDATION),))


@pytest.mark.parametrize(
    "observations",
    [
        (obs("VID01", 2, 0), obs("VID01", 1, 2)),
        (obs("VID01", 1, 0), obs("VID01", 1, 2)),
    ],
)
def test_graph_rejects_non_increasing_frame_ids_per_video(
    observations: tuple[PhaseObservation, ...],
) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        build_phase_transition_graph(observations)


def test_graph_requires_an_eligible_training_observation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_phase_transition_graph(())


def test_graph_contract_requires_stripped_nonempty_source_video_ids() -> None:
    with pytest.raises(ValueError, match="stripped"):
        PhaseTransitionGraph(
            transitions=((0, 0),),
            source_video_ids=(" VID01 ",),
            version="phase_transition_train_v1",
            sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="at least one"):
        PhaseTransitionGraph(
            transitions=((0, 0),),
            source_video_ids=(),
            version="phase_transition_train_v1",
            sha256="0" * 64,
        )


def test_identical_ordered_input_has_a_deterministic_hash() -> None:
    observations = (obs("VID02", 1, 3), obs("VID02", 4, 4))

    assert build_phase_transition_graph(observations).sha256 == (
        build_phase_transition_graph(observations).sha256
    )


def test_load_rejects_a_tampered_stored_hash(tmp_path: Path) -> None:
    output = write_phase_transition_graph(
        build_phase_transition_graph((obs("VID01", 1, 0),)),
        tmp_path / "phase-transition.json",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["sha256"] = "0" * 64
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        load_phase_transition_graph(output)


def test_loaded_graph_is_immutable_and_round_trips(tmp_path: Path) -> None:
    graph = build_phase_transition_graph(
        (obs("VID02", 1, 1), obs("VID02", 2, 2))
    )

    loaded = load_phase_transition_graph(
        write_phase_transition_graph(graph, tmp_path / "phase-transition.json")
    )

    assert loaded == graph
    with pytest.raises(FrozenInstanceError):
        loaded.transitions = ((0, 0),)  # type: ignore[misc]


def test_cli_adapter_uses_training_records_and_excludes_missing_phase_labels() -> None:
    script = _script_module()
    requested_video_ids: list[str] = []
    adapter = SimpleNamespace(
        entries={
            "VID01": SimpleNamespace(video_id="VID01", split=DatasetSplit.TRAINING),
            "VID30": SimpleNamespace(video_id="VID30", split=DatasetSplit.VALIDATION),
            "VID40": SimpleNamespace(video_id="VID40", split=DatasetSplit.TESTING),
        }
    )

    def iter_video(video_id: str):
        requested_video_ids.append(video_id)
        return iter(
            (
                SimpleNamespace(
                    inference=SimpleNamespace(source_split=DatasetSplit.TRAINING),
                    frame_supervision=SimpleNamespace(
                        video_id="VID01",
                        frame_id=1,
                        phase_id=0,
                        mask=SimpleNamespace(phase=True),
                    ),
                ),
                SimpleNamespace(
                    inference=SimpleNamespace(source_split=DatasetSplit.TRAINING),
                    frame_supervision=SimpleNamespace(
                        video_id="VID01",
                        frame_id=2,
                        phase_id=None,
                        mask=SimpleNamespace(phase=False),
                    ),
                ),
            )
        )

    adapter.iter_video = iter_video

    assert tuple(script._training_phase_observations(adapter)) == (obs("VID01", 1, 0),)
    assert requested_video_ids == ["VID01"]
