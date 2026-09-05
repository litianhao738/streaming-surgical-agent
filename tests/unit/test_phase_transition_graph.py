"""Behavioral tests for frozen train-only phase transition artifacts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from surgical_agent.data.schemas import (
    BoundingBox,
    DatasetSplit,
    EvaluationInstanceTarget,
    EvaluationTarget,
    LabelMask,
    TrackIds,
)
from surgical_agent.research.signals.contracts import PhaseTransitionGraph
from surgical_agent.research.signals.phase_graph import (
    PhaseObservation,
    build_phase_instrument_ivt_prior_from_training_adapter,
    build_phase_ivt_compatibility_from_training_adapter,
    build_phase_transition_graph,
    build_phase_transition_graph_from_training_adapter,
    iter_training_phase_ivt_targets,
    iter_training_phase_observations,
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
    graph = build_phase_transition_graph((obs("VID02", 1, 1), obs("VID02", 2, 2)))

    loaded = load_phase_transition_graph(
        write_phase_transition_graph(graph, tmp_path / "phase-transition.json")
    )

    assert loaded == graph
    with pytest.raises(FrozenInstanceError):
        loaded.transitions = ((0, 0),)  # type: ignore[misc]


def test_training_adapter_helper_uses_only_official_training_records() -> None:
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

    assert tuple(iter_training_phase_observations(adapter)) == (obs("VID01", 1, 0),)
    assert requested_video_ids == ["VID01"]


def _instance_prior_record(phase, *, ivt_mask=True, derived_ivt=None):
    ivt = 13 if phase == 4 else 94
    instance = EvaluationInstanceTarget(
        instrument_id=0,
        verb_id=8 if ivt == 13 else 9,
        target_id=0 if ivt == 13 else 14,
        triplet_id=ivt,
        phase_id=phase,
        operator_id=0,
        bbox=BoundingBox(0.1, 0.1, 0.2, 0.2),
        tracks=TrackIds(1, 1, 1),
        mask=LabelMask(True, True, True, ivt_mask, True),
    )
    return SimpleNamespace(
        inference=SimpleNamespace(source_split=DatasetSplit.TRAINING),
        frame_supervision=SimpleNamespace(
            video_id="VID01",
            frame_id=phase + 1,
            phase_id=phase,
            triplet_ids=() if derived_ivt is None else (derived_ivt,),
            mask=SimpleNamespace(phase=True, ivt=derived_ivt is not None),
        ),
        evaluation=EvaluationTarget("VID01", phase + 1, (instance,)),
    )


def _instance_prior_adapter(records):
    requested = []

    def iter_video(video):
        requested.append(video)
        return iter(records)

    return SimpleNamespace(
        entries={
            "VID01": SimpleNamespace(video_id="VID01", split=DatasetSplit.TRAINING),
            "VID30": SimpleNamespace(video_id="VID30", split=DatasetSplit.VALIDATION),
        },
        iter_video=iter_video,
    ), requested


def test_phase_ivt_support_includes_masked_instance_route_not_just_frame_targets():
    adapter, requested = _instance_prior_adapter(
        [_instance_prior_record(phase) for phase in range(7)]
    )
    support = build_phase_ivt_compatibility_from_training_adapter(adapter)
    prior = build_phase_instrument_ivt_prior_from_training_adapter(adapter)
    assert support[4] == (13,) and prior[(4, 0)] == (13,)
    assert requested == ["VID01", "VID01"]
    assert (
        tuple(
            iter_training_phase_ivt_targets(adapter, include_instance_supervision=False)
        )
        == ()
    )


def test_prior_instance_fallback_honors_masks_and_does_not_double_count():
    adapter, _ = _instance_prior_adapter(
        [
            _instance_prior_record(0, ivt_mask=False),
            _instance_prior_record(4, derived_ivt=12),
        ]
    )
    targets = tuple(iter_training_phase_ivt_targets(adapter))
    assert len(targets) == 1 and targets[0].triplet_ids == (12,)


def test_prior_instance_route_rejects_non_training_records():
    record = _instance_prior_record(0)
    record.inference.source_split = DatasetSplit.VALIDATION
    adapter, _ = _instance_prior_adapter([record])
    with pytest.raises(ValueError, match="non-training"):
        tuple(iter_training_phase_ivt_targets(adapter))


def test_training_adapter_helper_skips_missing_phase_labels() -> None:
    adapter = SimpleNamespace(
        entries={
            "VID01": SimpleNamespace(video_id="VID01", split=DatasetSplit.TRAINING)
        }
    )
    adapter.iter_video = lambda _video_id: iter(
        (
            SimpleNamespace(
                inference=SimpleNamespace(source_split=DatasetSplit.TRAINING),
                frame_supervision=None,
            ),
            SimpleNamespace(
                inference=SimpleNamespace(source_split=DatasetSplit.TRAINING),
                frame_supervision=SimpleNamespace(
                    video_id="VID01",
                    frame_id=1,
                    phase_id=None,
                    mask=SimpleNamespace(phase=True),
                ),
            ),
            SimpleNamespace(
                inference=SimpleNamespace(source_split=DatasetSplit.TRAINING),
                frame_supervision=SimpleNamespace(
                    video_id="VID01",
                    frame_id=2,
                    phase_id=3,
                    mask=SimpleNamespace(phase=False),
                ),
            ),
            SimpleNamespace(
                inference=SimpleNamespace(source_split=DatasetSplit.TRAINING),
                frame_supervision=SimpleNamespace(
                    video_id="VID01",
                    frame_id=3,
                    phase_id=4,
                    mask=SimpleNamespace(phase=True),
                ),
            ),
        )
    )

    graph = build_phase_transition_graph_from_training_adapter(adapter)

    assert graph.source_video_ids == ("VID01",)
    assert graph.transitions == ((4, 4),)


def test_training_adapter_helper_rejects_non_training_record() -> None:
    adapter = SimpleNamespace(
        entries={
            "VID01": SimpleNamespace(video_id="VID01", split=DatasetSplit.TRAINING)
        }
    )
    adapter.iter_video = lambda _video_id: iter(
        (
            SimpleNamespace(
                inference=SimpleNamespace(source_split=DatasetSplit.VALIDATION),
                frame_supervision=SimpleNamespace(
                    video_id="VID01",
                    frame_id=1,
                    phase_id=0,
                    mask=SimpleNamespace(phase=True),
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="non-training"):
        tuple(iter_training_phase_observations(adapter))


def test_candidate_prior_uses_only_jointly_masked_training_labels() -> None:
    requested_video_ids: list[str] = []
    adapter = SimpleNamespace(
        entries={
            "VID01": SimpleNamespace(video_id="VID01", split=DatasetSplit.TRAINING),
            "VID30": SimpleNamespace(video_id="VID30", split=DatasetSplit.VALIDATION),
        }
    )

    def frame(frame_id: int, ivt_ids: tuple[int, ...], *, ivt_mask: bool = True):
        return SimpleNamespace(
            inference=SimpleNamespace(source_split=DatasetSplit.TRAINING),
            frame_supervision=SimpleNamespace(
                frame_id=frame_id,
                phase_id=0,
                triplet_ids=ivt_ids,
                mask=SimpleNamespace(phase=True, ivt=ivt_mask),
            ),
        )

    def iter_video(video_id: str):
        requested_video_ids.append(video_id)
        return iter(
            (
                frame(1, (94,)),
                frame(2, (17,)),
                frame(3, (94,)),
                frame(4, (7,), ivt_mask=False),
            )
        )

    adapter.iter_video = iter_video

    prior = build_phase_instrument_ivt_prior_from_training_adapter(adapter)

    assert prior[(0, 0)] == (94, 17)
    assert requested_video_ids == ["VID01"]
