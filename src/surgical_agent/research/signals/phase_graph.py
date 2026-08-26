"""Deterministic train-only phase transition graph artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.signals.contracts import PhaseTransitionGraph

GRAPH_SCHEMA_VERSION = "phase_transition_graph_v1"
GRAPH_VERSION = "phase_transition_train_v1"


@dataclass(frozen=True)
class PhaseObservation:
    """One phase-supervised frame eligible for graph construction."""

    video_id: str
    frame_id: int
    phase_id: int
    split: DatasetSplit

    def __post_init__(self) -> None:
        if (
            not isinstance(self.video_id, str)
            or not self.video_id
            or self.video_id != self.video_id.strip()
        ):
            raise ValueError("video_id must be a stripped non-empty string")
        if (
            not isinstance(self.frame_id, int)
            or isinstance(self.frame_id, bool)
            or self.frame_id < 0
        ):
            raise ValueError("frame_id must be a non-negative integer")
        lower, upper = TASK_ID_BOUNDS["phase"]
        if (
            not isinstance(self.phase_id, int)
            or isinstance(self.phase_id, bool)
            or not lower <= self.phase_id <= upper
        ):
            raise ValueError("phase_id is outside the phase range")
        if not isinstance(self.split, DatasetSplit):
            raise TypeError("split must be a DatasetSplit")


def canonical_graph_payload(
    transitions: tuple[tuple[int, int], ...],
    source_video_ids: tuple[str, ...],
) -> dict[str, object]:
    """Return the hashable, schema-versioned graph payload without its digest."""

    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "version": GRAPH_VERSION,
        "source_video_ids": list(source_video_ids),
        "transitions": [list(edge) for edge in transitions],
    }


def _canonical_json(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _payload_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def build_phase_transition_graph(
    observations: Iterable[PhaseObservation],
) -> PhaseTransitionGraph:
    """Build a sorted graph solely from strictly ordered training observations."""

    observations_by_video: dict[str, list[PhaseObservation]] = defaultdict(list)
    for observation in observations:
        if not isinstance(observation, PhaseObservation):
            raise TypeError("observations must be PhaseObservation instances")
        if observation.split is not DatasetSplit.TRAINING:
            raise ValueError("phase transition graphs accept training observations only")
        observations_by_video[observation.video_id].append(observation)
    if not observations_by_video:
        raise ValueError("at least one eligible training observation is required")

    transitions: set[tuple[int, int]] = set()
    for video_id in sorted(observations_by_video):
        previous: PhaseObservation | None = None
        for observation in observations_by_video[video_id]:
            if previous is not None:
                if observation.frame_id <= previous.frame_id:
                    raise ValueError(
                        f"frame IDs for {video_id} must be strictly increasing"
                    )
                transitions.add((previous.phase_id, observation.phase_id))
            transitions.add((observation.phase_id, observation.phase_id))
            previous = observation

    sorted_transitions = tuple(sorted(transitions))
    source_video_ids = tuple(sorted(observations_by_video))
    payload = canonical_graph_payload(sorted_transitions, source_video_ids)
    return PhaseTransitionGraph(
        transitions=sorted_transitions,
        source_video_ids=source_video_ids,
        version=GRAPH_VERSION,
        sha256=_payload_sha256(payload),
    )


def write_phase_transition_graph(
    graph: PhaseTransitionGraph,
    path: str | Path,
) -> Path:
    """Atomically persist a graph whose digest matches its canonical payload."""

    if not isinstance(graph, PhaseTransitionGraph):
        raise TypeError("graph must be a PhaseTransitionGraph")
    payload = canonical_graph_payload(graph.transitions, graph.source_video_ids)
    if graph.version != GRAPH_VERSION:
        raise ValueError(f"graph version must be {GRAPH_VERSION}")
    if graph.sha256 != _payload_sha256(payload):
        raise ValueError("graph sha256 does not match its canonical payload")

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {**payload, "sha256": graph.sha256}
    encoded = json.dumps(artifact, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
    try:
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _require_artifact_fields(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("phase transition graph artifact must be a JSON object")
    required = {
        "schema_version",
        "version",
        "source_video_ids",
        "transitions",
        "sha256",
    }
    if set(raw) != required:
        raise ValueError("phase transition graph artifact has unexpected fields")
    if raw["schema_version"] != GRAPH_SCHEMA_VERSION:
        raise ValueError(f"unsupported graph schema version: {raw['schema_version']!r}")
    if raw["version"] != GRAPH_VERSION:
        raise ValueError(f"unsupported graph version: {raw['version']!r}")
    return raw


def load_phase_transition_graph(path: str | Path) -> PhaseTransitionGraph:
    """Load a graph and reject non-canonical or digest-tampered artifacts."""

    artifact_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid phase transition graph artifact: {artifact_path}") from exc
    artifact = _require_artifact_fields(raw)
    transitions_raw = artifact["transitions"]
    source_video_ids_raw = artifact["source_video_ids"]
    if not isinstance(transitions_raw, list) or not isinstance(source_video_ids_raw, list):
        raise TypeError("phase transition graph artifact has invalid graph collections")
    if any(
        not isinstance(edge, list)
        or len(edge) != 2
        or any(not isinstance(phase_id, int) or isinstance(phase_id, bool) for phase_id in edge)
        for edge in transitions_raw
    ):
        raise ValueError("phase transition graph artifact has invalid transitions")
    if any(not isinstance(video_id, str) for video_id in source_video_ids_raw):
        raise ValueError("phase transition graph artifact has invalid source video IDs")
    if not isinstance(artifact["sha256"], str):
        raise TypeError("phase transition graph artifact has invalid sha256")

    graph = PhaseTransitionGraph(
        transitions=tuple((edge[0], edge[1]) for edge in transitions_raw),
        source_video_ids=tuple(source_video_ids_raw),
        version=artifact["version"],
        sha256=artifact["sha256"],
    )
    expected_sha256 = _payload_sha256(
        canonical_graph_payload(graph.transitions, graph.source_video_ids)
    )
    if graph.sha256 != expected_sha256:
        raise ValueError("phase transition graph sha256 does not match its payload")
    return graph
