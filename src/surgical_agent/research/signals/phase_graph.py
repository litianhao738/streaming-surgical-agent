"""Deterministic train-only phase transition graph artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit, FrameSupervisionTarget
from surgical_agent.evaluation.frame_ground_truth import aggregate_evaluation_target
from surgical_agent.research.signals.contracts import PhaseTransitionGraph
from surgical_agent.research.signals.frame_evidence import load_ivt_components

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


def iter_training_phase_observations(
    adapter: CholecTrack20DatasetAdapter,
) -> Iterator[PhaseObservation]:
    """Re-enumerate only official Training phase supervision from an adapter."""

    entries = sorted(adapter.entries.values(), key=lambda entry: entry.video_id)
    for entry in entries:
        if entry.split is not DatasetSplit.TRAINING:
            continue
        for record in adapter.iter_video(entry.video_id):
            if record.inference.source_split is not DatasetSplit.TRAINING:
                raise ValueError("training graph adapter yielded a non-training record")
            target = record.frame_supervision
            if target is None or not target.mask.phase or target.phase_id is None:
                continue
            if target.video_id != entry.video_id:
                raise ValueError("training graph adapter yielded a cross-video record")
            yield PhaseObservation(
                video_id=entry.video_id,
                frame_id=target.frame_id,
                phase_id=target.phase_id,
                split=DatasetSplit.TRAINING,
            )


def build_phase_transition_graph_from_training_adapter(
    adapter: CholecTrack20DatasetAdapter,
) -> PhaseTransitionGraph:
    """Build the expected graph by re-enumerating official Training labels."""

    return build_phase_transition_graph(iter_training_phase_observations(adapter))


def iter_training_phase_ivt_targets(
    adapter: CholecTrack20DatasetAdapter,
    *,
    include_instance_supervision: bool = True,
) -> Iterator[FrameSupervisionTarget]:
    """Use both valid supervision routes once per Training timestamp.

    Raw instance-supervised videos deliberately expose phase-only frame targets.
    A false IVT mask on that convenience target does not mean their separate
    instance annotation lacks IVT. Aggregate the latter with all-instance masks;
    never replace an explicitly masked field with an inferred negative.
    """
    entries = sorted(adapter.entries.values(), key=lambda entry: entry.video_id)
    for entry in entries:
        if entry.split is not DatasetSplit.TRAINING:
            continue
        for record in adapter.iter_video(entry.video_id):
            if record.inference.source_split is not DatasetSplit.TRAINING:
                raise ValueError("phase-IVT builder received a non-training record")
            target = record.frame_supervision
            if target is not None and target.mask.phase and target.mask.ivt:
                yield target
                continue
            evaluation = getattr(record, "evaluation", None)
            if (
                not include_instance_supervision
                or evaluation is None
                or not evaluation.instance_supervision_available
            ):
                continue
            aggregated = aggregate_evaluation_target(
                evaluation, source="training_phase_ivt_prior_v2"
            )
            if aggregated.mask.phase and aggregated.mask.ivt:
                yield aggregated


def build_phase_ivt_compatibility_from_training_adapter(
    adapter: CholecTrack20DatasetAdapter,
    *,
    include_instance_supervision: bool = True,
) -> Mapping[int, tuple[int, ...]]:
    """Build phase-to-IVT support using official Training supervision only."""

    allowed: dict[int, set[int]] = defaultdict(set)
    for target in iter_training_phase_ivt_targets(
        adapter,
        include_instance_supervision=include_instance_supervision,
    ):
        if (
            target is None
            or not target.mask.phase
            or target.phase_id is None
            or not target.mask.ivt
        ):
            continue
        allowed[target.phase_id].update(target.triplet_ids)
    phase_lower, phase_upper = TASK_ID_BOUNDS["phase"]
    missing = set(range(phase_lower, phase_upper + 1)) - set(allowed)
    if missing:
        raise ValueError(
            f"Training phase-IVT support is missing phases: {sorted(missing)}"
        )
    return MappingProxyType(
        {phase_id: tuple(sorted(allowed[phase_id])) for phase_id in sorted(allowed)}
    )


def build_phase_instrument_ivt_prior_from_training_adapter(
    adapter: CholecTrack20DatasetAdapter,
    *,
    per_pair_limit: int = 4,
    include_instance_supervision: bool = True,
) -> Mapping[tuple[int, int], tuple[int, ...]]:
    """Rank IVTs for each phase/instrument pair using masked Training GT only.

    A frame contributes only when both Phase and IVT supervision are valid.  A
    missing task is therefore ignored rather than interpreted as a negative.
    Instrument IDs are derived from the valid IVT ontology tuple, so this does
    not require an independently valid instrument mask.
    """

    if (
        not isinstance(per_pair_limit, int)
        or isinstance(per_pair_limit, bool)
        or per_pair_limit <= 0
    ):
        raise ValueError("per_pair_limit must be a positive integer")
    counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    components = load_ivt_components()
    for target in iter_training_phase_ivt_targets(
        adapter,
        include_instance_supervision=include_instance_supervision,
    ):
        if (
            target is None
            or not target.mask.phase
            or target.phase_id is None
            or not target.mask.ivt
        ):
            continue
        for ivt_id in target.triplet_ids:
            instrument_id, _verb_id, _target_id = components[ivt_id]
            counts[(target.phase_id, instrument_id)][ivt_id] += 1
    if not counts:
        raise ValueError("Training contains no jointly valid phase-IVT supervision")
    return MappingProxyType(
        {
            key: tuple(
                ivt_id
                for ivt_id, _count in sorted(
                    counter.items(), key=lambda item: (-item[1], item[0])
                )[:per_pair_limit]
            )
            for key, counter in sorted(counts.items())
        }
    )


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
            raise ValueError(
                "phase transition graphs accept training observations only"
            )
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
        raise ValueError(
            f"invalid phase transition graph artifact: {artifact_path}"
        ) from exc
    artifact = _require_artifact_fields(raw)
    transitions_raw = artifact["transitions"]
    source_video_ids_raw = artifact["source_video_ids"]
    if not isinstance(transitions_raw, list) or not isinstance(
        source_video_ids_raw, list
    ):
        raise TypeError("phase transition graph artifact has invalid graph collections")
    if any(
        not isinstance(edge, list)
        or len(edge) != 2
        or any(
            not isinstance(phase_id, int) or isinstance(phase_id, bool)
            for phase_id in edge
        )
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
