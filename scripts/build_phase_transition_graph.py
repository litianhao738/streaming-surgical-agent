"""Build an immutable train-only phase transition graph from CholecTrack20."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.signals.phase_graph import (
    PhaseObservation,
    build_phase_transition_graph,
    write_phase_transition_graph,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--derived-manifest", type=Path)
    return parser


def _training_phase_observations(
    adapter: CholecTrack20DatasetAdapter,
) -> Iterator[PhaseObservation]:
    """Yield only available phase labels from official training videos."""

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
            yield PhaseObservation(
                video_id=target.video_id,
                frame_id=target.frame_id,
                phase_id=target.phase_id,
                split=DatasetSplit.TRAINING,
            )


def run(args: argparse.Namespace) -> Path:
    """Construct and write one graph using only the supplied dataset root."""

    adapter = CholecTrack20DatasetAdapter(
        args.dataset_root,
        derived_manifest_path=args.derived_manifest,
    )
    graph = build_phase_transition_graph(_training_phase_observations(adapter))
    return write_phase_transition_graph(graph, args.output)


def main() -> None:
    output = run(_parser().parse_args())
    print(f"Phase transition graph written: {output}")


if __name__ == "__main__":
    main()
