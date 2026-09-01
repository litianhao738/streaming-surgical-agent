"""Build an immutable train-only phase transition graph from CholecTrack20."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.signals.phase_graph import (
    build_phase_transition_graph_from_training_adapter,
    write_phase_transition_graph,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--derived-manifest", type=Path)
    return parser


def run(args: argparse.Namespace) -> Path:
    """Construct and write one graph using only the supplied dataset root."""

    adapter = CholecTrack20DatasetAdapter(
        args.dataset_root,
        derived_manifest_path=args.derived_manifest,
    )
    graph = build_phase_transition_graph_from_training_adapter(adapter)
    return write_phase_transition_graph(graph, args.output)


def main() -> None:
    output = run(_parser().parse_args())
    print(f"Phase transition graph written: {output}")


if __name__ == "__main__":
    main()
