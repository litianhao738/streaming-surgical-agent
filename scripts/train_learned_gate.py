"""Train both demo Learned Benefit Gate variants from frozen Training-only JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from surgical_agent.research.gate.learned import fit_learned_gate_pair, load_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training/learned_gate"),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = fit_learned_gate_pair(
        load_examples(args.examples),
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print("LEARNED_GATE_TRAINING_COMPLETE " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
