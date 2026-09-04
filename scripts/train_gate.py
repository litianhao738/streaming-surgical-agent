"""Train video-cross-fitted bootstrap G0 models from paired D0 examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.research.gate.oof_dataset import load_oof_examples
from surgical_agent.training.gate_trainer import fit_cross_fitted_bootstrap_gates


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--examples", type=Path, required=True)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/gate/formal",
    )
    result.add_argument("--seed", type=int, default=3407)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    summary = dict(
        fit_cross_fitted_bootstrap_gates(
            load_oof_examples(args.examples),
            output_dir=args.output_dir,
            seed=args.seed,
        )
    )
    print("FORMAL_GATE_TRAINING_COMPLETE " + json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
