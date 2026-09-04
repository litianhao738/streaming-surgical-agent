"""Run a non-deployable threshold diagnostic on held-out Training videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.research.gate.calibration import calibrate_formal_gate
from surgical_agent.research.gate.oof_dataset import load_oof_examples


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--calibration-examples", type=Path, required=True)
    result.add_argument("--artifact", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--verification-cost", type=float, default=0.05)
    result.add_argument("--false-positive-harm", type=float, default=1.0)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    summary = dict(
        calibrate_formal_gate(
            load_oof_examples(args.calibration_examples),
            artifact_path=args.artifact,
            output_dir=args.output_dir,
            verification_cost=args.verification_cost,
            false_positive_harm=args.false_positive_harm,
        )
    )
    print("FORMAL_GATE_CALIBRATION_COMPLETE " + json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
