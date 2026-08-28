"""Evaluate one completed frame-prediction run against local authorized GT."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from surgical_agent.evaluation.offline_artifacts import OfflineEvaluationError
from surgical_agent.evaluation.offline_frame import evaluate_completed_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--authorize-test-gt-evaluation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_completed_run(
            args.run_dir,
            args.dataset_root,
            output_dir=args.output_dir,
            authorize_test_gt_evaluation=args.authorize_test_gt_evaluation,
        )
    except OfflineEvaluationError as error:
        print(f"offline evaluation failed: {error}", file=sys.stderr)
        return 2
    print(result.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
