"""Run causal V3 inference with a rule Gate or an explicit learned Gate."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dataset_api_pipeline import (
    build_parser as build_dataset_parser,
)
from scripts.run_dataset_api_pipeline import (
    main as dataset_main,
)
from surgical_agent.systems.v3_streaming import (
    NO_TRAINING_V3_PROFILE,
    select_v3_profile,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_dataset_parser(fixed_profile=NO_TRAINING_V3_PROFILE)
    args = parser.parse_args(argv)
    args.pipeline_profile = select_v3_profile(args.gate_artifact)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    dataset_main(
        argv,
        fixed_profile=NO_TRAINING_V3_PROFILE,
        learned_when_artifact_present=True,
    )

if __name__ == "__main__":
    main()
