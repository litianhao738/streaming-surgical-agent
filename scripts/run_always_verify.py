"""Run the canonical dataset API pipeline with Always Verify fixed on."""

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

_PROFILE = "always_verify"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_dataset_parser(fixed_profile=_PROFILE).parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    dataset_main(argv, fixed_profile=_PROFILE)

if __name__ == "__main__":
    main()
