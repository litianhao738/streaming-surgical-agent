"""Verify the read-only CholecTrack20 single-root AutoDL bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.config.loader import load_yaml, resolve_dataset_root
from surgical_agent.data.portability import (
    assert_output_outside_dataset_root,
    verify_single_root,
)


def parser() -> argparse.ArgumentParser:
    """Build the intentionally thin read-only verifier command parser."""

    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path)
    result.add_argument(
        "--bundle",
        type=Path,
        default=PROJECT_ROOT / "configs/data/cholectrack20_autodl_bundle.yaml",
    )
    result.add_argument("--output", type=Path)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run verification and optionally write the relative-path-only report."""

    data_config = load_yaml(PROJECT_ROOT / "configs/data/cholectrack20.yaml")
    root = resolve_dataset_root(data_config, cli_root=args.dataset_root)
    report = verify_single_root(root, args.bundle)
    if args.output is not None:
        atomic_write_json(assert_output_outside_dataset_root(args.output, root), report)
    print("CholecTrack20 single-root portability: PASS")
    return report


def main() -> None:
    """Run the verifier command."""

    run(parser().parse_args())


if __name__ == "__main__":
    main()
