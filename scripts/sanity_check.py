"""Read-only P1/P2 dataset contract preflight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.config.loader import resolve_dataset_root
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    root = resolve_dataset_root(
        {"root": None, "root_env": "CHOLECTRACK20_ROOT"},
        cli_root=args.dataset_root,
    )
    adapter = CholecTrack20DatasetAdapter(root)
    split_counts = {
        split: sum(entry.split.value == split for entry in adapter.entries.values())
        for split in ("training", "validation", "testing")
    }
    print(f"DATA CONTRACT PASS: root={root} splits={split_counts}")

if __name__ == "__main__":
    main()
