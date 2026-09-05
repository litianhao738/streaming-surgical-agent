"""Migrate stored tracker IDs to gap-reset association; no training or inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from surgical_agent.tracking.reassociation import reassociate_tracker_bundle


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--oof-index",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/tracker_oof5/oof/index.json",
    )
    result.add_argument(
        "--full-artifact",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/tracker/predicted_tracks.json",
    )
    result.add_argument(
        "--full-training-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/tracker_oof5/full",
    )
    result.add_argument(
        "--oof-config",
        type=Path,
        default=PROJECT_ROOT / "configs/tracker/fasterrcnn_mobilenet_v3_5090_oof5.yaml",
    )
    result.add_argument(
        "--full-config",
        type=Path,
        default=PROJECT_ROOT / "configs/tracker/fasterrcnn_mobilenet_v3_5090.yaml",
    )
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--max-frame-id-gap", type=int, default=25)
    return result


def main() -> None:
    args = parser().parse_args()
    result = reassociate_tracker_bundle(
        oof_index_path=args.oof_index,
        full_artifact_path=args.full_artifact,
        full_training_directory=args.full_training_directory,
        oof_config_path=args.oof_config,
        full_config_path=args.full_config,
        output_root=args.output_root,
        max_frame_id_gap=args.max_frame_id_gap,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_root": str(args.output_root.resolve()),
                "artifact_count": len(result["artifacts"]),
                "model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
