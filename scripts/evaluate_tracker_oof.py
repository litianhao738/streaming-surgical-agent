"""Audit and evaluate video-held-out Tracker predictions without GPU inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.config.loader import load_yaml, resolve_dataset_root
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.tracking.box_policy import BBoxPolicy
from surgical_agent.tracking.oof_evaluation import evaluate_tracker_oof


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path)
    result.add_argument(
        "--tracker-oof-index",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/tracker_oof5/oof/index.json",
    )
    result.add_argument(
        "--tracker-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs/tracker/fasterrcnn_mobilenet_v3_5090_oof5.yaml"
        ),
    )
    result.add_argument(
        "--output",
        type=Path,
        help="default: heldout_evaluation.json beside the OOF index",
    )
    result.add_argument(
        "--gt-bbox-policy",
        choices=[policy.value for policy in BBoxPolicy],
        default=BBoxPolicy.LEGACY_STRICT_V1.value,
        help="GT box handling; legacy default reproduces old denominators, v2 clips intersecting boxes",
    )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_config = load_yaml(PROJECT_ROOT / "configs/data/cholectrack20.yaml")
    dataset_root = resolve_dataset_root(dataset_config, cli_root=args.dataset_root)
    index_path = args.tracker_oof_index.expanduser().resolve()
    tracker_config_path = args.tracker_config.expanduser().resolve()
    output = (
        index_path.parent / (
            "heldout_evaluation.json"
            if args.gt_bbox_policy == BBoxPolicy.LEGACY_STRICT_V1.value
            else f"heldout_evaluation_{args.gt_bbox_policy}.json"
        )
        if args.output is None
        else args.output.expanduser().resolve()
    )
    output = output.resolve()
    if dataset_root == output or dataset_root in output.parents:
        raise ValueError("Tracker evaluation output must be outside the dataset root")

    print("TRACKER_OOF_EVALUATION_START", flush=True)
    adapter = CholecTrack20DatasetAdapter(dataset_root)
    report = evaluate_tracker_oof(
        adapter=adapter,
        index_path=index_path,
        tracker_config_path=tracker_config_path,
        gt_bbox_policy=args.gt_bbox_policy,
    )
    atomic_write_json(output, report)
    overall = report["overall_pooled"]
    if not isinstance(overall, dict) or not isinstance(overall.get("metrics"), dict):
        raise TypeError("Tracker OOF evaluator returned an invalid report")
    summary = {
        "gt_bbox_policy": report["metric_protocol"]["gt_bbox_policy"],
        "gt_box_audit": overall["gt_box_audit"],
        "output": str(output),
        "scored_videos": report["coverage"]["scored_video_count"],
        "scored_frames": overall["scored_frame_count"],
        "macro_ap50": overall["metrics"]["macro_ap50"],
        "precision": overall["metrics"]["precision"],
        "recall": overall["metrics"]["recall"],
        "f1": overall["metrics"]["f1"],
    }
    print("TRACKER_OOF_EVALUATION_COMPLETE " + json.dumps(summary, sort_keys=True))
    return report


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
