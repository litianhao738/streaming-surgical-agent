"""Build paired T0/T1 video-grouped examples for the formal Benefit Gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.research.gate.oof_dataset import (
    build_paired_oof_examples,
    load_counterfactual_records,
    partition_oof_examples,
    write_oof_examples,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--counterfactuals", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--folds", type=int, default=3)
    result.add_argument(
        "--diagnostic-fold",
        "--calibration-fold",
        dest="diagnostic_fold",
        type=int,
        default=0,
    )
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    records = load_counterfactual_records(args.counterfactuals)
    examples = build_paired_oof_examples(records, fold_count=args.folds)
    training, diagnostic = partition_oof_examples(
        examples,
        calibration_fold=args.diagnostic_fold,
    )
    output = write_oof_examples(args.output, training)
    diagnostic_output = args.output.with_name(
        args.output.stem + ".diagnostic" + args.output.suffix
    )
    write_oof_examples(diagnostic_output, diagnostic)
    manifest = {
        "schema_version": "formal_gate_oof_dataset_manifest_v2",
        "source_split": "Training",
        "counterfactuals_sha256": sha256_file(args.counterfactuals),
        "example_count": len(examples),
        "fit_example_count": len(training),
        "training_internal_diagnostic_example_count": len(diagnostic),
        "observation_count": len(records),
        "video_ids": sorted({item.video_id for item in records}),
        "folds": args.folds,
        "training_internal_diagnostic_fold": args.diagnostic_fold,
        "fit_video_ids": sorted({item.video_id for item in training}),
        "training_internal_diagnostic_video_ids": sorted(
            {item.video_id for item in diagnostic}
        ),
        "paired_tracker_views": ["T0", "T1"],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "training_internal_diagnostic_output": str(diagnostic_output),
        "training_internal_diagnostic_output_sha256": sha256_file(
            diagnostic_output
        ),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    atomic_write_json(manifest_path, manifest)
    print("GATE_OOF_DATASET_COMPLETE " + json.dumps(manifest, sort_keys=True))
    return manifest


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
