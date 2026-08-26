"""Run the evidence-gated P2 local baseline from data to atomic artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import yaml

from surgical_agent.artifacts.manifest import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    sha256_mapping,
    sha256_source_tree,
)
from surgical_agent.artifacts.provenance import capture_runtime_provenance
from surgical_agent.config.loader import load_experiment_config
from surgical_agent.data.dataset import (
    CholecTrack20DatasetAdapter,
    CurrentFrameTensorDataset,
    ResolvedSample,
    collate_smoke_batch,
)
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.inference.frame_result_writer import FrameResultWriter
from surgical_agent.models.baseline import LocalSmokeModel
from surgical_agent.runtime.device import resolve_device
from surgical_agent.runtime.seed import seed_everything
from surgical_agent.systems.baseline_system import P2BaselineSystem
from surgical_agent.training.checkpoint import load_checkpoint, save_checkpoint
from surgical_agent.training.trainer import LocalSmokeTrainer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/p2",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--skip-full-granularity-audit", action="store_true")
    return parser


def _sample_id(record: ResolvedSample) -> str:
    return f"{record.inference.video_id}:{record.inference.target_frame_id}"


def _mask_summary(records: tuple[ResolvedSample, ...]) -> dict[str, Any]:
    counts = Counter()
    routes = Counter(record.provenance.route for record in records)
    for record in records:
        target = record.frame_supervision
        for task in ("instrument", "verb", "target", "ivt", "phase"):
            counts[task] += int(target is not None and getattr(target.mask, task))
    return {
        "schema_version": "p2_sample_mask_summary_v1",
        "sample_ids": [_sample_id(record) for record in records],
        "routes": dict(sorted(routes.items())),
        "valid_frame_task_counts": dict(sorted(counts.items())),
    }


def _assert_checkpoint_equal(first: LocalSmokeModel, second: LocalSmokeModel) -> None:
    second_state = second.state_dict()
    for name, value in first.state_dict().items():
        if not torch.equal(value.detach().cpu(), second_state[name].detach().cpu()):
            raise AssertionError(f"Checkpoint round-trip changed parameter {name}")


def _render_report(
    *,
    run_id: str,
    train_result: Any,
    metric_summary: dict[str, object],
    output_dir: Path,
    full_audit: bool,
) -> str:
    metrics = metric_summary["metrics"]
    return f"""# P2 Local Smoke Report

- Status: PASS
- Run ID: `{run_id}`
- Scope: engineering smoke only; no paper-performance claim
- Real optimizer step: PASS, finite loss `{train_result.total_loss:.6f}`
- Finite gradients: `{train_result.gradients_finite}`
- Checkpoint round-trip: PASS
- Canonical pipeline: PASS
- Test used for training/selection: NO
- Full per-video granularity audit: `{full_audit}`

## Task support

```json
{json.dumps(train_result.valid_counts, sort_keys=True)}
```

## Engineering smoke metrics

```json
{json.dumps(metrics, sort_keys=True)}
```

Artifacts: `{output_dir}`

P3 remains out of scope. Exact API provider, endpoint, and provider-returned model
identifier must be frozen before API implementation starts. P4 paper evaluation
granularity remains deferred to the target-granularity review.
"""


def run(args: argparse.Namespace) -> Path:
    resolved = load_experiment_config(
        args.config,
        dataset_root=args.dataset_root,
    )
    project = resolved.get("project", {})
    if project.get("current_phase") != "P2":
        raise RuntimeError("Resolved config is not authorized for P2")
    seed = int(project.get("seed", 42))
    deterministic = bool(resolved.get("runtime", {}).get("deterministic", True))
    requested_device = args.device or str(
        resolved.get("runtime", {}).get("device", "auto")
    )
    device = resolve_device(requested_device)
    seed_everything(seed, deterministic=deterministic)
    data = resolved["data"]
    dataset_root = Path(data["root"])
    manifest_value = data.get("derived_supervision_manifest", "repair_manifest.json")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = dataset_root / manifest_path
    pipeline_config = resolved.get("pipeline", {})
    if not isinstance(pipeline_config, dict):
        raise TypeError("Resolved pipeline config must be a mapping")
    max_causal_frames = pipeline_config.get("max_causal_frames", 3)
    if (
        not isinstance(max_causal_frames, int)
        or isinstance(max_causal_frames, bool)
        or not 1 <= max_causal_frames <= 3
    ):
        raise RuntimeError("pipeline.max_causal_frames must be an integer in 1..3")
    adapter = CholecTrack20DatasetAdapter(
        dataset_root,
        derived_manifest_path=manifest_path,
        causal_window_size=max_causal_frames,
    )

    config_hash = sha256_mapping(resolved)
    source_tree_hash = sha256_source_tree(PROJECT_ROOT)
    run_id = args.run_id or (
        datetime.now(UTC).strftime("p2_%Y%m%dT%H%M%SZ_") + config_hash[:8]
    )
    output_dir = args.output_root.expanduser().resolve() / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_text(
        output_dir / "resolved_config.yaml",
        yaml.safe_dump(resolved, sort_keys=True, allow_unicode=False),
    )

    smoke = resolved.get("smoke", {})
    train_videos = tuple(smoke.get("required_training_videos", ("VID02", "VID31")))
    validation_videos = tuple(
        smoke.get("required_validation_videos", ("VID30",))
    )
    train_records = adapter.collect(train_videos, samples_per_video=1)
    if any(
        record.inference.source_split is not DatasetSplit.TRAINING
        for record in train_records
    ):
        raise RuntimeError("Non-training split entered the optimizer subset")
    tensor_dataset = CurrentFrameTensorDataset(
        train_records,
        image_size=args.image_size,
    )
    train_batch = collate_smoke_batch(
        [tensor_dataset[index] for index in range(len(tensor_dataset))]
    )
    model = LocalSmokeModel(pretrained=False).to(device)
    learning_rate = float(
        resolved.get("train", {})
        .get("optimization", {})
        .get("learning_rate", 1e-4)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_result = LocalSmokeTrainer(
        model,
        optimizer,
        device=device,
    ).train_step(train_batch)

    checkpoint_path = output_dir / "checkpoints/local_smoke.pt"
    checkpoint_metadata = {
        "config_sha256": config_hash,
        "source_tree_sha256": source_tree_hash,
        "repair_manifest_sha256": sha256_file(manifest_path),
        "seed": seed,
        "device": str(device),
        "dtype": str(train_batch.images.dtype),
        "batch_sample_ids": list(train_result.batch_sample_ids),
        "loss_semantics": "SMOKE_ONLY_NOT_GATE_ERROR",
    }
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        step=1,
        metadata=checkpoint_metadata,
    )
    reloaded_model = LocalSmokeModel(pretrained=False).to(device)
    reloaded_optimizer = torch.optim.Adam(
        reloaded_model.parameters(),
        lr=learning_rate,
    )
    loaded = load_checkpoint(
        checkpoint_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        map_location=device,
    )
    if loaded.step != 1 or loaded.metadata != checkpoint_metadata:
        raise AssertionError("Checkpoint metadata round-trip failed")
    _assert_checkpoint_equal(model, reloaded_model)

    validation_limit = int(smoke.get("max_validation_samples", 4))
    inference_records = (
        adapter.collect(train_videos, samples_per_video=1)
        + tuple(
            sample
            for video_id in validation_videos
            for sample in adapter.iter_video(
                video_id,
                max_samples=validation_limit,
            )
        )
    )
    writer = FrameResultWriter(output_dir / "inference", run_id=run_id)
    runtime_provenance = capture_runtime_provenance(
        device=device,
        seed=seed,
        deterministic=deterministic,
    )
    baseline = P2BaselineSystem(
        reloaded_model,
        device=device,
        writer=writer,
        image_size=args.image_size,
    )
    run_result = baseline.run(
        inference_records,
        run_id=run_id,
        manifest_metadata={
            "paper_metric_eligible": False,
        },
    )
    atomic_write_json(output_dir / "smoke_metrics.json", run_result.metric_summary)
    atomic_write_json(
        output_dir / "sample_mask_summary.json",
        _mask_summary(inference_records),
    )
    atomic_write_json(
        output_dir / "runtime_provenance.json",
        {
            **runtime_provenance,
            "config_sha256": config_hash,
            "source_tree_sha256": source_tree_hash,
            "repair_manifest_sha256": sha256_file(manifest_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "train_step": {
                "total_loss": train_result.total_loss,
                "task_losses": train_result.task_losses,
                "valid_counts": train_result.valid_counts,
                "batch_sample_ids": train_result.batch_sample_ids,
                "gradients_finite": train_result.gradients_finite,
            },
            "test_training_or_selection_access": False,
        },
    )
    audited_videos = (
        None
        if not args.skip_full_granularity_audit
        else (*train_videos, *validation_videos)
    )
    granularity_audit = adapter.target_granularity_audit(audited_videos)
    atomic_write_json(
        output_dir / "p2_target_granularity_audit.json",
        granularity_audit,
    )
    report = _render_report(
        run_id=run_id,
        train_result=train_result,
        metric_summary=run_result.metric_summary,
        output_dir=output_dir,
        full_audit=not args.skip_full_granularity_audit,
    )
    atomic_write_text(output_dir / "P2_REPORT.md", report)
    print(f"P2 PASS: {output_dir}")
    return output_dir


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
