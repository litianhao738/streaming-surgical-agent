"""Train the causal five-task local Joint Perception model."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.cli.progress import progress_bar
from surgical_agent.config.loader import load_yaml, resolve_dataset_root
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.models.perception import (
    CausalJointPerceptionModel,
    JointPerceptionModelConfig,
)
from surgical_agent.perception.local_joint import checkpoint_payload
from surgical_agent.training.perception_data import (
    CausalFrameTensorDataset,
    build_perception_records,
    collate_perception_batch,
    compute_class_weights,
)
from surgical_agent.training.perception_trainer import (
    JointPerceptionTrainer,
    ValidationAccumulator,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    result.add_argument("--dataset-root", type=Path)
    result.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/perception/local_joint_mobilenet_v3.yaml",
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/perception",
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--epochs", type=int)
    result.add_argument("--max-train-batches", type=int)
    result.add_argument("--max-validation-batches", type=int)
    result.add_argument("--no-pretrained", action="store_true")
    result.add_argument("--no-progress", action="store_true")
    return result


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value: str) -> torch.device:
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return selected


def _positive_int(mapping: dict[str, Any], name: str) -> int:
    value = mapping.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate(
    model: CausalJointPerceptionModel,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    threshold_grid: tuple[float, ...],
    max_batches: int | None,
    progress_enabled: bool,
) -> dict[str, object]:
    model.eval()
    accumulator = ValidationAccumulator()
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    with torch.inference_mode(), progress_bar(
        total=total,
        description="Joint perception validation",
        unit="batch",
        enabled=progress_enabled,
    ) as progress:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            logits = model(batch.frames.to(device, non_blocking=True))
            accumulator.update(logits, batch.targets)
            progress.update()
    return accumulator.compute(threshold_grid=threshold_grid)


def run(args: argparse.Namespace) -> Path:
    raw = load_yaml(args.config)
    data_config = load_yaml(PROJECT_ROOT / "configs/data/cholectrack20.yaml")
    dataset_root = resolve_dataset_root(data_config, cli_root=args.dataset_root)
    seed = int(raw.get("seed", 42))
    _seed(seed)
    device = _device(args.device)

    window_size = _positive_int(raw, "window_size")
    image_size = _positive_int(raw, "image_size")
    if window_size > 6:
        raise ValueError("window_size must not exceed six causal frames")
    model_section = raw.get("model")
    optimization = raw.get("optimization")
    if not isinstance(model_section, dict) or not isinstance(optimization, dict):
        raise TypeError("perception config requires model and optimization mappings")
    full_epochs = _positive_int(optimization, "epochs")
    full_batch_size = _positive_int(optimization, "batch_size")
    full_workers = int(optimization.get("num_workers", 0))
    if full_workers < 0:
        raise ValueError("num_workers must be non-negative")
    learning_rate = float(optimization.get("learning_rate", 3e-4))
    weight_decay = float(optimization.get("weight_decay", 1e-4))
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate or weight_decay is invalid")
    epochs = args.epochs or (1 if args.mode == "smoke" else full_epochs)
    batch_size = 2 if args.mode == "smoke" else full_batch_size
    num_workers = 0 if args.mode == "smoke" else full_workers
    max_samples = (
        int(raw.get("smoke_samples_per_video", 4))
        if args.mode == "smoke"
        else None
    )
    max_train_batches = args.max_train_batches
    max_validation_batches = args.max_validation_batches
    if args.mode == "smoke":
        max_train_batches = 1 if max_train_batches is None else max_train_batches
        max_validation_batches = (
            1 if max_validation_batches is None else max_validation_batches
        )
    use_pretrained = args.mode == "full" and not args.no_pretrained

    adapter = CholecTrack20DatasetAdapter(
        dataset_root,
        derived_manifest_path=dataset_root / "repair_manifest.json",
        causal_window_size=window_size,
    )
    training_videos = tuple(
        sorted(
            video_id
            for video_id, entry in adapter.entries.items()
            if entry.split is DatasetSplit.TRAINING
        )
    )
    validation_videos = tuple(
        sorted(
            video_id
            for video_id, entry in adapter.entries.items()
            if entry.split is DatasetSplit.VALIDATION
        )
    )
    training_records = build_perception_records(
        adapter,
        training_videos,
        allowed_split=DatasetSplit.TRAINING,
        max_samples_per_video=max_samples,
    )
    validation_records = build_perception_records(
        adapter,
        validation_videos,
        allowed_split=DatasetSplit.VALIDATION,
        max_samples_per_video=max_samples,
    )
    class_weight_values = compute_class_weights(
        training_records,
        max_positive_weight=float(optimization.get("max_positive_weight", 20.0)),
    )
    class_weights = {
        task: torch.tensor(values, dtype=torch.float32)
        for task, values in class_weight_values.items()
    }
    train_dataset = CausalFrameTensorDataset(
        training_records,
        image_size=image_size,
        window_size=window_size,
    )
    validation_dataset = CausalFrameTensorDataset(
        validation_records,
        image_size=image_size,
        window_size=window_size,
    )
    generator = torch.Generator().manual_seed(seed)
    common_loader = {
        "num_workers": num_workers,
        "collate_fn": collate_perception_batch,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        **common_loader,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_loader,
    )

    model_config = JointPerceptionModelConfig(
        architecture=str(model_section.get("architecture", "mobilenet_v3_small")),
        visual_feature_dim=int(model_section.get("visual_feature_dim", 256)),
        temporal_hidden_dim=int(model_section.get("temporal_hidden_dim", 384)),
        temporal_layers=int(model_section.get("temporal_layers", 1)),
        dropout=float(model_section.get("dropout", 0.2)),
        pretrained=use_pretrained,
    )
    model = CausalJointPerceptionModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    task_weights_value = raw.get("task_weights", {})
    if not isinstance(task_weights_value, dict):
        raise TypeError("task_weights must be a mapping")
    task_weights = {
        task: float(value) for task, value in task_weights_value.items()
    }
    trainer = JointPerceptionTrainer(
        model,
        optimizer,
        device=device,
        class_weights=class_weights,
        task_weights=task_weights,
        gradient_clip_norm=float(optimization.get("gradient_clip_norm", 5.0)),
    )
    threshold_grid = tuple(
        float(value)
        for value in raw.get("threshold_grid", (0.2, 0.3, 0.4, 0.5, 0.6))
    )
    if not threshold_grid or any(
        not 0.0 <= value <= 1.0 for value in threshold_grid
    ):
        raise ValueError("threshold_grid must contain probabilities")

    output_dir = args.output_root.expanduser().resolve() / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    history: list[dict[str, object]] = []
    best_score = -1.0
    best_validation: dict[str, object] | None = None
    total_steps = 0
    started = time.perf_counter()
    train_batches = len(train_loader)
    if max_train_batches is not None:
        train_batches = min(train_batches, max_train_batches)
    for epoch in range(epochs):
        with progress_bar(
            total=train_batches,
            description=f"Joint perception train {epoch + 1}/{epochs}",
            unit="batch",
            enabled=not args.no_progress,
        ) as progress:
            train_result = trainer.train_epoch(
                train_loader,
                max_batches=max_train_batches,
                progress=progress,
            )
        total_steps += train_result.optimizer_steps
        validation = _validate(
            model,
            validation_loader,
            device=device,
            threshold_grid=threshold_grid,
            max_batches=max_validation_batches,
            progress_enabled=not args.no_progress,
        )
        metrics = validation["metrics"]
        if not isinstance(metrics, dict):
            raise TypeError("validation metrics are invalid")
        score = float(metrics["five_task_macro"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_mean_loss": train_result.mean_loss,
                "train_task_losses": dict(train_result.task_losses),
                "validation": validation,
            }
        )
        if score > best_score:
            best_score = score
            best_validation = validation
            metadata = {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_split": "training",
                "training_video_ids": list(training_videos),
                "validation_video_ids": list(validation_videos),
                "testing_accessed": False,
                "repair_manifest_sha256": sha256_file(
                    dataset_root / "repair_manifest.json"
                ),
                "config_sha256": sha256_file(args.config),
                "best_validation_macro": best_score,
            }
            thresholds = validation["thresholds"]
            if not isinstance(thresholds, dict):
                raise TypeError("validation thresholds are invalid")
            _atomic_torch_save(
                checkpoint_path,
                checkpoint_payload(
                    model=model,
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch + 1,
                    optimizer_steps=total_steps,
                    thresholds=thresholds,
                    image_size=image_size,
                    window_size=window_size,
                    metadata=metadata,
                ),
            )
    assert best_validation is not None
    atomic_write_json(output_dir / "validation_metrics.json", best_validation)
    manifest = {
        "schema_version": "local_joint_perception_training_v1",
        "status": "COMPLETE",
        "mode": args.mode,
        "architecture": model_config.architecture,
        "pretrained_initialization": use_pretrained,
        "training_video_ids": list(training_videos),
        "validation_video_ids": list(validation_videos),
        "testing_accessed": False,
        "training_samples": len(training_records),
        "validation_samples": len(validation_records),
        "epochs": epochs,
        "optimizer_steps": total_steps,
        "best_validation_macro": best_score,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "duration_seconds": time.perf_counter() - started,
        "history": history,
    }
    atomic_write_json(output_dir / "training_manifest.json", manifest)
    print(
        f"PERCEPTION_{args.mode.upper()}_COMPLETE "
        + "summary="
        + json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "validation_metrics": str(output_dir / "validation_metrics.json"),
                "best_validation_macro": best_score,
            },
            sort_keys=True,
        )
    )
    return output_dir


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
