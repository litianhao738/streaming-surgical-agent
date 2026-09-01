"""Train and export the causal predicted instrument tracker on AutoDL."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

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
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.tracking.artifact_writer import write_predicted_track_artifact
from surgical_agent.tracking.associator import CausalHungarianAssociator
from surgical_agent.tracking.config import (
    TrackerTrainingConfig,
    load_tracker_training_config,
)
from surgical_agent.tracking.contracts import PredictedTrackFrame
from surgical_agent.tracking.detector import (
    build_instrument_detector,
    calculate_detection_metrics,
    decode_detections,
    load_tracker_checkpoint,
    save_tracker_checkpoint,
)
from surgical_agent.tracking.training_data import (
    InstrumentDetectionDataset,
    build_detection_training_records,
    detection_collate,
    deterministic_video_folds,
    supervision_qualified_training_video_ids,
)

PROVIDER_NAME = "torchvision_fasterrcnn_hungarian_v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("smoke", "full", "oof"), default="smoke")
    result.add_argument("--dataset-root", type=Path)
    result.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/tracker/fasterrcnn_mobilenet_v3.yaml",
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/tracker",
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--epochs", type=int)
    result.add_argument("--max-train-batches", type=int)
    result.add_argument("--max-prediction-frames", type=int)
    result.add_argument("--no-pretrained", action="store_true")
    result.add_argument(
        "--no-progress",
        action="store_true",
        help="disable dynamic progress bars",
    )
    return result


def _seed_everything(seed: int) -> None:
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


def _train(
    *,
    adapter: CholecTrack20DatasetAdapter,
    config: TrackerTrainingConfig,
    config_path: Path,
    output_dir: Path,
    training_video_ids: tuple[str, ...],
    excluded_video_ids: tuple[str, ...],
    device: torch.device,
    epochs: int,
    max_train_batches: int | None,
    max_records: int | None,
    use_pretrained: bool,
    mode: str,
    progress_enabled: bool = False,
    progress_file: IO[str] | None = None,
) -> tuple[torch.nn.Module, Path, dict[str, object]]:
    records = build_detection_training_records(
        adapter,
        training_video_ids,
        max_samples_per_video=64 if mode == "smoke" else None,
        progress_enabled=progress_enabled,
        progress_file=progress_file,
    )
    if max_records is not None:
        records = records[:max_records]
    dataset = InstrumentDetectionDataset(records)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers if mode != "smoke" else 0,
        collate_fn=detection_collate,
        generator=generator,
    )
    model = build_instrument_detector(config, use_pretrained=use_pretrained).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    batches_per_epoch = len(loader)
    if max_train_batches is not None:
        batches_per_epoch = min(batches_per_epoch, max_train_batches)
    with progress_bar(
        total=epochs * batches_per_epoch,
        description=f"Tracker {mode} train",
        unit="batch",
        enabled=progress_enabled,
        file=progress_file,
    ) as progress:
        for epoch in range(epochs):
            for batch_index, (images, targets) in enumerate(loader):
                if max_train_batches is not None and batch_index >= max_train_batches:
                    break
                images = [image.to(device) for image in images]
                targets = [
                    {name: value.to(device) for name, value in target.items()}
                    for target in targets
                ]
                loss_mapping = model(images, targets)
                loss = sum(loss_mapping.values())
                if not torch.isfinite(loss).item():
                    raise RuntimeError("tracker training produced a non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)
                progress.set_postfix(
                    epoch=f"{epoch + 1}/{epochs}",
                    loss=f"{loss_value:.4f}",
                    refresh=False,
                )
                progress.update()
    if not losses:
        raise RuntimeError("tracker training completed no optimizer steps")
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision_version = "UNAVAILABLE"
    metadata: dict[str, object] = {
        "architecture": config.architecture,
        "initial_weights": "COCO_V1" if use_pretrained else "NONE",
        "num_classes": config.num_classes,
        "training_video_ids": list(training_video_ids),
        "excluded_video_ids": list(excluded_video_ids),
        "dataset_repair_manifest_sha256": sha256_file(
            adapter.dataset_root / "repair_manifest.json"
        ),
        "tracker_config_sha256": sha256_file(config_path),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "seed": config.seed,
        "epochs_completed": epochs,
        "optimizer_steps": len(losses),
        "mode": mode,
    }
    checkpoint_path = save_tracker_checkpoint(
        output_dir / "checkpoint.pt", model=model, metadata=metadata
    )
    manifest = {
        "schema_version": "predicted_tracker_training_manifest_v1",
        **metadata,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_samples": len(records),
        "mean_training_loss": sum(losses) / len(losses),
        "last_training_loss": losses[-1],
        "duration_seconds": time.perf_counter() - started,
        "config": asdict(config),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_dir / "training_manifest.json", manifest)
    return model, checkpoint_path, manifest


def _predict_videos(
    *,
    model: torch.nn.Module,
    adapter: CholecTrack20DatasetAdapter,
    video_ids: tuple[str, ...],
    config: TrackerTrainingConfig,
    device: torch.device,
    max_frames: int | None,
    progress_enabled: bool = False,
    progress_file: IO[str] | None = None,
) -> tuple[
    dict[str, tuple[DatasetSplit, tuple[PredictedTrackFrame, ...]]],
    dict[tuple[str, int], list[tuple[int, tuple[float, float, float, float], float]]],
]:
    media_loader = CausalApiMediaLoader()
    artifact_videos: dict[
        str, tuple[DatasetSplit, tuple[PredictedTrackFrame, ...]]
    ] = {}
    raw_predictions: dict[
        tuple[str, int], list[tuple[int, tuple[float, float, float, float], float]]
    ] = {}
    samples_by_video: dict[str, tuple[object, ...]] = {}
    with progress_bar(
        total=len(video_ids),
        description="Tracker prediction setup",
        unit="video",
        enabled=progress_enabled,
        file=progress_file,
    ) as setup_progress:
        for video_id in video_ids:
            setup_progress.set_postfix_str(video_id, refresh=False)
            samples_by_video[video_id] = tuple(
                adapter.iter_inference_video(video_id, max_samples=max_frames)
            )
            setup_progress.update()
    total_frames = sum(len(samples) for samples in samples_by_video.values())
    model.eval()
    with torch.inference_mode(), progress_bar(
        total=total_frames,
        description="Tracker prediction",
        unit="frame",
        enabled=progress_enabled,
        file=progress_file,
    ) as progress:
        for video_id in video_ids:
            entry = adapter.entries[video_id]
            associator = CausalHungarianAssociator(
                iou_threshold=config.association_iou_threshold,
                max_age=config.max_age,
            )
            associator.reset(video_id)
            frames: list[PredictedTrackFrame] = []
            for sample in samples_by_video[video_id]:
                progress.set_postfix_str(
                    f"{video_id}:{sample.target_frame_id}",
                    refresh=False,
                )
                loaded = media_loader.load(sample)
                image = loaded.frames[-1].to(device)
                height, width = int(image.shape[-2]), int(image.shape[-1])
                output = model([image])[0]
                detections = decode_detections(
                    output,
                    width=width,
                    height=height,
                    score_threshold=config.score_threshold,
                )
                tracks = associator.update(sample.target_frame_id, detections)
                frames.append(PredictedTrackFrame(sample.target_frame_id, tracks))
                raw_predictions[(video_id, sample.target_frame_id)] = [
                    (item.instrument_id, item.bbox_tlwh, item.score)
                    for item in detections
                ]
                progress.update()
            if not frames:
                raise RuntimeError(f"tracker prediction produced no frames for {video_id}")
            artifact_videos[video_id] = (entry.split, tuple(frames))
    return artifact_videos, raw_predictions


def _validation_targets(
    adapter: CholecTrack20DatasetAdapter,
    video_ids: tuple[str, ...],
    *,
    progress_enabled: bool = False,
    progress_file: IO[str] | None = None,
) -> dict[tuple[str, int], list[tuple[int, tuple[float, float, float, float]]]]:
    targets: dict[
        tuple[str, int], list[tuple[int, tuple[float, float, float, float]]]
    ] = {}
    with progress_bar(
        total=len(video_ids),
        description="Tracker validation GT",
        unit="video",
        enabled=progress_enabled,
        file=progress_file,
    ) as progress:
        for video_id in video_ids:
            progress.set_postfix_str(video_id, refresh=False)
            for resolved in adapter.iter_video(video_id):
                evaluation = resolved.evaluation
                if evaluation is None:
                    continue
                targets[(video_id, resolved.inference.target_frame_id)] = [
                    (
                        instance.instrument_id,
                        (
                            instance.bbox.x,
                            instance.bbox.y,
                            instance.bbox.width,
                            instance.bbox.height,
                        ),
                    )
                    for instance in evaluation.instances
                    if instance.mask.instrument
                    and instance.bbox.has_positive_extent
                    and instance.bbox.is_inside_unit_frame
                ]
            progress.update()
    return targets


def _write_metrics(
    *,
    path: Path,
    predictions: dict[
        tuple[str, int], list[tuple[int, tuple[float, float, float, float], float]]
    ],
    targets: dict[
        tuple[str, int], list[tuple[int, tuple[float, float, float, float]]]
    ],
) -> None:
    keys = sorted(set(targets) & set(predictions))
    if not keys:
        raise RuntimeError("tracker metrics have no aligned prediction/GT frames")
    metrics = calculate_detection_metrics(
        [predictions.get(key, []) for key in keys],
        [targets[key] for key in keys],
        num_classes=7,
    )
    atomic_write_json(path, metrics)


def _source_model_identifier(config: TrackerTrainingConfig, manifest: dict[str, object]) -> str:
    return (
        f"{config.architecture}:{manifest['initial_weights']}:"
        f"ct20:{manifest['checkpoint_sha256']}"
    )


def _run_full_or_smoke(
    args: argparse.Namespace,
    *,
    adapter: CholecTrack20DatasetAdapter,
    config: TrackerTrainingConfig,
    config_path: Path,
    device: torch.device,
    qualified: tuple[str, ...],
    output_root: Path,
) -> dict[str, object]:
    smoke = args.mode == "smoke"
    output_dir = output_root / ("smoke" if smoke else "full")
    epochs = args.epochs or (1 if smoke else config.epochs)
    max_batches = args.max_train_batches
    if smoke and max_batches is None:
        max_batches = 1
    model, checkpoint, manifest = _train(
        adapter=adapter,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        training_video_ids=qualified[:1] if smoke else qualified,
        excluded_video_ids=(),
        device=device,
        epochs=epochs,
        max_train_batches=max_batches,
        max_records=2 if smoke else None,
        use_pretrained=not args.no_pretrained,
        mode=args.mode,
        progress_enabled=not args.no_progress,
    )
    if smoke:
        prediction_ids = tuple(
            video_id
            for video_id, entry in sorted(adapter.entries.items())
            if entry.split is DatasetSplit.VALIDATION
        )[:1]
        max_frames = args.max_prediction_frames or 2
    else:
        prediction_ids = tuple(
            video_id
            for video_id, entry in sorted(adapter.entries.items())
            if entry.split in {DatasetSplit.VALIDATION, DatasetSplit.TESTING}
            or video_id == "VID31"
        )
        max_frames = args.max_prediction_frames
    artifact_videos, raw_predictions = _predict_videos(
        model=model,
        adapter=adapter,
        video_ids=prediction_ids,
        config=config,
        device=device,
        max_frames=max_frames,
        progress_enabled=not args.no_progress,
    )
    artifact_path = output_dir / "predicted_tracks.json" if smoke else output_root / "predicted_tracks.json"
    write_predicted_track_artifact(
        artifact_path,
        provider=PROVIDER_NAME,
        source_model_identifier=_source_model_identifier(config, manifest),
        checkpoint_path=checkpoint,
        inference_config_path=config_path,
        repair_manifest_path=adapter.dataset_root / "repair_manifest.json",
        videos=artifact_videos,
    )
    validation_ids = tuple(
        video_id
        for video_id in prediction_ids
        if adapter.entries[video_id].split is DatasetSplit.VALIDATION
    )
    _write_metrics(
        path=output_dir / "validation_detection_metrics.json",
        predictions=raw_predictions,
        targets=_validation_targets(
            adapter,
            validation_ids,
            progress_enabled=not args.no_progress,
        ),
    )
    return {
        "mode": args.mode,
        "checkpoint": str(checkpoint),
        "predicted_track_artifact": str(artifact_path),
        "prediction_video_ids": list(prediction_ids),
    }


def _run_oof(
    args: argparse.Namespace,
    *,
    adapter: CholecTrack20DatasetAdapter,
    config: TrackerTrainingConfig,
    config_path: Path,
    device: torch.device,
    qualified: tuple[str, ...],
    output_root: Path,
) -> dict[str, object]:
    fold_outputs: list[dict[str, object]] = []
    folds = deterministic_video_folds(qualified, config.oof_folds)
    for fold_index, held_out in enumerate(folds):
        training_ids = tuple(video_id for video_id in qualified if video_id not in held_out)
        output_dir = output_root / "oof" / f"fold_{fold_index}"
        model, checkpoint, manifest = _train(
            adapter=adapter,
            config=config,
            config_path=config_path,
            output_dir=output_dir,
            training_video_ids=training_ids,
            excluded_video_ids=held_out,
            device=device,
            epochs=args.epochs or config.epochs,
            max_train_batches=args.max_train_batches,
            max_records=None,
            use_pretrained=not args.no_pretrained,
            mode="oof",
            progress_enabled=not args.no_progress,
        )
        videos, _ = _predict_videos(
            model=model,
            adapter=adapter,
            video_ids=held_out,
            config=config,
            device=device,
            max_frames=args.max_prediction_frames,
            progress_enabled=not args.no_progress,
        )
        artifact = write_predicted_track_artifact(
            output_dir / "predicted_tracks.json",
            provider=PROVIDER_NAME,
            source_model_identifier=_source_model_identifier(config, manifest),
            checkpoint_path=checkpoint,
            inference_config_path=config_path,
            repair_manifest_path=adapter.dataset_root / "repair_manifest.json",
            videos=videos,
        )
        fold_outputs.append(
            {
                "fold_index": fold_index,
                "training_video_ids": list(training_ids),
                "held_out_video_ids": list(held_out),
                "artifact": str(artifact),
            }
        )
    full_checkpoint = output_root / "full/checkpoint.pt"
    if not full_checkpoint.is_file():
        raise RuntimeError("OOF VID31 export requires the completed full checkpoint")
    full_model = build_instrument_detector(config, use_pretrained=False).to(device)
    full_metadata = load_tracker_checkpoint(
        full_checkpoint,
        model=full_model,
        map_location=device,
    )
    if "VID31" in full_metadata.get("training_video_ids", []):
        raise RuntimeError("full checkpoint illegally consumed VID31 instance supervision")
    vid31_videos, _ = _predict_videos(
        model=full_model,
        adapter=adapter,
        video_ids=("VID31",),
        config=config,
        device=device,
        max_frames=args.max_prediction_frames,
        progress_enabled=not args.no_progress,
    )
    vid31_artifact = write_predicted_track_artifact(
        output_root / "oof/vid31/predicted_tracks.json",
        provider=PROVIDER_NAME,
        source_model_identifier=(
            f"{config.architecture}:{full_metadata['initial_weights']}:ct20-full"
        ),
        checkpoint_path=full_checkpoint,
        inference_config_path=config_path,
        repair_manifest_path=adapter.dataset_root / "repair_manifest.json",
        videos=vid31_videos,
    )
    return {
        "mode": "oof",
        "folds": fold_outputs,
        "vid31_artifact": str(vid31_artifact),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = args.config.expanduser().resolve()
    config = load_tracker_training_config(config_path)
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config = replace(config, epochs=args.epochs)
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        raise ValueError("--max-train-batches must be positive")
    if args.max_prediction_frames is not None and args.max_prediction_frames <= 0:
        raise ValueError("--max-prediction-frames must be positive")
    dataset_config = load_yaml(PROJECT_ROOT / "configs/data/cholectrack20.yaml")
    dataset_root = resolve_dataset_root(dataset_config, cli_root=args.dataset_root)
    output_root = args.output_root.expanduser().resolve()
    if dataset_root == output_root or dataset_root in output_root.parents:
        raise ValueError("tracker outputs must be outside the dataset root")
    device = _device(args.device)
    _seed_everything(config.seed)
    adapter = CholecTrack20DatasetAdapter(dataset_root)
    qualified = supervision_qualified_training_video_ids(adapter)
    if not qualified:
        raise RuntimeError("no supervision-qualified tracker training videos")
    if args.mode in {"smoke", "full"}:
        summary = _run_full_or_smoke(
            args,
            adapter=adapter,
            config=config,
            config_path=config_path,
            device=device,
            qualified=qualified,
            output_root=output_root,
        )
    else:
        summary = _run_oof(
            args,
            adapter=adapter,
            config=config,
            config_path=config_path,
            device=device,
            qualified=qualified,
            output_root=output_root,
        )
    atomic_write_json(output_root / f"{args.mode}_run_summary.json", summary)
    print(
        "TRACKER_"
        + args.mode.upper()
        + "_COMPLETE summary="
        + json.dumps(summary, sort_keys=True)
    )
    return summary


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
