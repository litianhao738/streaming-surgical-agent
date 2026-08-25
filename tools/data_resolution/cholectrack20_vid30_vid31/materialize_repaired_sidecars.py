"""Materialize reviewed VID30/VID31 sidecars without changing raw release files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = Path("D:/cholec_dataset")
DEFAULT_VID30_SOURCE = (
    PROJECT_ROOT
    / "artifacts/data_resolution/cholectrack20_vid30_vid31/derived/VID30/"
    "vid30_candidate_validation.json"
)
DEFAULT_VID31_PHASE_SOURCE = (
    PROJECT_ROOT
    / "artifacts/data_resolution/cholectrack20_vid30_vid31/derived/VID31/"
    "vid31_image_phase_supervision.json"
)
DEFAULT_VID31_IVT_SOURCE = (
    PROJECT_ROOT
    / "artifacts/data_resolution/cholect50_vid31/"
    "vid31_frame_level_ivt_supervision.json"
)

EXPECTED_VIDEO_DIRECTORIES = {
    "Training": (
        "VID02",
        "VID04",
        "VID103",
        "VID11",
        "VID13",
        "VID17",
        "VID23",
        "VID31",
        "VID37",
        "VID96",
    ),
    "Validation": ("VID110", "VID30"),
    "Testing": (
        "VID01",
        "VID06",
        "VID07",
        "VID111",
        "VID12",
        "VID25",
        "VID39",
        "VID92",
    ),
}


class MaterializationError(RuntimeError):
    """Raised when a sidecar cannot be installed without changing raw data."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError(f"Expected a JSON object: {path}")
    return value


def _validate_layout(dataset_root: Path) -> dict[str, list[str]]:
    observed: dict[str, list[str]] = {}
    for split, expected in EXPECTED_VIDEO_DIRECTORIES.items():
        split_root = dataset_root / split
        if not split_root.is_dir():
            raise MaterializationError(f"Missing official split directory: {split_root}")
        video_ids = sorted(path.name for path in split_root.iterdir() if path.is_dir())
        if video_ids != sorted(expected):
            raise MaterializationError(
                f"{split} video directories differ from the frozen 20-video release: "
                f"{video_ids}"
            )
        observed[split] = video_ids
    return observed


def _validate_sources(
    vid30_source: Path,
    vid31_phase_source: Path,
    vid31_ivt_source: Path,
) -> None:
    vid30 = _read_json(vid30_source)
    if len(vid30.get("annotations", {})) != 2717:
        raise MaterializationError("VID30 repaired candidate must contain 2,717 frames")
    video = vid30.get("video", {})
    if not isinstance(video, dict) or video.get("name", video.get("id")) != "VID30":
        raise MaterializationError("VID30 repaired candidate has the wrong video ID")

    vid31_phase = _read_json(vid31_phase_source)
    if vid31_phase.get("schema_version") != "ct20_vid30_vid31_derived_supervision_v1":
        raise MaterializationError("VID31 phase sidecar has an unsupported schema")
    if len(vid31_phase.get("phase_by_track20_image_frame_id", {})) != 3732:
        raise MaterializationError("VID31 phase sidecar must contain 3,732 aligned frames")

    vid31_ivt = _read_json(vid31_ivt_source)
    if vid31_ivt.get("schema_version") != "ct20_vid31_cholect50_frame_ivt_v1":
        raise MaterializationError("VID31 frame-level IVT sidecar has an unsupported schema")
    if vid31_ivt.get("supervision_granularity") != "FRAME_LEVEL_MULTI_LABEL":
        raise MaterializationError("VID31 IVT labels must remain frame-level")
    if len(vid31_ivt.get("frames", {})) != 3732:
        raise MaterializationError("VID31 IVT sidecar must contain 3,732 aligned frames")
    if vid31_ivt.get("instance_level_bbox_available") is not False:
        raise MaterializationError("VID31 IVT sidecar must not expose instance bboxes")


def copy_sidecar(source: Path, target: Path, *, replace: bool) -> str:
    """Copy one generated file atomically and reject silent replacement."""

    source_hash = sha256(source)
    if target.exists():
        if sha256(target) == source_hash:
            return source_hash
        if not replace:
            raise MaterializationError(
                f"Refusing to replace a different existing sidecar: {target}"
            )
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        if sha256(temporary) != source_hash:
            raise MaterializationError(f"Copied sidecar failed hash verification: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash


def build_manifest(
    *,
    dataset_root: Path,
    video_directories: dict[str, list[str]],
    vid30_target: Path,
    vid31_phase_target: Path,
    vid31_ivt_target: Path,
    source_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build the runtime manifest and record every upstream/local checksum."""

    vid30_raw = dataset_root / "Validation/VID30/vid30.json"
    vid31_raw = dataset_root / "Training/VID31/vid31.json"
    field_names = (
        "instrument",
        "verb",
        "target",
        "triplet",
        "phase",
        "bbox",
        "operator",
        "track_ids",
        "visual_conditions",
    )
    return {
        "schema_version": "ct20_vid30_vid31_derived_supervision_v1",
        "materialization_version": "ct20_local_sidecars_v1",
        "access_date": datetime.now(tz=UTC).date().isoformat(),
        "raw_dataset_root": str(dataset_root),
        "raw_dataset_modified": False,
        "policy": "EXPLICIT_DERIVED_SOURCES_ONLY",
        "dataset_contract": {
            "official_video_count": 20,
            "video_directories": video_directories,
            "added_video_directories": [],
            "added_splits": [],
            "added_classes": [],
            "verified_class_id_ranges": {
                "instrument": [0, 6],
                "verb": [0, 9],
                "target": [0, 14],
                "triplet": [0, 99],
                "phase": [0, 6],
            },
        },
        "videos": {
            "VID30": {
                "split": "validation",
                "media_source": str(dataset_root / "Validation/VID30/Frames"),
                "annotation_source": str(vid30_target),
                "field_supervision": {
                    field: field != "visual_conditions" for field in field_names
                },
                "status": "CANDIDATE_REPAIRED_VALIDATION",
            },
            "VID31": {
                "split": "training",
                "media_source": str(dataset_root / "Training/VID31/Frames"),
                "phase_source": str(vid31_phase_target),
                "frame_level_action_source": str(vid31_ivt_target),
                "frame_level_action_granularity": "FRAME_LEVEL_MULTI_LABEL",
                "field_supervision": {
                    field: field == "phase" for field in field_names
                },
                "status": "PHASE_AND_FRAME_LEVEL_IVT_TRAINING_INSTANCE_TASKS_DISABLED",
            },
        },
        "usage_constraints": {
            "VID30": "candidate validation only; visual-condition labels disabled",
            "VID31": (
                "phase plus frame-level IVT training only; bbox, operator, "
                "instance association, and track labels disabled"
            ),
            "negative_or_out_of_range_labels": "task-wise mask; no new class",
        },
        "source_sha256": {
            "track20_vid30_json": sha256(vid30_raw),
            "track20_vid31_json": sha256(vid31_raw),
            **{name: sha256(path) for name, path in source_paths.items()},
        },
        "materialized_sha256": {
            "vid30_repaired": sha256(vid30_target),
            "vid31_phase_repaired": sha256(vid31_phase_target),
            "vid31_frame_ivt_repaired": sha256(vid31_ivt_target),
        },
        "provenance": {name: str(path.resolve()) for name, path in source_paths.items()},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    vid30_source = args.vid30_source.expanduser().resolve()
    vid31_phase_source = args.vid31_phase_source.expanduser().resolve()
    vid31_ivt_source = args.vid31_ivt_source.expanduser().resolve()

    video_directories = _validate_layout(dataset_root)
    _validate_sources(vid30_source, vid31_phase_source, vid31_ivt_source)

    vid30_target = dataset_root / "Validation/VID30/vid30_repaired.json"
    vid31_phase_target = dataset_root / "Training/VID31/vid31_phase_repaired.json"
    vid31_ivt_target = dataset_root / "Training/VID31/vid31_frame_ivt_repaired.json"
    source_paths = {
        "vid30_candidate_artifact": vid30_source,
        "vid31_phase_artifact": vid31_phase_source,
        "vid31_frame_ivt_artifact": vid31_ivt_source,
        "cholec80_video30_phase": Path("D:/cholec80_30_31/video30-phase.txt"),
        "cholec80_video31_phase": Path("D:/cholec80_30_31/video31-phase.txt"),
        "cholect50_vid31_labels": Path("D:/CholecT50/CholecT50/labels/VID31.json"),
        "cholect50_label_mapping": Path("D:/CholecT50/CholecT50/label_mapping.txt"),
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise MaterializationError(f"Missing provenance inputs: {missing}")

    copy_sidecar(vid30_source, vid30_target, replace=args.replace_generated)
    copy_sidecar(vid31_phase_source, vid31_phase_target, replace=args.replace_generated)
    copy_sidecar(vid31_ivt_source, vid31_ivt_target, replace=args.replace_generated)

    if _validate_layout(dataset_root) != video_directories:
        raise MaterializationError("Video directory layout changed during materialization")
    manifest = build_manifest(
        dataset_root=dataset_root,
        video_directories=video_directories,
        vid30_target=vid30_target,
        vid31_phase_target=vid31_phase_target,
        vid31_ivt_target=vid31_ivt_target,
        source_paths=source_paths,
    )
    manifest_path = dataset_root / "repair_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--vid30-source", type=Path, default=DEFAULT_VID30_SOURCE)
    parser.add_argument(
        "--vid31-phase-source", type=Path, default=DEFAULT_VID31_PHASE_SOURCE
    )
    parser.add_argument("--vid31-ivt-source", type=Path, default=DEFAULT_VID31_IVT_SOURCE)
    parser.add_argument(
        "--replace-generated",
        action="store_true",
        help="Replace only generated sidecars when their hashes differ.",
    )
    return parser.parse_args()


def main() -> None:
    manifest = run(parse_args())
    print(json.dumps(manifest["materialized_sha256"], indent=2))


if __name__ == "__main__":
    main()
