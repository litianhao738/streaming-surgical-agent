"""Resolve CholecTrack20 VID30/VID31 identity without mutating source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.fft import dctn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.data.label_policy import classify_ivt_consistency

SCRIPT_VERSION = "ct20_vid30_vid31_resolution_v1"
TARGET_VIDEO_IDS = (17, 30, 31)
CHOLEC80_PHASE_NAME_TO_ID = {
    "Preparation": 0,
    "CalotTriangleDissection": 1,
    "ClippingCutting": 2,
    "GallbladderDissection": 3,
    "GallbladderPackaging": 4,
    "CleaningCoagulation": 5,
    "GallbladderRetraction": 6,
}
PAIRINGS = (
    (30, 30),
    (30, 31),
    (31, 30),
    (31, 31),
)
BOX_COLORS = (
    "#00ff88",
    "#ffcc00",
    "#00b7ff",
    "#ff5c8a",
    "#ffffff",
    "#b388ff",
    "#ff7a00",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _video_paths(dataset_root: Path, video_id: int) -> tuple[Path, Path]:
    directory_name = f"VID{video_id:02d}"
    candidates = list(dataset_root.glob(f"*/{directory_name}"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {directory_name} directory under {dataset_root}, found {candidates}"
        )
    video_dir = candidates[0]
    annotation_path = video_dir / f"vid{video_id:02d}.json"
    frames_dir = video_dir / "Frames"
    if not annotation_path.is_file() or not frames_dir.is_dir():
        raise FileNotFoundError(
            f"VID{video_id:02d} requires {annotation_path.name} and Frames/"
        )
    return annotation_path, frames_dir


def _annotation_index(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise TypeError("annotations must be a JSON object")
    result: dict[int, list[dict[str, Any]]] = {}
    for raw_frame_id, raw_instances in annotations.items():
        if not isinstance(raw_instances, list):
            raise TypeError(f"Frame {raw_frame_id} must contain a list")
        result[int(raw_frame_id)] = raw_instances
    return result


def _png_index(frames_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in frames_dir.glob("*.png"):
        result[int(path.stem)] = path
    return result


def _sample_quantiles(values: list[int], count: int) -> list[int]:
    if count <= 0 or not values:
        return []
    ordered = sorted(set(values))
    if len(ordered) <= count:
        return ordered
    indexes = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    return [ordered[int(index)] for index in indexes]


def _flatten(index: dict[int, list[dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (frame_id, instance)
        for frame_id in sorted(index)
        for instance in index[frame_id]
    ]


def _duplicate_diagnostic(
    left: dict[int, list[dict[str, Any]]],
    right: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    left_rows = _flatten(left)
    right_rows = _flatten(right)
    if len(left_rows) != len(right_rows):
        return {
            "row_counts_equal": False,
            "left_rows": len(left_rows),
            "right_rows": len(right_rows),
        }
    differing_fields: Counter[str] = Counter()
    frame_ids_equal = True
    for (left_frame, left_instance), (right_frame, right_instance) in zip(
        left_rows, right_rows, strict=True
    ):
        frame_ids_equal &= left_frame == right_frame
        for field in set(left_instance) | set(right_instance):
            if left_instance.get(field) != right_instance.get(field):
                differing_fields[field] += 1
    all_fields = set(left_rows[0][1]) if left_rows else set()
    equal_fields = sorted(all_fields - set(differing_fields))
    left_core = [
        (frame_id, {field: instance[field] for field in equal_fields})
        for frame_id, instance in left_rows
    ]
    right_core = [
        (frame_id, {field: instance[field] for field in equal_fields})
        for frame_id, instance in right_rows
    ]
    return {
        "row_counts_equal": True,
        "rows": len(left_rows),
        "frame_ids_equal": frame_ids_equal,
        "equal_fields": equal_fields,
        "differing_fields": dict(sorted(differing_fields.items())),
        "left_core_sha256": _json_sha256(left_core),
        "right_core_sha256": _json_sha256(right_core),
        "core_values_equal": left_core == right_core,
    }


def _draw_frame(
    image_path: Path,
    instances: list[dict[str, Any]],
    *,
    title: str,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    for instance in instances:
        bbox = instance.get("tool_bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, box_width, box_height = (float(value) for value in bbox)
        if box_width <= 0 or box_height <= 0:
            continue
        left = max(0, min(width - 1, round(x * width)))
        top = max(0, min(height - 1, round(y * height)))
        right = max(0, min(width - 1, round((x + box_width) * width)))
        bottom = max(0, min(height - 1, round((y + box_height) * height)))
        instrument = int(instance.get("instrument", -1))
        color = BOX_COLORS[instrument % len(BOX_COLORS)]
        draw.rectangle((left, top, right, bottom), outline=color, width=3)
        label = (
            f"I{instrument} V{instance.get('verb')} T{instance.get('target')} "
            f"IVT{instance.get('triplet')} tr{instance.get('visibility_track')}"
        )
        text_box = draw.textbbox((left, top), label, font=font)
        text_height = text_box[3] - text_box[1]
        label_top = max(0, top - text_height - 4)
        draw.rectangle(
            (left, label_top, min(width - 1, left + text_box[2] + 4), top),
            fill="black",
        )
        draw.text((left + 2, label_top + 1), label, fill=color, font=font)
    banner_height = 24
    canvas = Image.new("RGB", (width, height + banner_height), "black")
    canvas.paste(image, (0, banner_height))
    ImageDraw.Draw(canvas).text((6, 6), title, fill="white", font=font)
    return canvas


def _contact_sheet(images: list[Image.Image], *, columns: int = 3) -> Image.Image:
    tile_width, tile_height = 420, 270
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#202020")
    for index, image in enumerate(images):
        tile = ImageOps.contain(image, (tile_width - 8, tile_height - 8))
        x = (index % columns) * tile_width + (tile_width - tile.width) // 2
        y = (index // columns) * tile_height + (tile_height - tile.height) // 2
        sheet.paste(tile, (x, y))
    return sheet


def _write_pairing_overlays(
    annotations: dict[int, dict[int, list[dict[str, Any]]]],
    media: dict[int, dict[int, Path]],
    output_dir: Path,
    sample_count: int,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for annotation_video, media_video in PAIRINGS:
        common_ids = sorted(
            set(annotations[annotation_video]) & set(media[media_video])
        )
        sample_ids = _sample_quantiles(common_ids, sample_count)
        rendered = [
            _draw_frame(
                media[media_video][frame_id],
                annotations[annotation_video][frame_id],
                title=(
                    f"frame={frame_id} annotation=VID{annotation_video:02d} "
                    f"media=VID{media_video:02d}"
                ),
            )
            for frame_id in sample_ids
        ]
        output_path = output_dir / (
            f"annotation_vid{annotation_video:02d}_on_media_vid{media_video:02d}.jpg"
        )
        if rendered:
            _contact_sheet(rendered).save(output_path, quality=92)
        rows.append(
            {
                "annotation_video_id": f"VID{annotation_video:02d}",
                "media_video_id": f"VID{media_video:02d}",
                "common_frame_count": len(common_ids),
                "sample_frame_ids": sample_ids,
                "contact_sheet": str(output_path.resolve()) if rendered else None,
                "interpretation": "MANUAL_VISUAL_REVIEW_REQUIRED",
            }
        )
    return rows


def _triplet_mapping(path: Path) -> dict[int, tuple[int, int, int]]:
    audit = _load_json(path)
    raw_mapping = audit["ontology"]["triplet"]["component_map"]
    return {
        int(triplet_id): tuple(int(value) for value in components)
        for triplet_id, components in raw_mapping.items()
    }


def _label_policy_audit(
    dataset_root: Path,
    triplet_mapping: dict[int, tuple[int, int, int]],
) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    negative_patterns: Counter[tuple[int, int, int]] = Counter()
    for annotation_path in sorted(dataset_root.glob("*/VID*/vid*.json")):
        payload = _load_json(annotation_path)
        video_id = str(payload["video"]["name"]).upper()
        for frame_id, instances in _annotation_index(payload).items():
            for instance in instances:
                verb_id = int(instance["verb"])
                target_id = int(instance["target"])
                triplet_id = int(instance["triplet"])
                if min(verb_id, target_id, triplet_id) < 0:
                    negative_patterns[(verb_id, target_id, triplet_id)] += 1
                result = classify_ivt_consistency(
                    instrument_id=int(instance["instrument"]),
                    verb_id=verb_id,
                    target_id=target_id,
                    triplet_id=triplet_id,
                    triplet_mapping=triplet_mapping,
                )
                statuses[result.status.value] += 1
                bucket = examples.setdefault(result.status.value, [])
                if len(bucket) < 5:
                    bucket.append(
                        {
                            "video_id": video_id,
                            "frame_id": frame_id,
                            "instrument_id": int(instance["instrument"]),
                            "verb_id": verb_id,
                            "target_id": target_id,
                            "triplet_id": triplet_id,
                            "notes": result.notes,
                        }
                    )
    return {
        "policy": (
            "Negative/out-of-range IDs disable only their task. Cross-task consistency "
            "uses a separate mask and never rewrites raw labels."
        ),
        "status_counts": dict(sorted(statuses.items())),
        "negative_patterns": [
            {
                "verb_id": pattern[0],
                "target_id": pattern[1],
                "triplet_id": pattern[2],
                "count": count,
            }
            for pattern, count in sorted(negative_patterns.items())
        ],
        "examples": examples,
    }


def _probe_frame_count(video_path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw_count in (None, "N/A"):
        raise RuntimeError(f"Cannot determine frame count for {video_path}")
    return int(raw_count)


def _extract_frame(video_path: Path, decoder_index: int, output_path: Path) -> None:
    filter_expression = f"select=eq(n\\,{decoder_index})"
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        filter_expression,
        "-frames:v",
        "1",
        "-fps_mode",
        "vfr",
        "-y",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    if not output_path.is_file():
        raise RuntimeError(
            f"ffmpeg did not extract decoder frame {decoder_index} from {video_path}"
        )


def _normalized_gray(image: Image.Image, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.asarray(
        image.convert("L").resize(size, Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0


def _perceptual_hash(image: Image.Image) -> np.ndarray:
    pixels = _normalized_gray(image, (32, 32))
    coefficients = dctn(pixels, type=2, norm="ortho")[:8, :8]
    values = coefficients.flatten()[1:]
    return values > np.median(values)


def _similarity(left: Image.Image, right: Image.Image) -> dict[str, float | int]:
    left_gray = _normalized_gray(left)
    right_gray = _normalized_gray(right)
    left_centered = left_gray.flatten() - float(left_gray.mean())
    right_centered = right_gray.flatten() - float(right_gray.mean())
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    correlation = (
        float(np.dot(left_centered, right_centered) / denominator)
        if denominator > 0
        else 0.0
    )
    return {
        "phash_hamming": int(
            np.count_nonzero(_perceptual_hash(left) != _perceptual_hash(right))
        ),
        "gray_correlation": correlation,
        "mean_absolute_error": float(np.mean(np.abs(left_gray - right_gray))),
    }


def _image_frame_index(frames_dir: Path) -> dict[int, Path]:
    """Index a complete upstream frame export without changing its frame IDs."""
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)
    paths = [
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in frames_dir.glob(pattern)
    ]
    result: dict[int, Path] = {}
    for path in paths:
        try:
            frame_id = int(path.stem)
        except ValueError as error:
            raise ValueError(f"Non-numeric upstream frame name: {path}") from error
        if frame_id in result:
            raise ValueError(f"Duplicate upstream frame ID {frame_id}: {path}")
        result[frame_id] = path
    if not result:
        raise FileNotFoundError(f"No JPG/JPEG/PNG frames found in {frames_dir}")
    return result


def _phase_index(phase_path: Path) -> dict[int, int]:
    """Load Cholec80's text phase export using its literal frame IDs."""
    if not phase_path.is_file():
        raise FileNotFoundError(phase_path)
    rows = phase_path.read_text(encoding="utf-8-sig").splitlines()
    if not rows or rows[0].strip() != "Frame\tPhase":
        raise ValueError(f"Unexpected Cholec80 phase-file header: {phase_path}")
    result: dict[int, int] = {}
    for row in rows[1:]:
        if not row.strip():
            continue
        raw_frame, raw_phase = row.split("\t", maxsplit=1)
        if raw_phase not in CHOLEC80_PHASE_NAME_TO_ID:
            raise ValueError(f"Unknown Cholec80 phase {raw_phase!r} in {phase_path}")
        result[int(raw_frame)] = CHOLEC80_PHASE_NAME_TO_ID[raw_phase]
    return result


def _phase_alignment_audit(
    annotations: dict[int, dict[int, list[dict[str, Any]]]],
    phase_paths: dict[int, Path],
) -> dict[str, Any]:
    """Compare Track20 phases to Cholec80's phase export without altering either."""
    result: dict[str, Any] = {
        "frame_id_rule": "cholec80_phase_frame_id = track20_annotation_frame_id - 2",
        "phase_name_to_track20_id": CHOLEC80_PHASE_NAME_TO_ID,
        "videos": {},
    }
    for video_id in (30, 31):
        upstream = _phase_index(phase_paths[video_id])
        compared = 0
        matches = 0
        missing_upstream = 0
        inconsistent_local = 0
        for frame_id, instances in annotations[video_id].items():
            local_values = {int(instance["phase"]) for instance in instances if "phase" in instance}
            if len(local_values) != 1:
                inconsistent_local += 1
                continue
            upstream_value = upstream.get(frame_id - 2)
            if upstream_value is None:
                missing_upstream += 1
                continue
            compared += 1
            matches += int(next(iter(local_values)) == upstream_value)
        result["videos"][f"VID{video_id:02d}"] = {
            "cholec80_phase_path": str(phase_paths[video_id].resolve()),
            "track20_annotation_frames": len(annotations[video_id]),
            "compared_frames": compared,
            "exact_matches": matches,
            "exact_match_rate": matches / compared if compared else None,
            "missing_upstream_phase_frames": missing_upstream,
            "local_inconsistent_phase_frames": inconsistent_local,
        }
    return result


def _raw_json_phase_projection(
    annotations: dict[int, dict[int, list[dict[str, Any]]]],
    phase_path: Path,
) -> dict[str, Any]:
    """Project Cholec80 phases onto raw JSON keys for audit comparison only."""
    upstream = _phase_index(phase_path)
    values = {
        str(frame_id): upstream[frame_id - 2]
        for frame_id in sorted(annotations[31])
        if frame_id - 2 in upstream
    }
    return {
        "schema_version": "ct20_vid31_raw_json_phase_projection_v1",
        "source": "Cholec80 video31-phase.txt",
        "source_path": str(phase_path.resolve()),
        "frame_id_rule": "cholec80_phase_frame_id = track20_annotation_frame_id - 2",
        "phase_name_to_id": CHOLEC80_PHASE_NAME_TO_ID,
        "track20_video": "VID31",
        "training_use": "FORBIDDEN: annotation keys are not the VID31 image timeline",
        "phase_by_track20_annotation_frame_id": values,
        "source_data_modified": False,
    }


def _best_frame_matches(
    track_image: Image.Image,
    upstream: dict[int, Path],
    upstream_hashes: np.ndarray,
    upstream_ids: list[int],
) -> list[tuple[int, dict[str, float | int]]]:
    """Return tied best upstream frames, with correlation breaking hash ties."""
    track_hash = _perceptual_hash(track_image)
    distances = np.count_nonzero(upstream_hashes != track_hash, axis=1)
    best_distance = int(distances.min())
    candidate_indexes = np.flatnonzero(distances == best_distance)[:8]
    matches = [
        (
            upstream_ids[int(index)],
            _similarity(track_image, Image.open(upstream[upstream_ids[int(index)]])),
        )
        for index in candidate_indexes
    ]
    return sorted(
        matches,
        key=lambda item: (-float(item[1]["gray_correlation"]), item[0]),
    )


def _upstream_frame_directory_audit(
    media: dict[int, dict[int, Path]],
    upstream_frame_dirs: dict[int, Path],
    output_dir: Path,
    sample_count: int,
) -> dict[str, Any]:
    """Establish identity from complete Cholec80 frame exports.

    Unlike a raw MP4, an exported-frame directory has its own naming convention.
    The audit therefore performs global nearest-frame retrieval first and only then
    reports the observed Track20-to-Cholec80 frame-number offsets.
    """
    upstream = {
        video_id: _image_frame_index(path)
        for video_id, path in upstream_frame_dirs.items()
    }
    upstream_ids = {video_id: sorted(index) for video_id, index in upstream.items()}
    upstream_hashes = {
        video_id: np.stack(
            [
                _perceptual_hash(Image.open(index[frame_id]))
                for frame_id in upstream_ids[video_id]
            ]
        )
        for video_id, index in upstream.items()
    }
    side_by_side_dir = output_dir / "upstream_frame_directory_comparisons"
    side_by_side_dir.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict[str, Any]] = []

    for track_video in (30, 31):
        sample_ids = _sample_quantiles(sorted(media[track_video]), sample_count)
        for upstream_video in (30, 31):
            matches: list[dict[str, Any]] = []
            tiles: list[Image.Image] = []
            for frame_id in sample_ids:
                track_image = Image.open(media[track_video][frame_id]).convert("RGB")
                upstream_frame_id, metric = _best_frame_matches(
                    track_image,
                    upstream[upstream_video],
                    upstream_hashes[upstream_video],
                    upstream_ids[upstream_video],
                )[0]
                matches.append(
                    {
                        "track20_frame_id": frame_id,
                        "cholec80_frame_id": upstream_frame_id,
                        "cholec80_minus_track20_frame_id": upstream_frame_id - frame_id,
                        **metric,
                    }
                )
                upstream_image = Image.open(
                    upstream[upstream_video][upstream_frame_id]
                ).convert("RGB")
                left = ImageOps.contain(track_image, (400, 240))
                right = ImageOps.contain(upstream_image, (400, 240))
                tile = Image.new("RGB", (820, 270), "black")
                tile.paste(left, (0, 30))
                tile.paste(right, (420, 30))
                ImageDraw.Draw(tile).text(
                    (6, 8),
                    (
                        f"Track20 VID{track_video:02d} frame {frame_id} | "
                        f"Cholec80 video{upstream_video:02d} frame {upstream_frame_id}"
                    ),
                    fill="white",
                    font=ImageFont.load_default(),
                )
                tiles.append(tile)

            contact_path = side_by_side_dir / (
                f"track20_vid{track_video:02d}_vs_cholec80_video{upstream_video:02d}.jpg"
            )
            _contact_sheet(tiles, columns=1).save(contact_path, quality=92)
            pair_rows.append(
                {
                    "track20_media": f"VID{track_video:02d}",
                    "cholec80_video": f"video{upstream_video:02d}",
                    "comparison_method": "GLOBAL_PERCEPTUAL_HASH_RETRIEVAL",
                    "sample_count": len(matches),
                    "sample_frame_ids": sample_ids,
                    "median_phash_hamming": median(
                        int(row["phash_hamming"]) for row in matches
                    ),
                    "median_gray_correlation": median(
                        float(row["gray_correlation"]) for row in matches
                    ),
                    "median_mean_absolute_error": median(
                        float(row["mean_absolute_error"]) for row in matches
                    ),
                    "observed_frame_id_offsets": sorted(
                        {int(row["cholec80_minus_track20_frame_id"]) for row in matches}
                    ),
                    "per_frame_matches": matches,
                    "contact_sheet": str(contact_path.resolve()),
                }
            )

    suggestions: dict[str, Any] = {}
    for track_video in (30, 31):
        candidates = [
            row for row in pair_rows if row["track20_media"] == f"VID{track_video:02d}"
        ]
        ranked = sorted(
            candidates,
            key=lambda row: (
                row["median_phash_hamming"],
                -row["median_gray_correlation"],
            ),
        )
        suggestions[f"VID{track_video:02d}"] = {
            "best_candidate_by_metrics": ranked[0]["cholec80_video"],
            "status": "EVIDENCE_GENERATED_REQUIRES_REVIEW",
            "reason": (
                "Frame names are preserved as external evidence; no Track20 asset or "
                "supervision field is remapped automatically."
            ),
        }
    return {
        "source_type": "COMPLETE_CHOLEC80_FRAME_DIRECTORIES",
        "frame_id_semantics": "PRESERVED_FROM_SUPPLIED_FILENAMES; NOT_REINTERPRETED",
        "upstream_frame_directories": {
            f"video{video_id:02d}": {
                "path": str(path.resolve()),
                "frame_count": len(upstream[video_id]),
                "min_frame_id": min(upstream[video_id]),
                "max_frame_id": max(upstream[video_id]),
            }
            for video_id, path in upstream_frame_dirs.items()
        },
        "pairings": pair_rows,
        "identity_suggestions": suggestions,
    }


def _upstream_identity_audit(
    media: dict[int, dict[int, Path]],
    upstream_videos: dict[int, Path],
    output_dir: Path,
    sample_count: int,
) -> dict[str, Any]:
    frame_counts = {
        video_id: _probe_frame_count(path)
        for video_id, path in upstream_videos.items()
    }
    cache: dict[tuple[int, int], Image.Image] = {}
    pair_rows: list[dict[str, Any]] = []
    side_by_side_dir = output_dir / "upstream_comparisons"
    side_by_side_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ct20_identity_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for track_video in (30, 31):
            for upstream_video in (30, 31):
                legal_ids = [
                    frame_id
                    for frame_id in media[track_video]
                    if 0 <= frame_id - 1 < frame_counts[upstream_video]
                ]
                sample_ids = _sample_quantiles(legal_ids, sample_count)
                metrics: list[dict[str, float | int]] = []
                comparison_tiles: list[Image.Image] = []
                for frame_id in sample_ids:
                    decoder_index = frame_id - 1
                    cache_key = (upstream_video, decoder_index)
                    if cache_key not in cache:
                        extracted_path = temp_dir / (
                            f"video{upstream_video:02d}_{decoder_index:06d}.png"
                        )
                        _extract_frame(
                            upstream_videos[upstream_video],
                            decoder_index,
                            extracted_path,
                        )
                        cache[cache_key] = Image.open(extracted_path).convert("RGB").copy()
                    track_image = Image.open(media[track_video][frame_id]).convert("RGB")
                    upstream_image = cache[cache_key]
                    metric = _similarity(track_image, upstream_image)
                    metrics.append({"frame_id": frame_id, **metric})

                    left = ImageOps.contain(track_image, (400, 240))
                    right = ImageOps.contain(upstream_image, (400, 240))
                    tile = Image.new("RGB", (820, 270), "black")
                    tile.paste(left, (0, 30))
                    tile.paste(right, (420, 30))
                    ImageDraw.Draw(tile).text(
                        (6, 8),
                        (
                            f"Track20 VID{track_video:02d} frame {frame_id} | "
                            f"Cholec80 video{upstream_video:02d} decoder {decoder_index}"
                        ),
                        fill="white",
                        font=ImageFont.load_default(),
                    )
                    comparison_tiles.append(tile)

                contact_path = side_by_side_dir / (
                    f"track20_vid{track_video:02d}_vs_cholec80_video{upstream_video:02d}.jpg"
                )
                if comparison_tiles:
                    _contact_sheet(comparison_tiles, columns=1).save(
                        contact_path, quality=92
                    )
                phash_values = [int(row["phash_hamming"]) for row in metrics]
                correlation_values = [float(row["gray_correlation"]) for row in metrics]
                mae_values = [float(row["mean_absolute_error"]) for row in metrics]
                pair_rows.append(
                    {
                        "track20_media": f"VID{track_video:02d}",
                        "cholec80_video": f"video{upstream_video:02d}",
                        "sample_count": len(metrics),
                        "sample_frame_ids": sample_ids,
                        "median_phash_hamming": median(phash_values) if metrics else None,
                        "max_phash_hamming": max(phash_values) if metrics else None,
                        "median_gray_correlation": (
                            median(correlation_values) if metrics else None
                        ),
                        "median_mean_absolute_error": (
                            median(mae_values) if metrics else None
                        ),
                        "per_frame_metrics": metrics,
                        "contact_sheet": str(contact_path.resolve()),
                    }
                )

    suggestions: dict[str, Any] = {}
    for track_video in (30, 31):
        candidates = [
            row for row in pair_rows if row["track20_media"] == f"VID{track_video:02d}"
        ]
        ranked = sorted(
            candidates,
            key=lambda row: (
                row["median_phash_hamming"],
                -row["median_gray_correlation"],
            ),
        )
        suggestions[f"VID{track_video:02d}"] = {
            "best_candidate_by_metrics": ranked[0]["cholec80_video"],
            "status": "EVIDENCE_GENERATED_REQUIRES_REVIEW",
            "reason": (
                "No raw asset is remapped automatically; review score margins and contact sheets."
            ),
        }
    return {
        "candidate_decoder_rule": "decoder_frame_index = annotation_frame_id - 1",
        "upstream_video_sha256": {
            f"video{video_id:02d}": _sha256(path)
            for video_id, path in upstream_videos.items()
        },
        "upstream_frame_counts": {
            f"video{video_id:02d}": count
            for video_id, count in frame_counts.items()
        },
        "pairings": pair_rows,
        "identity_suggestions": suggestions,
    }


def _supervision_manifest(local_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "ct20_field_level_supervision_manifest_v1",
        "source_data_policy": "READ_ONLY_NO_RENAME_NO_MOVE_NO_OVERWRITE",
        "unlisted_video_policy": "USE_CANONICAL_TASK_MASKS",
        "runtime_policy": (
            "Keep all raw assets. Enable supervision per field only after identity evidence."
        ),
        "videos": {
            "VID30": {
                "raw_assets_retained": True,
                "media_usage": "UNSUPERVISED_OR_MANUAL_REVIEW_ONLY",
                "field_supervision": {
                    "instrument": False,
                    "verb": False,
                    "target": False,
                    "triplet": False,
                    "phase": False,
                    "bbox": False,
                    "operator": False,
                    "track_ids": False,
                    "visual_conditions": False,
                },
                "status": "PENDING_CHOLECT80_PIXEL_IDENTITY_AND_PAIRING_REVIEW",
                "evidence_ref": "local_identity_audit.json",
            },
            "VID31": {
                "raw_assets_retained": True,
                "media_usage": "UNSUPERVISED_OR_MANUAL_REVIEW_ONLY",
                "field_supervision": {
                    "instrument": False,
                    "verb": False,
                    "target": False,
                    "triplet": False,
                    "phase": False,
                    "bbox": False,
                    "operator": False,
                    "track_ids": False,
                    "visual_conditions": False,
                },
                "phase_evidence": (
                    "Cholec80 same-ID phase agreement is high in the provenance audit, but "
                    "media pixel identity remains pending."
                ),
                "status": "PENDING_CHOLECT80_PIXEL_IDENTITY_AND_FIELD_LEVEL_REVIEW",
                "evidence_ref": "local_identity_audit.json",
            },
        },
        "local_evidence_sha256": _json_sha256(local_audit),
    }


def _derived_supervision_plan(
    local_audit: dict[str, Any],
    upstream: dict[str, Any] | None,
    phase_alignment: dict[str, Any] | None,
) -> dict[str, Any]:
    """Record conservative downstream use without editing Track20 source assets."""
    plan: dict[str, Any] = {
        "schema_version": "ct20_vid30_vid31_derived_supervision_plan_v1",
        "source_data_policy": "READ_ONLY_NO_RENAME_NO_MOVE_NO_OVERWRITE",
        "training_code_changed": False,
        "decision": (
            "Do not consume VID30/VID31 raw JSON as ordinary paired supervision. "
            "Keep original files; use only field-specific sources explicitly listed here."
        ),
        "videos": {
            "VID30": {
                "media": "Track20 VID30 Frames, pixel identity supported as Cholec80 video30",
                "raw_json_supervision": "DISABLED: core annotations duplicate Track20 VID17",
                "formal_tracking_supervision": "DISABLED",
                "formal_ivt_supervision": "DISABLED",
                "formal_phase_supervision": "DISABLED: local phase agreement with video30 is insufficient",
                "allowed_nontraining_use": "UNSUPERVISED_VISUAL_DATA_OR_MANUAL_QA",
            },
            "VID31": {
                "media": "Track20 VID31 Frames, pixel identity supported as Cholec80 video31",
                "raw_json_core_supervision": "DISABLED: core geometry has an unresolved cross-pairing signal",
                "formal_tracking_supervision": "DISABLED",
                "formal_ivt_supervision": "DISABLED",
                "formal_phase_supervision": (
                    "DERIVE_FROM_CHOLEC80_VIDEO31_PHASE_FILE_PER_SAMPLE; do not use the "
                    "raw Track20 phase value where it disagrees."
                ),
            },
        },
        "candidate_not_authorized_for_formal_training": {
            "media": "Track20 VID30 Frames",
            "core_label_source": "Track20 VID31 JSON at identical frame IDs",
            "reason": (
                "Local overlay review supports this candidate, but it remains a derived "
                "cross-file pairing rather than an official Track20 pairing."
            ),
        },
        "evidence_refs": {
            "local_audit_sha256": _json_sha256(local_audit),
            "upstream_identity": "upstream_identity_audit.json" if upstream else None,
            "phase_alignment": "upstream_phase_alignment.json" if phase_alignment else None,
        },
    }
    if phase_alignment is not None:
        plan["videos"]["VID31"]["phase_alignment"] = phase_alignment["videos"][
            "VID31"
        ]
    return plan


def _render_report(local_audit: dict[str, Any], upstream: dict[str, Any] | None) -> str:
    vid30 = local_audit["videos"]["VID30"]
    vid31 = local_audit["videos"]["VID31"]
    duplicate = local_audit["vid17_vid30_duplicate"]
    lines = [
        "# VID30/VID31 Resolution Report",
        "",
        "## Local decision",
        "",
        "- Source dataset files were read only and remain unchanged.",
        f"- VID30 annotation frames: {vid30['annotation_frame_count']}; PNGs: {vid30['png_count']}.",
        f"- VID31 annotation frames: {vid31['annotation_frame_count']}; PNGs: {vid31['png_count']}.",
        f"- VID30 PNG IDs equal VID31 annotation IDs: {local_audit['relations']['vid30_png_ids_equal_vid31_annotation_ids']}.",
        f"- VID17/VID30 core values equal: {duplicate.get('core_values_equal')}.",
        "- The raw JSON files remain blocked as ordinary paired supervision; the field-level plan records any supported derived use.",
        "",
        "## Label policy",
        "",
        "Negative or out-of-range IDs disable only their task. IVT component consistency is",
        "tracked separately; specimen-bag scope differences are not rewritten as errors.",
        "",
        "## Cholec80 identity",
        "",
    ]
    if upstream is None:
        lines.extend(
            [
                "`PENDING`: provide Cholec80 video30 and video31 MP4 files and rerun this tool.",
                "The complete Cholec80 dataset is not required for this identity step.",
            ]
        )
    else:
        pairing_rows = upstream["pairings"]
        for video_id, result in upstream["identity_suggestions"].items():
            row = next(
                item
                for item in pairing_rows
                if item["track20_media"] == video_id
                and item["cholec80_video"] == result["best_candidate_by_metrics"]
            )
            lines.append(
                f"- {video_id}: {result['best_candidate_by_metrics']} "
                f"({result['status']}); median pHash distance "
                f"{row['median_phash_hamming']}, correlation "
                f"{row['median_gray_correlation']:.3f}."
            )
        lines.extend(
            [
                "",
                (
                    "The derived-supervision plan keeps VID30 labels disabled. VID31 core "
                    "labels remain disabled; its phase labels must be derived per sample from "
                    "the Cholec80 video31 phase file when that source was supplied."
                ),
            ]
        )
        lines.append("")
        lines.append(
            "Review the score matrix and side-by-side contact sheets before authorizing any derived mapping."
        )
    lines.extend(
        [
            "",
            "## Remaining evidence boundary",
            "",
            "Cholec80 can verify media identity and phase timing, but it cannot reconstruct missing",
            "Track20 bounding boxes or three-perspective track IDs.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads: dict[int, dict[str, Any]] = {}
    annotations: dict[int, dict[int, list[dict[str, Any]]]] = {}
    media: dict[int, dict[int, Path]] = {}
    paths: dict[int, dict[str, Path]] = {}
    for video_id in TARGET_VIDEO_IDS:
        annotation_path, frames_dir = _video_paths(dataset_root, video_id)
        payloads[video_id] = _load_json(annotation_path)
        annotations[video_id] = _annotation_index(payloads[video_id])
        media[video_id] = _png_index(frames_dir)
        paths[video_id] = {
            "annotation": annotation_path,
            "frames": frames_dir,
        }

    overlay_rows = _write_pairing_overlays(
        annotations,
        media,
        output_dir / "local_pairing_overlays",
        args.sample_count,
    )
    videos = {
        f"VID{video_id:02d}": {
            "annotation_path": str(paths[video_id]["annotation"]),
            "annotation_sha256": _sha256(paths[video_id]["annotation"]),
            "frames_dir": str(paths[video_id]["frames"]),
            "annotation_frame_count": len(annotations[video_id]),
            "png_count": len(media[video_id]),
            "common_count": len(set(annotations[video_id]) & set(media[video_id])),
            "annotation_only_count": len(
                set(annotations[video_id]) - set(media[video_id])
            ),
            "png_only_count": len(set(media[video_id]) - set(annotations[video_id])),
        }
        for video_id in TARGET_VIDEO_IDS
    }

    triplet_mapping = _triplet_mapping(args.cross_audit.expanduser().resolve())
    local_audit: dict[str, Any] = {
        "schema_version": SCRIPT_VERSION,
        "dataset_root": str(dataset_root),
        "source_data_modified": False,
        "videos": videos,
        "relations": {
            "vid30_png_ids_equal_vid31_annotation_ids": (
                set(media[30]) == set(annotations[31])
            ),
            "vid30_annotation_ids_equal_vid17_annotation_ids": (
                set(annotations[30]) == set(annotations[17])
            ),
        },
        "vid17_vid30_duplicate": _duplicate_diagnostic(
            annotations[17], annotations[30]
        ),
        "candidate_pairings": overlay_rows,
        "label_policy_audit": _label_policy_audit(dataset_root, triplet_mapping),
    }

    upstream: dict[str, Any] | None = None
    supplied_videos = (args.cholec80_video30, args.cholec80_video31)
    supplied_frame_dirs = (args.cholec80_frames30, args.cholec80_frames31)
    if any(supplied_videos) and not all(supplied_videos):
        raise ValueError("Provide both --cholec80-video30 and --cholec80-video31")
    if any(supplied_frame_dirs) and not all(supplied_frame_dirs):
        raise ValueError("Provide both --cholec80-frames30 and --cholec80-frames31")
    if all(supplied_videos) and all(supplied_frame_dirs):
        raise ValueError("Provide either Cholec80 MP4s or frame directories, not both")
    if all(supplied_videos):
        upstream_videos = {
            30: args.cholec80_video30.expanduser().resolve(),
            31: args.cholec80_video31.expanduser().resolve(),
        }
        for path in upstream_videos.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        upstream = _upstream_identity_audit(
            {30: media[30], 31: media[31]},
            upstream_videos,
            output_dir,
            args.sample_count,
        )
    if all(supplied_frame_dirs):
        upstream_frame_dirs = {
            30: args.cholec80_frames30.expanduser().resolve(),
            31: args.cholec80_frames31.expanduser().resolve(),
        }
        upstream = _upstream_frame_directory_audit(
            {30: media[30], 31: media[31]},
            upstream_frame_dirs,
            output_dir,
            args.sample_count,
        )

    supplied_phase_files = (args.cholec80_phase30, args.cholec80_phase31)
    if any(supplied_phase_files) and not all(supplied_phase_files):
        raise ValueError("Provide both --cholec80-phase30 and --cholec80-phase31")
    phase_alignment: dict[str, Any] | None = None
    raw_json_phase_projection: dict[str, Any] | None = None
    if all(supplied_phase_files):
        phase31_path = args.cholec80_phase31.expanduser().resolve()
        phase_alignment = _phase_alignment_audit(
            annotations,
            {
                30: args.cholec80_phase30.expanduser().resolve(),
                31: phase31_path,
            },
        )
        raw_json_phase_projection = _raw_json_phase_projection(
            annotations,
            phase31_path,
        )

    (output_dir / "local_identity_audit.json").write_text(
        json.dumps(local_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "field_level_supervision_manifest.json").write_text(
        json.dumps(_supervision_manifest(local_audit), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    if upstream is not None:
        (output_dir / "upstream_identity_audit.json").write_text(
            json.dumps(upstream, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if phase_alignment is not None:
        (output_dir / "upstream_phase_alignment.json").write_text(
            json.dumps(phase_alignment, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if raw_json_phase_projection is not None:
        (output_dir / "vid31_raw_json_phase_projection.json").write_text(
            json.dumps(raw_json_phase_projection, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (output_dir / "derived_supervision_plan.json").write_text(
        json.dumps(
            _derived_supervision_plan(local_audit, upstream, phase_alignment),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "RESOLUTION_REPORT.md").write_text(
        _render_report(local_audit, upstream), encoding="utf-8"
    )
    return {
        "local_audit": str((output_dir / "local_identity_audit.json").resolve()),
        "supervision_manifest": str(
            (output_dir / "field_level_supervision_manifest.json").resolve()
        ),
        "derived_supervision_plan": str(
            (output_dir / "derived_supervision_plan.json").resolve()
        ),
        "upstream_status": "COMPLETE" if upstream is not None else "PENDING_CHOLECT80",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data_resolution/cholectrack20_vid30_vid31"),
    )
    parser.add_argument(
        "--cross-audit",
        type=Path,
        default=Path(
            "dataset_reports/cholectrack20_cross_dataset_provenance_audit.json"
        ),
    )
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--cholec80-video30", type=Path)
    parser.add_argument("--cholec80-video31", type=Path)
    parser.add_argument("--cholec80-frames30", type=Path)
    parser.add_argument("--cholec80-frames31", type=Path)
    parser.add_argument("--cholec80-phase30", type=Path)
    parser.add_argument("--cholec80-phase31", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(_parse_args()), indent=2))
