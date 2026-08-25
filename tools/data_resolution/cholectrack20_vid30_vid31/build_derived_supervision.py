"""Build the reviewed VID30/VID31 supervision package without mutating raw data."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.data_resolution.cholectrack20_vid30_vid31.resolve_identity import (
    _contact_sheet,
    _draw_frame,
    _image_frame_index,
    _load_json,
    _phase_index,
    _png_index,
    _sample_quantiles,
    _sha256,
    _similarity,
    _video_paths,
)

SCHEMA_VERSION = "ct20_vid30_vid31_derived_supervision_v1"
VID30_CHOLEC80_FRAME_OFFSET = -11677
VID31_CHOLEC80_FRAME_OFFSET = -6152
CORE_FIELDS = (
    "instrument",
    "verb",
    "target",
    "triplet",
    "tool_bbox",
    "operator",
    "intraoperative_track",
    "intracorporeal_track",
    "visibility_track",
)


def _fixed_offset_metrics(
    track_media: dict[int, Path],
    upstream_media: dict[int, Path],
    *,
    offset: int,
    sample_count: int,
) -> dict[str, Any]:
    legal_ids = sorted(
        frame_id for frame_id in track_media if frame_id + offset in upstream_media
    )
    sample_ids = _sample_quantiles(legal_ids, sample_count)
    rows: list[dict[str, Any]] = []
    for frame_id in sample_ids:
        upstream_id = frame_id + offset
        metric = _similarity(
            Image.open(track_media[frame_id]),
            Image.open(upstream_media[upstream_id]),
        )
        rows.append(
            {
                "track20_frame_id": frame_id,
                "cholec80_frame_id": upstream_id,
                **metric,
            }
        )
    return {
        "offset": offset,
        "eligible_frame_count": len(legal_ids),
        "sample_count": len(rows),
        "median_phash_hamming": median(
            int(row["phash_hamming"]) for row in rows
        ),
        "median_gray_correlation": median(
            float(row["gray_correlation"]) for row in rows
        ),
        "median_mean_absolute_error": median(
            float(row["mean_absolute_error"]) for row in rows
        ),
        "samples": rows,
    }


def _dominant_offset_evidence(
    identity_audit: dict[str, Any],
    *,
    track_video: str,
    cholec80_video: str,
    expected_offset: int,
) -> dict[str, Any]:
    row = next(
        item
        for item in identity_audit["pairings"]
        if item["track20_media"] == track_video
        and item["cholec80_video"] == cholec80_video
    )
    matches = row["per_frame_matches"]
    exact = sum(
        int(item["cholec80_minus_track20_frame_id"] == expected_offset)
        for item in matches
    )
    rate = exact / len(matches)
    if len(matches) < 100 or rate < 0.75:
        raise ValueError(
            f"Insufficient offset evidence for {track_video}: "
            f"samples={len(matches)}, exact_rate={rate:.3f}"
        )
    return {
        "sample_count": len(matches),
        "expected_offset": expected_offset,
        "global_nearest_exact_offset_count": exact,
        "global_nearest_exact_offset_rate": rate,
    }


def _bbox_statistics(annotations: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    total = 0
    positive = 0
    inside = 0
    missing = 0
    for instances in annotations.values():
        for instance in instances:
            total += 1
            bbox = instance.get("tool_bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                missing += 1
                continue
            x, y, width, height = (float(value) for value in bbox)
            if width > 0 and height > 0:
                positive += 1
            if x >= 0 and y >= 0 and x + width <= 1 and y + height <= 1:
                inside += 1
    return {
        "instances": total,
        "positive_extent_boxes": positive,
        "inside_unit_frame_boxes": inside,
        "missing_or_malformed_boxes": missing,
    }


def _write_candidate_review_sheets(
    annotations: dict[int, list[dict[str, Any]]],
    media: dict[int, Path],
    output_dir: Path,
    *,
    sample_count: int,
    page_size: int = 20,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_ids = _sample_quantiles(sorted(set(annotations) & set(media)), sample_count)
    rendered = [
        _draw_frame(
            media[frame_id],
            annotations[frame_id],
            title=f"VID30 candidate validation frame={frame_id} core=VID31 JSON",
        )
        for frame_id in frame_ids
    ]
    paths: list[str] = []
    for page_index, start in enumerate(range(0, len(rendered), page_size), start=1):
        output_path = output_dir / f"vid30_candidate_bbox_review_{page_index:02d}.jpg"
        _contact_sheet(rendered[start : start + page_size], columns=4).save(
            output_path,
            quality=92,
        )
        paths.append(str(output_path.resolve()))
    return paths


def _build_vid30_candidate(
    vid31_payload: dict[str, Any],
    phase30: dict[int, int],
) -> tuple[dict[str, Any], dict[str, int]]:
    derived = copy.deepcopy(vid31_payload)
    derived["info"]["version"] = "1.0-derived-vid30-candidate"
    derived["video"]["name"] = "VID30"
    derived["video"]["split"] = "validation"
    phase_counts: dict[str, int] = {}
    for raw_frame_id, instances in derived["annotations"].items():
        track_frame_id = int(raw_frame_id)
        upstream_frame_id = track_frame_id + VID30_CHOLEC80_FRAME_OFFSET
        if upstream_frame_id not in phase30:
            raise ValueError(
                f"Missing Cholec80 video30 phase for Track20 frame {track_frame_id} "
                f"(upstream {upstream_frame_id})"
            )
        phase_id = phase30[upstream_frame_id]
        phase_counts[str(phase_id)] = phase_counts.get(str(phase_id), 0) + 1
        for instance in instances:
            instance["phase"] = phase_id
    derived["derivation"] = {
        "schema_version": SCHEMA_VERSION,
        "status": "CANDIDATE_REPAIRED_VALIDATION_SUPERVISION",
        "media_source": "Track20 Validation/VID30/Frames",
        "core_annotation_source": "Track20 Training/VID31/vid31.json",
        "core_fields": list(CORE_FIELDS),
        "phase_source": "Cholec80 video30-phase.txt",
        "phase_frame_rule": (
            "cholec80_frame_id = track20_vid30_image_frame_id - 11677"
        ),
        "visual_condition_supervision": "DISABLED",
        "raw_dataset_modified": False,
    }
    return derived, phase_counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    cholec80_root = args.cholec80_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vid30_json, vid30_frames_dir = _video_paths(dataset_root, 30)
    vid31_json, vid31_frames_dir = _video_paths(dataset_root, 31)
    vid30_media = _png_index(vid30_frames_dir)
    vid31_media = _png_index(vid31_frames_dir)
    vid31_payload = _load_json(vid31_json)
    vid31_annotations = {
        int(frame_id): instances
        for frame_id, instances in vid31_payload["annotations"].items()
    }
    if set(vid30_media) != set(vid31_annotations):
        raise ValueError("VID30 media IDs must exactly equal VID31 annotation IDs")

    cholec80_media = {
        30: _image_frame_index(cholec80_root / "30"),
        31: _image_frame_index(cholec80_root / "31"),
    }
    phase_paths = {
        30: cholec80_root / "video30-phase.txt",
        31: cholec80_root / "video31-phase.txt",
    }
    phases = {video_id: _phase_index(path) for video_id, path in phase_paths.items()}
    identity_audit_path = args.identity_audit.expanduser().resolve()
    identity_audit = _load_json(identity_audit_path)
    offset_evidence = {
        "VID30": _dominant_offset_evidence(
            identity_audit,
            track_video="VID30",
            cholec80_video="video30",
            expected_offset=VID30_CHOLEC80_FRAME_OFFSET,
        ),
        "VID31": _dominant_offset_evidence(
            identity_audit,
            track_video="VID31",
            cholec80_video="video31",
            expected_offset=VID31_CHOLEC80_FRAME_OFFSET,
        ),
    }
    fixed_offset_metrics = {
        "VID30": _fixed_offset_metrics(
            vid30_media,
            cholec80_media[30],
            offset=VID30_CHOLEC80_FRAME_OFFSET,
            sample_count=args.sample_count,
        ),
        "VID31": _fixed_offset_metrics(
            vid31_media,
            cholec80_media[31],
            offset=VID31_CHOLEC80_FRAME_OFFSET,
            sample_count=args.sample_count,
        ),
    }

    derived_vid30, vid30_phase_counts = _build_vid30_candidate(
        vid31_payload,
        phases[30],
    )
    derived_root = output_dir / "derived"
    vid30_output_dir = derived_root / "VID30"
    vid30_output_dir.mkdir(parents=True, exist_ok=True)
    vid30_candidate_path = vid30_output_dir / "vid30_candidate_validation.json"
    vid30_candidate_path.write_text(
        json.dumps(derived_vid30, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    vid31_phase_values = {
        str(frame_id): {
            "phase_id": phases[31][frame_id + VID31_CHOLEC80_FRAME_OFFSET],
            "cholec80_frame_id": frame_id + VID31_CHOLEC80_FRAME_OFFSET,
        }
        for frame_id in sorted(vid31_media)
        if frame_id + VID31_CHOLEC80_FRAME_OFFSET in phases[31]
        and frame_id + VID31_CHOLEC80_FRAME_OFFSET in cholec80_media[31]
    }
    vid31_phase_path = derived_root / "VID31" / "vid31_image_phase_supervision.json"
    vid31_phase_path.parent.mkdir(parents=True, exist_ok=True)
    vid31_phase_payload = {
        "schema_version": SCHEMA_VERSION,
        "video_id": "VID31",
        "split": "training",
        "media_source": str(vid31_frames_dir.resolve()),
        "phase_source": str(phase_paths[31].resolve()),
        "frame_rule": "cholec80_frame_id = track20_vid31_image_frame_id - 6152",
        "phase_by_track20_image_frame_id": vid31_phase_values,
        "excluded_track20_image_frame_ids": sorted(
            set(vid31_media) - {int(value) for value in vid31_phase_values}
        ),
        "non_phase_supervision": "DISABLED",
        "raw_dataset_modified": False,
    }
    vid31_phase_path.write_text(
        json.dumps(vid31_phase_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    review_paths = _write_candidate_review_sheets(
        vid31_annotations,
        vid30_media,
        output_dir / "qa" / "vid30_candidate_bbox",
        sample_count=args.sample_count,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "raw_dataset_root": str(dataset_root),
        "raw_dataset_modified": False,
        "policy": "EXPLICIT_DERIVED_SOURCES_ONLY",
        "videos": {
            "VID30": {
                "split": "validation",
                "media_source": str(vid30_frames_dir.resolve()),
                "annotation_source": str(vid30_candidate_path.resolve()),
                "field_supervision": {
                    "instrument": True,
                    "verb": True,
                    "target": True,
                    "triplet": True,
                    "phase": True,
                    "bbox": True,
                    "operator": True,
                    "track_ids": True,
                    "visual_conditions": False,
                },
                "status": "CANDIDATE_REPAIRED_VALIDATION",
            },
            "VID31": {
                "split": "training",
                "media_source": str(vid31_frames_dir.resolve()),
                "phase_source": str(vid31_phase_path.resolve()),
                "field_supervision": {
                    "instrument": False,
                    "verb": False,
                    "target": False,
                    "triplet": False,
                    "phase": True,
                    "bbox": False,
                    "operator": False,
                    "track_ids": False,
                    "visual_conditions": False,
                },
                "status": "PHASE_ONLY_TRAINING",
            },
        },
        "evidence": {
            "identity_audit": str(identity_audit_path),
            "offset_evidence": offset_evidence,
            "fixed_offset_metrics": fixed_offset_metrics,
            "vid30_bbox_statistics": _bbox_statistics(
                derived_vid30["annotations"]
            ),
            "vid30_phase_frame_counts": vid30_phase_counts,
            "vid30_bbox_review_sheets": review_paths,
            "vid30_manual_review": str(
                (output_dir / "VID30_CANDIDATE_QA_REVIEW.md").resolve()
            ),
        },
        "source_sha256": {
            "track20_vid30_json": _sha256(vid30_json),
            "track20_vid31_json": _sha256(vid31_json),
            "cholec80_video30_phase": _sha256(phase_paths[30]),
            "cholec80_video31_phase": _sha256(phase_paths[31]),
        },
        "derived_sha256": {
            "vid30_candidate_validation": _sha256(vid30_candidate_path),
            "vid31_image_phase_supervision": _sha256(vid31_phase_path),
        },
    }
    manifest_path = output_dir / "derived_supervision_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    record = [
        "# CholecTrack20 VID30/VID31 Derived Supervision Record",
        "",
        "## Source-data guarantee",
        "",
        f"- Raw dataset root: `{dataset_root}`.",
        "- No raw CholecTrack20 or Cholec80 file was renamed, moved, overwritten, or deleted.",
        "- Every derived source and source hash is recorded in `derived_supervision_manifest.json`.",
        "",
        "## VID30 validation",
        "",
        f"- Media: original Track20 VID30 PNGs ({len(vid30_media)} frames).",
        "- Core fields: copied from Track20 VID31 JSON at identical frame IDs.",
        "- Phase: replaced from Cholec80 video30 with frame offset `-11677`.",
        "- Visual-condition supervision: disabled.",
        f"- Candidate annotation: `{vid30_candidate_path}`.",
        f"- QA: {args.sample_count} dispersed bbox-overlay frames across {len(review_paths)} sheets.",
        "- Status: candidate repaired validation supervision; not an official maintainer correction.",
        "",
        "## VID31 training",
        "",
        f"- Media: original Track20 VID31 PNGs ({len(vid31_media)} frames).",
        f"- Phase-supervised frames: {len(vid31_phase_values)}.",
        f"- Excluded image frames without aligned Cholec80 phase: {len(vid31_phase_payload['excluded_track20_image_frame_ids'])}.",
        "- Phase: Cholec80 video31 with frame offset `-6152`.",
        "- Detection, IVT, operator, bbox, tracking, and visual-condition supervision: disabled.",
        f"- Phase source: `{vid31_phase_path}`.",
        "",
        "## Runtime boundary",
        "",
        "Training and evaluation code must load `derived_supervision_manifest.json` explicitly.",
        "The ordinary raw parser remains unchanged and must not silently substitute these sources.",
    ]
    record_path = output_dir / "DERIVED_SUPERVISION_MODIFICATION_RECORD.md"
    record_path.write_text("\n".join(record) + "\n", encoding="utf-8")
    return {
        "manifest": str(manifest_path.resolve()),
        "modification_record": str(record_path.resolve()),
        "vid30_candidate": str(vid30_candidate_path.resolve()),
        "vid31_phase_supervision": str(vid31_phase_path.resolve()),
        "raw_dataset_modified": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cholec80-root", type=Path, required=True)
    parser.add_argument(
        "--identity-audit",
        type=Path,
        default=Path(
            "artifacts/data_resolution/cholectrack20_vid30_vid31/"
            "upstream_identity_audit.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data_resolution/cholectrack20_vid30_vid31"),
    )
    parser.add_argument("--sample-count", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(_parse_args()), indent=2, ensure_ascii=False))
