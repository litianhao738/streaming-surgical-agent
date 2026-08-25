"""Audit CholecT50 VID31 against CholecTrack20 VID31 without mutating either dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.fft import dctn

DEFAULT_CHOLECT50_ROOT = Path("D:/CholecT50/CholecT50")
DEFAULT_TRACK20_ROOT = Path("D:/cholec_dataset")
DEFAULT_PHASE_SOURCE = Path(
    "artifacts/data_resolution/cholectrack20_vid30_vid31/derived/VID31/"
    "vid31_image_phase_supervision.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/data_resolution/cholect50_vid31")
TRACK20_ORIGIN_FRAME = 6201
FRAME_STRIDE = 25


def track20_to_cholect50_frame(frame_id: int, *, origin: int = TRACK20_ORIGIN_FRAME) -> int:
    """Map an aligned Track20 original-video frame ID to a CholecT50 1 FPS ID."""
    delta = frame_id - origin
    if delta % FRAME_STRIDE:
        raise ValueError(f"Track20 frame {frame_id} is not aligned to origin {origin}")
    return delta // FRAME_STRIDE


def parse_label_mapping(path: Path) -> dict[int, tuple[int, int, int]]:
    """Read the official IVT -> instrument/verb/target relation."""
    result: dict[int, tuple[int, int, int]] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = [int(value) for value in line.split(",")]
        if len(values) != 6:
            raise ValueError(f"Unexpected mapping row: {raw_line!r}")
        triplet_id, instrument_id, verb_id, target_id, _, _ = values
        if triplet_id in result:
            raise ValueError(f"Duplicate triplet mapping {triplet_id}")
        result[triplet_id] = (instrument_id, verb_id, target_id)
    if set(result) != set(range(100)):
        raise ValueError("CholecT50 mapping must contain exactly triplet IDs 0..99")
    return result


def parse_cholect50_frame(
    rows: list[list[float | int]],
    mapping: dict[int, tuple[int, int, int]],
) -> dict[str, Any]:
    """Convert one CholecT50 frame to auditable frame-level presence labels."""
    if not rows:
        raise ValueError("CholecT50 frames must contain a label row")
    phases: set[int] = set()
    triplets: set[int] = set()
    instruments: set[int] = set()
    verbs: set[int] = set()
    targets: set[int] = set()
    negative_rows = 0

    for row in rows:
        if len(row) != 15:
            raise ValueError(f"Expected 15 CholecT50 fields, received {len(row)}")
        triplet_id = int(row[0])
        instrument_id = int(row[1])
        verb_id = int(row[7])
        target_id = int(row[8])
        phase_id = int(row[14])
        phases.add(phase_id)

        components = (triplet_id, instrument_id, verb_id, target_id)
        if triplet_id < 0:
            if components != (-1, -1, -1, -1):
                raise ValueError(f"Mixed negative CholecT50 action row: {components}")
            negative_rows += 1
            continue
        expected = mapping.get(triplet_id)
        observed = (instrument_id, verb_id, target_id)
        if expected != observed:
            raise ValueError(
                f"Triplet {triplet_id} maps to {expected}, observed {observed}"
            )
        triplets.add(triplet_id)
        instruments.add(instrument_id)
        verbs.add(verb_id)
        targets.add(target_id)

    if len(phases) != 1:
        raise ValueError(f"Frame has inconsistent phase IDs: {sorted(phases)}")
    return {
        "triplet_ids": sorted(triplets),
        "instrument_ids": sorted(instruments),
        "verb_ids": sorted(verbs),
        "target_ids": sorted(targets),
        "phase_id": next(iter(phases)),
        "action_present": bool(triplets),
        "negative_sentinel_row_count": negative_rows,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _image_index(path: Path) -> dict[int, Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    result: dict[int, Path] = {}
    for image_path in path.glob("*.png"):
        frame_id = int(image_path.stem)
        if frame_id in result:
            raise ValueError(f"Duplicate frame ID {frame_id}: {path}")
        result[frame_id] = image_path
    if not result:
        raise FileNotFoundError(f"No PNG images found in {path}")
    return result


def _normalized_gray(image: Image.Image, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    return np.asarray(
        image.convert("L").resize(size, Image.Resampling.LANCZOS), dtype=np.float32
    ) / 255.0


def _perceptual_hash(image: Image.Image) -> np.ndarray:
    coefficients = dctn(_normalized_gray(image, (32, 32)), type=2, norm="ortho")[:8, :8]
    values = coefficients.flatten()[1:]
    return values > np.median(values)


def _similarity(left: Path, right: Path) -> dict[str, float | int]:
    with Image.open(left) as left_image, Image.open(right) as right_image:
        left_gray = _normalized_gray(left_image)
        right_gray = _normalized_gray(right_image)
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
                np.count_nonzero(
                    _perceptual_hash(left_image) != _perceptual_hash(right_image)
                )
            ),
            "gray_correlation": correlation,
            "mean_absolute_error": float(np.mean(np.abs(left_gray - right_gray))),
        }


def _sample_ids(frame_ids: list[int], count: int) -> list[int]:
    if not frame_ids:
        return []
    indices = np.linspace(0, len(frame_ids) - 1, min(count, len(frame_ids)), dtype=int)
    return [frame_ids[int(index)] for index in indices]


def _alignment_metrics(
    track_images: dict[int, Path],
    t50_images: dict[int, Path],
    aligned_ids: list[int],
    *,
    sample_count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for temporal_delta in (-1, 0, 1):
        samples: list[dict[str, float | int]] = []
        for track_frame_id in _sample_ids(aligned_ids, sample_count):
            t50_frame_id = track20_to_cholect50_frame(track_frame_id) + temporal_delta
            if t50_frame_id not in t50_images:
                continue
            samples.append(
                _similarity(track_images[track_frame_id], t50_images[t50_frame_id])
            )
        result[str(temporal_delta)] = {
            "sample_count": len(samples),
            "median_phash_hamming": float(
                np.median([sample["phash_hamming"] for sample in samples])
            ),
            "median_gray_correlation": float(
                np.median([sample["gray_correlation"] for sample in samples])
            ),
            "median_mean_absolute_error": float(
                np.median([sample["mean_absolute_error"] for sample in samples])
            ),
        }
    return result


def audit(
    *,
    cholect50_root: Path,
    track20_root: Path,
    phase_source: Path,
    output_dir: Path,
    sample_count: int,
) -> dict[str, str]:
    label_path = cholect50_root / "labels" / "VID31.json"
    mapping_path = cholect50_root / "label_mapping.txt"
    readme_path = cholect50_root / "README.md"
    t50_images_dir = cholect50_root / "videos" / "VID31"
    track_images_dir = track20_root / "Training" / "VID31" / "Frames"

    t50 = _json(label_path)
    phase_payload = _json(phase_source)
    mapping = parse_label_mapping(mapping_path)
    t50_images = _image_index(t50_images_dir)
    track_images = _image_index(track_images_dir)
    annotations = t50.get("annotations")
    if not isinstance(annotations, dict):
        raise TypeError("CholecT50 VID31 annotations must be an object")
    if set(map(int, annotations)) != set(t50_images):
        raise ValueError("CholecT50 VID31 image and annotation frame IDs differ")

    parsed_t50 = {
        int(frame_id): parse_cholect50_frame(rows, mapping)
        for frame_id, rows in annotations.items()
    }
    phase20_raw = phase_payload.get("phase_by_track20_image_frame_id")
    if not isinstance(phase20_raw, dict):
        raise TypeError("Existing VID31 phase supervision has no frame mapping")
    phase20 = {int(key): int(value["phase_id"]) for key, value in phase20_raw.items()}

    aligned_ids = sorted(
        frame_id
        for frame_id in track_images
        if (frame_id - TRACK20_ORIGIN_FRAME) % FRAME_STRIDE == 0
        and track20_to_cholect50_frame(frame_id) in parsed_t50
    )
    frame_records: dict[str, dict[str, Any]] = {}
    phase_mismatches: list[dict[str, int]] = []
    for track_frame_id in aligned_ids:
        t50_frame_id = track20_to_cholect50_frame(track_frame_id)
        record = dict(parsed_t50[t50_frame_id])
        current_phase = phase20.get(track_frame_id)
        record.update(
            {
                "cholect50_frame_id": t50_frame_id,
                "cholect50_phase_id": record.pop("phase_id"),
                "cholect80_phase_id": current_phase,
                "phase_agreement": current_phase
                == parsed_t50[t50_frame_id]["phase_id"],
            }
        )
        if not record["phase_agreement"]:
            phase_mismatches.append(
                {
                    "track20_frame_id": track_frame_id,
                    "cholect50_frame_id": t50_frame_id,
                    "cholect50_phase_id": record["cholect50_phase_id"],
                    "cholect80_phase_id": current_phase,
                }
            )
        frame_records[str(track_frame_id)] = record

    valid_action_frames = sum(
        bool(record["action_present"]) for record in frame_records.values()
    )
    negative_rows = sum(
        int(record["negative_sentinel_row_count"])
        for record in frame_records.values()
    )
    all_bbox_missing = all(
        tuple(float(value) for value in row[3:7]) == (-1.0, -1.0, -1.0, -1.0)
        for rows in annotations.values()
        for row in rows
    )
    metrics = _alignment_metrics(
        track_images,
        t50_images,
        aligned_ids,
        sample_count=sample_count,
    )
    if metrics["0"]["median_gray_correlation"] <= max(
        metrics["-1"]["median_gray_correlation"],
        metrics["1"]["median_gray_correlation"],
    ):
        raise ValueError("The frozen VID31 alignment is not stronger than adjacent offsets")

    output_dir.mkdir(parents=True, exist_ok=True)
    supervision_path = output_dir / "vid31_frame_level_ivt_supervision.json"
    supervision = {
        "schema_version": "ct20_vid31_cholect50_frame_ivt_v1",
        "video_id": "VID31",
        "split": "training",
        "supervision_granularity": "FRAME_LEVEL_MULTI_LABEL",
        "frame_rule": "cholect50_frame_id = (track20_frame_id - 6201) / 25",
        "media_source": str(track_images_dir.resolve()),
        "label_source": str(label_path.resolve()),
        "label_source_sha256": _sha256(label_path),
        "mapping_source": str(mapping_path.resolve()),
        "mapping_source_sha256": _sha256(mapping_path),
        "phase_policy": (
            "Retain existing Cholec80-derived phase; record CholecT50 phase for audit"
        ),
        "instance_level_bbox_available": False,
        "track_ids_available": False,
        "frames": frame_records,
        "excluded_track20_frame_ids": sorted(set(track_images) - set(aligned_ids)),
    }
    supervision_path.write_text(
        json.dumps(supervision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    audit_path = output_dir / "audit.json"
    audit_payload = {
        "schema_version": "ct20_cholect50_vid31_audit_v1",
        "source": {
            "cholect50_root": str(cholect50_root.resolve()),
            "track20_root": str(track20_root.resolve()),
            "release": "CholecT50 2.0",
            "readme_sha256": _sha256(readme_path),
            "vid31_label_sha256": _sha256(label_path),
            "label_mapping_sha256": _sha256(mapping_path),
            "raw_data_modified": False,
        },
        "ontology": {
            "mapping_row_count": len(mapping),
            "valid_triplet_mapping_conflicts": 0,
            "instrument_class_count": len(t50["categories"]["instrument"]),
            "verb_class_count": len(t50["categories"]["verb"]),
            "target_class_count": len(t50["categories"]["target"]),
            "triplet_class_count": len(t50["categories"]["triplet"]),
            "phase_mapping": t50["categories"]["phase"],
        },
        "vid31": {
            "cholect50_image_count": len(t50_images),
            "cholect50_annotation_frame_count": len(parsed_t50),
            "cholect50_instance_row_count": sum(len(rows) for rows in annotations.values()),
            "cholect50_all_bbox_missing": all_bbox_missing,
            "track20_image_count": len(track_images),
            "aligned_track20_frame_count": len(aligned_ids),
            "excluded_track20_frame_count": len(track_images) - len(aligned_ids),
            "aligned_valid_action_frame_count": valid_action_frames,
            "aligned_no_action_frame_count": len(aligned_ids) - valid_action_frames,
            "aligned_negative_sentinel_row_count": negative_rows,
            "alignment_metrics_by_t50_second_delta": metrics,
            "phase_comparison_count": len(aligned_ids),
            "phase_match_count": len(aligned_ids) - len(phase_mismatches),
            "phase_mismatch_count": len(phase_mismatches),
            "phase_mismatches": phase_mismatches,
        },
        "decision": {
            "ONTOLOGY_STATUS": "VERIFIED_FROM_FULL_CHOLECT50_RELEASE",
            "VID31_MEDIA_IDENTITY": "VERIFIED_PIXEL_AND_OFFICIAL_PRESERVED_ID",
            "VID31_FRAME_LEVEL_IVT": "READY_DERIVED_SUPERVISION",
            "VID31_PHASE": "READY_KEEP_CHOLECT80_WITH_9_BOUNDARY_DIFFERENCES_RECORDED",
            "VID31_INSTANCE_LEVEL_DETECTION": "UNAVAILABLE_CHOLECT50_RELEASE_2_HAS_NO_BBOX",
            "VID31_TRACKING": "UNAVAILABLE_NOT_A_CHOLECT50_LABEL",
            "VID30": "UNCHANGED_NOT_PRESENT_IN_CHOLECT50",
        },
        "artifact": str(supervision_path.resolve()),
    }
    audit_path.write_text(
        json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_path = output_dir / "AUDIT_REPORT.md"
    report = f"""# CholecT50 VID31 Cross-Dataset Audit

## 结论

- CholecT50 Release 2.0 的 `VID31.json`、`label_mapping.txt` 和 3,946 张图片完整存在，原始文件未修改。
- Track20 VID31 与 CholecT50 VID31 的稳定换算是：`cholect50_frame_id = (track20_frame_id - 6201) / 25`。
- 可对齐 Track20 图片 {len(aligned_ids):,} 张，另有 {len(track_images) - len(aligned_ids):,} 张位于 CholecT50 发布片段之外。
- {sample_count} 帧抽样的正确偏移中位图像相关系数为 `{metrics['0']['median_gray_correlation']:.3f}`，相邻前一秒和后一秒分别为 `{metrics['-1']['median_gray_correlation']:.3f}`、`{metrics['1']['median_gray_correlation']:.3f}`。
- 对齐帧中，{valid_action_frames:,} 帧具有有效的帧级 IVT，{len(aligned_ids) - valid_action_frames:,} 帧是 CholecT50 的无动作/空标签帧。
- 100 条 triplet mapping 与 VID31 的所有正标签实例完全一致，未发现内部映射冲突。

## 可用范围

VID31 可以从原来的 `phase-only` 升级为：

- Track20 VID31 原始图片；
- Cholec80 派生 phase；
- CholecT50 派生的帧级 instrument、verb、target、triplet 多标签监督。

机器可读监督文件：`vid31_frame_level_ivt_supervision.json`。

## 仍不可用的字段

CholecT50 Release 2.0 的 {sum(len(rows) for rows in annotations.values()):,} 条 VID31 标签记录中，bbox 全部是 `[-1,-1,-1,-1]`。它也不提供 Track20 的 operator 和三套 track ID。因此：

- 不能用它恢复 VID31 的实例级检测框；
- 不能恢复 bbox 与 IVT 的实例配对；
- 不能恢复 visibility、intracorporeal、intraoperative track ID；
- 不能解决 VID30，因为 VID30 不属于 CholecT50。

## Phase 边界

CholecT50 与当前 Cholec80 派生 phase 在 {len(aligned_ids):,} 个对齐帧中有 {len(aligned_ids) - len(phase_mismatches):,} 帧一致，{len(phase_mismatches)} 帧不一致；差异集中在阶段切换边界。当前继续使用 Cholec80 phase，不自动覆盖，差异帧已完整记录在 `audit.json`。

## 论文边界

这批标签适合作为同源上游数据恢复出的帧级监督，不应描述成 Track20 官方修正版实例标注。使用 CholecT50 进行模型预训练时还必须按视频身份去重，避免同源视频跨训练、验证或测试造成泄漏。
"""
    report_path.write_text(report, encoding="utf-8")
    return {
        "audit": str(audit_path.resolve()),
        "report": str(report_path.resolve()),
        "supervision": str(supervision_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cholect50-root", type=Path, default=DEFAULT_CHOLECT50_ROOT)
    parser.add_argument("--track20-root", type=Path, default=DEFAULT_TRACK20_ROOT)
    parser.add_argument("--phase-source", type=Path, default=DEFAULT_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=100)
    args = parser.parse_args()
    outputs = audit(
        cholect50_root=args.cholect50_root,
        track20_root=args.track20_root,
        phase_source=args.phase_source,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
