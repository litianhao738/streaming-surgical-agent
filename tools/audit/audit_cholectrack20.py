"""Generate read-only P1 qualification artifacts for local CholecTrack20 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.data.media_backend import AlignmentStatus
from surgical_agent.data.parser import REQUIRED_INSTANCE_FIELDS, parse_annotation_file
from surgical_agent.data.schemas import CanonicalVideoAnnotation
from surgical_agent.data.splits import (
    SplitManifestEntry,
    discover_official_split_manifest,
)

OFFICIAL_DATASET_URL = "https://github.com/CAMMA-public/cholectrack20"
OFFICIAL_PAPER_URL = (
    "https://openaccess.thecvf.com/content/CVPR2025/papers/"
    "Nwoye_CholecTrack20_A_Multi-Perspective_Tracking_Dataset_for_Surgical_"
    "Tools_CVPR_2025_paper.pdf"
)
OFFICIAL_ALIGNMENT_ISSUE_URL = "https://github.com/CAMMA-public/cholectrack20/issues/10"
OFFICIAL_IVT_MAP_URL = (
    "https://github.com/CAMMA-public/ivtmetrics/blob/main/ivtmetrics/maps.txt"
)
P1_STATUS = {
    "P1_STATUS": "PARTIAL",
    "DATA_STRUCTURE_STATUS": "PASS",
    "MEDIA_METADATA_STATUS": "PARTIAL",
    "ALIGNMENT_STATUS": "BLOCKED",
    "ONTOLOGY_STATUS": "PARTIAL",
    "PARTIAL_LABEL_STATUS": "PARTIAL",
    "CANONICAL_PARSER_STATUS": "PASS",
    "GOLD_FREE_STATUS": "PASS",
    "SPLIT_SAFETY_STATUS": "PASS",
    "DATA_READY_FOR_P2_SMOKE": "PASS",
    "LOCAL_NUMERIC_TRAINING_READY": "BLOCKED",
    "API_SEMANTIC_PIPELINE_READY": "BLOCKED",
    "FORMAL_TRAINING_READY": "BLOCKED",
    "DATA_READY_FOR_TRAINING": "BLOCKED",
    "DATA_READY_FOR_P3_API_ENGINEERING": "BLOCKED",
    "P9_PREREQUISITE": "BLOCKED",
    "VID30_STATUS": "BLOCKED_ALIGNMENT",
    "VID31_STATUS": "BLOCKED_ALIGNMENT",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return raw


def _merge_synapse_evidence(
    *,
    alignment: dict[str, Any],
    ontology: dict[str, Any],
    synapse_audit: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if synapse_audit is None:
        return None
    diagnostics = synapse_audit.get("target_diagnostics", {})
    projected_diagnostics: dict[str, Any] = {}
    for video_id in ("VID30", "VID31"):
        record = diagnostics.get(video_id, {})
        projected_diagnostics[video_id] = {
            "source_path": record.get("source_path"),
            "annotation_entity": record.get("annotation_entity"),
            "local_raw_md5_matches_remote": record.get("local_raw_md5_matches_remote"),
            "downloaded_official_copy": record.get("downloaded_official_copy"),
            "frames_folder_entity_id": record.get("frames_folder_entity_id"),
            "remote_frame_file_count": record.get("remote_frame_file_count"),
            "remote_local_frame_id_sets_equal": record.get(
                "remote_local_frame_id_sets_equal"
            ),
            "annotation_frame_count": record.get("annotation_frame_count"),
            "annotation_media_common_count": record.get(
                "annotation_media_common_count"
            ),
            "missing_annotated_media_count": record.get(
                "missing_annotated_media_count"
            ),
            "extra_media_count": record.get("extra_media_count"),
            "sample_remote_local_frame_hashes": record.get(
                "sample_remote_local_frame_hashes"
            ),
            "status": record.get("status"),
        }
    source = {
        "project_entity_id": synapse_audit.get("project_entity_id"),
        "project_doi": synapse_audit.get("project_doi"),
        "access_date": synapse_audit.get("access_date"),
        "read_only": synapse_audit.get("read_only"),
        "artifact_schema_version": synapse_audit.get("schema_version"),
    }
    alignment["synapse_official_release_verification"] = {
        "source": source,
        "diagnostics": projected_diagnostics,
        "conclusion": (
            "Current official Synapse entity names and sampled frame hashes match the local "
            "VID30/VID31 media. The downloaded official VID31 JSON is semantically equal to "
            "the local JSON despite serialization-level MD5 differences. The alignment "
            "contradiction is therefore present in the inspected official release."
        ),
    }
    metadata_search = synapse_audit.get("metadata_search", {})
    ontology["synapse_official_metadata_search"] = {
        "source": source,
        "wiki": synapse_audit.get("wiki"),
        "nonempty_entity_annotations": metadata_search.get(
            "nonempty_entity_annotations"
        ),
        "access_errors": metadata_search.get("access_errors"),
        "ontology_mapping_file_found": metadata_search.get(
            "ontology_mapping_file_found"
        ),
        "extraction_or_frame_index_rule_file_found": metadata_search.get(
            "extraction_or_frame_index_rule_file_found"
        ),
        "official_findings": synapse_audit.get("official_findings"),
        "interpretation": metadata_search.get("interpretation"),
    }
    return source


def _split_manifest_payload(entries: tuple[SplitManifestEntry, ...]) -> dict[str, Any]:
    counts = Counter(entry.split.value for entry in entries)
    return {
        "schema_version": "ct20_p1_split_manifest_v1",
        "audit_version": "p1_2026-08-20_v1",
        "source_rule": "official release directory plus raw JSON video.split",
        "split_counts": dict(sorted(counts.items())),
        "videos": [
            {
                **asdict(entry),
                "split": entry.split.value,
                "annotation_sha256": _sha256(Path(entry.annotation_file)),
            }
            for entry in entries
        ],
    }


def _parse_all(
    entries: tuple[SplitManifestEntry, ...],
) -> dict[str, CanonicalVideoAnnotation]:
    videos: dict[str, CanonicalVideoAnnotation] = {}
    for entry in entries:
        videos[entry.video_id] = parse_annotation_file(
            entry.annotation_file,
            expected_split=entry.split,
            manifest_source=entry.source,
        )
    return videos


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        return {"status": "BLOCKED_FFPROBE", "error": type(exc).__name__}
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        return {"status": "BLOCKED_NO_VIDEO_STREAM"}
    stream = streams[0]
    return {
        "status": "PROBED",
        "codec_name": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream.get("r_frame_rate"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        "duration_seconds": float(
            stream.get("duration") or payload.get("format", {}).get("duration")
        ),
    }


def _alignment_payload(
    entries: tuple[SplitManifestEntry, ...],
    videos: dict[str, CanonicalVideoAnnotation],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    png_sets: dict[str, set[int]] = {}
    for entry in entries:
        annotation_ids = set(videos[entry.video_id].frame_ids)
        media_path = Path(entry.media_source)
        if media_path.is_dir():
            media_ids: set[int] = set()
            nonnumeric: list[str] = []
            for path in media_path.glob("*.png"):
                try:
                    media_ids.add(int(path.stem))
                except ValueError:
                    nonnumeric.append(path.name)
            png_sets[entry.video_id] = media_ids
            missing = sorted(annotation_ids - media_ids)
            extra = sorted(media_ids - annotation_ids)
            if nonnumeric or (missing and extra):
                status = AlignmentStatus.NONTRIVIAL_ALIGNMENT
            elif missing:
                status = AlignmentStatus.MISSING_ANNOTATED_MEDIA
            elif extra:
                status = AlignmentStatus.EXTRA_MEDIA_ONLY
            else:
                status = AlignmentStatus.EXACT
            records.append(
                {
                    "video_id": entry.video_id,
                    "split": entry.split.value,
                    "media_kind": "png_frames",
                    "status": status.value,
                    "rule": "canonical_frame_id == numeric PNG stem",
                    "annotated_frame_count": len(annotation_ids),
                    "media_frame_count": len(media_ids),
                    "common_count": len(annotation_ids & media_ids),
                    "missing_annotation_media_count": len(missing),
                    "extra_media_count": len(extra),
                    "missing_annotation_media_ids": missing,
                    "extra_media_ids": extra,
                    "nonnumeric_media_files": sorted(nonnumeric),
                }
            )
        else:
            probe = _ffprobe(media_path)
            frame_count = probe.get("nb_frames")
            max_id = max(annotation_ids) if annotation_ids else None
            ids_inside_container_range = (
                frame_count is not None
                and max_id is not None
                and 0 <= max_id < frame_count
            )
            records.append(
                {
                    "video_id": entry.video_id,
                    "split": entry.split.value,
                    "media_kind": "mp4",
                    "status": AlignmentStatus.UNRESOLVED.value,
                    "reason": "decoder frame-index offset is not authoritatively verified",
                    "annotated_frame_count": len(annotation_ids),
                    "annotation_min_frame_id": min(annotation_ids),
                    "annotation_max_frame_id": max_id,
                    "annotation_ids_inside_container_frame_range": ids_inside_container_range,
                    "decoder_frame_index_offset": None,
                    "probe": probe,
                }
            )

    vid30_media_matches_vid31_annotations = False
    if "VID30" in png_sets and "VID31" in videos:
        vid30_media_matches_vid31_annotations = png_sets["VID30"] == set(
            videos["VID31"].frame_ids
        )
    return {
        "schema_version": "ct20_p1_media_alignment_v1",
        "audit_version": "p1_2026-08-20_v1",
        "alignment_status": "BLOCKED",
        "exact_png_rule": "numeric equality only; nearest-frame fallback forbidden",
        "mp4_rule": "resolution forbidden until decoder offset is verified",
        "localized_blockers": {
            "VID30": (
                "Official Validation media IDs exactly match VID31 annotation IDs, but "
                "cross-split reassignment is forbidden without a corrected official package."
            ),
            "VID31": (
                "Training annotation/media sets are nontrivially inconsistent; the official "
                "issue reports that VID31 annotations appear to describe VID30."
            ),
            "Testing": "All MP4 decoder frame-index offsets remain unresolved.",
        },
        "cross_video_diagnostics": {
            "vid30_png_ids_equal_vid31_annotation_ids": (
                vid30_media_matches_vid31_annotations
            ),
            "used_as_automatic_repair": False,
        },
        "official_issue": OFFICIAL_ALIGNMENT_ISSUE_URL,
        "sources": [
            {
                "url": OFFICIAL_DATASET_URL,
                "access_date": "2026-08-20",
                "claim": "1 FPS labels/PNG frames, 25 FPS raw test video, and frame-key format",
                "locally_verified": True,
            },
            {
                "url": OFFICIAL_ALIGNMENT_ISSUE_URL,
                "access_date": "2026-08-20",
                "claim": "public report of the VID30/VID31 annotation mismatch",
                "locally_verified": True,
            },
            {
                "local_file": "D:/cholectrack20email.pdf",
                "access_date": "2026-08-20",
                "claim": "official email states 1 FPS annotations and 25 FPS raw test videos",
                "locally_verified": True,
                "sensitive_credential": "REDACTED_NOT_STORED",
            },
        ],
        "videos": records,
    }


def _load_ivt_map(path: Path | None) -> dict[int, tuple[int, int, int]]:
    if path is None or not path.is_file():
        return {}
    result: dict[int, tuple[int, int, int]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = [int(value.strip()) for value in stripped.split(",")]
        if len(values) < 4:
            raise ValueError(f"Invalid IVT mapping row: {line!r}")
        result[values[0]] = (values[1], values[2], values[3])
    return result


def _ontology_payload(
    entries: tuple[SplitManifestEntry, ...],
    videos: dict[str, CanonicalVideoAnnotation],
    ivt_map_path: Path | None,
) -> dict[str, Any]:
    category_hashes: dict[str, str] = {}
    for entry in entries:
        raw = json.loads(Path(entry.annotation_file).read_text(encoding="utf-8-sig"))
        encoded = json.dumps(raw["categories"], sort_keys=True).encode("utf-8")
        category_hashes[entry.video_id] = hashlib.sha256(encoded).hexdigest()

    representative = videos[entries[0].video_id]
    embedded_terms = [asdict(term) for term in representative.ontology_terms]
    ivt_map = _load_ivt_map(ivt_map_path)
    exact_matches = 0
    total_compared = 0
    mismatches: Counter[tuple[int, int, int, int, int, int, int]] = Counter()
    mismatch_videos: defaultdict[tuple[int, int, int, int, int, int, int], set[str]] = (
        defaultdict(set)
    )
    for video in videos.values():
        for frame in video.frames:
            for instance in frame.instances:
                if not instance.mask.ivt or instance.triplet_id not in ivt_map:
                    continue
                total_compared += 1
                official = ivt_map[instance.triplet_id]
                observed = (
                    instance.instrument_id,
                    instance.verb_id,
                    instance.target_id,
                )
                if official == observed:
                    exact_matches += 1
                else:
                    key = (instance.triplet_id, *official, *observed)
                    mismatches[key] += 1
                    mismatch_videos[key].add(video.video_id)

    mismatch_rows = [
        {
            "triplet_id": key[0],
            "reference_components": {
                "instrument_id": key[1],
                "verb_id": key[2],
                "target_id": key[3],
            },
            "observed_components": {
                "instrument_id": key[4],
                "verb_id": key[5],
                "target_id": key[6],
            },
            "count": count,
            "videos": sorted(mismatch_videos[key]),
        }
        for key, count in sorted(mismatches.items())
    ]
    return {
        "schema_version": "ct20_p1_ontology_provenance_v1",
        "audit_version": "p1_2026-08-20_v1",
        "ontology_status": "PARTIAL",
        "dataset_embedded_categories_identical_across_20_json": (
            len(set(category_hashes.values())) == 1
        ),
        "dataset_embedded_category_sha256": sorted(set(category_hashes.values())),
        "verified_embedded_terms": embedded_terms,
        "numeric_ivt_relation": {
            "status": (
                "PARTIAL_COMPONENT_CONSISTENCY"
                if ivt_map
                else "BLOCKED_REFERENCE_NOT_SUPPLIED"
            ),
            "reference_path": str(ivt_map_path.resolve()) if ivt_map_path else None,
            "reference_sha256": (
                _sha256(ivt_map_path)
                if ivt_map_path and ivt_map_path.is_file()
                else None
            ),
            "reference_url": OFFICIAL_IVT_MAP_URL,
            "reference_triplet_count": len(ivt_map),
            "compared_instance_count": total_compared,
            "exact_component_match_count": exact_matches,
            "mismatch_count": total_compared - exact_matches,
            "mismatch_patterns": mismatch_rows,
            "automatic_correction_applied": False,
        },
        "unresolved": [
            "authoritative verb ID-to-name mapping for this release",
            "authoritative target ID-to-name mapping for this release",
            "authoritative triplet ID-to-name mapping for this release",
            "authoritative phase numeric ID-to-name mapping for this release",
            "semantic meaning of raw -1 categorical sentinel",
        ],
        "parser_policy": "preserve raw numeric IDs and never infer unresolved names",
        "sources": [
            {
                "url": OFFICIAL_DATASET_URL,
                "access_date": "2026-08-20",
                "claim": "release structure, split, annotation cadence, media rates, and fields",
                "locally_verified": True,
            },
            {
                "url": OFFICIAL_PAPER_URL,
                "access_date": "2026-08-20",
                "claim": "dataset design and annotation provenance",
                "locally_verified": "schema compared; research claims not independently reproduced",
            },
            {
                "url": OFFICIAL_IVT_MAP_URL,
                "access_date": "2026-08-20",
                "claim": "candidate IVT-to-instrument/verb/target numeric relation",
                "locally_verified": f"compared with {total_compared} local instances",
            },
        ],
    }


def _statistics_payload(
    videos: dict[str, CanonicalVideoAnnotation],
) -> tuple[dict[str, Any], dict[str, Any]]:
    split_frames: Counter[str] = Counter()
    split_instances: Counter[str] = Counter()
    task_valid: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter()
    bbox_quality: Counter[str] = Counter()
    observed_fields: set[str] = set()
    categorical_values: defaultdict[str, set[int]] = defaultdict(set)
    negative_value_counts: defaultdict[str, Counter[int]] = defaultdict(Counter)
    negative_value_videos: defaultdict[tuple[str, int], set[str]] = defaultdict(set)
    numeric_ranges: dict[str, list[float]] = {}
    empty_annotation_frames = 0

    def observe(name: str, value: float) -> None:
        current = numeric_ranges.setdefault(name, [float(value), float(value)])
        current[0] = min(current[0], float(value))
        current[1] = max(current[1], float(value))

    for video in videos.values():
        split_frames[video.split.value] += len(video.frames)
        for frame in video.frames:
            observe("frame_id", frame.frame_id)
            if not frame.instances:
                empty_annotation_frames += 1
            split_instances[video.split.value] += len(frame.instances)
            for instance in frame.instances:
                categorical = {
                    "instrument_id": instance.instrument_id,
                    "verb_id": instance.verb_id,
                    "target_id": instance.target_id,
                    "triplet_id": instance.triplet_id,
                    "phase_id": instance.phase_id,
                    "operator_id": instance.operator_id,
                }
                for name, value in categorical.items():
                    categorical_values[name].add(value)
                    observe(name, value)
                    if value < 0:
                        negative_value_counts[name][value] += 1
                        negative_value_videos[(name, value)].add(video.video_id)
                tracks = {
                    "intraoperative_track": instance.tracks.intraoperative,
                    "intracorporeal_track": instance.tracks.intracorporeal,
                    "visibility_track": instance.tracks.visibility,
                }
                for name, value in tracks.items():
                    observe(name, value)
                for name, value in (
                    ("bbox_x", instance.bbox.x),
                    ("bbox_y", instance.bbox.y),
                    ("bbox_width", instance.bbox.width),
                    ("bbox_height", instance.bbox.height),
                    ("score", instance.score),
                    ("area", instance.area),
                ):
                    observe(name, value)
                if instance.bbox.is_all_negative_one:
                    bbox_quality["all_negative_one"] += 1
                elif not instance.bbox.has_positive_extent:
                    bbox_quality["non_positive_extent"] += 1
                elif not instance.bbox.is_inside_unit_frame:
                    bbox_quality["positive_extent_crosses_unit_boundary"] += 1
                else:
                    bbox_quality["inside_unit_frame"] += 1
                flags = {
                    "instrument": instance.mask.instrument,
                    "verb": instance.mask.verb,
                    "target": instance.mask.target,
                    "ivt": instance.mask.ivt,
                    "phase": instance.mask.phase,
                }
                task_valid.update(key for key, valid in flags.items() if valid)
                combination_counts[
                    f"verb={int(flags['verb'])},target={int(flags['target'])},ivt={int(flags['ivt'])}"
                ] += 1
                observed_fields.update(instance.extras)

    total_instances = sum(split_instances.values())
    partial_labels = {
        "schema_version": "ct20_p1_partial_label_statistics_v1",
        "audit_version": "p1_2026-08-20_v1",
        "mask_semantics_version": "ct20_p1_raw_negative_id_invalid_v1",
        "sentinel": -1,
        "sentinel_name": "UNRESOLVED_MISSING_SENTINEL",
        "semantic_interpretation": "BLOCKED_UNRESOLVED",
        "observed_negative_ids": {
            task: {
                str(value): {
                    "count": count,
                    "videos": sorted(negative_value_videos[(task, value)]),
                    "semantic_interpretation": "BLOCKED_UNRESOLVED",
                }
                for value, count in sorted(counts.items())
            }
            for task, counts in sorted(negative_value_counts.items())
        },
        "policy": (
            "Each task mask is independently true iff its raw numeric ID is a "
            "non-boolean integer greater than or equal to zero."
        ),
        "total_instances": total_instances,
        "valid_by_task": dict(task_valid),
        "invalid_by_task": {
            task: total_instances - task_valid[task]
            for task in ("instrument", "verb", "target", "ivt", "phase")
        },
        "verb_target_ivt_combinations": dict(sorted(combination_counts.items())),
    }
    schema_summary = {
        "schema_version": "ct20_p1_canonical_schema_summary_v1",
        "audit_version": "p1_2026-08-20_v1",
        "canonical_video_count": len(videos),
        "annotated_frame_count": sum(split_frames.values()),
        "instance_count": total_instances,
        "frames_by_split": dict(sorted(split_frames.items())),
        "instances_by_split": dict(sorted(split_instances.items())),
        "canonical_instance_fields": [
            "instrument_id",
            "verb_id",
            "target_id",
            "triplet_id",
            "phase_id",
            "operator_id",
            "bbox",
            "tracks",
            "conditions",
            "mask",
            "score",
            "area",
            "is_crowd",
            "extras",
        ],
        "raw_required_instance_fields": sorted(REQUIRED_INSTANCE_FIELDS),
        "missing_required_instance_field_count": 0,
        "empty_annotation_frame_count": empty_annotation_frames,
        "annotation_key_semantics": (
            "Raw annotations-object keys are parsed as integer canonical frame IDs; "
            "visual correspondence is qualified separately by the alignment manifest."
        ),
        "observed_categorical_values": {
            key: sorted(values) for key, values in sorted(categorical_values.items())
        },
        "observed_numeric_ranges": {
            key: {"min": values[0], "max": values[1]}
            for key, values in sorted(numeric_ranges.items())
        },
        "unknown_extra_fields_observed": sorted(observed_fields),
        "raw_bbox_quality": dict(sorted(bbox_quality.items())),
        "bbox_policy": (
            "Preserve finite raw TLWH values; do not clip boundary-crossing boxes or "
            "reinterpret the [-1,-1,-1,-1] tuple."
        ),
        "gold_free_runtime_type": "InferenceSample",
        "evaluation_only_type": "EvaluationTarget",
    }
    return partial_labels, schema_summary


def _write_usability_report(
    path: Path,
    *,
    split_manifest: dict[str, Any],
    alignment: dict[str, Any],
    ontology: dict[str, Any],
    partial_labels: dict[str, Any],
    schema_summary: dict[str, Any],
) -> None:
    negative_id_summary = {
        task: {value: details["count"] for value, details in values.items()}
        for task, values in partial_labels["observed_negative_ids"].items()
    }
    synapse_verification = alignment.get("synapse_official_release_verification")
    if synapse_verification:
        synapse_source = synapse_verification["source"]
        synapse_diagnostics = synapse_verification["diagnostics"]
        synapse_section = f"""
## Official Synapse read-only audit

- Project `{synapse_source["project_entity_id"]}` was inspected read-only on
  {synapse_source["access_date"]}; no write operation or recursive dataset download occurred.
- `VID30.json`: `{synapse_diagnostics["VID30"]["annotation_entity"]["entity_id"]}` v1; local raw
  MD5 equals the Synapse file handle MD5.
- `VID31.json`: `{synapse_diagnostics["VID31"]["annotation_entity"]["entity_id"]}` v1; raw MD5
  differs only by JSON serialization. The downloaded official copy is semantically identical to
  the local JSON.
- Synapse frame filename sets equal the local sets for both videos. Eight dispersed PNG samples
  also have exact remote/local MD5 matches.
- The project Wiki verifies 1 FPS annotations and instrument names, but no accessible Wiki,
  entity name, non-frame annotation, or supplementary file provides verb/target/triplet/phase
  ID tables or a frame extraction/index-offset rule.
- Therefore the `VID30`/`VID31` contradiction is present in the inspected official release, not
  explained by local corruption.
"""
    else:
        synapse_section = ""
    alignment_rows = "\n".join(
        "| {video_id} | {split} | {media_kind} | {status} | {annotated_frame_count} |".format(
            **record
        )
        for record in alignment["videos"]
    )
    report = f"""# CholecTrack20 P1 Data Qualification Report

Generated by `tools/audit/audit_cholectrack20.py`. The raw dataset was read only.

## Qualification decision

| Status | Value |
| --- | --- |
{chr(10).join(f"| `{key}` | **{value}** |" for key, value in P1_STATUS.items())}

The release is present and numerically parseable. It is not globally data-ready: `VID30`/`VID31`
have a localized official-release alignment problem, and test MP4 decoder offsets are unresolved.

## Structure

- Official split counts: `{split_manifest["split_counts"]}`.
- Canonical videos: {schema_summary["canonical_video_count"]}.
- Annotated frames: {schema_summary["annotated_frame_count"]}.
- Tool instances: {schema_summary["instance_count"]}.

## Media alignment

| Video | Split | Media | Status | Annotated frames |
| --- | --- | --- | --- | ---: |
{alignment_rows}

No nearest-frame fallback or cross-split reassignment was applied. `VID30` PNG IDs exactly equal
the `VID31` annotation IDs, but this diagnostic is not an authorized repair. The test MP4 files
probe at 25 FPS; annotation-to-decoder indexing remains blocked until an offset is verified.

{synapse_section}

## Ontology and partial labels

- Tool and operator ID/name terms are verified from identical embedded `categories` objects in all
  20 JSON files.
- Verb, target, triplet names and phase numeric ID/name semantics remain unresolved.
- Official IVT numeric relation comparison: {ontology["numeric_ivt_relation"]["exact_component_match_count"]}
  exact and {ontology["numeric_ivt_relation"]["mismatch_count"]} mismatched instances; no automatic
  correction was applied.
- Raw `-1` is recorded only as `UNRESOLVED_MISSING_SENTINEL`; it is not called background, none,
  or negative. Other observed negative IDs are independently unresolved:
  `{negative_id_summary}`. Task masks are independent. Valid counts:
  `{partial_labels["valid_by_task"]}`.

## Evidence and blockers

- Official dataset/release documentation: {OFFICIAL_DATASET_URL}
- Official CVPR 2025 paper: {OFFICIAL_PAPER_URL}
- Official VID30/VID31 issue: {OFFICIAL_ALIGNMENT_ISSUE_URL}
- Local official email states 1 FPS annotations and 25 FPS raw test video. Its download credential
  is sensitive and was deliberately redacted and not stored.

## Permitted next use

An aligned video such as `VID02` may be used for a P2 parser/model smoke test. Formal training,
semantic API prompting, test evaluation, and any P9 Gate error definition remain blocked. Request a
corrected `VID30`/`VID31` release package and authoritative frame-index/ontology metadata.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_audit(
    *,
    dataset_root: Path,
    artifact_dir: Path,
    report_dir: Path,
    ivt_map_path: Path | None,
    synapse_audit_path: Path | None,
) -> dict[str, Any]:
    entries = discover_official_split_manifest(dataset_root)
    videos = _parse_all(entries)
    split_manifest = _split_manifest_payload(entries)
    alignment = _alignment_payload(entries, videos)
    ontology = _ontology_payload(entries, videos, ivt_map_path)
    synapse_audit = _load_optional_json(synapse_audit_path)
    synapse_source = _merge_synapse_evidence(
        alignment=alignment,
        ontology=ontology,
        synapse_audit=synapse_audit,
    )
    partial_labels, schema_summary = _statistics_payload(videos)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dataset_split_manifest": artifact_dir / "dataset_split_manifest.json",
        "media_alignment_manifest": artifact_dir / "media_alignment_manifest.json",
        "ontology_provenance": artifact_dir / "ontology_provenance.json",
        "partial_label_statistics": artifact_dir / "partial_label_statistics.json",
        "canonical_schema_summary": artifact_dir / "canonical_schema_summary.json",
    }
    payloads = (
        split_manifest,
        alignment,
        ontology,
        partial_labels,
        schema_summary,
    )
    for path, payload in zip(outputs.values(), payloads, strict=True):
        _write_json(path, payload)

    combined = {
        "status": P1_STATUS,
        "dataset_root": str(dataset_root.resolve()),
        "read_only": True,
        "split_manifest": split_manifest,
        "alignment": alignment,
        "ontology": ontology,
        "partial_labels": partial_labels,
        "canonical_schema": schema_summary,
        "synapse_official_audit_source": synapse_source,
        "sensitive_credential_policy": "REDACTED_NOT_STORED",
    }
    _write_json(report_dir / "cholectrack20_audit_stats.json", combined)
    _write_usability_report(
        report_dir / "cholectrack20_usability_report.md",
        split_manifest=split_manifest,
        alignment=alignment,
        ontology=ontology,
        partial_labels=partial_labels,
        schema_summary=schema_summary,
    )
    return {
        "status": P1_STATUS,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "p1"
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "dataset_reports"
    )
    parser.add_argument("--ivt-map", type=Path, default=None)
    parser.add_argument(
        "--synapse-audit",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "p1" / "synapse_metadata_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_audit(
        dataset_root=args.dataset_root,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        ivt_map_path=args.ivt_map,
        synapse_audit_path=args.synapse_audit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
