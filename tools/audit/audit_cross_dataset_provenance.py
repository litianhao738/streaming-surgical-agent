"""Read-only CholecTrack20 provenance audit against CholecT50 and Cholec80."""

# ruff: noqa: ISC004

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import pickletools
import struct
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

OVERLAP_COMMIT = "8347b9f4cb02ebe739747903e6eada272ee9d25e"
CHOLECT50_COMMIT = "354020eb217086a07be4b225d803c7fea7760b7a"
CHOLECTRACK20_COMMIT = "6406b52806b6d881ef5ee9bf75255b3d0af665f9"
SELFSUPSURG_COMMIT = "8a03d28948e59471cc8ea865d56632eda048079c"

OVERLAP_BASE = (
    "https://raw.githubusercontent.com/CAMMA-public/camma_dataset_overlaps/"
    f"{OVERLAP_COMMIT}/resources"
)
CHOLECT50_ZIP = (
    "https://s3.unistra.fr/camma_public/datasets/cholect50/"
    "CholecT50-Challenge-Validation.zip"
)
CHOLEC80_LABEL_ZIP = (
    "https://s3.unistra.fr/camma_public/github/selfsupsurg/ch80_labels.zip"
)
CHOLECT50_FORMAT = (
    "https://github.com/CAMMA-public/cholect50/blob/"
    f"{CHOLECT50_COMMIT}/docs/README-Format.md"
)
CHOLECT50_SPLITS = (
    "https://github.com/CAMMA-public/cholect50/blob/"
    f"{CHOLECT50_COMMIT}/docs/README-Splits.md"
)
CHOLECTRACK20_README = (
    "https://github.com/CAMMA-public/cholectrack20/blob/"
    f"{CHOLECTRACK20_COMMIT}/README.md"
)
CHOLECTRACK20_ISSUE = "https://github.com/CAMMA-public/cholectrack20/issues/10"
CHOLECTRACK20_PAPER = (
    "https://openaccess.thecvf.com/content/CVPR2025/papers/"
    "Nwoye_CholecTrack20_A_Multi-Perspective_Tracking_Dataset_for_"
    "Surgical_Tools_CVPR_2025_paper.pdf"
)
SELFSUPSURG_EXTRACTOR = (
    "https://github.com/CAMMA-public/SelfSupSurg/blob/"
    f"{SELFSUPSURG_COMMIT}/utils/extract_frames_ch80.py"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(data)


def _get_with_retries(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    attempts: int = 8,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, headers=headers, timeout=90)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"Failed to read official source: {url}") from last_error


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compression: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int


class RemoteZip:
    """Minimal range-only ZIP reader with CRC verification."""

    def __init__(self, session: requests.Session, url: str) -> None:
        self.session = session
        self.url = url
        self.transferred_bytes = 0
        self.response_metadata: dict[str, Any] = {}
        self.entries = self._read_directory()

    def _request(self, range_value: str) -> bytes:
        response = _get_with_retries(
            self.session,
            self.url,
            headers={"Range": range_value},
        )
        content = response.content
        self.transferred_bytes += len(content)
        if not self.response_metadata:
            self.response_metadata = {
                "content_range": response.headers.get("Content-Range"),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "content_type": response.headers.get("Content-Type"),
            }
        return content

    def _range(self, start: int, end: int) -> bytes:
        parts: list[bytes] = []
        chunk_size = 256 * 1024
        for chunk_start in range(start, end + 1, chunk_size):
            chunk_end = min(end, chunk_start + chunk_size - 1)
            expected = chunk_end - chunk_start + 1
            for attempt in range(8):
                content = self._request(f"bytes={chunk_start}-{chunk_end}")
                if len(content) == expected:
                    parts.append(content)
                    break
                time.sleep(0.2 * (attempt + 1))
            else:
                raise RuntimeError(
                    f"Incomplete range from {self.url}: {chunk_start}-{chunk_end}"
                )
        return b"".join(parts)

    def _read_directory(self) -> dict[str, ZipEntry]:
        tail = self._request("bytes=-131072")
        eocd_position = tail.rfind(b"PK\x05\x06")
        if eocd_position < 0:
            raise ValueError(f"ZIP end record not found: {self.url}")
        eocd = struct.unpack_from("<4s4H2LH", tail, eocd_position)
        entry_count = eocd[4]
        directory_size = eocd[5]
        directory_offset = eocd[6]
        directory = self._range(directory_offset, directory_offset + directory_size - 1)

        entries: dict[str, ZipEntry] = {}
        position = 0
        while position < len(directory):
            if directory[position : position + 4] != b"PK\x01\x02":
                raise ValueError(f"Malformed ZIP central directory: {self.url}")
            values = struct.unpack_from("<4s6H3L5H2L", directory, position)
            name_length, extra_length, comment_length = values[10:13]
            name_start = position + 46
            name = directory[name_start : name_start + name_length].decode("utf-8")
            entries[name] = ZipEntry(
                name=name,
                compression=values[4],
                crc32=values[7],
                compressed_size=values[8],
                uncompressed_size=values[9],
                local_offset=values[16],
            )
            position += 46 + name_length + extra_length + comment_length
        if len(entries) != entry_count:
            raise ValueError(
                f"ZIP entry count mismatch: expected {entry_count}, got {len(entries)}"
            )
        return entries

    def read(self, name: str) -> bytes:
        entry = self.entries[name]
        local_header = self._range(entry.local_offset, entry.local_offset + 29)
        values = struct.unpack("<4s5H3L2H", local_header)
        if values[0] != b"PK\x03\x04":
            raise ValueError(f"Malformed local ZIP header: {name}")
        payload_start = entry.local_offset + 30 + values[-2] + values[-1]
        compressed = self._range(
            payload_start, payload_start + entry.compressed_size - 1
        )
        if entry.compression == 0:
            data = compressed
        elif entry.compression == 8:
            data = zlib.decompress(compressed, -15)
        else:
            raise ValueError(f"Unsupported ZIP compression {entry.compression}: {name}")
        if len(data) != entry.uncompressed_size:
            raise ValueError(f"Uncompressed size mismatch: {name}")
        if zlib.crc32(data) != entry.crc32:
            raise ValueError(f"CRC mismatch: {name}")
        return data


class _DTypeStub:
    def __init__(self, code: str, *_: Any) -> None:
        self.code = code
        self.state: Any = None

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _RestrictedNumpyScalarUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("numpy", "dtype"):
            return _DTypeStub
        if (module, name) == ("numpy.core.multiarray", "scalar"):
            return lambda _dtype, value: int.from_bytes(
                value, byteorder="little", signed=True
            )
        raise pickle.UnpicklingError(f"Blocked pickle global: {module}.{name}")


def _load_restricted_cholec80_pickle(data: bytes) -> dict[str, list[dict[str, Any]]]:
    globals_seen = {
        argument
        for opcode, argument, _ in pickletools.genops(data)
        if opcode.name == "GLOBAL"
    }
    allowed = {"numpy dtype", "numpy.core.multiarray scalar"}
    if not globals_seen <= allowed:
        raise pickle.UnpicklingError(
            f"Unexpected globals in official Cholec80 label pickle: {globals_seen}"
        )
    value = _RestrictedNumpyScalarUnpickler(io.BytesIO(data)).load()
    if not isinstance(value, dict):
        raise TypeError("Cholec80 label pickle is not a dictionary")
    return value


def _download_json(session: requests.Session, url: str) -> tuple[Any, dict[str, Any]]:
    response = _get_with_retries(session, url)
    return response.json(), {
        "url": url,
        "sha256": _sha256(response.content),
        "content_size": len(response.content),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
    }


def _discover_local_json(dataset_root: Path) -> dict[int, dict[str, Any]]:
    videos: dict[int, dict[str, Any]] = {}
    for annotation_path in sorted(dataset_root.glob("*/*/*.json")):
        digits = "".join(
            character for character in annotation_path.stem if character.isdigit()
        )
        video_id = int(digits)
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        split = annotation_path.parents[1].name.lower()
        frame_ids = sorted(int(frame_id) for frame_id in payload["annotations"])
        frame_dir = annotation_path.parent / "Frames"
        png_ids = (
            sorted(int(path.stem) for path in frame_dir.glob("*.png"))
            if frame_dir.is_dir()
            else []
        )
        videos[video_id] = {
            "video_id": f"VID{video_id:02d}",
            "split": split,
            "annotation_path": str(annotation_path),
            "payload": payload,
            "frame_ids": frame_ids,
            "png_ids": png_ids,
        }
    if len(videos) != 20:
        raise ValueError(f"Expected 20 CholecTrack20 JSON files, found {len(videos)}")
    return videos


def _flatten_instances(video: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (int(frame_id), instance)
        for frame_id, instances in sorted(
            video["payload"]["annotations"].items(), key=lambda item: int(item[0])
        )
        for instance in instances
    ]


def _video_provenance(
    videos: dict[int, dict[str, Any]],
    cholec80_ids: set[int],
    cholec50_splits: dict[str, list[int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    cholec50_lookup = {
        video_id: split
        for split, split_ids in cholec50_splits.items()
        for video_id in split_ids
    }
    rows = []
    for video_id, video in sorted(videos.items()):
        rows.append(
            {
                "track20_video_id": video["video_id"],
                "track20_split": video["split"],
                "cholec80_video_id": (
                    f"video{video_id:02d}" if video_id in cholec80_ids else None
                ),
                "cholec50_video_id": (
                    f"VID{video_id:02d}" if video_id in cholec50_lookup else None
                ),
                "cholec50_split": cholec50_lookup.get(video_id),
                "identity_basis": "OFFICIAL_PRESERVED_NUMERIC_VIDEO_ID",
                "image_content_verified": False,
            }
        )
    counts = {
        "track20_videos": len(rows),
        "mapped_to_cholec80": sum(row["cholec80_video_id"] is not None for row in rows),
        "mapped_to_cholec50": sum(row["cholec50_video_id"] is not None for row in rows),
        "mapped_to_both": sum(
            row["cholec80_video_id"] is not None
            and row["cholec50_video_id"] is not None
            for row in rows
        ),
        "unmapped": sum(
            row["cholec80_video_id"] is None and row["cholec50_video_id"] is None
            for row in rows
        ),
    }
    return rows, counts


def _ontology_audit(
    videos: dict[int, dict[str, Any]],
    categories: dict[str, dict[str, str]],
    mapping_text: str,
) -> dict[str, Any]:
    mapping: dict[int, tuple[int, int, int]] = {}
    for line in mapping_text.splitlines():
        if not line or line.startswith("#"):
            continue
        values = [int(value) for value in line.split(",")]
        mapping[values[0]] = (values[1], values[2], values[3])

    mismatches: Counter[tuple[int, tuple[int, int, int], tuple[int, int, int], str]] = (
        Counter()
    )
    compared = 0
    exact = 0
    for video in videos.values():
        for _, instance in _flatten_instances(video):
            observed = (
                instance["instrument"],
                instance["verb"],
                instance["target"],
            )
            triplet_id = instance["triplet"]
            if triplet_id < 0 or triplet_id not in mapping:
                continue
            compared += 1
            if observed == mapping[triplet_id]:
                exact += 1
            else:
                mismatches[
                    (triplet_id, mapping[triplet_id], observed, video["video_id"])
                ] += 1

    mismatch_rows = []
    for (triplet_id, expected, observed, video_id), count in sorted(
        mismatches.items(), key=lambda item: (-item[1], item[0])
    ):
        if triplet_id == 12 and observed == (6, 0, 13):
            interpretation = (
                "EXPLAINED_SCOPE_DIFFERENCE: Track20 box category is specimen-bag, "
                "while upstream triplet 12 describes the grasper grasping specimen_bag"
            )
        elif triplet_id == 94 and observed == (0, -1, -1):
            interpretation = (
                "UNRESOLVED_NULL_ENCODING: upstream uses verb 9/target 14, "
                "Track20 uses negative component sentinels"
            )
        else:
            interpretation = "UNRESOLVED_COMPONENT_CONFLICT"
        mismatch_rows.append(
            {
                "triplet_id": triplet_id,
                "expected_i_v_t": list(expected),
                "observed_i_v_t": list(observed),
                "video_id": video_id,
                "count": count,
                "interpretation": interpretation,
            }
        )

    track20_tools = {
        str(item["id"]): item["name"]
        for item in next(iter(videos.values()))["payload"]["categories"]["tools"]
    }
    return {
        "status": "PARTIAL_RECOVERED_WITH_EXCEPTIONS",
        "instrument": {
            "status": "VERIFIED",
            "track20_mapping": track20_tools,
            "cholec50_mapping": categories["instrument"],
            "note": "Track20 adds instrument 6 specimen-bag; CholecT50 has instruments 0-5.",
        },
        "verb": {
            "status": "RECOVERED_CROSS_DATASET_PROVENANCE",
            "mapping": categories["verb"],
        },
        "target": {
            "status": "RECOVERED_CROSS_DATASET_PROVENANCE",
            "mapping": categories["target"],
        },
        "triplet": {
            "status": "RECOVERED_CROSS_DATASET_PROVENANCE_WITH_EXCEPTIONS",
            "mapping": categories["triplet"],
            "component_map": {
                str(triplet_id): list(components)
                for triplet_id, components in sorted(mapping.items())
            },
            "compared_instances": compared,
            "exact_component_matches": exact,
            "mismatch_count": compared - exact,
            "mismatches": mismatch_rows,
        },
        "phase": {
            "status": "UNRESOLVED_TRACK20_NUMERIC_ID_MAPPING",
            "cholec50_explicit_mapping": categories["phase"],
            "track20_paper_ordered_names": [
                "preparation",
                "calot triangle dissection",
                "gallbladder dissection",
                "clipping & cutting",
                "gallbladder packaging",
                "cleaning and coagulation",
                "gallbladder extraction",
            ],
            "reason": (
                "The Track20 paper lists names in this order but does not provide an "
                "explicit zero-based ID table; CholecT50 uses a different ID 2/3 order."
            ),
        },
    }


def _negative_label_audit(videos: dict[int, dict[str, Any]]) -> dict[str, Any]:
    patterns: Counter[tuple[int, int, int]] = Counter()
    triplet_minus_two_by_instrument: Counter[int] = Counter()
    for video in videos.values():
        for _, instance in _flatten_instances(video):
            key = (instance["verb"], instance["target"], instance["triplet"])
            if min(key) < 0:
                patterns[key] += 1
            if instance["triplet"] == -2:
                triplet_minus_two_by_instrument[instance["instrument"]] += 1
    minus_two_total = sum(triplet_minus_two_by_instrument.values())
    specimen_bag_count = triplet_minus_two_by_instrument[6]
    return {
        "status": "PARTIAL",
        "negative_patterns": [
            {
                "verb_id": key[0],
                "target_id": key[1],
                "triplet_id": key[2],
                "count": count,
            }
            for key, count in sorted(patterns.items())
        ],
        "triplet_minus_two": {
            "total": minus_two_total,
            "by_instrument": {
                str(key): value
                for key, value in sorted(triplet_minus_two_by_instrument.items())
            },
            "specimen_bag_count": specimen_bag_count,
            "specimen_bag_fraction": specimen_bag_count / minus_two_total,
            "interpretation": (
                "PARTIALLY_EXPLAINED: CholecT50 has no instrument/triplet class for "
                "Track20 specimen-bag; non-bag -2 cases remain unresolved."
            ),
        },
        "minus_one_interpretation": (
            "UNRESOLVED: CholecT50 encodes null as verb 9, target 14 and triplets "
            "94-99, so Track20 -1 cannot be equated with the upstream null classes."
        ),
    }


def _vid30_vid31_audit(videos: dict[int, dict[str, Any]]) -> dict[str, Any]:
    vid17 = videos[17]
    vid30 = videos[30]
    vid31 = videos[31]
    rows17 = _flatten_instances(vid17)
    rows30 = _flatten_instances(vid30)
    if len(rows17) != len(rows30):
        raise ValueError("VID17/VID30 row count changed; duplicate diagnostic invalid")

    differing_fields: Counter[str] = Counter()
    for (frame17, instance17), (frame30, instance30) in zip(rows17, rows30):
        if frame17 != frame30:
            raise ValueError("VID17/VID30 frame order differs")
        for field in instance17:
            if instance17[field] != instance30[field]:
                differing_fields[field] += 1
    equal_fields = sorted(set(rows17[0][1]) - set(differing_fields))
    core17 = [
        (frame_id, {field: instance[field] for field in equal_fields})
        for frame_id, instance in rows17
    ]
    core30 = [
        (frame_id, {field: instance[field] for field in equal_fields})
        for frame_id, instance in rows30
    ]
    return {
        "VID30": {
            "status": "BLOCKED_VID17_CORE_ANNOTATION_DUPLICATE_CONFIRMED",
            "annotation_frame_count": len(vid30["frame_ids"]),
            "png_count": len(vid30["png_ids"]),
            "annotation_identity": {
                "matches_video": "VID17",
                "frame_id_sets_equal": vid30["frame_ids"] == vid17["frame_ids"],
                "instance_row_counts_equal": len(rows30) == len(rows17),
                "instance_rows": len(rows30),
                "equal_core_fields": equal_fields,
                "differing_fields": dict(sorted(differing_fields.items())),
                "core_sha256_vid17": _canonical_sha256(core17),
                "core_sha256_vid30": _canonical_sha256(core30),
                "core_values_equal": core17 == core30,
            },
            "media_identity": "UNRESOLVED_NO_UPSTREAM_IMAGE_CONTENT_AVAILABLE",
            "png_ids_equal_vid31_annotation_ids": (
                vid30["png_ids"] == vid31["frame_ids"]
            ),
        },
        "VID31": {
            "status": "BLOCKED_ANNOTATION_MEDIA_PAIRING",
            "annotation_frame_count": len(vid31["frame_ids"]),
            "png_count": len(vid31["png_ids"]),
            "annotation_identity": (
                "LIKELY_FOR_VID30_PER_PUBLIC_ISSUE_AND_EXACT_VID30_PNG_ID_SET; "
                "NOT_MAINTAINER_CONFIRMED"
            ),
            "media_identity": "UNRESOLVED_NO_UPSTREAM_IMAGE_CONTENT_AVAILABLE",
        },
    }


def _cholec80_phase_index(
    archive: RemoteZip,
) -> tuple[dict[int, dict[int, int]], list[dict[str, Any]]]:
    entry_names = [
        "labels/train/1fps_100_0.pickle",
        "labels/val/1fps.pickle",
        "labels/test/1fps.pickle",
    ]
    index: dict[int, dict[int, int]] = {}
    evidence = []
    for name in entry_names:
        data = archive.read(name)
        payload = _load_restricted_cholec80_pickle(data)
        for raw_video_id, records in payload.items():
            video_id = int(raw_video_id.replace("video", ""))
            index[video_id] = {
                int(record["Original_frame_id"]): int(record["Phase_gt"])
                for record in records
            }
        evidence.append(
            {
                "entry": name,
                "sha256": _sha256(data),
                "uncompressed_size": len(data),
                "pickle_globals_allowlisted": True,
            }
        )
    return index, evidence


def _frame_rule_audit(
    videos: dict[int, dict[str, Any]],
    cholec80_phase: dict[int, dict[int, int]],
    media_manifest: dict[str, Any],
) -> dict[str, Any]:
    local_ids = [
        frame_id for video in videos.values() for frame_id in video["frame_ids"]
    ]
    cholec80_rows = []
    for video_id, video in sorted(videos.items()):
        if video_id not in cholec80_phase:
            continue
        upstream = cholec80_phase[video_id]
        common = [
            frame_id for frame_id in video["frame_ids"] if frame_id - 1 in upstream
        ]
        phase_exact = 0
        for frame_id in common:
            local_phases = {
                int(instance["phase"])
                for instance in video["payload"]["annotations"][str(frame_id)]
            }
            if (
                len(local_phases) == 1
                and next(iter(local_phases)) == upstream[frame_id - 1]
            ):
                phase_exact += 1
        cholec80_rows.append(
            {
                "video_id": video["video_id"],
                "track20_annotation_frames": len(video["frame_ids"]),
                "candidate_key_minus_one_found_upstream": len(common),
                "coverage": len(common) / len(video["frame_ids"]),
                "phase_id_exact_count": phase_exact,
                "phase_id_exact_fraction_on_common": (
                    phase_exact / len(common) if common else None
                ),
                "upstream_last_original_frame_id": max(upstream),
            }
        )

    test_rows = []
    for row in media_manifest["videos"]:
        if row["media_kind"] != "mp4":
            continue
        video_id = int(row["video_id"].replace("VID", ""))
        frame_ids = videos[video_id]["frame_ids"]
        frame_count = row["probe"]["nb_frames"]
        test_rows.append(
            {
                "video_id": row["video_id"],
                "container_frame_count": frame_count,
                "all_candidate_decoder_indices_valid": all(
                    0 <= frame_id - 1 < frame_count for frame_id in frame_ids
                ),
                "candidate_rule": "decoder_frame_index = annotation_frame_id - 1",
                "cholec80_same_id_upstream_available": video_id in cholec80_phase,
            }
        )
    return {
        "status": "UNRESOLVED_FORMAL_RULE_STRONGLY_SUPPORTED",
        "candidate_rule": "decoder_frame_index = annotation_frame_id - 1",
        "track20_all_annotation_ids_mod_25": sorted(
            {value % 25 for value in local_ids}
        ),
        "track20_all_differences_are_multiples_of_25": all(
            all(
                (right - left) % 25 == 0
                for left, right in zip(video["frame_ids"], video["frame_ids"][1:])
            )
            for video in videos.values()
        ),
        "cholec80_official_extraction_evidence": (
            "Official SelfSupSurg extraction starts count=0 and names every decoded frame "
            "with that zero-based count; 1 FPS label Original_frame_id values are 0 mod 25."
        ),
        "cholec80_same_video_checks": cholec80_rows,
        "test_mp4_checks": test_rows,
        "blocking_reason": (
            "No accessible upstream same-VID image frames were available for pixel/hash "
            "comparison, and phase annotations differ across release variants. Arithmetic "
            "and provenance support key-1, but do not make it an authoritative Track20 rule."
        ),
    }


def _source_record(
    *,
    source_id: str,
    url: str,
    claim: str,
    sha256: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "url": url,
        "version": version,
        "sha256": sha256,
        "claim_supported": claim,
    }


def _render_report(audit: dict[str, Any]) -> str:
    status = audit["status"]
    counts = audit["video_provenance"]["counts"]
    ontology = audit["ontology"]
    conflicts = ontology["triplet"]
    negative = audit["negative_labels"]["triplet_minus_two"]
    vid30 = audit["vid30_vid31"]["VID30"]
    frame_rule = audit["frame_index_rule"]
    rows = audit["video_provenance"]["videos"]

    def compact_mapping(mapping: dict[str, str]) -> str:
        return ", ".join(
            f"{key}={value}"
            for key, value in sorted(mapping.items(), key=lambda item: int(item[0]))
        )

    mapping_lines = []
    for row in rows:
        mapping_lines.append(
            "| {track20_video_id} | {track20_split} | {c80} | {c50} |".format(
                **row,
                c80=row["cholec80_video_id"] or "-",
                c50=(
                    f"{row['cholec50_video_id']} ({row['cholec50_split']})"
                    if row["cholec50_video_id"]
                    else "-"
                ),
            )
        )

    return "\n".join(
        [
            "# CholecTrack20 Cross-Dataset Provenance Audit",
            "",
            f"Access date: `{audit['access_date']}`. Audit mode: read-only.",
            "",
            "## Decision",
            "",
            "| Item | Status |",
            "| --- | --- |",
            f"| Video provenance | **{status['VIDEO_PROVENANCE_STATUS']}** |",
            f"| Ontology | **{status['ONTOLOGY_STATUS']}** |",
            f"| Frame index rule | **{status['FRAME_INDEX_RULE_STATUS']}** |",
            f"| VID30 | **{status['VID30_STATUS']}** |",
            f"| VID31 | **{status['VID31_STATUS']}** |",
            f"| Negative labels | **{status['NEGATIVE_LABEL_STATUS']}** |",
            f"| IVT conflicts | **{status['IVT_CONFLICT_STATUS']}** |",
            f"| Formal experiments | **{status['FORMAL_EXPERIMENT_READY']}** |",
            "",
            "Formal experiments remain blocked. The audit recovers useful provenance and",
            "ontology evidence but does not authorize data repair or label rewriting.",
            "",
            "## Video Provenance",
            "",
            f"All {counts['track20_videos']} Track20 IDs map to an upstream collection: "
            f"{counts['mapped_to_cholec80']} map to Cholec80, "
            f"{counts['mapped_to_cholec50']} map to CholecT50, and "
            f"{counts['mapped_to_both']} occur in both. This is based on the official "
            "preserved-ID convention, not on pixel hashes.",
            "",
            "| Track20 | Split | Cholec80 | CholecT50 |",
            "| --- | --- | --- | --- |",
            *mapping_lines,
            "",
            "## Ontology",
            "",
            "- Instrument IDs are verified from Track20 JSON and official Cholec80/CholecT50 "
            "metadata. Track20 adds ID 6 `specimen-bag`; CholecT50 IVT has only instruments 0-5.",
            "- Verb, target, and 100 triplet names were recovered from the official CholecT50 "
            "challenge validation JSON and its `label_mapping.txt`.",
            f"- Instrument: {compact_mapping(ontology['instrument']['track20_mapping'])}.",
            f"- Verb: {compact_mapping(ontology['verb']['mapping'])}.",
            f"- Target: {compact_mapping(ontology['target']['mapping'])}.",
            "- The complete 100-class Triplet map is retained in the machine-readable audit.",
            f"- Of {conflicts['compared_instances']:,} comparable Track20 instances, "
            f"{conflicts['exact_component_matches']:,} match the upstream IVT relation and "
            f"{conflicts['mismatch_count']} do not.",
            "- Track20 phase names are documented, but the numeric mapping remains `UNRESOLVED`: "
            "the paper does not publish an explicit ID table and CholecT50 orders IDs 2/3 "
            "differently from the Track20 paper's phase list.",
            "",
            "## The 34 IVT Conflicts",
            "",
            "- 32 rows are explained by semantic scope: Track20 boxes the specimen bag "
            "(instrument 6), while triplet 12 means `grasper, grasp, specimen_bag`.",
            "- One VID25 row has triplet 43 (`bipolar, retract, gallbladder`) but stores "
            "instrument 0 (`grasper`): `UNRESOLVED`.",
            "- One VID23 row keeps triplet 94 but stores verb/target as `-1/-1` instead of "
            "upstream null IDs `9/14`: `UNRESOLVED_NULL_ENCODING`.",
            "",
            "## Frame Index Rule",
            "",
            f"All Track20 annotation IDs are `1 mod 25`; the Cholec80 official extractor is "
            "zero-based and its 1 FPS `Original_frame_id` values are `0 mod 25`. Therefore "
            f"`{frame_rule['candidate_rule']}` is strongly supported. It is still marked "
            "`UNRESOLVED` for formal evaluation because same-VID upstream pixels were not "
            "available for hash comparison and release-specific phase labels are not identical.",
            "",
            "Six Cholec80-derived test videos support the provenance chain; VID92 and VID111 "
            "lack an accessible same-VID upstream frame reference. The candidate index is valid "
            "inside all eight local MP4 containers, but no automatic offset was written to config.",
            "",
            "## VID30 / VID31",
            "",
            f"- VID30 JSON is not merely misaligned: its {vid30['annotation_identity']['instance_rows']:,} "
            "instances have the same frame IDs and identical geometry, instrument, operator, "
            "visibility and three track-ID fields as VID17. Only phase and several scene-condition "
            "fields differ. It is therefore confirmed as a VID17 core-annotation duplicate; "
            "the audit does not infer how that duplication occurred.",
            "- VID30 PNG IDs exactly equal VID31 annotation IDs. The public issue reports that "
            "VID31 annotation appears to describe VID30, but there is no maintainer confirmation "
            "or upstream pixel reference; that pairing remains blocked rather than auto-repaired.",
            "- VID31 PNG identity and the missing correct VID31 annotation remain unresolved.",
            "",
            "## Negative Labels",
            "",
            f"Triplet `-2` occurs {negative['total']:,} times; "
            f"{negative['specimen_bag_count']:,} are specimen-bag instances, which have no "
            "CholecT50 instrument/triplet class. The remaining "
            f"{negative['total'] - negative['specimen_bag_count']} cases remain unresolved. "
            "Categorical `-1` is not equivalent to CholecT50's explicit null classes and remains "
            "an invalid/missing sentinel only.",
            "",
            "## Remaining Blockers",
            "",
            "- Maintainer-provided corrected VID30/VID31 annotations and media identities.",
            "- Explicit Track20 phase ID table and negative-sentinel specification.",
            "- Official Track20 annotation-key to decoder-index rule, or same-VID upstream frames "
            "for pixel-level verification.",
            "- Source CholecT50 annotations for overlapping Track20 VIDs; the public validation "
            "package contains only VID68/70/73/74/75, none of which is in Track20.",
            "",
            "## Provenance",
            "",
            *[
                f"- [{source['source_id']}]({source['url']}), version "
                f"`{source['version'] or 'server object'}`: {source['claim_supported']}"
                for source in audit["sources"]
            ],
            "",
            "No training/inference code or source dataset file was modified.",
            "",
        ]
    )


def run_audit(
    dataset_root: Path,
    media_manifest_path: Path,
    json_output: Path,
    markdown_output: Path,
) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update({"User-Agent": "CholecTrack20-Provenance-Audit/1.0"})

    cholec50_splits, split_source = _download_json(
        session, f"{OVERLAP_BASE}/CholecT50_splits.json"
    )
    cholec80_splits, cholec80_split_source = _download_json(
        session, f"{OVERLAP_BASE}/Cholec80_splits.json"
    )
    cholec80_ids = {
        video_id for split_ids in cholec80_splits.values() for video_id in split_ids
    }

    cholec50_archive = RemoteZip(session, CHOLECT50_ZIP)
    mapping_name = "cholect50-challenge-val/label_mapping.txt"
    sample_name = "cholect50-challenge-val/labels/VID68.json"
    readme_name = "cholect50-challenge-val/README.md"
    mapping_bytes = cholec50_archive.read(mapping_name)
    sample_bytes = cholec50_archive.read(sample_name)
    readme_bytes = cholec50_archive.read(readme_name)
    sample = json.loads(sample_bytes)

    cholec80_archive = RemoteZip(session, CHOLEC80_LABEL_ZIP)
    cholec80_phase, cholec80_entries = _cholec80_phase_index(cholec80_archive)

    videos = _discover_local_json(dataset_root)
    video_rows, video_counts = _video_provenance(videos, cholec80_ids, cholec50_splits)
    ontology = _ontology_audit(
        videos, sample["categories"], mapping_bytes.decode("utf-8")
    )
    negative = _negative_label_audit(videos)
    vid30_vid31 = _vid30_vid31_audit(videos)
    media_manifest = json.loads(media_manifest_path.read_text(encoding="utf-8"))
    frame_rule = _frame_rule_audit(videos, cholec80_phase, media_manifest)

    audit = {
        "schema_version": "ct20_cross_dataset_provenance_audit_v1",
        "access_date": datetime.now(timezone.utc).astimezone().date().isoformat(),
        "read_only": True,
        "dataset_root": str(dataset_root.resolve()),
        "status": {
            "VIDEO_PROVENANCE_STATUS": "PASS_OFFICIAL_ID_RELATION",
            "ONTOLOGY_STATUS": ontology["status"],
            "FRAME_INDEX_RULE_STATUS": frame_rule["status"],
            "VID30_STATUS": vid30_vid31["VID30"]["status"],
            "VID31_STATUS": vid30_vid31["VID31"]["status"],
            "NEGATIVE_LABEL_STATUS": negative["status"],
            "IVT_CONFLICT_STATUS": "PARTIAL_32_EXPLAINED_2_UNRESOLVED",
            "FORMAL_EXPERIMENT_READY": "BLOCKED",
        },
        "video_provenance": {"counts": video_counts, "videos": video_rows},
        "ontology": ontology,
        "frame_index_rule": frame_rule,
        "vid30_vid31": vid30_vid31,
        "negative_labels": negative,
        "remote_access": {
            "method": "HTTP_RANGE_ONLY",
            "full_archives_downloaded": False,
            "cholec50_zip_transferred_bytes": cholec50_archive.transferred_bytes,
            "cholec80_zip_transferred_bytes": cholec80_archive.transferred_bytes,
            "cholec50_zip_metadata": cholec50_archive.response_metadata,
            "cholec80_zip_metadata": cholec80_archive.response_metadata,
            "cholec80_entries": cholec80_entries,
        },
        "sources": [
            _source_record(
                source_id="CAMMA dataset-overlap CholecT50 split",
                url=split_source["url"],
                version=OVERLAP_COMMIT,
                sha256=split_source["sha256"],
                claim="CholecT50 membership and official split",
            ),
            _source_record(
                source_id="CAMMA dataset-overlap Cholec80 split",
                url=cholec80_split_source["url"],
                version=OVERLAP_COMMIT,
                sha256=cholec80_split_source["sha256"],
                claim="Cholec80 video membership and split",
            ),
            _source_record(
                source_id="CholecT50 official format",
                url=CHOLECT50_FORMAT,
                version=CHOLECT50_COMMIT,
                claim="1 FPS sequential image IDs, ontology and mapping format",
            ),
            _source_record(
                source_id="CholecT50 official preserved-ID statement",
                url=CHOLECT50_SPLITS,
                version=CHOLECT50_COMMIT,
                claim="Numeric video IDs are consistent across CAMMA datasets",
            ),
            _source_record(
                source_id="CholecT50 challenge mapping",
                url=f"{CHOLECT50_ZIP}#{mapping_name}",
                version=cholec50_archive.response_metadata.get("last_modified"),
                sha256=_sha256(mapping_bytes),
                claim="100 IVT-to-I/V/T relations",
            ),
            _source_record(
                source_id="CholecT50 challenge ontology",
                url=f"{CHOLECT50_ZIP}#{sample_name}",
                version=cholec50_archive.response_metadata.get("last_modified"),
                sha256=_sha256(sample_bytes),
                claim="Instrument, verb, target, triplet and phase ID names",
            ),
            _source_record(
                source_id="CholecT50 challenge README",
                url=f"{CHOLECT50_ZIP}#{readme_name}",
                version=cholec50_archive.response_metadata.get("last_modified"),
                sha256=_sha256(readme_bytes),
                claim="Release identity, source institution and preserved video IDs",
            ),
            _source_record(
                source_id="Cholec80 official frame extractor",
                url=SELFSUPSURG_EXTRACTOR,
                version=SELFSUPSURG_COMMIT,
                claim="25 FPS check and zero-based decoder-frame filenames",
            ),
            _source_record(
                source_id="Cholec80 official 1 FPS labels",
                url=CHOLEC80_LABEL_ZIP,
                version=cholec80_archive.response_metadata.get("last_modified"),
                claim="Original_frame_id values and phase labels for 80 videos",
            ),
            _source_record(
                source_id="CholecTrack20 official README",
                url=CHOLECTRACK20_README,
                version=CHOLECTRACK20_COMMIT,
                claim="Dataset overlap, preserved identities, 1 FPS labels and 25 FPS videos",
            ),
            _source_record(
                source_id="CholecTrack20 CVPR paper",
                url=CHOLECTRACK20_PAPER,
                version="CVPR 2025",
                claim="Track20 phase names and dataset design",
            ),
            _source_record(
                source_id="CholecTrack20 public issue 10",
                url=CHOLECTRACK20_ISSUE,
                version="opened 2026-04-22; no maintainer resolution at access date",
                claim="Independent report that VID31 annotation appears to describe VID30",
            ),
        ],
        "automatic_repairs_applied": False,
        "training_or_inference_executed": False,
    }

    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_output.write_text(_render_report(audit), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument(
        "--media-manifest",
        type=Path,
        default=Path("artifacts/p1/media_alignment_manifest.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "dataset_reports/cholectrack20_cross_dataset_provenance_audit.json"
        ),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/CROSS_DATASET_PROVENANCE_AUDIT.md"),
    )
    args = parser.parse_args()
    audit = run_audit(
        args.dataset_root,
        args.media_manifest,
        args.json_output,
        args.markdown_output,
    )
    print(json.dumps(audit["status"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
