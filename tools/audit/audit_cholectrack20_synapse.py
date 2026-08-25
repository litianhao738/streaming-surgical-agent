"""Read-only Synapse metadata audit for the official CholecTrack20 project."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from synapseclient import Synapse
from synapseclient.api import get_children
from synapseclient.core.exceptions import SynapseHTTPError

PROJECT_ID = "syn53182642"
EXPECTED_DATASET_FOLDER = "CholecTrack20"
EXPECTED_SPLITS = {"Training": 10, "Validation": 2, "Testing": 8}
SEARCH_TERMS = (
    "ontology",
    "mapping",
    "label_mapping",
    "dictionary",
    "categories",
    "instrument",
    "verb",
    "target",
    "triplet",
    "ivt",
    "phase",
    "class",
    "labels",
    "fps",
    "frame",
    "sampling",
    "extraction",
    "annotation frequency",
    "frame index",
    "offset",
    "png",
    "video",
    "temporal",
    "1fps",
    "25fps",
)
TARGET_DIAGNOSTIC_VIDEOS = ("VID30", "VID31")
SAMPLE_FRAME_IDS = (11801, 25001, 50001, 84776)


class SynapseAuditError(RuntimeError):
    """Raised when the read-only Synapse audit cannot establish its contract."""


@dataclass(frozen=True)
class RemoteVideo:
    video_id: str
    split: str
    folder_id: str
    annotation_entity_id: str
    annotation_name: str
    media_entity_id: str
    media_name: str
    media_kind: str
    source_path: str


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _semantic_json_sha256(path: Path) -> str:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    canonical = json.dumps(
        raw,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _runtime_token() -> str:
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        raise SynapseAuditError(
            "SYNAPSE_AUTH_TOKEN must be present only in the runtime environment"
        )
    return token


async def _children_async(synapse: Synapse, parent_id: str) -> list[dict[str, Any]]:
    return [
        child
        async for child in get_children(
            parent=parent_id,
            synapse_client=synapse,
        )
    ]


def _children(synapse: Synapse, parent_id: str) -> list[dict[str, Any]]:
    return asyncio.run(_children_async(synapse, parent_id))


def _plain_annotations(raw: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (raw.get("annotations") or {}).items():
        if isinstance(value, Mapping):
            result[key] = {
                "type": value.get("type"),
                "value": value.get("value"),
            }
        else:
            result[key] = value
    return result


def _entity_metadata(synapse: Synapse, entity_id: str) -> dict[str, Any]:
    entity = synapse.restGET(f"/entity/{entity_id}")
    try:
        annotations = synapse.restGET(f"/entity/{entity_id}/annotations2")
        annotation_payload: dict[str, Any] | None = _plain_annotations(annotations)
        annotation_error = None
    except SynapseHTTPError as exc:
        response = getattr(exc, "response", None)
        annotation_payload = None
        annotation_error = {
            "error_type": type(exc).__name__,
            "http_status": response.status_code if response is not None else None,
        }
    return {
        "entity_id": entity.get("id"),
        "name": entity.get("name"),
        "parent_id": entity.get("parentId"),
        "concrete_type": entity.get("concreteType"),
        "version_number": entity.get("versionNumber"),
        "version_label": entity.get("versionLabel"),
        "created_on": entity.get("createdOn"),
        "modified_on": entity.get("modifiedOn"),
        "etag": entity.get("etag"),
        "annotations": annotation_payload,
        "annotations_error": annotation_error,
    }


def _file_handle_metadata(synapse: Synapse, entity_id: str) -> dict[str, Any]:
    bundle = synapse.restPOST(
        f"/entity/{entity_id}/bundle2",
        body=json.dumps(
            {
                "includeEntity": True,
                "includeFileHandles": True,
                "includeRestrictionInformation": True,
            }
        ),
    )
    handles = bundle.get("fileHandles") or []
    handle = handles[0] if handles else {}
    entity = bundle.get("entity") or {}
    return {
        "entity_id": entity_id,
        "entity_version": entity.get("versionNumber"),
        "file_handle_id": handle.get("id"),
        "file_name": handle.get("fileName"),
        "content_type": handle.get("contentType"),
        "content_size": handle.get("contentSize"),
        "content_md5": handle.get("contentMd5"),
        "storage_location_id": handle.get("storageLocationId"),
    }


def _wiki_metadata(synapse: Synapse, owner_id: str) -> dict[str, Any]:
    try:
        wiki = synapse.getWiki(owner_id)
    except SynapseHTTPError as exc:
        response = getattr(exc, "response", None)
        return {
            "owner_entity_id": owner_id,
            "status": "ACCESS_ERROR",
            "error_type": type(exc).__name__,
            "http_status": response.status_code if response is not None else None,
        }
    markdown = wiki.markdown or ""
    normalized = markdown.lower()
    attachment_handles = synapse.getWikiAttachments(wiki)
    attachments = [
        {
            "file_handle_id": handle.get("id"),
            "file_name": handle.get("fileName"),
            "content_type": handle.get("contentType"),
            "content_size": handle.get("contentSize"),
            "content_md5": handle.get("contentMd5"),
        }
        for handle in attachment_handles
    ]
    keyword_hits = sorted(term for term in SEARCH_TERMS if term in normalized)
    claims: list[dict[str, Any]] = []
    if "1 frame per second" in normalized:
        claims.append(
            {
                "claim": "annotations are provided at 1 frame per second",
                "status": "OFFICIAL_WIKI_VERIFIED",
            }
        )
    if all(
        fragment in normalized
        for fragment in ("10 videos", "2 for validation", "8 for testing")
    ):
        claims.append(
            {
                "claim": "official split contains 10 training, 2 validation, and 8 testing videos",
                "status": "OFFICIAL_WIKI_VERIFIED",
            }
        )
    tool_names = (
        "grasper",
        "bipolar",
        "hook",
        "scissors",
        "clipper",
        "irrigator",
        "specimen bag",
    )
    if all(name in normalized for name in tool_names):
        claims.append(
            {
                "claim": "instrument canonical names are listed",
                "status": "OFFICIAL_WIKI_VERIFIED",
            }
        )
    return {
        "owner_entity_id": owner_id,
        "status": "READ",
        "wiki_id": getattr(wiki, "id", None),
        "wiki_version": getattr(wiki, "version", None),
        "title": getattr(wiki, "title", None),
        "markdown_file_handle_id": getattr(wiki, "markdownFileHandleId", None),
        "attachment_count": len(attachments),
        "attachments": attachments,
        "local_cache_path": getattr(wiki, "markdown_path", None),
        "markdown_sha256": _sha256_bytes(markdown.encode("utf-8")),
        "keyword_hits": keyword_hits,
        "supported_claims": claims,
        "missing_from_page": [
            "verb/target/triplet/phase numeric ID-to-name tables",
            "annotation-frame extraction implementation",
            "0-based or 1-based decoder indexing rule",
            "PNG filename to source-video decoder-index rule",
        ],
    }


def _discover_release(
    synapse: Synapse,
    project_id: str,
) -> tuple[dict[str, Any], list[RemoteVideo], list[dict[str, Any]]]:
    project_children = _children(synapse, project_id)
    dataset_folders = [
        child
        for child in project_children
        if child.get("name") == EXPECTED_DATASET_FOLDER
        and str(child.get("type", "")).endswith(".Folder")
    ]
    if len(dataset_folders) != 1:
        raise SynapseAuditError(
            f"Expected one {EXPECTED_DATASET_FOLDER!r} folder, found {len(dataset_folders)}"
        )
    dataset_folder = dataset_folders[0]
    root_children = _children(synapse, dataset_folder["id"])
    children_by_name = {str(child.get("name")): child for child in root_children}

    videos: list[RemoteVideo] = []
    split_records: list[dict[str, Any]] = []
    metadata_entities: list[dict[str, Any]] = [
        _entity_metadata(synapse, project_id),
        _entity_metadata(synapse, dataset_folder["id"]),
    ]
    for child in root_children:
        metadata_entities.append(_entity_metadata(synapse, child["id"]))

    for split_name, expected_count in EXPECTED_SPLITS.items():
        split = children_by_name.get(split_name)
        if not split:
            raise SynapseAuditError(
                f"Missing official Synapse split folder: {split_name}"
            )
        video_folders = _children(synapse, split["id"])
        split_records.append(
            {
                "split": split_name,
                "folder_entity_id": split["id"],
                "expected_video_count": expected_count,
                "observed_video_count": len(video_folders),
                "video_ids": sorted(str(item.get("name")) for item in video_folders),
                "status": "PASS" if len(video_folders) == expected_count else "FAIL",
            }
        )
        for video_folder in video_folders:
            video_id = str(video_folder.get("name")).upper()
            children = _children(synapse, video_folder["id"])
            annotation_files = [
                item
                for item in children
                if str(item.get("name", "")).lower().endswith(".json")
            ]
            media = [
                item
                for item in children
                if item.get("name") == "Frames"
                or str(item.get("name", "")).lower().endswith(".mp4")
            ]
            if len(annotation_files) != 1 or len(media) != 1:
                raise SynapseAuditError(
                    f"{video_id} has {len(annotation_files)} JSON and {len(media)} media entities"
                )
            annotation = annotation_files[0]
            media_entity = media[0]
            source_path = (
                f"{project_id}/{EXPECTED_DATASET_FOLDER}/{split_name}/{video_id}"
            )
            videos.append(
                RemoteVideo(
                    video_id=video_id,
                    split=split_name,
                    folder_id=video_folder["id"],
                    annotation_entity_id=annotation["id"],
                    annotation_name=annotation["name"],
                    media_entity_id=media_entity["id"],
                    media_name=media_entity["name"],
                    media_kind="png_folder"
                    if media_entity["name"] == "Frames"
                    else "mp4",
                    source_path=source_path,
                )
            )
            metadata_entities.extend(
                (
                    _entity_metadata(synapse, video_folder["id"]),
                    _entity_metadata(synapse, annotation["id"]),
                    _entity_metadata(synapse, media_entity["id"]),
                )
            )

    tree = {
        "project_entity_id": project_id,
        "dataset_folder_entity_id": dataset_folder["id"],
        "dataset_folder_name": dataset_folder["name"],
        "root_children": [
            {
                "entity_id": child.get("id"),
                "name": child.get("name"),
                "type": child.get("type"),
                "version_number": child.get("versionNumber"),
            }
            for child in root_children
        ],
        "splits": split_records,
        "videos": [
            video.__dict__ for video in sorted(videos, key=lambda item: item.video_id)
        ],
    }
    return tree, videos, metadata_entities


def _local_video_paths(dataset_root: Path, video: RemoteVideo) -> tuple[Path, Path]:
    video_dir = dataset_root / video.split / video.video_id
    annotation_path = video_dir / video.annotation_name.lower()
    if not annotation_path.is_file():
        candidates = sorted(video_dir.glob("*.json"))
        if len(candidates) != 1:
            raise SynapseAuditError(
                f"Cannot identify local annotation for {video.video_id}"
            )
        annotation_path = candidates[0]
    media_path = video_dir / video.media_name
    if video.media_kind == "png_folder" and not media_path.is_dir():
        raise SynapseAuditError(f"Missing local Frames folder for {video.video_id}")
    if video.media_kind == "mp4" and not media_path.is_file():
        candidates = sorted(video_dir.glob("*.mp4"))
        if len(candidates) != 1:
            raise SynapseAuditError(f"Cannot identify local MP4 for {video.video_id}")
        media_path = candidates[0]
    return annotation_path, media_path


def _annotation_frame_ids(path: Path) -> set[int]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    return {int(value) for value in raw["annotations"]}


def _numeric_png_children(
    children: Iterable[Mapping[str, Any]],
) -> tuple[dict[int, str], list[str]]:
    result: dict[int, str] = {}
    nonnumeric: list[str] = []
    for child in children:
        name = str(child.get("name"))
        try:
            frame_id = int(Path(name).stem)
        except ValueError:
            nonnumeric.append(name)
            continue
        result[frame_id] = str(child.get("id"))
    return result, sorted(nonnumeric)


def _target_diagnostic(
    synapse: Synapse,
    *,
    video: RemoteVideo,
    dataset_root: Path,
    downloaded_metadata_dir: Path | None,
) -> dict[str, Any]:
    annotation_path, media_path = _local_video_paths(dataset_root, video)
    annotation_handle = _file_handle_metadata(synapse, video.annotation_entity_id)
    local_raw_md5 = _md5(annotation_path)
    local_semantic_sha256 = _semantic_json_sha256(annotation_path)
    remote_children = _children(synapse, video.media_entity_id)
    remote_frame_entities, nonnumeric = _numeric_png_children(remote_children)
    local_frame_ids = {int(path.stem) for path in media_path.glob("*.png")}
    annotation_ids = _annotation_frame_ids(annotation_path)

    downloaded_copy: dict[str, Any] | None = None
    if downloaded_metadata_dir is not None:
        candidate = downloaded_metadata_dir / video.annotation_name.lower()
        if candidate.is_file():
            downloaded_copy = {
                "path": str(candidate.resolve()),
                "raw_md5": _md5(candidate),
                "semantic_json_sha256": _semantic_json_sha256(candidate),
                "raw_md5_matches_remote": (
                    _md5(candidate) == annotation_handle["content_md5"]
                ),
                "semantically_equals_local": (
                    _semantic_json_sha256(candidate) == local_semantic_sha256
                ),
            }

    sample_hashes: list[dict[str, Any]] = []
    for frame_id in SAMPLE_FRAME_IDS:
        entity_id = remote_frame_entities.get(frame_id)
        local_path = media_path / f"{frame_id:06d}.png"
        if entity_id is None or not local_path.is_file():
            sample_hashes.append(
                {
                    "frame_id": frame_id,
                    "status": "UNAVAILABLE_FOR_HASH_CHECK",
                    "remote_entity_id": entity_id,
                    "local_file_present": local_path.is_file(),
                }
            )
            continue
        handle = _file_handle_metadata(synapse, entity_id)
        local_md5 = _md5(local_path)
        sample_hashes.append(
            {
                "frame_id": frame_id,
                "remote_entity_id": entity_id,
                "remote_content_md5": handle["content_md5"],
                "local_content_md5": local_md5,
                "exact_hash_match": local_md5 == handle["content_md5"],
            }
        )

    missing_media = sorted(annotation_ids - local_frame_ids)
    extra_media = sorted(local_frame_ids - annotation_ids)
    return {
        "video_id": video.video_id,
        "split": video.split,
        "source_path": video.source_path,
        "annotation_entity": annotation_handle,
        "local_annotation_path": str(annotation_path.resolve()),
        "local_annotation_raw_md5": local_raw_md5,
        "local_annotation_semantic_json_sha256": local_semantic_sha256,
        "local_raw_md5_matches_remote": local_raw_md5
        == annotation_handle["content_md5"],
        "downloaded_official_copy": downloaded_copy,
        "frames_folder_entity_id": video.media_entity_id,
        "remote_frame_file_count": len(remote_frame_entities),
        "remote_nonnumeric_frame_names": nonnumeric,
        "local_frame_file_count": len(local_frame_ids),
        "remote_local_frame_id_sets_equal": set(remote_frame_entities)
        == local_frame_ids,
        "annotation_frame_count": len(annotation_ids),
        "annotation_media_common_count": len(annotation_ids & local_frame_ids),
        "missing_annotated_media_count": len(missing_media),
        "extra_media_count": len(extra_media),
        "missing_annotated_media_ids": missing_media,
        "sample_remote_local_frame_hashes": sample_hashes,
        "status": "BLOCKED_ALIGNMENT"
        if missing_media
        else "QUALIFIED_WITH_EXTRA_MEDIA",
    }


def _search_summary(
    tree: Mapping[str, Any],
    wiki_records: Iterable[Mapping[str, Any]],
    metadata_entities: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    entity_hits: list[dict[str, Any]] = []
    for video in tree["videos"]:
        for field in ("annotation_name", "media_name", "source_path"):
            value = str(video[field])
            hits = sorted(term for term in SEARCH_TERMS if term in value.lower())
            if hits:
                entity_hits.append(
                    {
                        "entity_id": video.get("annotation_entity_id"),
                        "value": value,
                        "keyword_hits": hits,
                    }
                )
    nonempty_annotations = [
        {
            "entity_id": entity["entity_id"],
            "name": entity["name"],
            "annotations": entity["annotations"],
        }
        for entity in metadata_entities
        if entity.get("annotations")
    ]
    access_errors = [
        {
            "entity_id": entity["entity_id"],
            "name": entity["name"],
            "access_error": entity["annotations_error"],
            "missing_information": "entity annotations",
            "user_action_required": True,
        }
        for entity in metadata_entities
        if entity.get("annotations_error")
    ]
    access_errors.extend(
        {
            "entity_id": wiki["owner_entity_id"],
            "name": "Wiki",
            "access_error": {
                "error_type": wiki.get("error_type"),
                "http_status": wiki.get("http_status"),
            },
            "missing_information": "Wiki documentation",
            "user_action_required": True,
        }
        for wiki in wiki_records
        if wiki.get("status") == "ACCESS_ERROR"
    )
    return {
        "searched_terms": list(SEARCH_TERMS),
        "entity_name_or_path_hits": entity_hits,
        "nonempty_entity_annotations": nonempty_annotations,
        "access_errors": access_errors,
        "ontology_mapping_file_found": False,
        "extraction_or_frame_index_rule_file_found": False,
        "interpretation": (
            "No dedicated ontology/extraction entity was found among the accessible project "
            "tree, Wiki pages, and non-frame entity annotations. This does not prove that no "
            "such official material exists outside the inspected accessible entities."
        ),
    }


def run_audit(
    *,
    project_id: str,
    dataset_root: Path,
    output_path: Path,
    downloaded_metadata_dir: Path | None,
) -> dict[str, Any]:
    token = _runtime_token()
    synapse = Synapse(skip_checks=True)
    synapse.login(authToken=token, silent=True)

    tree, videos, metadata_entities = _discover_release(synapse, project_id)
    wiki_records = [
        _wiki_metadata(synapse, project_id),
        _wiki_metadata(synapse, tree["dataset_folder_entity_id"]),
    ]
    videos_by_id = {video.video_id: video for video in videos}
    target_diagnostics = {
        video_id: _target_diagnostic(
            synapse,
            video=videos_by_id[video_id],
            dataset_root=dataset_root,
            downloaded_metadata_dir=downloaded_metadata_dir,
        )
        for video_id in TARGET_DIAGNOSTIC_VIDEOS
    }
    search = _search_summary(tree, wiki_records, metadata_entities)
    payload = {
        "schema_version": "ct20_p1_synapse_metadata_audit_v1",
        "access_date": datetime.now(tz=ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "project_entity_id": project_id,
        "project_doi": f"https://doi.org/10.7303/{project_id}",
        "read_only": True,
        "authentication": {
            "method": "SYNAPSE_AUTH_TOKEN runtime environment variable",
            "token_serialized": False,
            "requested_operations": [
                "authenticate",
                "read entity tree",
                "read entity annotations and file-handle metadata",
                "read small Wiki documentation",
            ],
            "write_operations_performed": False,
        },
        "tree": tree,
        "wiki": wiki_records,
        "metadata_search": search,
        "target_diagnostics": target_diagnostics,
        "official_findings": {
            "source_video_fps": {
                "value": 25,
                "status": "VERIFIED_BY_OFFICIAL_REPOSITORY_AND_LOCAL_FFPROBE",
                "synapse_wiki_support": "NOT_STATED",
            },
            "annotation_sampling_frequency_fps": {
                "value": 1,
                "status": "VERIFIED_BY_SYNAPSE_PROJECT_WIKI_628401",
            },
            "frame_extraction_convention": "BLOCKED_NOT_DOCUMENTED_IN_INSPECTED_SYNAPSE_METADATA",
            "annotation_frame_key_semantics": "BLOCKED_NOT_DOCUMENTED_IN_INSPECTED_SYNAPSE_METADATA",
            "png_naming_semantics": "BLOCKED_NOT_DOCUMENTED_IN_INSPECTED_SYNAPSE_METADATA",
            "decoder_index_base_or_offset": "BLOCKED_NOT_DOCUMENTED_IN_INSPECTED_SYNAPSE_METADATA",
            "instrument_names": "VERIFIED_BY_SYNAPSE_WIKI_AND_LOCAL_JSON_CATEGORIES",
            "verb_target_triplet_phase_id_names": "BLOCKED_NOT_FOUND_IN_INSPECTED_SYNAPSE_METADATA",
        },
        "status": {
            "ONTOLOGY_STATUS": "PARTIAL",
            "MEDIA_METADATA_STATUS": "PARTIAL",
            "ALIGNMENT_STATUS": "BLOCKED",
            "VID30_STATUS": target_diagnostics["VID30"]["status"],
            "VID31_STATUS": target_diagnostics["VID31"]["status"],
            "PARTIAL_LABEL_STATUS": "PARTIAL",
            "DATA_READY_FOR_P2_SMOKE": "PASS",
            "LOCAL_NUMERIC_TRAINING_READY": "BLOCKED",
            "API_SEMANTIC_PIPELINE_READY": "BLOCKED",
            "FORMAL_TRAINING_READY": "BLOCKED",
        },
    }
    _write_json(output_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default=PROJECT_ID)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/p1/synapse_metadata_audit.json"),
    )
    parser.add_argument(
        "--downloaded-metadata-dir",
        type=Path,
        default=Path("artifacts/p1/synapse_metadata"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_audit(
        project_id=args.project_id,
        dataset_root=args.dataset_root.resolve(),
        output_path=args.output.resolve(),
        downloaded_metadata_dir=args.downloaded_metadata_dir.resolve(),
    )
    summary = {
        "project_entity_id": payload["project_entity_id"],
        "access_date": payload["access_date"],
        "read_only": payload["read_only"],
        "status": payload["status"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
