"""Derived VID30/VID31 supervision contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surgical_agent.data.derived_supervision import (
    DERIVED_SCHEMA_VERSION,
    DerivedSupervisionError,
    load_derived_supervision_manifest,
    load_image_phase_supervision,
)
from surgical_agent.data.qualification import SUPERVISION_FIELDS


def _fields(*enabled: str) -> dict[str, bool]:
    return {field: field in enabled for field in SUPERVISION_FIELDS}


def test_derived_manifest_requires_explicit_field_sources(tmp_path: Path) -> None:
    media = tmp_path / "frames"
    media.mkdir()
    phase = tmp_path / "phase.json"
    phase.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": DERIVED_SCHEMA_VERSION,
                "raw_dataset_root": str(tmp_path),
                "raw_dataset_modified": False,
                "policy": "EXPLICIT_DERIVED_SOURCES_ONLY",
                "videos": {
                    "VID31": {
                        "split": "training",
                        "media_source": str(media),
                        "phase_source": str(phase),
                        "field_supervision": _fields("phase"),
                        "status": "PHASE_ONLY_TRAINING",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_derived_supervision_manifest(manifest)
    vid31 = loaded.video("vid31")
    assert vid31.allows("phase")
    assert not vid31.allows("bbox")
    assert vid31.phase_source == phase.resolve()


def test_derived_manifest_rejects_core_fields_without_annotation(tmp_path: Path) -> None:
    media = tmp_path / "frames"
    media.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": DERIVED_SCHEMA_VERSION,
                "raw_dataset_root": str(tmp_path),
                "raw_dataset_modified": False,
                "policy": "EXPLICIT_DERIVED_SOURCES_ONLY",
                "videos": {
                    "VID30": {
                        "split": "validation",
                        "media_source": str(media),
                        "field_supervision": _fields("bbox"),
                        "status": "INVALID",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DerivedSupervisionError, match="annotation_source"):
        load_derived_supervision_manifest(manifest)


def test_image_phase_supervision_validates_phase_range(tmp_path: Path) -> None:
    path = tmp_path / "phase.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": DERIVED_SCHEMA_VERSION,
                "phase_by_track20_image_frame_id": {
                    "26": {"phase_id": 2, "cholec80_frame_id": 24}
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_image_phase_supervision(path) == {26: 2}

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["phase_by_track20_image_frame_id"]["26"]["phase_id"] = 7
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DerivedSupervisionError, match="outside"):
        load_image_phase_supervision(path)
