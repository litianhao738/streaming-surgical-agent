"""Local integration checks for the explicit VID30/VID31 derived package."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from surgical_agent.data.derived_supervision import (
    load_derived_supervision_manifest,
    load_frame_level_ivt_supervision,
    load_image_phase_supervision,
)
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.parser import parse_annotation_file
from surgical_agent.data.schemas import DatasetSplit

DATASET_ROOT = Path(
    os.environ.get("CHOLECTRACK20_ROOT", "__external_dataset_not_configured__")
)
CHOLEC80_ROOT = Path(
    os.environ.get("CHOLEC80_30_31_ROOT", "__optional_cholec80_not_configured__")
)
pytestmark = pytest.mark.skipif(
    not DATASET_ROOT.is_dir(),
    reason="CholecTrack20 local-data integration fixture is unavailable",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_derived_vid30_validation_and_vid31_phase_package() -> None:
    manifest_path = DATASET_ROOT / "repair_manifest.json"
    manifest = load_derived_supervision_manifest(
        manifest_path,
        dataset_root_override=DATASET_ROOT,
    )

    vid30_source = manifest.video("VID30")
    assert vid30_source.allows("bbox")
    assert vid30_source.allows("track_ids")
    assert vid30_source.allows("phase")
    assert not vid30_source.allows("visual_conditions")
    assert vid30_source.annotation_source is not None
    vid30 = parse_annotation_file(
        vid30_source.annotation_source,
        expected_split=DatasetSplit.VALIDATION,
    )
    assert vid30.video_id == "VID30"
    assert len(vid30.frames) == 2717
    assert sum(len(frame.instances) for frame in vid30.frames) == 4836

    vid31_source = manifest.video("VID31")
    assert vid31_source.allows("phase")
    assert not vid31_source.allows("bbox")
    assert not vid31_source.allows("triplet")
    assert vid31_source.phase_source is not None
    assert vid31_source.frame_level_action_source is not None
    assert vid31_source.frame_level_action_granularity == "FRAME_LEVEL_MULTI_LABEL"
    phases = load_image_phase_supervision(vid31_source.phase_source)
    assert len(phases) == 3732

    assert vid31_source.frame_level_action_source is not None
    frame_ivt = load_frame_level_ivt_supervision(vid31_source.frame_level_action_source)
    assert len(frame_ivt) == 3732
    assert sum(target.action_present for target in frame_ivt.values()) == 3715
    assert sum(not target.phase_agreement for target in frame_ivt.values()) == 9


def test_materialized_dataset_sidecars_are_runtime_loadable() -> None:
    manifest_path = DATASET_ROOT / "repair_manifest.json"
    manifest = load_derived_supervision_manifest(
        manifest_path,
        dataset_root_override=DATASET_ROOT,
    )

    vid30_source = manifest.video("VID30")
    assert vid30_source.annotation_source == (
        DATASET_ROOT / "Validation/VID30/vid30_repaired.json"
    ).resolve()
    vid30 = parse_annotation_file(
        vid30_source.annotation_source,
        expected_split=DatasetSplit.VALIDATION,
    )
    assert len(vid30.frames) == 2717

    vid31_source = manifest.video("VID31")
    assert vid31_source.phase_source is not None
    assert len(load_image_phase_supervision(vid31_source.phase_source)) == 3732
    assert vid31_source.frame_level_action_source is not None
    assert (
        len(load_frame_level_ivt_supervision(vid31_source.frame_level_action_source))
        == 3732
    )


def test_materialized_dataset_contract_preserves_release_layout_and_hashes() -> None:
    root = DATASET_ROOT
    raw = json.loads((root / "repair_manifest.json").read_text(encoding="utf-8"))
    contract = raw["dataset_contract"]

    assert contract["official_video_count"] == 20
    assert contract["added_video_directories"] == []
    assert contract["added_splits"] == []
    assert contract["added_classes"] == []
    assert sum(len(video_ids) for video_ids in contract["video_directories"].values()) == 20

    materialized_paths = {
        "vid30_repaired": root / "Validation/VID30/vid30_repaired.json",
        "vid31_phase_repaired": root / "Training/VID31/vid31_phase_repaired.json",
        "vid31_frame_ivt_repaired": (
            root / "Training/VID31/vid31_frame_ivt_repaired.json"
        ),
    }
    for name, path in materialized_paths.items():
        assert _sha256(path) == raw["materialized_sha256"][name]

    assert _sha256(root / "Validation/VID30/vid30.json") == raw["source_sha256"][
        "track20_vid30_json"
    ]
    assert _sha256(root / "Training/VID31/vid31.json") == raw["source_sha256"][
        "track20_vid31_json"
    ]


def test_derived_manifest_hashes_still_match_read_only_sources() -> None:
    if not CHOLEC80_ROOT.is_dir():
        pytest.skip("Optional upstream Cholec80 phase sources are unavailable")
    manifest_path = DATASET_ROOT / "repair_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = raw["source_sha256"]
    assert _sha256(DATASET_ROOT / "Validation/VID30/vid30.json") == expected[
        "track20_vid30_json"
    ]
    assert _sha256(DATASET_ROOT / "Training/VID31/vid31.json") == expected[
        "track20_vid31_json"
    ]
    assert _sha256(CHOLEC80_ROOT / "video30-phase.txt") == expected[
        "cholec80_video30_phase"
    ]
    assert _sha256(CHOLEC80_ROOT / "video31-phase.txt") == expected[
        "cholec80_video31_phase"
    ]


def test_runtime_routes_need_only_cholectrack20_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHOLEC80_30_31_ROOT", raising=False)
    adapter = CholecTrack20DatasetAdapter(DATASET_ROOT)
    vid30 = next(adapter.iter_video("VID30", max_samples=1))
    vid31 = next(adapter.iter_video("VID31", max_samples=1))
    root = DATASET_ROOT.resolve()
    for sample in (vid30, vid31):
        for media_ref in sample.inference.media_refs:
            Path(media_ref).resolve().relative_to(root)
    Path(vid30.provenance.annotation_source).resolve().relative_to(root)
    Path(vid31.provenance.phase_source).resolve().relative_to(root)
    Path(vid31.provenance.frame_action_source).resolve().relative_to(root)
