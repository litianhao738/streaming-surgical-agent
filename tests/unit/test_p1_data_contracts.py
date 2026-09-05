"""P1 canonical data, alignment, masking, and Gold-free contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from surgical_agent.data.causal_window import build_causal_frame_ids
from surgical_agent.data.masks import (
    UNRESOLVED_MISSING_SENTINEL,
    canonical_label_mask,
    is_valid_categorical_id,
)
from surgical_agent.data.media_backend import (
    AlignmentStatus,
    ExactFrameFolderResolver,
    MediaAlignmentError,
    Mp4FrameResolver,
)
from surgical_agent.data.parser import AnnotationSchemaError, parse_annotation_file
from surgical_agent.data.schemas import DatasetSplit, EvaluationTarget, InferenceSample
from surgical_agent.data.splits import (
    SplitSafetyError,
    discover_official_split_manifest,
    require_train_only,
)
from surgical_agent.data.targets import build_evaluation_target, build_inference_sample
from surgical_agent.inference.engine import require_inference_sample


def _instance(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "instrument": 0,
        "verb": 2,
        "target": 1,
        "phase": 3,
        "triplet": -1,
        "tool_bbox": [0.2, -0.002, 0.3, 0.4],
        "operator": 1,
        "iscrowd": 0,
        "area": 0.12,
        "score": 1.0,
        "intraoperative_track": 4,
        "intracorporeal_track": 5,
        "visibility_track": 6,
        "visibility": 1,
        "crowded": 0,
        "visible": 1,
        "occluded": 0,
        "bleeding": 0,
        "smoke": 0,
        "blurred": 0,
        "undercoverage": 0,
        "reflection": 0,
        "stainedlens": 0,
    }
    record.update(overrides)
    return record


def _annotation_payload(
    *,
    video_id: str = "VID02",
    split: str = "training",
    instances: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "info": {"dataset": "CholecTrack20", "version": 1.0},
        "video": {
            "name": video_id,
            "num_frames": 1,
            "split": split,
            "height": 480,
            "width": 854,
        },
        "categories": {
            "tools": [{"id": 0, "name": "grasper"}],
            "operators": [{"id": 1, "name": "main-surgeon-left-hand (MSLH)"}],
        },
        "annotations": {"26": instances if instances is not None else [_instance()]},
    }


def _write_annotation(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_canonical_parser_preserves_raw_ids_and_boundary_box(tmp_path: Path) -> None:
    annotation_path = tmp_path / "vid02.json"
    _write_annotation(annotation_path, _annotation_payload())

    video = parse_annotation_file(
        annotation_path,
        expected_split=DatasetSplit.TRAINING,
    )

    instance = video.frames[0].instances[0]
    assert video.video_id == "VID02"
    assert video.frame_ids == (26,)
    assert instance.bbox.y == -0.002
    assert not instance.bbox.is_inside_unit_frame
    assert instance.verb_id == 2
    assert instance.triplet_id == UNRESOLVED_MISSING_SENTINEL
    assert instance.mask.verb
    assert instance.mask.target
    assert not instance.mask.ivt
    assert {(term.task, term.canonical_name) for term in video.ontology_terms} == {
        ("instrument", "grasper"),
        ("operator", "main-surgeon-left-hand (MSLH)"),
    }


def test_parser_rejects_missing_required_instance_field(tmp_path: Path) -> None:
    malformed = _instance()
    malformed.pop("visibility_track")
    annotation_path = tmp_path / "bad.json"
    _write_annotation(
        annotation_path,
        _annotation_payload(instances=[malformed]),
    )

    with pytest.raises(AnnotationSchemaError, match="visibility_track"):
        parse_annotation_file(annotation_path)


def test_partial_label_masks_are_independent_and_conservative() -> None:
    mask = canonical_label_mask(
        instrument_id=0,
        verb_id=2,
        target_id=1,
        triplet_id=-1,
        phase_id=3,
    )
    assert mask.instrument and mask.verb and mask.target and mask.phase
    assert not mask.ivt
    assert not is_valid_categorical_id(-1)
    assert not is_valid_categorical_id(-2)
    assert not is_valid_categorical_id(True)


def test_exact_png_resolver_forbids_nearest_frame_fallback(tmp_path: Path) -> None:
    (tmp_path / "25.png").touch()
    (tmp_path / "27.png").touch()
    resolver = ExactFrameFolderResolver(
        video_id="VID02",
        split=DatasetSplit.TRAINING,
        frames_dir=tmp_path,
    )

    assert resolver.resolve(25).original_media_id == 25
    with pytest.raises(MediaAlignmentError, match="fallback is forbidden"):
        resolver.resolve(26)


def test_mp4_resolver_fails_closed_without_verified_offset(tmp_path: Path) -> None:
    media_path = tmp_path / "vid01.mp4"
    media_path.touch()
    resolver = Mp4FrameResolver(
        video_id="VID01",
        split=DatasetSplit.TESTING,
        media_path=media_path,
        frame_count=100,
        decoder_index_offset=None,
        alignment_version="BLOCKED_UNVERIFIED",
    )

    with pytest.raises(MediaAlignmentError, match="offset is unresolved"):
        resolver.resolve(26)


def test_mp4_resolver_applies_verified_annotation_id_minus_one_rule(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "vid01.mp4"
    media_path.touch()
    resolver = Mp4FrameResolver(
        video_id="VID01",
        split=DatasetSplit.TESTING,
        media_path=media_path,
        frame_count=100,
        decoder_index_offset=-1,
        alignment_version="ct20_test_mp4_annotation_id_minus_1_v1",
    )

    reference = resolver.resolve(26)

    assert reference.decoder_frame_index == 25
    assert reference.alignment_rule == "decoder_index = frame_id + -1"
    assert reference.status is AlignmentStatus.EXACT
    with pytest.raises(MediaAlignmentError, match="outside"):
        resolver.resolve(0)


def test_causal_window_is_ordered_and_excludes_future_frames() -> None:
    assert build_causal_frame_ids(
        [76, 1, 51, 26, 101],
        target_frame_id=76,
        max_frames=3,
    ) == (26, 51, 76)


def test_causal_window_resets_at_a_missing_expected_frame() -> None:
    available = [1, 26, 51, 101, 126, 151]

    assert build_causal_frame_ids(
        available,
        target_frame_id=101,
        max_frames=3,
        expected_frame_id_step=25,
    ) == (101,)
    assert build_causal_frame_ids(
        available,
        target_frame_id=126,
        max_frames=3,
        expected_frame_id_step=25,
    ) == (101, 126)
    assert build_causal_frame_ids(
        available,
        target_frame_id=151,
        max_frames=3,
        expected_frame_id_step=25,
    ) == (101, 126, 151)


def test_gold_free_runtime_and_evaluation_targets_are_separate(tmp_path: Path) -> None:
    annotation_path = tmp_path / "vid02.json"
    _write_annotation(annotation_path, _annotation_payload())
    frame = parse_annotation_file(annotation_path).frames[0]

    runtime = build_inference_sample(
        frame,
        causal_frame_ids=(26,),
        media_refs=("26.png",),
        alignment_version="ct20_exact_png_stem_v1",
    )
    target = build_evaluation_target(frame)

    assert isinstance(runtime, InferenceSample)
    assert isinstance(target, EvaluationTarget)
    assert "instances" not in runtime.__dataclass_fields__
    assert require_inference_sample(runtime) is runtime
    with pytest.raises(TypeError, match="only InferenceSample"):
        require_inference_sample(target)


def test_official_split_manifest_and_train_only_guard(tmp_path: Path) -> None:
    split_specs = (
        ("Training", "VID02", "training"),
        ("Validation", "VID30", "validation"),
        ("Testing", "VID01", "testing"),
    )
    for directory, video_id, split in split_specs:
        video_dir = tmp_path / directory / video_id
        payload = _annotation_payload(video_id=video_id, split=split)
        _write_annotation(video_dir / f"{video_id.lower()}.json", payload)
        _write_annotation(video_dir / f"{video_id.lower()}_repaired.json", payload)
        if split == "testing":
            (video_dir / f"{video_id.lower()}.mp4").touch()
        else:
            (video_dir / "Frames").mkdir()

    entries = discover_official_split_manifest(tmp_path)

    assert {entry.video_id for entry in entries} == {"VID01", "VID02", "VID30"}
    assert all("_repaired" not in entry.annotation_file for entry in entries)
    require_train_only(
        entry for entry in entries if entry.split is DatasetSplit.TRAINING
    )
    with pytest.raises(SplitSafetyError, match="held-out videos"):
        require_train_only(entries)
