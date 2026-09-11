import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.train_final_only_gate import fit_pilot
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.gate.final_only_training import (
    FEATURE_VERSION,
    extract_features,
    mask_tracker,
)


def rows():
    result = []
    for video in ("VID02", "VID04", "VID11"):
        for frame in range(1, 7):
            positive = frame % 2
            h0 = {"instrument": [0], "verb": [0], "target": [0] if positive else [],
                  "ivt": [0], "phase": [1]}
            features = extract_features(h0, target_frame_id=frame, causal_frame_ids=[frame])
            result.append({"sample_id": f"{video}:{frame}", "video_id": video, "frame_id": frame,
                           "source_split": "Training", "policy_id": "synthetic-fixture-only",
                           "feature_version": FEATURE_VERSION,
                           "features_no_tracker": mask_tracker(features), "features_with_tracker": features,
                           "labels": {"benefit": positive, "utility_delta": 0.25 if positive else -0.25,
                                      "any_head_harm": not positive}})
    return result


def test_cpu_pilot_grouping_serialization_and_non_deployment():
    report = fit_pilot(rows())
    assert report["deployable"] is False
    assert report["validation_calibrated"] is False
    for value in report["variants"].values():
        assert len(value["oof_scores"]) == 18
        for fold in value["folds"]:
            assert fold["held_out_video_id"] not in fold["fit_video_ids"]
    serialized = json.loads(json.dumps(report))
    assert serialized["artifacts"]["no_tracker"]["policy_id"] == "synthetic-fixture-only"


def test_unfit_realistic_one_positive_video_refuses_instead_of_creating_dummy_model():
    data = rows()
    for row in data:
        row["labels"]["benefit"] = int(row["video_id"] == "VID02" and row["frame_id"] == 1)
    with pytest.raises(ValueError, match="not ready"):
        fit_pilot(data)


def test_collection_manifest_binds_training_mask_images_and_identity(monkeypatch, tmp_path):
    from scripts import run_recent_mean_panel_trial as trial

    image = tmp_path / "frame.png"
    image.write_bytes(b"test image bytes only")
    sample = SimpleNamespace(target_frame_id=51, causal_frame_ids=[51], media_refs=[image])
    adapter = SimpleNamespace(entries={"VID103": SimpleNamespace(split=DatasetSplit.TRAINING)},
                              iter_inference_video=lambda video: iter([sample]),
                              iter_video=lambda video, frame_ids: iter([object()]))
    mask = dict.fromkeys(("instrument", "verb", "target", "ivt", "phase"), True)
    monkeypatch.setattr(trial, "_mask_only", lambda _: mask)
    monkeypatch.setattr(trial, "build_gemini_base", lambda *a: object())
    monkeypatch.setattr(trial, "canonical_request_metadata", lambda _: SimpleNamespace(to_mapping=lambda: {"frozen": True}))
    row = {"video_id": "VID103", "frame_id": 51, "anchor_frame_id": 51, "causal_frame_ids": [51],
           "images": [{"sha256": trial.sha(image)}], "gt_availability_only": mask}
    manifest = {"schema_version": "final_only_gate_collection_selection_v1",
                "no_gt_label_values_used_for_selection": True, "selection": [row], "selection_rule": "fixed-time"}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = trial.load_gate_selection(path, adapter)
    assert result[0]["request_metadata"] == {"frozen": True}
    adapter.entries["VID103"].split = DatasetSplit.TESTING
    with pytest.raises(ValueError, match="Training"):
        trial.load_gate_selection(path, adapter)
    adapter.entries["VID103"].split = DatasetSplit.TRAINING
    image.write_bytes(b"changed pixels")
    with pytest.raises(ValueError, match="hash"):
        trial.load_gate_selection(path, adapter)


def test_check_command_rejects_features_modified_after_gt_join(tmp_path):
    from scripts.train_final_only_gate import run
    from surgical_agent.artifacts.manifest import sha256_file

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    data = rows()
    frozen = [{key: value for key, value in row.items() if key != "labels"} for row in data]
    feature_path = prepared / "features_before_gt.json"
    feature_path.write_text(json.dumps(frozen), encoding="utf-8")
    data[0]["features_no_tracker"]["h0_target_count"] = 99
    data[0]["features_with_tracker"]["h0_target_count"] = 99
    example_path = prepared / "training_examples.json"
    example_path.write_text(json.dumps(data), encoding="utf-8")
    manifest = {"schema_version": "final_only_gate_preparation_v1", "policy_id": "synthetic-fixture-only",
                "training_examples_sha256": sha256_file(example_path),
                "features_before_gt_sha256": sha256_file(feature_path)}
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="after the GT join"):
        run(SimpleNamespace(prepared=prepared, target="benefit", check_only=True))
