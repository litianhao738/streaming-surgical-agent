"""Single-root CholecTrack20 AutoDL bundle contract checks."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from surgical_agent.data.portability import (
    PortabilityError,
    assert_output_outside_dataset_root,
    assert_within_root,
    validate_bundle,
    verify_single_root,
)


@pytest.fixture
def cholectrack20_root() -> Path:
    value = os.environ.get("CHOLECTRACK20_ROOT")
    if value is None or not Path(value).is_dir():
        pytest.skip("CholecTrack20 local-data integration fixture is unavailable")
    return Path(value).resolve()


def _valid_bundle() -> dict[str, object]:
    return yaml.safe_load(
        Path("configs/data/cholectrack20_autodl_bundle.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_resolved_path_containment_rejects_sibling_prefix(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sibling = tmp_path / "data-escape"
    root.mkdir()
    sibling.mkdir()

    with pytest.raises(PortabilityError, match="outside"):
        assert_within_root(sibling, root, role="media")


def test_bundle_manifest_contains_only_relative_runtime_paths() -> None:
    raw = _valid_bundle()

    assert raw["schema_version"] == "cholectrack20_autodl_bundle_v1"
    assert raw["root_env"] == "CHOLECTRACK20_ROOT"
    assert raw["official_split_video_counts"] == {
        "Training": 10,
        "Validation": 2,
        "Testing": 8,
    }
    for relative in raw["required_relative_paths"].values():
        assert not Path(relative).is_absolute()


def test_bundle_validation_rejects_unknown_v1_key_before_dataset_access(
    tmp_path: Path,
) -> None:
    bundle = _valid_bundle()
    bundle["unexpected"] = "not allowed"
    bundle_path = tmp_path / "bundle.yaml"
    bundle_path.write_text(yaml.safe_dump(bundle), encoding="utf-8")

    with pytest.raises(PortabilityError, match="unknown keys"):
        verify_single_root(tmp_path / "missing-dataset", bundle_path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda bundle: bundle.__setitem__("dataset", "Cholec80"),
            "dataset",
        ),
        (
            lambda bundle: bundle.__setitem__(
                "runtime_external_data_roots_allowed", True
            ),
            "runtime_external_data_roots_allowed",
        ),
        (
            lambda bundle: bundle["official_split_video_counts"].__setitem__(
                "Training", True
            ),
            "official_split_video_counts",
        ),
        (
            lambda bundle: bundle["required_relative_paths"].__setitem__(
                "repair_manifest", "../repair_manifest.json"
            ),
            "required_relative_paths",
        ),
    ],
)
def test_bundle_validation_rejects_contradictory_runtime_contract(
    mutator: object,
    message: str,
) -> None:
    bundle = deepcopy(_valid_bundle())
    mutator(bundle)

    with pytest.raises(PortabilityError, match=message):
        validate_bundle(bundle)


def test_output_destination_rejects_dataset_descendant_without_writing(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    with pytest.raises(PortabilityError, match="output"):
        assert_output_outside_dataset_root(dataset_root / "report.json", dataset_root)


def test_output_destination_allows_path_outside_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output = tmp_path / "reports" / "bundle.json"

    assert assert_output_outside_dataset_root(output, dataset_root) == output.resolve()


def test_output_destination_rejects_existing_symlink_to_dataset_root(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    symlink = tmp_path / "reports-link"
    try:
        symlink.symlink_to(dataset_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Windows symlink creation is unavailable: {exc}")

    with pytest.raises(PortabilityError, match="output"):
        assert_output_outside_dataset_root(symlink / "bundle.json", dataset_root)


def test_single_root_report_never_serializes_absolute_runtime_paths(
    cholectrack20_root: Path,
) -> None:
    report = verify_single_root(
        cholectrack20_root,
        Path("configs/data/cholectrack20_autodl_bundle.yaml"),
    )
    rendered = json.dumps(report, sort_keys=True)

    assert report["status"] == "PASS"
    assert str(cholectrack20_root) not in rendered
    assert report["runtime_data_root_env"] == "CHOLECTRACK20_ROOT"
    runtime_paths = report["runtime_paths"]
    assert "VID30:derived_media" in runtime_paths
    assert "VID31:derived_media" in runtime_paths
    assert runtime_paths["VID30:derived_media"] == "Validation/VID30/Frames"
    assert runtime_paths["VID31:derived_media"] == "Training/VID31/Frames"
    sampled_refs = report["sampled_media_refs"]
    assert sampled_refs
    for relative in [*runtime_paths.values(), *sampled_refs.values()]:
        assert not Path(str(relative)).is_absolute()
