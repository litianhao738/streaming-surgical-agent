"""Single-root CholecTrack20 AutoDL bundle contract checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from surgical_agent.data.portability import (
    PortabilityError,
    assert_within_root,
    verify_single_root,
)


@pytest.fixture
def cholectrack20_root() -> Path:
    value = os.environ.get("CHOLECTRACK20_ROOT")
    if value is None or not Path(value).is_dir():
        pytest.skip("CholecTrack20 local-data integration fixture is unavailable")
    return Path(value).resolve()


def test_resolved_path_containment_rejects_sibling_prefix(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sibling = tmp_path / "data-escape"
    root.mkdir()
    sibling.mkdir()

    with pytest.raises(PortabilityError, match="outside"):
        assert_within_root(sibling, root, role="media")


def test_bundle_manifest_contains_only_relative_runtime_paths() -> None:
    raw = yaml.safe_load(
        Path("configs/data/cholectrack20_autodl_bundle.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert raw["schema_version"] == "cholectrack20_autodl_bundle_v1"
    assert raw["root_env"] == "CHOLECTRACK20_ROOT"
    assert raw["official_split_video_counts"] == {
        "Training": 10,
        "Validation": 2,
        "Testing": 8,
    }
    for relative in raw["required_relative_paths"].values():
        assert not Path(relative).is_absolute()


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
