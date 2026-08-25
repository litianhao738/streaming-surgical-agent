"""Cross-machine configuration and derived-manifest path tests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from surgical_agent.artifacts.manifest import sha256_mapping, sha256_source_tree
from surgical_agent.config.loader import load_experiment_config, resolve_dataset_root
from surgical_agent.data.derived_supervision import (
    DERIVED_SCHEMA_VERSION,
    load_derived_supervision_manifest,
)
from surgical_agent.data.qualification import SUPERVISION_FIELDS


def test_dataset_root_priority_is_cli_then_environment_then_config(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "cli"
    environment = tmp_path / "environment"
    configured = tmp_path / "configured"
    for path in (cli, environment, configured):
        path.mkdir()
    config = {
        "root": str(configured),
        "root_env": "CHOLECTRACK20_ROOT",
    }
    assert resolve_dataset_root(
        config,
        cli_root=cli,
        environment={"CHOLECTRACK20_ROOT": str(environment)},
    ) == cli.resolve()
    assert resolve_dataset_root(
        config,
        environment={"CHOLECTRACK20_ROOT": str(environment)},
    ) == environment.resolve()
    assert resolve_dataset_root(config, environment={}) == configured.resolve()


def test_experiment_reference_composition_resolves_external_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (tmp_path / "base.yaml").write_text(
        "project:\n  current_phase: P2\nruntime:\n  device: cpu\n",
        encoding="utf-8",
    )
    (tmp_path / "data.yaml").write_text(
        "dataset: cholectrack20\nroot: null\nroot_env: CHOLECTRACK20_ROOT\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "base_config: base.yaml\ndata_config: data.yaml\nexperiment: smoke\n",
        encoding="utf-8",
    )

    loaded = load_experiment_config(experiment, dataset_root=root)

    assert Path(loaded["data"]["root"]) == root.resolve()
    assert loaded["data"]["root_resolved_from"] == "cli"
    assert loaded["project"]["current_phase"] == "P2"


def test_manifest_paths_rebase_from_windows_root(tmp_path: Path) -> None:
    media = tmp_path / "Training/VID31/Frames"
    media.mkdir(parents=True)
    phase = tmp_path / "Training/VID31/phase.json"
    phase.write_text("{}", encoding="utf-8")
    fields = {name: name == "phase" for name in SUPERVISION_FIELDS}
    manifest_path = tmp_path / "repair_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": DERIVED_SCHEMA_VERSION,
                "raw_dataset_root": "D:\\cholec_dataset",
                "raw_dataset_modified": False,
                "policy": "EXPLICIT_DERIVED_SOURCES_ONLY",
                "videos": {
                    "VID31": {
                        "split": "training",
                        "media_source": "D:\\cholec_dataset\\Training\\VID31\\Frames",
                        "phase_source": "D:\\cholec_dataset\\Training\\VID31\\phase.json",
                        "field_supervision": fields,
                        "status": "PHASE_ONLY",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_derived_supervision_manifest(
        manifest_path,
        dataset_root_override=tmp_path,
    )

    assert manifest.raw_dataset_root == tmp_path.resolve()
    assert manifest.video("VID31").media_source == media.resolve()
    assert manifest.video("VID31").phase_source == phase.resolve()


def test_config_hash_canonicalizes_yaml_dates() -> None:
    assert sha256_mapping({"verified_on": date(2026, 8, 22)}) == sha256_mapping(
        {"verified_on": "2026-08-22"}
    )


def test_source_tree_hash_excludes_non_runtime_text_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "src/module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "scripts/API.txt").write_text("not-runtime", encoding="utf-8")
    (tmp_path / "configs/base.yaml").write_text("seed: 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    before = sha256_source_tree(tmp_path)
    (tmp_path / "scripts/API.txt").write_text("changed", encoding="utf-8")
    assert sha256_source_tree(tmp_path) == before
    (tmp_path / "src/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert sha256_source_tree(tmp_path) != before
