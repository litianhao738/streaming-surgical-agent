"""Read-only verification for the CholecTrack20 single-root AutoDL bundle."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from surgical_agent.artifacts.manifest import sha256_file
from surgical_agent.config.loader import load_yaml
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.derived_supervision import load_derived_supervision_manifest
from surgical_agent.data.splits import discover_official_split_manifest


class PortabilityError(RuntimeError):
    """Raised when the AutoDL bundle is incomplete or escapes its root."""


def assert_within_root(path: Path, root: Path, *, role: str) -> Path:
    """Resolve one runtime path and reject paths outside the dataset root."""

    resolved_root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PortabilityError(f"{role} resolves outside CHOLECTRACK20_ROOT") from exc
    return resolved


def relative_runtime_path(path: Path, root: Path, *, role: str) -> str:
    """Return a root-contained runtime path in portable POSIX notation."""

    resolved = assert_within_root(path, root, role=role)
    return resolved.relative_to(root.expanduser().resolve()).as_posix()


def require_path(value: Path | None, role: str) -> Path:
    """Require an explicit derived runtime path without fallback guessing."""

    if value is None:
        raise PortabilityError(f"required runtime path is missing: {role}")
    return value


def verify_single_root(
    dataset_root: str | Path,
    bundle_path: str | Path,
) -> dict[str, object]:
    """Validate that all runtime CholecTrack20 inputs stay under one root."""

    root = Path(dataset_root).expanduser().resolve()
    bundle = load_yaml(bundle_path)
    entries = discover_official_split_manifest(root)
    repair_path = assert_within_root(
        root / str(bundle["required_relative_paths"]["repair_manifest"]),
        root,
        role="repair_manifest",
    )
    derived = load_derived_supervision_manifest(
        repair_path,
        dataset_root_override=root,
    )
    adapter = CholecTrack20DatasetAdapter(
        root,
        derived_manifest_path=repair_path,
    )
    split_counts = Counter(entry.split.value for entry in entries)
    expected_counts = {
        name.casefold(): int(count)
        for name, count in bundle["official_split_video_counts"].items()
    }
    if dict(sorted(split_counts.items())) != dict(sorted(expected_counts.items())):
        raise PortabilityError("official split counts do not match the bundle")
    if len(entries) != 20 or len({entry.video_id for entry in entries}) != 20:
        raise PortabilityError("official bundle must contain 20 unique videos")

    runtime_paths: dict[str, Path] = {"repair_manifest": repair_path}
    for entry in entries:
        runtime_paths[f"{entry.video_id}:annotation"] = Path(entry.annotation_file)
        runtime_paths[f"{entry.video_id}:media"] = Path(entry.media_source)

    vid30 = derived.video("VID30")
    vid31 = derived.video("VID31")
    runtime_paths["VID30:repaired_annotation"] = require_path(
        vid30.annotation_source,
        "VID30 annotation",
    )
    runtime_paths["VID31:phase"] = require_path(vid31.phase_source, "VID31 phase")
    runtime_paths["VID31:frame_ivt"] = require_path(
        vid31.frame_level_action_source,
        "VID31 frame IVT",
    )
    for role, path in runtime_paths.items():
        assert_within_root(path, root, role=role)
        if not path.exists():
            raise PortabilityError(f"required runtime path is missing: {role}")

    expected_manifest_hash = str(bundle["repair_manifest_sha256"])
    if sha256_file(repair_path) != expected_manifest_hash:
        raise PortabilityError("repair_manifest SHA-256 mismatch")

    required_relative = {
        str(role): str(value)
        for role, value in bundle["required_relative_paths"].items()
    }
    observed_materialized = {
        "repair_manifest": relative_runtime_path(
            repair_path,
            root,
            role="repair_manifest",
        ),
        "vid30_repaired": relative_runtime_path(
            require_path(vid30.annotation_source, "VID30 annotation"),
            root,
            role="VID30 annotation",
        ),
        "vid31_phase_repaired": relative_runtime_path(
            require_path(vid31.phase_source, "VID31 phase"),
            root,
            role="VID31 phase",
        ),
        "vid31_frame_ivt_repaired": relative_runtime_path(
            require_path(vid31.frame_level_action_source, "VID31 frame IVT"),
            root,
            role="VID31 frame IVT",
        ),
    }
    if observed_materialized != required_relative:
        raise PortabilityError("materialized paths do not match the bundle contract")

    adapter.collect(("VID02", "VID31"), samples_per_video=1)
    tuple(adapter.iter_video("VID30", max_samples=1))
    return {
        "schema_version": "cholectrack20_single_root_report_v1",
        "status": "PASS",
        "runtime_data_root_env": "CHOLECTRACK20_ROOT",
        "official_video_count": len(entries),
        "split_counts": dict(sorted(split_counts.items())),
        "runtime_paths": {
            role: relative_runtime_path(path, root, role=role)
            for role, path in sorted(runtime_paths.items())
        },
        "repair_manifest_sha256": expected_manifest_hash,
        "external_runtime_dependencies": [],
        "provenance_or_regeneration_only": ["Cholec80", "CholecT50"],
        "dataset_modified": False,
        "p4_contract_frozen": False,
    }
