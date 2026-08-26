"""Read-only verification for the CholecTrack20 single-root AutoDL bundle."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from surgical_agent.artifacts.manifest import sha256_file
from surgical_agent.config.loader import load_yaml
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.derived_supervision import load_derived_supervision_manifest
from surgical_agent.data.splits import discover_official_split_manifest

BUNDLE_SCHEMA_VERSION = "cholectrack20_autodl_bundle_v1"
EXPECTED_BUNDLE_KEYS = frozenset(
    {
        "schema_version",
        "dataset",
        "root_env",
        "official_split_video_counts",
        "required_relative_paths",
        "repair_manifest_sha256",
        "hash_authority",
        "provenance_or_regeneration_only",
        "optional_upstream_environment_variables",
        "runtime_external_data_roots_allowed",
        "dataset_mutation_allowed",
    }
)
EXPECTED_SPLIT_COUNTS = {"Training": 10, "Validation": 2, "Testing": 8}
EXPECTED_REQUIRED_RELATIVE_PATHS = {
    "repair_manifest": "repair_manifest.json",
    "vid30_repaired": "Validation/VID30/vid30_repaired.json",
    "vid31_phase_repaired": "Training/VID31/vid31_phase_repaired.json",
    "vid31_frame_ivt_repaired": "Training/VID31/vid31_frame_ivt_repaired.json",
}
EXPECTED_PROVENANCE_ONLY = ["Cholec80", "CholecT50"]
EXPECTED_OPTIONAL_UPSTREAM_ENVIRONMENT_VARIABLES = ["CHOLEC80_30_31_ROOT"]


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


def _require_exact_bundle_value(
    bundle: dict[str, object],
    key: str,
    expected: object,
) -> None:
    if bundle[key] != expected:
        raise PortabilityError(f"bundle {key} does not match the strict v1 contract")


def validate_bundle(bundle: dict[str, object]) -> dict[str, object]:
    """Validate the complete strict-v1 bundle contract before data access.

    Strict v1 intentionally rejects unknown keys so an AutoDL bundle cannot
    silently claim a capability that this verifier has not audited.
    """

    actual_keys = frozenset(bundle)
    unknown_keys = sorted(actual_keys - EXPECTED_BUNDLE_KEYS)
    missing_keys = sorted(EXPECTED_BUNDLE_KEYS - actual_keys)
    if unknown_keys or missing_keys:
        details = []
        if unknown_keys:
            details.append(f"unknown keys: {unknown_keys}")
        if missing_keys:
            details.append(f"missing keys: {missing_keys}")
        raise PortabilityError("bundle key set violates strict v1 contract: " + "; ".join(details))

    _require_exact_bundle_value(bundle, "schema_version", BUNDLE_SCHEMA_VERSION)
    _require_exact_bundle_value(bundle, "dataset", "CholecTrack20")
    _require_exact_bundle_value(bundle, "root_env", "CHOLECTRACK20_ROOT")
    _require_exact_bundle_value(bundle, "hash_authority", "repair_manifest.json")
    _require_exact_bundle_value(
        bundle,
        "provenance_or_regeneration_only",
        EXPECTED_PROVENANCE_ONLY,
    )
    _require_exact_bundle_value(
        bundle,
        "optional_upstream_environment_variables",
        EXPECTED_OPTIONAL_UPSTREAM_ENVIRONMENT_VARIABLES,
    )
    _require_exact_bundle_value(
        bundle,
        "official_split_video_counts",
        EXPECTED_SPLIT_COUNTS,
    )
    _require_exact_bundle_value(
        bundle,
        "required_relative_paths",
        EXPECTED_REQUIRED_RELATIVE_PATHS,
    )
    for key in (
        "runtime_external_data_roots_allowed",
        "dataset_mutation_allowed",
    ):
        if bundle[key] is not False:
            raise PortabilityError(f"bundle {key} must be false")

    manifest_hash = bundle["repair_manifest_sha256"]
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise PortabilityError("bundle repair_manifest_sha256 must be a SHA-256 hex string")
    try:
        int(manifest_hash, 16)
    except ValueError as exc:
        raise PortabilityError(
            "bundle repair_manifest_sha256 must be a SHA-256 hex string"
        ) from exc

    for role, relative in EXPECTED_REQUIRED_RELATIVE_PATHS.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not relative:
            raise PortabilityError(f"bundle required_relative_paths is unsafe: {role}")
    return bundle


def assert_output_outside_dataset_root(output: Path, dataset_root: Path) -> Path:
    """Resolve an output destination and reject any route into dataset data."""

    resolved_output = output.expanduser().resolve()
    resolved_root = dataset_root.expanduser().resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError:
        return resolved_output
    raise PortabilityError("output resolves inside CHOLECTRACK20_ROOT")


def verify_single_root(
    dataset_root: str | Path,
    bundle_path: str | Path,
) -> dict[str, object]:
    """Validate that all runtime CholecTrack20 inputs stay under one root."""

    root = Path(dataset_root).expanduser().resolve()
    bundle = validate_bundle(load_yaml(bundle_path))
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
    runtime_paths["VID30:derived_media"] = require_path(
        vid30.media_source,
        "VID30 derived media",
    )
    runtime_paths["VID31:derived_media"] = require_path(
        vid31.media_source,
        "VID31 derived media",
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

    sampled_records = (
        *adapter.collect(("VID02", "VID31"), samples_per_video=1),
        *tuple(adapter.iter_video("VID30", max_samples=1)),
    )
    sampled_media_refs: dict[str, Path] = {}
    for sample in sampled_records:
        for index, media_ref in enumerate(sample.inference.media_refs):
            role = (
                f"{sample.inference.video_id}:{sample.inference.target_frame_id}"
                f":media:{index}"
            )
            media_path = Path(media_ref)
            assert_within_root(media_path, root, role=role)
            if not media_path.exists():
                raise PortabilityError(f"required runtime path is missing: {role}")
            sampled_media_refs[role] = media_path
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
        "sampled_media_refs": {
            role: relative_runtime_path(path, root, role=role)
            for role, path in sorted(sampled_media_refs.items())
        },
        "repair_manifest_sha256": expected_manifest_hash,
        "external_runtime_dependencies": [],
        "provenance_or_regeneration_only": ["Cholec80", "CholecT50"],
        "dataset_modified": False,
        "p4_contract_frozen": False,
    }
