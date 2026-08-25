"""Safety checks for installing generated annotations beside raw release files."""

from pathlib import Path

import pytest

from tools.data_resolution.cholectrack20_vid30_vid31.materialize_repaired_sidecars import (
    MaterializationError,
    copy_sidecar,
    sha256,
)


def test_copy_sidecar_is_idempotent_and_hash_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")

    copied_hash = copy_sidecar(source, target, replace=False)
    assert copied_hash == sha256(source) == sha256(target)
    assert copy_sidecar(source, target, replace=False) == copied_hash


def test_copy_sidecar_refuses_silent_replacement(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")
    target.write_text('{"value": 2}\n', encoding="utf-8")

    with pytest.raises(MaterializationError, match="Refusing to replace"):
        copy_sidecar(source, target, replace=False)
