from pathlib import Path

import pytest

from surgical_agent.api.credentials import (
    SecretValue,
    assert_secret_absent,
    load_api_key_file,
    resolve_api_key,
)


def test_api_key_file_splits_only_the_first_equals(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    key_name = "REQUESTY_API_" + "KEY"
    key_value = "alpha" + "=beta"
    path.write_text(f"{key_name}={key_value}\n", encoding="utf-8")
    secret = load_api_key_file(path)
    assert secret.reveal() == key_value
    assert str(secret) == "<redacted>"
    assert repr(secret) == "SecretValue(<redacted>)"


def test_api_key_file_errors_never_include_file_content(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    invalid_content = "sensitive" + "-without-" + "separator"
    path.write_text(f"{invalid_content}\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_api_key_file(path)
    assert invalid_content not in str(caught.value)


def test_api_key_inputs_are_mutually_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    key_name = "REQUESTY_API_" + "KEY"
    file_value = "val" + "ue"
    direct_value = "dir" + "ect"
    path.write_text(f"{key_name}={file_value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_api_key(api_key=direct_value, api_key_file=path)


def test_exact_secret_scan_reports_only_the_path(tmp_path: Path) -> None:
    needle = "needle" + "-value"
    secret = SecretValue(needle)
    safe = tmp_path / "safe.json"
    safe.write_text('{"value":"safe"}', encoding="utf-8")
    assert_secret_absent(secret, (safe,))
    leaked = tmp_path / "leaked.json"
    leaked.write_text(f'{{"value":"{needle}"}}', encoding="utf-8")
    with pytest.raises(RuntimeError) as caught:
        assert_secret_absent(secret, (leaked,))
    expected_message = f"credential value found in persisted files: {sorted([str(leaked)])}"
    assert str(caught.value) == expected_message
    assert needle not in str(caught.value)
