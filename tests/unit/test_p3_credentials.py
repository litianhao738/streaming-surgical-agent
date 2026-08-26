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
    path.write_text("REQUESTY_API_KEY=alpha=beta\n", encoding="utf-8")
    secret = load_api_key_file(path)
    assert secret.reveal() == "alpha=beta"
    assert str(secret) == "<redacted>"
    assert repr(secret) == "SecretValue(<redacted>)"


def test_api_key_file_errors_never_include_file_content(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    path.write_text("sensitive-without-separator\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_api_key_file(path)
    assert "sensitive-without-separator" not in str(caught.value)


def test_api_key_inputs_are_mutually_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    path.write_text("REQUESTY_API_KEY=value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_api_key(api_key="value", api_key_file=path)


def test_exact_secret_scan_reports_only_the_path(tmp_path: Path) -> None:
    secret = SecretValue("needle-value")
    safe = tmp_path / "safe.json"
    safe.write_text('{"value":"safe"}', encoding="utf-8")
    assert_secret_absent(secret, (safe,))
    leaked = tmp_path / "leaked.json"
    leaked.write_text('{"value":"needle-value"}', encoding="utf-8")
    with pytest.raises(RuntimeError) as caught:
        assert_secret_absent(secret, (leaked,))
    assert str(leaked) in str(caught.value)
    assert "needle-value" not in str(caught.value)
