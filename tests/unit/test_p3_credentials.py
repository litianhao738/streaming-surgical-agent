from pathlib import Path

import pytest

from surgical_agent.api.credentials import (
    SecretValue,
    assert_secret_absent,
    load_api_key_file,
    resolve_api_key,
)


def _runtime_raw_key(*, padding: str, length: int = 32) -> str:
    prefix = "".join(chr(value) for value in (82, 65, 87, 45, 75, 69, 89, 45))
    return prefix + "x" * (length - len(prefix) - len(padding)) + padding


def test_api_key_file_splits_only_the_first_equals(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    key_name = "OPENROUTER_API_" + "KEY"
    key_value = "alpha" + "=beta"
    path.write_text(f"{key_name}={key_value}\n", encoding="utf-8")
    secret = load_api_key_file(path)
    assert secret.reveal() == key_value
    assert str(secret) == "<redacted>"
    assert repr(secret) == "SecretValue(<redacted>)"


@pytest.mark.parametrize("padding", ("=", "=="))
def test_api_key_file_accepts_long_raw_padded_key(
    tmp_path: Path, padding: str
) -> None:
    raw_key = _runtime_raw_key(padding=padding)
    path = tmp_path / "key.txt"
    path.write_text(raw_key + "\n", encoding="utf-8")

    secret = load_api_key_file(path)

    assert secret.reveal() == raw_key


def test_api_key_file_accepts_openrouter_key_before_documentation(
    tmp_path: Path,
) -> None:
    raw_key = "sk-or-v1-" + "x" * 64
    path = tmp_path / "key.txt"
    path.write_text(
        raw_key
        + "\n\nExample:\n"
        + "fetch('https://openrouter.ai/api/v1/chat/completions', {})\n",
        encoding="utf-8",
    )

    secret = load_api_key_file(path)

    assert secret.reveal() == raw_key


def test_api_key_file_rejects_multiple_openrouter_keys(tmp_path: Path) -> None:
    first = "sk-or-v1-" + "x" * 64
    second = "sk-or-v1-" + "y" * 64
    path = tmp_path / "key.txt"
    path.write_text(f"{first}\n{second}\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        load_api_key_file(path)

    assert first not in str(caught.value)
    assert second not in str(caught.value)


def test_api_key_file_rejects_short_raw_padded_key(tmp_path: Path) -> None:
    raw_key = _runtime_raw_key(padding="=", length=31)
    path = tmp_path / "key.txt"
    path.write_text(raw_key + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_api_key_file(path)


@pytest.mark.parametrize(
    "raw_key",
    (
        _runtime_raw_key(padding="===", length=35),
        _runtime_raw_key(padding="=", length=32)[:-1] + " \t=",
        _runtime_raw_key(padding="=", length=32)[:-1],
    ),
)
def test_api_key_file_rejects_malformed_raw_key_without_echoing_content(
    tmp_path: Path, raw_key: str
) -> None:
    path = tmp_path / "key.txt"
    path.write_text(raw_key + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        load_api_key_file(path)

    assert raw_key not in str(caught.value)


def test_api_key_file_rejects_empty_assignment(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    key_name = "OPENROUTER_API_" + "KEY"
    path.write_text(f"{key_name}=\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_api_key_file(path)


def test_api_key_file_errors_never_include_file_content(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    invalid_content = "sensitive" + "-without-" + "separator"
    path.write_text(f"{invalid_content}\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_api_key_file(path)
    assert invalid_content not in str(caught.value)


def test_api_key_inputs_are_mutually_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "key.txt"
    key_name = "OPENROUTER_API_" + "KEY"
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
    expected_message = (
        f"credential value found in persisted files: {sorted([str(leaked)])}"
    )
    assert str(caught.value) == expected_message
    assert needle not in str(caught.value)
