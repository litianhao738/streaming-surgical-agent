"""Credential loading and exact-value leak checks without serialization."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class SecretValue:
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("API key must be non-empty")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __str__(self) -> str:
        return "<redacted>"

    def __repr__(self) -> str:
        return "SecretValue(<redacted>)"


def load_api_key_file(path: str | Path) -> SecretValue:
    source = Path(path).expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) != 1 or "=" not in lines[0]:
        raise ValueError("API key file must contain one NAME=value line")
    line = lines[0]
    first_equals = line.find("=")
    prefix = line[:first_equals]
    suffix = line[first_equals + 1 :]

    trailing_padding = len(line) - len(line.rstrip("="))
    if trailing_padding in (1, 2):
        body = line[:-trailing_padding]
        if (
            len(line) >= 32
            and "=" not in body
            and not any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in line
            )
        ):
            return SecretValue(line)

    # A long delimiter-free prefix followed only by padding-like characters is
    # a malformed raw credential, not a NAME=value assignment.
    if (
        len(line) >= 32
        and "=" not in prefix
        and all(
            character == "="
            or character.isspace()
            or ord(character) < 32
            or ord(character) == 127
            for character in suffix
        )
    ):
        raise ValueError("API key file contains malformed credential padding")

    name, value = line.split("=", 1)
    if not name.strip() or not value:
        raise ValueError("API key file must contain one non-empty NAME=value line")
    return SecretValue(value)


def resolve_api_key(
    *,
    api_key: str | None,
    api_key_file: str | Path | None,
) -> SecretValue:
    if api_key is not None and api_key_file is not None:
        raise ValueError("--api-key and --api-key-file are mutually exclusive")
    if api_key_file is not None:
        return load_api_key_file(api_key_file)
    if api_key is not None:
        return SecretValue(api_key)
    raise ValueError("real API mode requires --api-key or --api-key-file")


def assert_secret_absent(secret: SecretValue, paths: Iterable[Path]) -> None:
    needle = secret.reveal().encode("utf-8")
    leaked: list[str] = []
    for path in paths:
        if path.is_file() and needle in path.read_bytes():
            leaked.append(str(path))
    if leaked:
        raise RuntimeError(f"credential value found in persisted files: {sorted(leaked)}")
