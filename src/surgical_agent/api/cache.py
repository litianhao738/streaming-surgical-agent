"""Atomic file cache for validated API response records."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from surgical_agent.api.contracts import ApiResponseRecord
from surgical_agent.api.errors import ApiCacheError
from surgical_agent.artifacts.manifest import atomic_write_text

CACHE_SCHEMA_VERSION = "api_response_cache_v1"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _record_dict(record: ApiResponseRecord) -> dict[str, Any]:
    return {
        name: _plain(getattr(record, name))
        for name in ApiResponseRecord.__dataclass_fields__
    }


class FileApiCache:
    """One immutable JSON entry per canonical request hash."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def path_for(self, request_hash: str) -> Path:
        if len(request_hash) != 64 or any(
            character not in "0123456789abcdef" for character in request_hash
        ):
            raise ApiCacheError("Cache key must be a lowercase SHA-256 digest")
        return self.root / f"{request_hash}.json"

    def get(self, request_hash: str) -> ApiResponseRecord | None:
        path = self.path_for(request_hash)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise ApiCacheError(f"Unsupported cache schema in {path}")
            if payload.get("request_hash") != request_hash:
                raise ApiCacheError(f"Cache filename/content mismatch in {path}")
            record = payload.get("response")
            if not isinstance(record, dict):
                raise ApiCacheError(f"Cache response is not an object in {path}")
            parsed = record.get("parsed_payload")
            if not isinstance(parsed, dict):
                raise ApiCacheError(f"Cache parsed_payload is not an object in {path}")
            return ApiResponseRecord(**record)
        except ApiCacheError:
            raise
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ApiCacheError(f"Invalid cache entry {path}: {exc}") from exc

    def put(self, record: ApiResponseRecord) -> Path:
        path = self.path_for(record.request_hash)
        content = json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "request_hash": record.request_hash,
                "response": _record_dict(record),
            },
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
        ) + "\n"
        if path.is_file():
            existing = path.read_text(encoding="utf-8")
            if existing != content:
                raise ApiCacheError(f"Refusing to overwrite divergent cache entry {path}")
            return path
        atomic_write_text(path, content)
        return path
