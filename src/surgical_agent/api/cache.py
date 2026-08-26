"""Atomic cache whose entries bind exact safe request provenance."""

from __future__ import annotations

import json
from pathlib import Path

from surgical_agent.api.contracts import (
    ApiResponseRecord,
    CanonicalRequestMetadata,
    canonical_json_bytes,
)
from surgical_agent.api.errors import ApiCacheError
from surgical_agent.artifacts.manifest import atomic_write_text

CACHE_SCHEMA_VERSION = "api_cache_entry_v2"


class FileApiCache:
    """One immutable, validated JSON envelope per canonical request hash."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def path_for(self, request_hash: str) -> Path:
        if len(request_hash) != 64 or any(
            character not in "0123456789abcdef" for character in request_hash
        ):
            raise ApiCacheError("Cache key must be a lowercase SHA-256 digest")
        return self.root / f"{request_hash}.json"

    def get(self, metadata: CanonicalRequestMetadata) -> ApiResponseRecord | None:
        path = self.path_for(metadata.request_hash)
        if not path.exists() and not path.is_symlink():
            return None
        if path.is_symlink() or not path.is_file():
            raise ApiCacheError("Present cache entry is not a regular file")
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict) or set(envelope) != {
                "schema_version",
                "request",
                "response",
            }:
                raise ApiCacheError("Cache envelope has invalid fields")
            if envelope["schema_version"] != CACHE_SCHEMA_VERSION:
                raise ApiCacheError("Cache envelope has unsupported schema_version")
            request_mapping = envelope["request"]
            if not isinstance(request_mapping, dict):
                raise ApiCacheError("Cache request metadata is not an object")
            try:
                cached_metadata = CanonicalRequestMetadata.from_mapping(request_mapping)
            except (KeyError, TypeError, ValueError) as exc:
                raise ApiCacheError("Cache request metadata is invalid") from exc
            if canonical_json_bytes(
                cached_metadata.to_mapping()
            ) != canonical_json_bytes(metadata.to_mapping()):
                raise ApiCacheError("Cache request metadata mismatch")
            response_mapping = envelope["response"]
            if not isinstance(response_mapping, dict):
                raise ApiCacheError("Cache response is not an object")
            response = ApiResponseRecord.from_persisted_mapping(response_mapping)
            if response.request_hash != metadata.request_hash:
                raise ApiCacheError("Cache response request hash mismatch")
            if (
                response.provider != metadata.provider
                or response.endpoint_identifier != metadata.endpoint_identifier
                or response.requested_model_identifier
                != metadata.requested_model_identifier
            ):
                raise ApiCacheError("Cache response identity mismatch")
            return response
        except ApiCacheError:
            raise
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ApiCacheError("Invalid present cache entry") from exc

    def put(
        self,
        metadata: CanonicalRequestMetadata,
        response: ApiResponseRecord,
    ) -> Path:
        if response.request_hash != metadata.request_hash:
            raise ApiCacheError("Cache response request hash mismatch")
        path = self.path_for(metadata.request_hash)
        content = (
            json.dumps(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "request": metadata.to_mapping(),
                    "response": response.to_persisted_mapping(),
                },
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise ApiCacheError("Present cache entry is not a regular file")
            try:
                existing = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise ApiCacheError("Unable to read existing cache entry") from exc
            if existing != content:
                raise ApiCacheError("Refusing to overwrite divergent cache entry")
            return path
        try:
            atomic_write_text(path, content)
        except OSError as exc:
            raise ApiCacheError("Unable to write cache entry") from exc
        return path
