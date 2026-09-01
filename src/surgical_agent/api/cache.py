"""Race-safe cache whose entries bind exact safe request provenance."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from surgical_agent.api.contracts import (
    ApiResponseRecord,
    CanonicalRequestMetadata,
    canonical_json_bytes,
)
from surgical_agent.api.errors import ApiCacheError

CACHE_SCHEMA_VERSION = "api_cache_entry_v3"


def _file_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


@dataclass(frozen=True)
class _VerifiedRead:
    content: str
    identity: tuple[int, int]


class FileApiCache:
    """One immutable, descriptor-verified envelope per canonical request hash.

    Entry identity is checked before open, after open, after read, and after parsing.
    This detects ordinary pathname replacement during an operation; it does not lock
    the cache directory or prevent another process from mutating it after return.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def path_for(self, request_hash: str) -> Path:
        if len(request_hash) != 64 or any(
            character not in "0123456789abcdef" for character in request_hash
        ):
            raise ApiCacheError("Cache key must be a lowercase SHA-256 digest")
        return self.root / f"{request_hash}.json"

    @staticmethod
    def _read_verified(path: Path) -> _VerifiedRead | None:
        try:
            before = os.lstat(path)
        except FileNotFoundError:
            try:
                os.lstat(path)
            except FileNotFoundError:
                return None
            except OSError:
                raise ApiCacheError(
                    "Cache entry could not be inspected safely"
                ) from None
            raise ApiCacheError("Cache entry raced from absent to present") from None
        except OSError:
            raise ApiCacheError("Cache entry could not be inspected safely") from None
        if not stat.S_ISREG(before.st_mode):
            raise ApiCacheError("Present cache entry is not a regular file")

        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor: int | None = None
        stream = None
        try:
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            after_open = os.lstat(path)
            if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(after_open.st_mode):
                raise ApiCacheError("Present cache entry is not a regular file")
            identities = {
                _file_identity(before),
                _file_identity(opened),
                _file_identity(after_open),
            }
            if len(identities) != 1:
                raise ApiCacheError("Cache entry raced during verified read")
            stream = os.fdopen(descriptor, "r", encoding="utf-8")
            descriptor = None
            content = stream.read()
            opened_after_read = os.fstat(stream.fileno())
            after_read = os.lstat(path)
            if (
                not stat.S_ISREG(opened_after_read.st_mode)
                or not stat.S_ISREG(after_read.st_mode)
                or _file_identity(opened_after_read) not in identities
                or _file_identity(after_read) not in identities
            ):
                raise ApiCacheError("Cache entry raced during verified read")
            return _VerifiedRead(content, _file_identity(opened_after_read))
        except ApiCacheError:
            raise
        except (OSError, UnicodeError):
            raise ApiCacheError(
                "Cache entry raced or could not be opened safely"
            ) from None
        finally:
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
            elif descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    @staticmethod
    def _verify_entry_identity(path: Path, expected: tuple[int, int]) -> None:
        try:
            current = os.lstat(path)
        except OSError:
            raise ApiCacheError("Cache entry raced after verified read") from None
        if not stat.S_ISREG(current.st_mode) or _file_identity(current) != expected:
            raise ApiCacheError("Cache entry raced after verified read")

    @staticmethod
    def _parse_envelope(
        content: str,
        metadata: CanonicalRequestMetadata,
    ) -> ApiResponseRecord:
        try:
            envelope = json.loads(content)
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
            FileApiCache._validate_response_binding(metadata, response)
            return response
        except ApiCacheError:
            raise
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ApiCacheError("Invalid present cache entry") from exc

    @staticmethod
    def _validate_response_binding(
        metadata: CanonicalRequestMetadata,
        response: ApiResponseRecord,
    ) -> None:
        if response.request_hash != metadata.request_hash:
            raise ApiCacheError("Cache response request hash mismatch")
        if (
            response.provider != metadata.provider
            or response.endpoint_identifier != metadata.endpoint_identifier
            or response.requested_model_identifier
            != metadata.requested_model_identifier
        ):
            raise ApiCacheError("Cache response identity mismatch")

    @staticmethod
    def _validate_origin(response: ApiResponseRecord) -> None:
        if (
            response.cache_hit
            or response.provider_call_count <= 0
            or response.retry_count != response.provider_call_count - 1
            or response.total_latency_ms is None
            or response.provider_cost != response.origin_provider_cost
        ):
            raise ApiCacheError("Cache can persist only a coherent origin response")

    def get(self, metadata: CanonicalRequestMetadata) -> ApiResponseRecord | None:
        path = self.path_for(metadata.request_hash)
        verified = self._read_verified(path)
        if verified is None:
            return None
        response = self._parse_envelope(verified.content, metadata)
        self._validate_origin(response)
        self._verify_entry_identity(path, verified.identity)
        return response

    @staticmethod
    def _render(
        metadata: CanonicalRequestMetadata,
        response: ApiResponseRecord,
    ) -> str:
        return (
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

    def put(
        self,
        metadata: CanonicalRequestMetadata,
        response: ApiResponseRecord,
    ) -> Path:
        self._validate_response_binding(metadata, response)
        self._validate_origin(response)
        content = self._render(metadata, response)
        path = self.path_for(metadata.request_hash)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise ApiCacheError("Unable to prepare cache directory") from None

        descriptor: int | None = None
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{metadata.request_hash}.",
                suffix=".tmp",
                dir=self.root,
                text=True,
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                descriptor = None
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            except OSError:
                raise ApiCacheError(
                    "Unable to install cache entry without overwrite"
                ) from None
            existing = self.get(metadata)
            if existing is None or canonical_json_bytes(
                existing.to_persisted_mapping()
            ) != canonical_json_bytes(response.to_persisted_mapping()):
                raise ApiCacheError("Refusing divergent concurrent cache entry")
            return path
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    raise ApiCacheError(
                        "Unable to clean temporary cache entry"
                    ) from None
