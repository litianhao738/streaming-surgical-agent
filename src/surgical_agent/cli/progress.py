"""Single-line progress bars for long-running command-line stages."""

from __future__ import annotations

from typing import IO, Any

from tqdm import tqdm


def progress_bar(
    *,
    total: int | None,
    description: str,
    unit: str,
    enabled: bool,
    file: IO[str] | None = None,
) -> tqdm[Any]:
    """Create one dynamic terminal line instead of per-item log messages."""

    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        disable=not enabled,
        dynamic_ncols=True,
        leave=True,
        mininterval=0.2,
        file=file,
    )


__all__ = ["progress_bar"]
