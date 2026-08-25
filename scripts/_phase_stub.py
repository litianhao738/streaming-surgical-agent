"""Shared fail-closed behavior for phase entry points not implemented yet."""

from __future__ import annotations


def blocked(script_name: str, phase: str) -> None:
    """Exit clearly instead of pretending an unimplemented phase succeeded."""

    raise SystemExit(
        f"BLOCKED: {script_name} belongs to {phase}; "
        "its implementation and phase tests have not passed yet."
    )
