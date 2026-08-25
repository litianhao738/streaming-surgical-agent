"""Single source for verified CholecTrack20 categorical task ranges."""

from __future__ import annotations

from types import MappingProxyType

TASK_ID_BOUNDS = MappingProxyType(
    {
        "instrument": (0, 6),
        "verb": (0, 9),
        "target": (0, 14),
        "ivt": (0, 99),
        "phase": (0, 6),
    }
)
TASK_CLASS_COUNTS = MappingProxyType(
    {
        task: upper - lower + 1
        for task, (lower, upper) in TASK_ID_BOUNDS.items()
    }
)
