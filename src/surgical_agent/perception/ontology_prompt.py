"""Compact, versioned CholecTrack20 ontology context for API prompts."""

from __future__ import annotations

import csv
from functools import lru_cache
from importlib.resources import files

_INSTRUMENTS = (
    "grasper",
    "bipolar",
    "hook",
    "scissors",
    "clipper",
    "irrigator",
    "specimen_bag",
)
_VERBS = (
    "grasp",
    "retract",
    "dissect",
    "coagulate",
    "clip",
    "cut",
    "aspirate",
    "irrigate",
    "pack",
    "null_verb",
)
_TARGETS = (
    "gallbladder",
    "cystic_plate",
    "cystic_duct",
    "cystic_artery",
    "cystic_pedicle",
    "blood_vessel",
    "fluid",
    "abdominal_wall_or_cavity",
    "liver",
    "adhesion",
    "omentum",
    "peritoneum",
    "gut",
    "specimen_bag",
    "null_target",
)
_PHASES = (
    "preparation",
    "calot_triangle_dissection",
    "clipping_and_cutting",
    "gallbladder_dissection",
    "gallbladder_packaging",
    "cleaning_and_coagulation",
    "gallbladder_extraction",
)

ACADEMIC_MEDICAL_CONTEXT = (
    "This input contains de-identified laparoscopic frames from an academic "
    "surgical-video benchmark; they depict a routine medical procedure, not "
    "interpersonal harm."
)


def _named_ids(names: tuple[str, ...]) -> str:
    return ",".join(f"{index}={name}" for index, name in enumerate(names))


@lru_cache(maxsize=1)
def load_prompt_ontology_text() -> str:
    """Return one deterministic ontology appendix shared by perception/verifiers."""

    resource = files("surgical_agent.research.signals.resources").joinpath(
        "ivt_components_v1.csv"
    )
    rows = list(csv.DictReader(resource.read_text(encoding="utf-8").splitlines()))
    if len(rows) != 100:
        raise RuntimeError("CholecTrack20 IVT resource must contain exactly 100 rows")
    ivt = ";".join(
        f"{int(row['ivt'])}=({int(row['instrument'])},{int(row['verb'])},{int(row['target'])})"
        for row in rows
    )
    return "\n".join(
        (
            (
                "Operational CholecTrack20 ontology. Names define numeric IDs; "
                "output IDs only."
            ),
            f"instrument:{_named_ids(_INSTRUMENTS)}",
            f"verb:{_named_ids(_VERBS)}",
            f"target:{_named_ids(_TARGETS)}",
            f"phase:{_named_ids(_PHASES)}",
            "IVT id=(instrument_id,verb_id,target_id):" + ivt,
            (
                "Select every visibly active frame-level label, not merely one tool. "
                "Every selected IVT must use the exact tuple above, and its component "
                "IDs must also be selected in instrument, verb, and target. IVT 94-99 "
                "mean a visible instrument with null_verb and null_target."
            ),
        )
    )


def add_academic_medical_context(prompt: str) -> str:
    """Prefix the unchanged benchmark vocabulary with its benign research context."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise TypeError("prompt must be non-empty text")
    return f"{ACADEMIC_MEDICAL_CONTEXT}\n{prompt}"
