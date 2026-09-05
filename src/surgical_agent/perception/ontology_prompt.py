"""Compact, versioned CholecTrack20 ontology context for API prompts."""

from __future__ import annotations

import csv
from collections.abc import Mapping
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
    "Authorized de-identified academic analysis of routine laparoscopic "
    "cholecystectomy benchmark frames for surgical workflow research."
)


def _named_ids(names: tuple[str, ...]) -> str:
    return ",".join(f"{index}={name}" for index, name in enumerate(names))


_TASK_NAMES: Mapping[str, tuple[str, ...]] = {
    "instrument": _INSTRUMENTS,
    "verb": _VERBS,
    "target": _TARGETS,
    "phase": _PHASES,
}


@lru_cache(maxsize=1)
def _ivt_rows() -> tuple[tuple[int, int, int, int], ...]:
    resource = files("surgical_agent.research.signals.resources").joinpath(
        "ivt_components_v1.csv"
    )
    rows = tuple(
        (
            int(row["ivt"]),
            int(row["instrument"]),
            int(row["verb"]),
            int(row["target"]),
        )
        for row in csv.DictReader(resource.read_text(encoding="utf-8").splitlines())
    )
    if len(rows) != 100 or tuple(row[0] for row in rows) != tuple(range(100)):
        raise RuntimeError("CholecTrack20 IVT resource must contain IDs 0 through 99")
    return rows


@lru_cache(maxsize=1)
def load_prompt_ontology_text() -> str:
    """Return one deterministic ontology appendix shared by perception/verifiers."""

    ivt = ";".join(
        f"{ivt_id}=({instrument_id},{verb_id},{target_id})"
        for ivt_id, instrument_id, verb_id, target_id in _ivt_rows()
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


def load_scoped_prompt_ontology_text(
    requested_tasks: tuple[str, ...],
    candidate_ids: Mapping[str, tuple[int, ...]],
    *,
    max_ivt_candidates: int = 20,
) -> str:
    """Return only ontology entries needed by one targeted verification call.

    The request already restricts selectable values to ``candidate_ids``.  A
    verifier for instrument or phase therefore does not need the unrelated
    100-row IVT table.  Interaction verification retains the complete names of
    its component vocabularies but receives only its bounded IVT candidates.
    """

    requested = tuple(dict.fromkeys(requested_tasks))
    if not requested or any(
        task not in {"instrument", "verb", "target", "ivt", "phase"}
        for task in requested
    ):
        raise ValueError("requested_tasks must contain known ontology tasks")
    if not isinstance(candidate_ids, Mapping) or any(
        task not in candidate_ids for task in requested
    ):
        raise ValueError("candidate_ids must cover every requested task")

    named_tasks = set(requested) - {"ivt"}
    if "ivt" in requested:
        named_tasks.update(("instrument", "verb", "target"))
    lines = [
        (
            "Operational CholecTrack20 ontology subset for the requested paths. "
            "Names define numeric IDs; output IDs only."
        )
    ]
    for task in ("instrument", "verb", "target", "phase"):
        if task in named_tasks:
            lines.append(f"{task}:{_named_ids(_TASK_NAMES[task])}")

    if "ivt" in requested:
        values = tuple(candidate_ids["ivt"])
        if (
            not values
            or len(values) > max_ivt_candidates
            or len(set(values)) != len(values)
            or any(type(value) is not int or not 0 <= value < 100 for value in values)
        ):
            raise ValueError("ivt candidate IDs must be one to twenty unique IDs")
        rows = _ivt_rows()
        lines.append(
            "IVT candidate id=(instrument_id,verb_id,target_id):"
            + ";".join(
                f"{ivt_id}=({rows[ivt_id][1]},{rows[ivt_id][2]},{rows[ivt_id][3]})"
                for ivt_id in values
            )
        )
        lines.append(
            "Every selected IVT must use an exact listed tuple and satisfy component "
            "closure in the requested instrument, verb and target fields."
        )
    else:
        lines.append(
            "Select only supplied candidate IDs for the requested target-frame field."
        )
    return "\n".join(lines)


def add_academic_medical_context(prompt: str) -> str:
    """Prefix the unchanged benchmark vocabulary with its benign research context."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise TypeError("prompt must be non-empty text")
    return f"{ACADEMIC_MEDICAL_CONTEXT}\n{prompt}"
