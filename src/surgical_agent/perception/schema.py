"""Strict wire-schema validation for joint frame perception."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.perception.contracts import EVIDENCE_REF_CODES

JOINT_PERCEPTION_SCHEMA_VERSION = "joint_perception_frame_v1"
TASK_LAYOUT = (
    ("instrument", 7),
    ("verb", 10),
    ("target", 15),
    ("ivt", 20),
    ("phase", 7),
)
_TASK_NAMES = tuple(task for task, _count in TASK_LAYOUT)
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        *_TASK_NAMES,
        "evidence_refs",
        "self_reported_confidence",
    }
)


def joint_perception_schema() -> dict[str, Any]:
    """Return a fresh copy of the packaged strict JSON schema."""

    resource = files("surgical_agent.perception.prompts").joinpath(
        "perception_schema.json"
    )
    schema = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):  # Defensive check for the package resource.
        raise TypeError("joint perception schema resource must be an object")
    return schema


def _invalid() -> None:
    from surgical_agent.api.errors import ApiSchemaError

    raise ApiSchemaError("Joint perception response violates the strict schema")


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _require_id(value: object, *, task: str) -> int:
    lower, upper = TASK_ID_BOUNDS[task]
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not lower <= value <= upper
    ):
        _invalid()
    return value


def _require_score(value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        _invalid()
    return float(value)


def _validate_topk(value: object, *, task: str, expected_count: int) -> set[int]:
    if not _is_sequence(value) or len(value) != expected_count:
        _invalid()
    candidate_ids: list[int] = []
    scores: list[float] = []
    for candidate in value:
        if not isinstance(candidate, Mapping) or set(candidate) != {"id", "score"}:
            _invalid()
        candidate_ids.append(_require_id(candidate["id"], task=task))
        scores.append(_require_score(candidate["score"]))
    if len(set(candidate_ids)) != len(candidate_ids):
        _invalid()
    if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
        _invalid()
    return set(candidate_ids)


def _validate_selected_ids(value: object, *, task: str, topk_ids: set[int]) -> None:
    if not _is_sequence(value):
        _invalid()
    selected_ids = [_require_id(item, task=task) for item in value]
    if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= topk_ids:
        _invalid()


def _validate_task(value: object, *, task: str, expected_count: int) -> None:
    if not isinstance(value, Mapping):
        _invalid()
    selected_key = "selected_id" if task == "phase" else "selected_ids"
    if set(value) != {selected_key, "topk"}:
        _invalid()
    topk_ids = _validate_topk(value["topk"], task=task, expected_count=expected_count)
    if task == "phase":
        if _require_id(value[selected_key], task=task) not in topk_ids:
            _invalid()
    else:
        _validate_selected_ids(value[selected_key], task=task, topk_ids=topk_ids)


def _validate_evidence_refs(value: object) -> None:
    if not _is_sequence(value):
        _invalid()
    for reference in value:
        if not isinstance(reference, Mapping) or set(reference) != {"frame_id", "code"}:
            _invalid()
        frame_id = reference["frame_id"]
        if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
            _invalid()
        if reference["code"] not in EVIDENCE_REF_CODES:
            _invalid()


def _validate_confidences(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_TASK_NAMES):
        _invalid()
    for task in _TASK_NAMES:
        _require_score(value[task])


def validate_joint_perception_payload(payload: Mapping[str, Any]) -> None:
    """Validate schema and semantic constraints without exposing provider content."""

    try:
        if not isinstance(payload, Mapping) or set(payload) != _ROOT_KEYS:
            _invalid()
        if payload["schema_version"] != JOINT_PERCEPTION_SCHEMA_VERSION:
            _invalid()
        for task, expected_count in TASK_LAYOUT:
            _validate_task(payload[task], task=task, expected_count=expected_count)
        _validate_evidence_refs(payload["evidence_refs"])
        _validate_confidences(payload["self_reported_confidence"])
    except (KeyError, OverflowError, TypeError, ValueError):
        _invalid()
