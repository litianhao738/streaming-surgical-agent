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
COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION = "joint_perception_compact_v1"
RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION = (
    "joint_perception_reliability_compact_v2"
)
GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION = (
    "joint_perception_gate_owned_compact_v1"
)
TASK_LAYOUT = (
    ("instrument", 7),
    ("verb", 10),
    ("target", 15),
    ("ivt", 20),
    ("phase", 7),
)
COMPACT_TASK_LAYOUT = (
    ("instrument", 3),
    ("verb", 4),
    ("target", 5),
    ("ivt", 8),
    ("phase", 3),
)
JOINT_PERCEPTION_SCHEMA_VERSIONS = frozenset(
    {
        JOINT_PERCEPTION_SCHEMA_VERSION,
        COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    }
)
_TASK_NAMES = tuple(task for task, _count in TASK_LAYOUT)
_V1_ROOT_KEYS = frozenset(
    {
        "schema_version",
        *_TASK_NAMES,
        "evidence_refs",
        "self_reported_confidence",
    }
)
_V2_ROOT_KEYS = frozenset({"schema_version", *_TASK_NAMES, "uncertainty"})
_GATE_OWNED_ROOT_KEYS = frozenset({"schema_version", *_TASK_NAMES})
_UNCERTAINTY_PATHS = {
    "/instrument/selected_ids": "instrument",
    "/verb/selected_ids": "verb",
    "/target/selected_ids": "target",
    "/ivt/selected_ids": "ivt",
    "/phase/selected_id": "phase",
}
UNCERTAINTY_REASONS = frozenset(
    {
        "LOW_VISUAL_CONFIDENCE",
        "CLOSE_ALTERNATIVES",
        "OCCLUSION",
        "MOTION_BLUR",
        "TEMPORAL_AMBIGUITY",
        "OTHER_VISUAL_AMBIGUITY",
    }
)


def task_layout_for_schema_version(
    schema_version: object,
) -> tuple[tuple[str, int], ...]:
    """Return the exact ranked-list sizes for one supported wire contract."""

    if schema_version == JOINT_PERCEPTION_SCHEMA_VERSION:
        return TASK_LAYOUT
    if schema_version == COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
        return COMPACT_TASK_LAYOUT
    if schema_version == RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
        return COMPACT_TASK_LAYOUT
    if schema_version == GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
        return COMPACT_TASK_LAYOUT
    _invalid()


def joint_perception_schema(
    schema_version: str = JOINT_PERCEPTION_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Return a fresh copy of the packaged strict JSON schema."""

    resources = {
        JOINT_PERCEPTION_SCHEMA_VERSION: "perception_schema.json",
        COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION: "perception_schema_compact.json",
        RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION: (
            "perception_schema_reliability_compact.json"
        ),
        GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION: (
            "perception_schema_gate_owned_compact.json"
        ),
    }
    try:
        resource_name = resources[schema_version]
    except KeyError as exc:
        raise ValueError("unsupported joint perception schema version") from exc
    resource = files("surgical_agent.perception.prompts").joinpath(
        resource_name
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


def _validate_topk(
    value: object,
    *,
    task: str,
    expected_count: int,
    confidence_key: str,
) -> set[int]:
    if not _is_sequence(value) or len(value) != expected_count:
        _invalid()
    candidate_ids: list[int] = []
    scores: list[float] = []
    for candidate in value:
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "id",
            confidence_key,
        }:
            _invalid()
        candidate_ids.append(_require_id(candidate["id"], task=task))
        scores.append(_require_score(candidate[confidence_key]))
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


def _validate_task(
    value: object,
    *,
    task: str,
    expected_count: int,
    confidence_key: str,
) -> set[int]:
    if not isinstance(value, Mapping):
        _invalid()
    selected_key = "selected_id" if task == "phase" else "selected_ids"
    if set(value) != {selected_key, "topk"}:
        _invalid()
    topk_ids = _validate_topk(
        value["topk"],
        task=task,
        expected_count=expected_count,
        confidence_key=confidence_key,
    )
    if task == "phase":
        if _require_id(value[selected_key], task=task) not in topk_ids:
            _invalid()
    else:
        _validate_selected_ids(value[selected_key], task=task, topk_ids=topk_ids)
    return topk_ids


def _validate_evidence_refs(value: object, *, max_count: int | None = None) -> None:
    if not _is_sequence(value) or (
        max_count is not None and len(value) > max_count
    ):
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


def _validate_uncertainty(
    value: object,
    *,
    task_topk_ids: Mapping[str, set[int]],
) -> None:
    if not _is_sequence(value) or len(value) > 5:
        _invalid()
    paths: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "reason",
            "alternative_ids",
        }:
            _invalid()
        path = item["path"]
        if not isinstance(path, str) or path not in _UNCERTAINTY_PATHS or path in paths:
            _invalid()
        paths.add(path)
        if item["reason"] not in UNCERTAINTY_REASONS:
            _invalid()
        alternatives = item["alternative_ids"]
        if not _is_sequence(alternatives):
            _invalid()
        task = _UNCERTAINTY_PATHS[path]
        alternative_ids = [_require_id(candidate, task=task) for candidate in alternatives]
        if (
            len(set(alternative_ids)) != len(alternative_ids)
            or not set(alternative_ids) <= task_topk_ids[task]
        ):
            _invalid()


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    schema_version: str,
    task_layout: tuple[tuple[str, int], ...],
    max_evidence_refs: int | None,
) -> None:
    try:
        if not isinstance(payload, Mapping) or set(payload) != _V1_ROOT_KEYS:
            _invalid()
        if payload["schema_version"] != schema_version:
            _invalid()
        for task, expected_count in task_layout:
            _validate_task(
                payload[task],
                task=task,
                expected_count=expected_count,
                confidence_key="score",
            )
        _validate_evidence_refs(
            payload["evidence_refs"], max_count=max_evidence_refs
        )
        _validate_confidences(payload["self_reported_confidence"])
    except (KeyError, OverflowError, TypeError, ValueError):
        _invalid()


def validate_joint_perception_payload(payload: Mapping[str, Any]) -> None:
    """Validate the original full-ranking wire contract."""

    _validate_payload(
        payload,
        schema_version=JOINT_PERCEPTION_SCHEMA_VERSION,
        task_layout=TASK_LAYOUT,
        max_evidence_refs=None,
    )


def validate_compact_joint_perception_payload(payload: Mapping[str, Any]) -> None:
    """Validate the bounded wire contract used for low-latency API rollouts."""

    _validate_payload(
        payload,
        schema_version=COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
        task_layout=COMPACT_TASK_LAYOUT,
        max_evidence_refs=6,
    )


def validate_reliability_compact_joint_perception_payload(
    payload: Mapping[str, Any],
) -> None:
    """Validate the v2 compact response with field-scoped uncertainty only."""

    try:
        if not isinstance(payload, Mapping) or set(payload) != _V2_ROOT_KEYS:
            _invalid()
        if payload["schema_version"] != RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
            _invalid()
        task_topk_ids = {
            task: _validate_task(
                payload[task],
                task=task,
                expected_count=expected_count,
                confidence_key="confidence",
            )
            for task, expected_count in COMPACT_TASK_LAYOUT
        }
        _validate_uncertainty(payload["uncertainty"], task_topk_ids=task_topk_ids)
    except (KeyError, OverflowError, TypeError, ValueError):
        _invalid()


def validate_gate_owned_compact_joint_perception_payload(
    payload: Mapping[str, Any],
) -> None:
    """Validate compact predictions whose reliability is owned by the Gate."""

    try:
        if not isinstance(payload, Mapping) or set(payload) != _GATE_OWNED_ROOT_KEYS:
            _invalid()
        if payload["schema_version"] != GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
            _invalid()
        for task, expected_count in COMPACT_TASK_LAYOUT:
            _validate_task(
                payload[task],
                task=task,
                expected_count=expected_count,
                confidence_key="score",
            )
    except (KeyError, OverflowError, TypeError, ValueError):
        _invalid()


def validate_joint_perception_payload_by_version(
    payload: Mapping[str, Any],
) -> None:
    """Dispatch strict semantic validation using the payload's declared version."""

    try:
        schema_version = payload["schema_version"]
    except (KeyError, TypeError):
        _invalid()
    if schema_version == JOINT_PERCEPTION_SCHEMA_VERSION:
        validate_joint_perception_payload(payload)
        return
    if schema_version == COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
        validate_compact_joint_perception_payload(payload)
        return
    if schema_version == RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
        validate_reliability_compact_joint_perception_payload(payload)
        return
    if schema_version == GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION:
        validate_gate_owned_compact_joint_perception_payload(payload)
        return
    _invalid()
