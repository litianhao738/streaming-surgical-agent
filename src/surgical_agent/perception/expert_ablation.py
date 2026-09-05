"""Single-task contracts for independent experts, not production repair."""

from collections.abc import Mapping
from functools import partial

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.schema import (
    COMPACT_TASK_LAYOUT, GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    joint_perception_schema, _validate_task,
)

TASK_COUNTS = dict(COMPACT_TASK_LAYOUT)


def expert_version(task):
    if task not in TASK_COUNTS:
        raise ValueError("unknown expert task")
    return f"joint_independent_{task}_expert_v1"


def expert_schema(task):
    schema = joint_perception_schema(GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION)
    schema["properties"] = {key: value for key, value in schema["properties"].items()
                            if key in ("schema_version", task)}
    schema["properties"]["schema_version"]["const"] = expert_version(task)
    schema["required"] = ["schema_version", task]
    return schema


def validate_expert(payload, *, task):
    if (not isinstance(payload, Mapping) or set(payload) != {"schema_version", task}
            or payload["schema_version"] != expert_version(task)):
        raise ApiSchemaError("Expert response violates the strict schema")
    _validate_task(payload[task], task=task, expected_count=TASK_COUNTS[task], confidence_key="score")


EXPERT_CONTRACTS = {
    expert_version(task): (expert_schema(task), partial(validate_expert, task=task))
    for task in TASK_COUNTS
}
