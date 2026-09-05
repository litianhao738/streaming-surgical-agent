"""Isolated final-label wire contract for the pure API ablation (no scores)."""

from collections.abc import Mapping

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.schema import (
    GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    joint_perception_schema,
)

FINAL_ONLY_SCHEMA_VERSION = "joint_perception_final_only_v1"
TASKS = ("instrument", "verb", "target", "ivt", "phase")


def final_only_schema():
    schema = joint_perception_schema(GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION)
    schema["properties"]["schema_version"]["const"] = FINAL_ONLY_SCHEMA_VERSION
    for task in TASKS:
        field = schema["properties"][task]
        del field["properties"]["topk"]
        field["required"].remove("topk")
    return schema


def validate_final_only(payload):
    def invalid():
        raise ApiSchemaError("Final-only response violates the strict schema")

    schema = final_only_schema()
    if not isinstance(payload, Mapping) or set(payload) != set(schema["required"]):
        invalid()
    if payload["schema_version"] != FINAL_ONLY_SCHEMA_VERSION:
        invalid()
    for task in TASKS:
        key = "selected_id" if task == "phase" else "selected_ids"
        field = payload[task]
        if not isinstance(field, Mapping) or set(field) != {key}:
            invalid()
        spec = schema["properties"][task]["properties"][key]
        values = [field[key]] if task == "phase" else field[key]
        if not isinstance(values, (list, tuple)):
            invalid()
        bound = spec if task == "phase" else spec["items"]
        if len(values) > spec.get("maxItems", 1):
            invalid()
        if any(type(v) is not int or not bound["minimum"] <= v <= bound["maximum"] for v in values):
            invalid()
        if len(set(values)) != len(values):
            invalid()
