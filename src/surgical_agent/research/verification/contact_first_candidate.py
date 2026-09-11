"""Append a frozen observation procedure without changing proposal contracts."""
import json
from copy import deepcopy
from pathlib import Path

VERSION = "contact_first_candidate_v1"
FIELD = "private_contact_audit"
TEMPLATE = Path(__file__).with_name("prompts") / (VERSION + ".json")


def append_contact_audit(body):
    result = deepcopy(body)
    text = result["messages"][0]["content"][0]
    if text.get("type") != "text":
        raise ValueError("first proposal content must be JSON text")
    packet = json.loads(text["text"])
    if next(iter(packet)) != "academic_context" or FIELD in packet:
        raise ValueError("academic context first; refuse duplicate audit procedure")
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    if template["version"] != VERSION:
        raise ValueError("unexpected observation template")
    packet[FIELD] = template
    text["text"] = json.dumps(packet, ensure_ascii=False)
    return result
