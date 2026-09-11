"""Isolated action-cue replacement with an unchanged proposal wire contract.

The local text counters are conservative experiment gates, not the Gemini
tokenizer or a guarantee about native input tokens, output tokens or charges.
This module makes no model calls and is not enabled by the default pipeline.
"""
import json
from copy import deepcopy
from pathlib import Path

from surgical_agent.research.verification.candidate_coordinator import proposal_schema

VERSION = "action_cue_candidate_v1"
TEMPLATE = Path(__file__).with_name("prompts") / f"{VERSION}.txt"
ORIGINAL_INSTRUCTIONS = (
    "Look across all visible tools and contact regions; multiple IVTs may coexist. "
    "Use only these causal images. You cannot accept, delete or change the final answer. "
    "Do not repeat candidate IDs already in the pool. Do not add alternatives merely to fill a quota. "
    "Return empty lists if no new candidate has visual support. Issues are uncertain model judgments, not ground truth. "
    "Return exactly the four candidate lists in the JSON schema, no commentary."
)


def _packet(body):
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("expected the frozen single-user proposal request")
    content = messages[0].get("content")
    if not isinstance(content, list) or not content or content[0].get("type") != "text":
        raise ValueError("first proposal content must be JSON text")
    packet = json.loads(content[0]["text"])
    if not isinstance(packet, dict) or not isinstance(packet.get("instructions"), str):
        raise TypeError("expected a JSON packet with string instructions")
    return packet


def _assert_instruction_only(control, treatment):
    before, after = _packet(control), _packet(treatment)
    if before["instructions"] == after["instructions"]:
        raise ValueError("instructions must change exactly once")
    restored = deepcopy(after)
    restored["instructions"] = before["instructions"]
    if json.dumps(before, ensure_ascii=False) != json.dumps(restored, ensure_ascii=False):
        raise ValueError("only the instructions field may change")
    restored_wire = deepcopy(treatment)
    restored_wire["messages"][0]["content"][0]["text"] = control["messages"][0]["content"][0]["text"]
    if restored_wire != control:
        raise ValueError("images, transport and all other request fields must stay exact")


def replace_action_cues(body):
    """Deep-copy a known proposal; change only its instructions string.

    Reject an unknown/previously modified template rather than silently applying
    this experiment to another protocol. Call token_audit on each completed pair
    before freezing paid requests.
    """
    before = _packet(body)
    if json.dumps(before, ensure_ascii=False) != body["messages"][0]["content"][0]["text"]:
        raise ValueError("expected the frozen proposal JSON text serialization")
    if next(iter(before)) != "academic_context":
        raise ValueError("academic context must remain the first packet field")
    if before["instructions"] != ORIGINAL_INSTRUCTIONS:
        raise ValueError("unknown or already replaced proposal instructions")
    if before.get("response_schema") != proposal_schema():
        raise ValueError("expected the unchanged four bounded candidate-list schema")
    required = {"current_prediction", "candidate_pool", "issues", "full_ontology", "label_boundaries"}
    if not required <= before.keys():
        raise ValueError("expected the complete frozen proposal context")
    revised = TEMPLATE.read_text(encoding="utf-8").strip()
    if not revised or len(revised.encode("utf-8")) >= len(ORIGINAL_INSTRUCTIONS.encode("utf-8")):
        raise ValueError("action-cue instructions must be strictly shorter in UTF-8 bytes")
    out = deepcopy(body)
    before["instructions"] = revised
    out["messages"][0]["content"][0]["text"] = json.dumps(before, ensure_ascii=False)
    _assert_instruction_only(body, out)
    return out


def _text_blocks(body):
    return [block["text"] for message in body["messages"]
            for block in message["content"] if block.get("type") == "text"]


def token_audit(control, treatment):
    """Return three full-message-text proxies, rejecting any increase.

    Image payloads/tokens, framing and provider-side schema overhead are not
    counted. They remain byte-for-byte unchanged by the separate wire check.
    cl100k_base/o200k_base are reference encodings, not Gemini's native counter.
    """
    _assert_instruction_only(control, treatment)
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("tiktoken is required for the frozen prompt proxy audit") from exc
    before, after = _text_blocks(control), _text_blocks(treatment)
    counters = {"utf8_bytes": lambda value: len(value.encode("utf-8"))}
    for name in ("cl100k_base", "o200k_base"):
        encoding = tiktoken.get_encoding(name)
        counters[name] = lambda value, encoding=encoding: len(encoding.encode(value, disallowed_special=()))
    result = {}
    for name, counter in counters.items():
        old_count, new_count = sum(map(counter, before)), sum(map(counter, after))
        result[name] = {"control": old_count, "treatment": new_count,
                        "delta": new_count - old_count, "nonincreasing": new_count <= old_count}
    if not all(item["nonincreasing"] for item in result.values()):
        raise ValueError(f"action-cue prompt exceeds a local text proxy budget: {result}")
    return {"version": VERSION, "changed_field": "messages[0].content[0].text.instructions",
            "scope": "sum of all message text blocks; images and provider overhead excluded",
            "provider_token_guarantee": False, "native_gemini_token_count": False,
            "counters": result, "all_nonincreasing": True}
