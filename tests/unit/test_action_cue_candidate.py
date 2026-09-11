"""Guard the paired experiment's only intervention and input budget."""
import ast
import json
from copy import deepcopy
from pathlib import Path

import pytest

from surgical_agent.research.verification.action_cue_candidate import (
    ORIGINAL_INSTRUCTIONS,
    TEMPLATE,
    replace_action_cues,
    token_audit,
)
from surgical_agent.research.verification.candidate_coordinator import proposal_schema


def body():
    packet = {"academic_context": "Research only", "instructions": ORIGINAL_INSTRUCTIONS,
              "current_prediction": {"verb": [0]}, "candidate_pool": {"propositions": []},
              "issues": [], "full_ontology": "Frozen ontology", "label_boundaries": "Frozen boundaries",
              "candidate_relation_hints": {"relations": [], "nested": {"z": 1, "a": 2}},
              "response_schema": proposal_schema()}
    return {"model": "google/gemini-3.8-flash", "max_tokens": 4096, "temperature": 0,
            "reasoning": {"effort": "low"}, "provider": {"only": ["google-ai-studio"]},
            "response_format": {"type": "json_schema", "json_schema": {"schema": proposal_schema()}},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": json.dumps(packet, ensure_ascii=False)},
                *[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i}",
                    "detail": "high" if i == 2 else "low"}} for i in range(3)]]}]}


def set_packet(wire, name, value):
    packet = json.loads(wire["messages"][0]["content"][0]["text"])
    packet[name] = value
    wire["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)


def test_frozen_original_matches_historical_proposer_source_without_import_side_effects():
    path = Path(__file__).resolve().parents[2] / "scripts/run_candidate_panel_trial.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "proposal_body")
    packet = next(node.value for node in function.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "packet" for target in node.targets))
    original = next(value for key, value in zip(packet.keys, packet.values, strict=True)
                    if isinstance(key, ast.Constant) and key.value == "instructions")
    assert ast.literal_eval(original) == ORIGINAL_INSTRUCTIONS


def test_only_instruction_changes_and_input_is_immutable():
    original = body()
    snapshot = deepcopy(original)
    changed = replace_action_cues(original)
    before = json.loads(original["messages"][0]["content"][0]["text"])
    after = json.loads(changed["messages"][0]["content"][0]["text"])
    assert after["instructions"] == TEMPLATE.read_text(encoding="utf-8").strip()
    after["instructions"] = before["instructions"]
    assert after == before and list(after) == list(before)
    changed["messages"][0]["content"][0]["text"] = original["messages"][0]["content"][0]["text"]
    assert changed == original == snapshot


@pytest.mark.parametrize("kind", ["unknown", "duplicate", "schema", "role", "image_first"])
def test_unknown_or_changed_protocol_fails_closed(kind):
    original = body()
    if kind == "unknown":
        set_packet(original, "instructions", ORIGINAL_INSTRUCTIONS + " Unknown rule.")
    elif kind == "duplicate":
        original = replace_action_cues(original)
    elif kind == "schema":
        set_packet(original, "response_schema", {"type": "object"})
    elif kind == "role":
        original["messages"][0]["role"] = "assistant"
    else:
        original["messages"][0]["content"].reverse()
    with pytest.raises(ValueError):
        replace_action_cues(original)


def test_all_text_proxies_strictly_drop_but_are_not_native_gemini_counts():
    original = body()
    changed = replace_action_cues(original)
    audit = token_audit(original, changed)
    assert audit["all_nonincreasing"]
    assert not audit["provider_token_guarantee"] and not audit["native_gemini_token_count"]
    assert set(audit["counters"]) == {"utf8_bytes", "cl100k_base", "o200k_base"}
    assert all(item["delta"] < 0 for item in audit["counters"].values())


@pytest.mark.parametrize("kind", ["image", "model", "schema", "graph", "ordering", "nested_ordering"])
def test_audit_rejects_any_intervention_besides_instructions(kind):
    original = body()
    changed = replace_action_cues(original)
    if kind == "image":
        changed["messages"][0]["content"][1]["image_url"]["detail"] = "high"
    elif kind == "model":
        changed["reasoning"]["effort"] = "minimal"
    elif kind == "schema":
        changed["response_format"]["json_schema"]["schema"] = {}
    elif kind == "graph":
        set_packet(changed, "candidate_relation_hints", {"relations": [7]})
    elif kind == "nested_ordering":
        set_packet(changed, "candidate_relation_hints", {"relations": [], "nested": {"a": 2, "z": 1}})
    else:
        text = changed["messages"][0]["content"][0]
        text["text"] = json.dumps(json.loads(text["text"]), sort_keys=True)
    with pytest.raises(ValueError):
        token_audit(original, changed)


def test_audit_rejects_a_longer_instruction_and_a_noop():
    original = body()
    changed = deepcopy(original)
    set_packet(changed, "instructions", ORIGINAL_INSTRUCTIONS * 4)
    with pytest.raises(ValueError, match="exceeds"):
        token_audit(original, changed)
    with pytest.raises(ValueError, match="must change"):
        token_audit(original, original)
