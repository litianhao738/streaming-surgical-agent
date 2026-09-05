import json
from dataclasses import replace

import pytest

from scripts.prepare_h0_cost_optimization import build_variant
from scripts.run_five_expert_ablation import BOUNDARY
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.providers.openrouter import OpenRouterTransport
from tests.unit.test_h0_frame_strategy_study import requests


def wire(request):
    transport = OpenRouterTransport(api_key=SecretValue("fixture"), endpoint_identifier=request.endpoint_identifier)
    return json.loads(transport._request_body(request))


def test_minimal_changes_only_reasoning_on_actual_wire():
    base = requests()["B"]
    before, after = wire(base), wire(build_variant(base, "original_minimal"))
    assert after.pop("reasoning") == {"effort": "minimal"}
    assert before.pop("reasoning") == {"effort": "low"}
    assert after == before


def test_lean_preserves_pixels_schema_ontology_and_label_boundaries():
    base = requests()["B"]
    lean = build_variant(base, "lean_low")
    before, after = wire(base), wire(lean)
    assert before["response_format"] == after["response_format"]
    assert before["messages"][1]["content"][1:] == after["messages"][1]["content"][1:]
    assert before["reasoning"] == after["reasoning"] == {"effort": "low"}
    system = lean.payload["system_text"]
    ontology = base.payload["system_text"].split("Operational CholecTrack20 ontology.", 1)[1].split("\n\nThe output JSON", 1)[0]
    assert ontology in system and BOUNDARY in system
    assert "Never carry a label into the target" in system
    assert "Predict ONLY the final target image" in system
    assert json.loads(lean.payload["input_text"]) == {"frame_ids": [76, 101, 126], "relative_seconds": [-2, -1, 0], "target_frame_id": 126}
    for key in set(before) - {"messages"}:
        assert before[key] == after[key]


def test_cost_variant_rejects_stateful_or_noncausal_sources():
    base = requests()["B"]
    payload = thaw_json(base.payload)
    data = json.loads(payload["input_text"])
    data["prior_finalized_prediction"] = {"phase": 1}
    payload["input_text"] = json.dumps(data)
    with pytest.raises(ValueError, match="forbids"):
        build_variant(replace(base, payload=payload), "lean_low")
    with pytest.raises(ValueError, match="identities"):
        build_variant(replace(base, images=tuple(reversed(base.images))), "lean_low")
