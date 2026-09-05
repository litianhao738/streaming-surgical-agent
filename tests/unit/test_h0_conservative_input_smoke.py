import json

from scripts.run_h0_conservative_input_smoke import conservative_request
from surgical_agent.api.schema import schema_for
from tests.unit.test_h0_cost_optimization import wire
from tests.unit.test_h0_frame_strategy_study import requests


def test_conservative_keeps_both_schemas_images_and_low():
    base = requests()["B"]
    candidate = conservative_request(base)
    before, after = wire(base), wire(candidate)
    assert before["messages"][1]["content"][1:] == after["messages"][1]["content"][1:]
    for key in set(before) - {"messages"}:
        assert before[key] == after[key]
    schema = json.dumps(schema_for(base.response_schema_version), sort_keys=True)
    assert candidate.payload["system_text"].endswith(schema)
    original = json.loads(base.payload["input_text"])
    cleaned = json.loads(candidate.payload["input_text"])
    for key in ("causal_frame_ids", "prior_finalized_prediction", "track_summary", "workflow_summary"):
        del original[key]
    assert original == cleaned
    assert "Operational CholecTrack20 ontology." in candidate.payload["system_text"]
