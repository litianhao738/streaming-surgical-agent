import json
import re
from dataclasses import replace

import pytest

from scripts.h0_prompt_refinement import (
    ARMS,
    ORIGINAL_ASSOCIATION,
    REFINED_ASSOCIATION,
    variant_request,
)
from scripts.run_five_expert_ablation import BOUNDARY
from surgical_agent.api.contracts import thaw_json
from tests.unit.test_h0_cost_optimization import wire
from tests.unit.test_h0_frame_strategy_study import requests


def test_schema_only_changes_no_wire_input_except_system_suffix():
    base = requests()["B"]
    candidate = variant_request(base, "schema_only")
    before, after = wire(base), wire(candidate)
    first_system = before["messages"][0].pop("content")
    second_system = after["messages"][0].pop("content")
    assert before == after
    assert first_system.split("Return exactly this JSON schema:\n")[0] == (
        second_system.removesuffix("Follow the supplied response_format JSON schema.")
    )
    assert "All six root keys are required." in second_system
    assert ORIGINAL_ASSOCIATION in second_system
    assert variant_request(base, "baseline") is base


@pytest.mark.parametrize("arm", ARMS)
def test_candidates_preserve_all_100_classes_null_boundary_and_five_heads(arm):
    base = requests()["B"]
    candidate = variant_request(base, arm)
    before, after = wire(base), wire(candidate)
    assert before["messages"][1]["content"][1:] == after["messages"][1]["content"][1:]
    for key in set(before) - {"messages"}:
        assert before[key] == after[key]
    original, system = base.payload["system_text"], candidate.payload["system_text"]
    tuples = lambda text: re.findall(r"\b(\d+)=\((\d+),(\d+),(\d+)\)", text)
    assert len(tuples(system)) == 100 and tuples(system) == tuples(original)
    ontology = original.split("Operational CholecTrack20 ontology.", 1)[1].split("\n\nThe output JSON", 1)[0]
    assert ontology in system and BOUNDARY in system
    assert "Never carry a label into the target" in system
    assert "Predict ONLY the final target image" in system
    assert "All six root keys are required." in system
    assert set(after["response_format"]["json_schema"]["schema"]["required"]) == {
        "schema_version", "instrument", "verb", "target", "ivt", "phase"
    }


def test_tuned_keeps_timing_and_nonempty_metadata_without_new_label_rules():
    base = requests()["B"]
    candidate = variant_request(base, "tuned")
    original = json.loads(base.payload["input_text"])
    cleaned = json.loads(candidate.payload["input_text"])
    for key in ("causal_frame_ids", "prior_finalized_prediction", "track_summary", "workflow_summary"):
        del original[key]
    assert original == cleaned
    assert REFINED_ASSOCIATION in candidate.payload["system_text"]
    assert "grasp" not in REFINED_ASSOCIATION and "retract" not in REFINED_ASSOCIATION


def test_refinement_rejects_future_state_and_unexpected_source():
    base = requests()["B"]
    with pytest.raises(ValueError, match="identities"):
        variant_request(replace(base, images=tuple(reversed(base.images))), "schema_only")
    payload = thaw_json(base.payload)
    payload["system_text"] += "unrecognized suffix"
    with pytest.raises(ValueError, match="suffix"):
        variant_request(replace(base, payload=payload), "schema_only")
    payload = thaw_json(base.payload)
    data = json.loads(payload["input_text"])
    data["prior_finalized_prediction"] = {"phase": 1}
    payload["input_text"] = json.dumps(data)
    with pytest.raises(ValueError, match="forbids"):
        variant_request(replace(base, payload=payload), "tuned")
