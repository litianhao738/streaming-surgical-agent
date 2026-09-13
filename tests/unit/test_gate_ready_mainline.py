"""Zero-provider acceptance of the H0/repair seam using the actual wire builders."""
from collections import Counter
from copy import deepcopy
from threading import RLock

import pytest

from scripts import run_gate_ready_mainline as ready
from scripts import run_prior_gated_joint_mainline as frozen
from surgical_agent.api.contracts import ApiImageInput, ApiRequest, canonical_json_bytes
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION
from surgical_agent.perception.main_h0 import load_main_h0_prompt
from surgical_agent.research.verification.prior_panel import fit_prior, video_counts


@pytest.fixture
def inputs():
    selected = {"key": "query_100", "video_id": "query", "frame_id": 100,
                "causal_frame_ids": [50, 75, 100]}
    base = ApiRequest(
        provider="openrouter", model_identifier="google/gemini-3.8-flash",
        endpoint_identifier="https://openrouter.ai/api/v1/chat/completions",
        prompt_version="joint_perception_main_h0_v1",
        response_schema_version=FINAL_ONLY_SCHEMA_VERSION,
        payload={"system_text": load_main_h0_prompt(), "input_text": "Synthetic causal H0 input.",
                 "image_details": ["low", "low", "high"],
                 "openrouter_routing_profile": "strict_google_ai_studio"},
        images=tuple(ApiImageInput(f"query_{frame}", "image/png", f"offline-{frame}".encode())
                     for frame in selected["causal_frame_ids"]),
        generation_parameters={"temperature": 0, "max_output_tokens": 4096, "reasoning": {"effort": "low"}},
    )
    row = {"frame_id": 100, "mask": dict.fromkeys(ready.old.TASKS, True),
           "gt": {"instrument": [0, 2], "verb": [1, 2], "target": [0, 1], "ivt": [17, 59], "phase": [3]}}
    # Synthetic other-video prior; no query GT or dataset files are read.
    counts = video_counts([row])
    prior = fit_prior({"other_a": counts, "other_b": counts, "other_c": counts}, "query")
    return base, selected, prior, deepcopy(ready.old.GATE)


class RecordingMock(ready.old.MockCalls):
    def __init__(self, replacements=None):
        super().__init__()
        self.wires, self.lock = {}, RLock()
        self.replacements = replacements or {}

    def call(self, target, stage, seat, body):
        with self.lock:
            identity = (target, stage, seat)
            assert identity not in self.wires, "No duplicate request or implicit retry"
            self.wires[identity] = canonical_json_bytes(body)
            response = super().call(target, stage, seat, body)
        replacement = self.replacements.get((stage, seat), response)
        if isinstance(replacement, Exception):
            raise replacement
        return deepcopy(replacement)


def without_timing(record):
    return {key: value for key, value in record.items() if key != "timing_seconds"}


@pytest.mark.parametrize("replacements", [
    {},
    {("proposal", "base"): None},
    {("proposal", "base"): {"invalid": "candidate schema"}},
    {("phase_recommendation", "base"): None},
    {("joint_r1", "gemini"): None},
    {("control_graph", "qwen"): None},
])
def test_gate_open_has_identical_requests_and_record_to_frozen(inputs, replacements):
    old_calls, new_calls = RecordingMock(replacements), RecordingMock(replacements)
    with ready.old.joint.roster.lightweight_protocol():
        old_record = frozen.run_target(old_calls, *inputs)
        new_record = ready.run_target(new_calls, *inputs, verify=True)
    assert len(new_calls.rows) == 13
    assert Counter(row["stage"] for row in new_calls.rows) == ready.STAGES
    assert old_calls.wires == new_calls.wires
    assert canonical_json_bytes(without_timing(old_record)) == canonical_json_bytes(without_timing(new_record))
    assert set(old_record["timing_seconds"]) == set(new_record["timing_seconds"])


def test_gate_closed_preserves_all_five_heads_with_one_request(inputs):
    calls = RecordingMock()
    result = ready.run_target(calls, *inputs, verify=False)
    assert Counter(row["stage"] for row in calls.rows) == {"h0": 1}
    assert set(result["h0"]) == set(ready.old.TASKS)
    assert all(prediction == result["h0"] for prediction in result["predictions"].values())
    assert result["verification"]["repair_calls"] == 0
    result["predictions"][ready.PRIMARY]["phase"].append(6)
    assert result["h0"]["phase"] == [3]
    assert result["predictions"]["h0"]["phase"] == [3]


def test_split_collection_uses_same_h0_once_and_only_twelve_dependent_calls(inputs):
    calls = RecordingMock()
    base, selected, prior, gate = inputs
    with ready.old.joint.roster.lightweight_protocol():
        raw = ready.generate_h0(calls, base, selected)
        sealed_raw = canonical_json_bytes(raw)
        assert len(calls.rows) == 1  # Features can be frozen at exactly this point.
        record = ready.repair_from_h0(calls, base, selected, prior, gate, raw, h0_seconds=1.25)
    assert len(calls.rows) == 13
    assert Counter(row["stage"] for row in calls.rows[1:]) == ready.REPAIR_STAGES
    assert canonical_json_bytes(raw) == sealed_raw
    assert record["h0_raw"] == raw
    assert record["timing_seconds"]["h0"] == 1.25


@pytest.mark.parametrize("failure", [None, {"wrong": "schema"}, ValueError("transport stopped")])
def test_h0_failure_is_not_retried_and_never_starts_repair(inputs, failure):
    calls = RecordingMock({("h0", "base"): failure})
    with pytest.raises((ApiSchemaError, ValueError, TypeError, KeyError)):
        ready.run_target(calls, *inputs)
    assert Counter(row["stage"] for row in calls.rows) == {"h0": 1}


def test_invalid_supplied_h0_is_rejected_without_any_call(inputs):
    calls = RecordingMock()
    with pytest.raises((ApiSchemaError, ValueError, TypeError, KeyError)):
        ready.repair_from_h0(calls, *inputs, None)
    assert not calls.rows


def test_query_contaminated_prior_is_rejected_before_h0(inputs):
    base, selected, prior, gate = inputs
    prior["fit_videos"].append(selected["video_id"])
    calls = RecordingMock()
    with pytest.raises(ValueError, match="query video"):
        ready.run_target(calls, base, selected, prior, gate)
    assert not calls.rows


def test_gate_decision_must_be_explicit_boolean(inputs):
    calls = RecordingMock()
    with pytest.raises(TypeError, match="boolean"):
        ready.run_target(calls, *inputs, verify="false")
    assert not calls.rows
