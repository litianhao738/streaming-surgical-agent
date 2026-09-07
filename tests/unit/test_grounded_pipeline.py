"""Offline regressions for the exact API flow, optional failures and request binding."""

import hashlib
import io
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from PIL import Image

from surgical_agent.api.contracts import ApiImageInput, ApiRequest, thaw_json
from surgical_agent.api.errors import ApiContractError, ApiTransportError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.schema import schema_for
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION
from surgical_agent.perception.main_h0 import load_main_h0_prompt, main_h0_input
from surgical_agent.perception.ontology_prompt import load_prompt_ontology_text
from surgical_agent.research.verification.grounded_pipeline import run_grounded_target
from surgical_agent.research.verification.grounded_repair import (
    LOCATOR_VERSION,
    PROPOSAL_VERSION,
    REVIEW_VERSION,
)


def _base(count=3):
    frames = tuple(range(101 - 25 * (count - 1), 102, 25))
    images = []
    for frame in frames:
        content = io.BytesIO()
        Image.new("RGB", (100, 80), (frame, 20, 30)).save(content, format="PNG")
        images.append(ApiImageInput(f"VID103:{frame}", "image/png", content.getvalue()))
    return ApiRequest(
        provider="openrouter", model_identifier="google/gemini-test",
        endpoint_identifier="https://openrouter.ai/api/v1/chat/completions",
        prompt_version="joint_perception_main_h0_v1",
        response_schema_version=FINAL_ONLY_SCHEMA_VERSION,
        payload={
            "system_text": load_main_h0_prompt(),
            "input_text": json.dumps(main_h0_input(video_id="VID103", target_frame_id=101, frame_ids=frames)),
            "image_details": ["low"] * (count - 1) + ["high"],
            "openrouter_image_detail_mode": "explicit_v1",
        },
        images=tuple(images),
        generation_parameters={"temperature": 0, "max_output_tokens": 4096, "reasoning": {"effort": "low"}},
    )


def _wire(labels=None):
    labels = labels or {"instrument": [2], "verb": [2], "target": [0], "ivt": [60], "phase": [3]}
    return {
        "schema_version": FINAL_ONLY_SCHEMA_VERSION,
        **{task: {"selected_ids": labels[task]} for task in ("instrument", "verb", "target", "ivt")},
        "phase": {"selected_id": labels["phase"][0]},
    }


def _responses(groups=((2, [59]),), slot="FIRST", labels=None):
    return {
        "h0": _wire(labels),
        "locator": {
            "schema_version": LOCATOR_VERSION, "all_visible_tools_covered": True,
            "instances": [{"instance_id": idx, "tip_box": [.2, .2, .4, .4]} for idx in range(1, len(groups) + 1)],
        },
        "proposal": {
            "schema_version": PROPOSAL_VERSION,
            "instances": [{"instance_id": idx, "instrument_id": instrument, "ivt_ids": ivts,
                           "support": "SUPPORTED", "contact_observation": "Visible contact region."}
                          for idx, (instrument, ivts) in enumerate(groups, 1)],
        },
        "review": {
            "schema_version": REVIEW_VERSION, "preferred": slot, "all_visible_tools_covered": True,
            "instances": [{"instance_id": idx, "crop_relevant": True, "instrument_identity_supported": True,
                           "target_identity_or_oov_supported": True, "action_or_oov_supported": True,
                           "distinguishing_observation": "Visible target contact distinguishes this interaction."}
                          for idx in range(1, len(groups) + 1)],
        },
    }


def _run(responses, base=None, slot="FIRST"):
    calls = []

    def call(stage, request):
        calls.append((stage, request))
        return deepcopy(responses.get(stage))

    return run_grounded_target(base or _base(), call, proposal_slot=slot), calls


@pytest.mark.parametrize("slot", ["FIRST", "SECOND"])
def test_complete_flow_preserves_exact_h0_and_binds_review_to_both_hypotheses(slot):
    base = _base()
    saved_hash = canonical_request_metadata(base).request_hash
    row, calls = _run(_responses(slot=slot), base, slot)
    assert [stage for stage, _ in calls] == ["h0", "locator", "proposal", "review"]
    assert calls[0][1] is base
    assert canonical_request_metadata(base).request_hash == saved_hash
    assert row["status"] == "OK" and row["admission"]["decision"] == "ACCEPT"
    assert set(row["final"]) == {"instrument", "verb", "target", "ivt", "phase"}
    assert row["h0"]["ivt"] == [60] and row["final"]["ivt"] == [59]
    assert row["final"]["phase"] == row["h0"]["phase"] == [3]
    review_request = calls[-1][1]
    review_data = json.loads(review_request.payload["input_text"])
    other_slot = "SECOND" if slot == "FIRST" else "FIRST"
    assert review_data["hypotheses"] == {slot: row["h1"], other_slot: row["h0"]}
    binding = row["review_binding"]
    assert binding["hypotheses"] == review_data["hypotheses"]
    assert binding["request_hash"] == row["review_request_hash"] == canonical_request_metadata(review_request).request_hash
    encoded = json.dumps(review_data["hypotheses"], sort_keys=True, separators=(",", ":")).encode()
    assert binding["hypotheses_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert binding["h0_request_hash"] == saved_hash and binding["proposal_slot"] == slot
    assert binding["target_frame_id"] == 101 and binding["video_id"] == "VID103"


@pytest.mark.parametrize("count", [1, 2, 3])
def test_true_short_windows_detail_target_crops_and_blind_localization(count):
    base = _base(count)
    payload = thaw_json(base.payload)
    data = json.loads(payload["input_text"])
    data.update(h0={"ivt": [60]}, gt={"ivt": [59]}, hypotheses={"FIRST": "must not leak"})
    data["track_summary"] = {"forbidden": "tracking answer"}
    data["workflow_summary"] = {"forbidden": "phase answer"}
    payload["input_text"] = json.dumps(data)
    base = replace(base, payload=payload)
    row, calls = _run(_responses(), base)
    assert calls[0][1] is base  # The adapter never edits a caller's H0 request.
    for stage, request in calls[1:]:
        stage_data = json.loads(request.payload["input_text"])
        assert stage_data["causal_frame_ids"] == data["causal_frame_ids"]
        assert stage_data["relative_seconds"] == list(range(1 - count, 1))
        assert not ({"gt", "h0", "track_summary", "workflow_summary", "prior_finalized_prediction"} & stage_data.keys())
        assert request.generation_parameters == base.generation_parameters
        crops_count = 0 if stage == "locator" else 1
        assert request.images[:count] == base.images
        assert thaw_json(request.payload["image_details"]) == ["low"] * (count - 1) + ["high"] * (1 + crops_count)
        schema_text = request.payload["system_text"].split("Required output schema:\n", 1)[1]
        assert json.loads(schema_text) == schema_for(request.response_schema_version)
        if stage != "review":
            assert "hypotheses" not in stage_data
        if stage == "locator":
            assert "LAST full original image" in request.payload["system_text"]
            assert "THIRD" not in request.payload["system_text"]
        else:
            assert load_prompt_ontology_text() in request.payload["system_text"]
            assert stage_data["image_order"].startswith(f"first {count} full causal images")
            with Image.open(io.BytesIO(request.images[-1].content)) as crop:
                assert crop.getpixel((0, 0)) == (101, 20, 30)
    assert row["crop_manifest"][0]["source_sha256"] == base.images[-1].sha256


def test_label_permutation_is_not_a_repair_and_never_spends_review():
    h0 = {"instrument": [2, 0], "verb": [2, 1], "target": [0], "ivt": [60, 17], "phase": [3]}
    row, calls = _run(_responses(groups=((0, [17]), (2, [60])), labels=h0))
    assert [stage for stage, _ in calls] == ["h0", "locator", "proposal"]
    assert row["final"] == row["h0"] == h0
    assert row["admission"] == {"decision": "KEEP", "reason": "NO_LABEL_CHANGE"}
    assert row["review_binding"] is None and row["review_request_hash"] is None


def test_overflowed_candidate_cannot_trigger_review_or_escape_final_only_schema():
    responses = _responses(groups=((0, [0, 1, 2]), (0, [3, 4, 5]), (0, [6, 7, 8])))
    row, calls = _run(responses)
    assert [stage for stage, _ in calls] == ["h0", "locator", "proposal"]
    assert row["h1"] is None and row["final"] == row["h0"]
    assert row["admission"]["reason"] == "CANDIDATE_OUTSIDE_FINAL_ONLY_CONTRACT"


@pytest.mark.parametrize("stage", ["h0", "locator", "proposal", "review"])
@pytest.mark.parametrize("bad_response", [None, {}])
def test_failure_at_any_api_stage_keeps_h0_or_reports_h0_failure(stage, bad_response):
    responses = _responses()
    responses[stage] = bad_response
    row, calls = _run(responses)
    assert calls[-1][0] == stage
    assert stage in row["stage_errors"]
    if stage == "h0":
        assert row["status"] == "H0_FAILURE" and row["final"] is None
    else:
        assert row["status"] == "OK" and row["final"] == row["h0"]
        assert row["admission"]["decision"] == "KEEP"


def test_transport_failure_is_safe_without_retry():
    calls = []

    def call(stage, request):
        calls.append(stage)
        if stage == "h0":
            return _wire()
        raise ApiTransportError("unavailable", code="timeout", retryable=True)

    row = run_grounded_target(_base(), call, proposal_slot="FIRST")
    assert calls == ["h0", "locator"]
    assert row["stage_errors"] == {"locator": "timeout"}
    assert row["final"] == row["h0"]


@pytest.mark.parametrize("case", ["incomplete", "empty", "degenerate"])
def test_bad_localization_does_not_spend_proposal_or_review(case):
    responses = _responses()
    locator = responses["locator"]
    if case == "incomplete":
        locator["all_visible_tools_covered"] = False
    elif case == "empty":
        locator["instances"] = []
    else:
        locator["instances"][0]["tip_box"] = [.4, .2, .2, .4]
    row, calls = _run(responses)
    assert [stage for stage, _ in calls] == ["h0", "locator"]
    assert row["final"] == row["h0"]


def test_review_preference_for_other_slot_never_accepts_proposal():
    row, _ = _run(_responses(slot="SECOND"), slot="FIRST")
    assert row["admission"]["decision"] == "KEEP" and row["final"] == row["h0"]


@pytest.mark.parametrize("case", ["future", "repeat", "missing_image", "old_schema", "invalid_slot"])
def test_bad_context_fails_before_any_paid_call(case):
    base = _base()
    payload = thaw_json(base.payload)
    data = json.loads(payload["input_text"])
    if case in {"future", "repeat"}:
        data["causal_frame_ids"][-1] = 126 if case == "future" else 76
        data["selected_image_frame_ids"] = data["causal_frame_ids"]
        payload["input_text"] = json.dumps(data)
        base = replace(base, payload=payload)
    elif case == "missing_image":
        base = replace(base, images=base.images[:-1])
    elif case == "old_schema":
        base = replace(base, response_schema_version="old_topk_schema")
    calls = []
    with pytest.raises((ApiContractError, ValueError)):
        run_grounded_target(base, lambda stage, request: calls.append(stage),
                            proposal_slot="INVALID" if case == "invalid_slot" else "FIRST")
    assert calls == []
