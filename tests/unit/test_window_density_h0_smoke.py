"""Audit the experimental 25-image wire contract and unconditional scoring."""

import base64
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts.run_pure_h0_smoke import FinalOnlyRequestBuilder, evaluate_offline
from scripts.run_window_density_h0_smoke import (
    GROUPS, ORDERS, WindowDensityTransport, assert_request_contract,
    augment_metrics, build_request, frame_ids, read_annotation_index,
)
from surgical_agent.api.contracts import ApiImageInput
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError
from surgical_agent.api.providers.openrouter import OpenRouterTransport
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.schemas import FrameSupervisionTarget, FrameTaskMask
from tests.unit.test_joint_api_vlm import _context


def requests():
    config = load_api_config("configs/perception/joint_openrouter_qwen38max0902_pure_h0_fixed3.yaml")
    base = FinalOnlyRequestBuilder(config=config).build(_context())
    images = {fid: ApiImageInput(f"cholectrack20:VID01:frame:{fid}", "image/png", f"image-{fid}".encode())
              for fid in range(1, 26)}
    return {group: build_request(base, images=images, group=group, target=25, fps=25) for group in GROUPS}


def test_dense_request_reaches_wire_as_25_ordered_images_without_relaxing_production():
    request = requests()["C"]
    transport = WindowDensityTransport(api_key=SecretValue("fixture"), endpoint_identifier=request.endpoint_identifier)
    transport._validate_request(request)
    body = json.loads(transport._request_body(request))
    wire_images = body["messages"][1]["content"][1:]
    assert len(wire_images) == 25
    for fid, item in enumerate(wire_images, 1):
        assert base64.b64decode(item["image_url"]["url"].split(",", 1)[1]) == f"image-{fid}".encode()
        assert item["image_url"]["detail"] == ("high" if fid == 25 else "low")
    normal = OpenRouterTransport(api_key=SecretValue("fixture"), endpoint_identifier=request.endpoint_identifier)
    with pytest.raises(ApiContractError, match="one to six"):
        normal._validate_request(request)


def test_groups_share_prompt_target_and_routing_but_have_truthful_temporal_evidence():
    built = requests()
    assert len({request.payload["system_text"] for request in built.values()}) == 1
    assert [len(built[group].images) for group in GROUPS] == [3, 3, 25]
    spans = []
    for group, request in built.items():
        data = json.loads(request.payload["input_text"])
        assert "group" not in data
        assert data["causal_frame_ids"] == list(frame_ids(group, 25))
        assert data["temporal_evidence"]["window_frame_count"] == len(request.images)
        assert data["prior_finalized_prediction"] is None
        spans.append(data["raw_video_sampling"]["window_span_seconds"])
    assert spans == [0.08, 0.96, 0.96]
    # Each group occurs twice in every within-target call position for six targets.
    for position in range(3):
        scheduled = [ORDERS[index % 3][position] for index in range(6)]
        assert all(scheduled.count(group) == 2 for group in GROUPS)


def test_mismatched_dense_image_order_and_stale_counts_are_rejected():
    request = requests()["C"]
    reordered = replace(request, images=(request.images[1], request.images[0], *request.images[2:]))
    with pytest.raises(ValueError, match="must agree"):
        assert_request_contract(reordered, tuple(range(1, 26)), 25)
    payload = dict(request.payload)
    data = json.loads(payload["input_text"])
    data["temporal_evidence"]["uploaded_image_count"] = 3
    payload["input_text"] = json.dumps(data)
    with pytest.raises(ValueError, match="Temporal evidence"):
        assert_request_contract(replace(request, payload=payload), tuple(range(1, 26)), 25)


def test_annotation_precheck_uses_keys_without_requiring_interpretable_labels(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"annotations": {"25": "not a label schema", "50": None}}))
    assert read_annotation_index(path, [25, 50])["exact_target_keys_present"] == [25, 50]
    with pytest.raises(ValueError, match="exact annotation keys"):
        read_annotation_index(path, [25, 75])


def test_failure_keeps_gt_in_exact_and_recall_denominators_while_missing_gt_is_masked():
    targets = [
        FrameSupervisionTarget("VID01", 25, (0,), (1,), (0,), (17,), 1,
            FrameTaskMask(True, True, True, True, True), "frame_multilabel", "test"),
        FrameSupervisionTarget("VID01", 50, (0,), (1,), (0,), (17,), 1,
            FrameTaskMask(True, False, True, True, True), "frame_multilabel", "test"),
    ]
    adapter = SimpleNamespace(iter_video=lambda video, frame_ids: iter(
        SimpleNamespace(inference=SimpleNamespace(target_frame_id=target.frame_id),
                        frame_supervision=target, evaluation=None) for target in targets))
    predictions = [
        {"frame_id": 25, "status": "OK", "selected_ids": {
            "instrument": [0], "verb": [1], "target": [0], "ivt": [17], "phase": [1]}},
        {"frame_id": 50, "status": "API_FAILURE"},
    ]
    evaluation = augment_metrics(evaluate_offline(adapter, "VID01", predictions))
    instrument = evaluation["tasks"]["instrument"]
    assert instrument["exact_accuracy_conditional_on_response"] == 1
    assert instrument["exact_accuracy_all_valid_gt"] == 0.5
    assert instrument["response_coverage"] == 0.5
    assert (instrument["tp"], instrument["fp"], instrument["fn"]) == (1, 0, 1)
    assert instrument["micro_f1"] == pytest.approx(2 / 3)
    assert evaluation["tasks"]["verb"]["valid_gt"] == 1
    assert evaluation["tasks"]["verb"]["micro_recall"] == 1
