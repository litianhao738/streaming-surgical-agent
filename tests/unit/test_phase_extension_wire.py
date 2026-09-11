"""Phase transport must bind real images without exposing existing answers."""

import base64
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from scripts.run_phase_extension_trial import (
    MODELS,
    PHASE_GUIDE,
    build_inputs,
    phase_schema,
    phase_wire,
)
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.verification.candidate_coordinator import SEATS


@pytest.fixture
def selected(tmp_path):
    images = []
    for index, frame in enumerate((251, 751, 1001)):
        path = tmp_path / f"{frame}.png"
        Image.new("RGB", (3, 2), (index * 70, 30, 80)).save(path)
        images.append({"frame_id": frame, "path": str(path)})
    return {
        "key": "VID_FIXTURE_1001", "video_id": "VID_FIXTURE", "frame_id": 1001,
        "causal_frame_ids": [251, 751, 1001], "images": images,
    }


@pytest.mark.parametrize("seat", SEATS)
def test_five_families_receive_only_phase_contract_and_actual_bound_images(seat, selected):
    selected.update(
        h0={"phase": [4], "ivt": [11]},
        gt={"phase": [2]},
        previous_five={"target": [8]},
        reviewer_feedback="ANSWER_LEAK_SENTINEL",
        prior_packet="PRIOR_LEAK_SENTINEL",
    )
    snapshot = deepcopy(selected)
    wire = phase_wire(seat, selected)
    assert selected == snapshot
    assert wire["model"] == MODELS[seat]
    assert wire["stream"] is False
    assert wire["max_tokens"] == 4096
    assert len(wire["messages"]) == 1
    assert wire["messages"][0]["role"] == "user"
    blocks = wire["messages"][0]["content"]
    assert len(blocks) == 4
    packet = json.loads(blocks[0]["text"])
    assert next(iter(packet)) == "academic_context"
    assert "academic" in packet["academic_context"]
    assert set(packet) == {
        "academic_context", "task", "instructions", "target_frame_id", "images",
        "phase_definitions", "current_image_index", "response_schema",
    }
    assert packet["phase_definitions"] == PHASE_GUIDE
    assert [p["id"] for p in PHASE_GUIDE] == list(range(7))
    assert packet["images"] == [
        {"index": 0, "frame_id": 251, "seconds_relative_to_target": -30.0},
        {"index": 1, "frame_id": 751, "seconds_relative_to_target": -10.0},
        {"index": 2, "frame_id": 1001, "seconds_relative_to_target": 0.0},
    ]
    assert packet["current_image_index"] == 2
    assert "LEAK_SENTINEL" not in blocks[0]["text"]
    assert str(selected["images"][0]["path"]) not in blocks[0]["text"]
    for index, block in enumerate(blocks[1:]):
        assert block["type"] == "image_url"
        im = block["image_url"]
        assert im["detail"] == ("high" if index == 2 else "low")
        prefix, content = im["url"].split(",", 1)
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(content) == Path(selected["images"][index]["path"]).read_bytes()
    schema = packet["response_schema"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for phase in (*range(7), None):
        assert validator.is_valid({"phase_id": phase, "image_indices": [2], "observation": "Current scene."})
    assert not validator.is_valid({"phase_id": 7, "image_indices": [2], "observation": "Out of range."})
    assert not validator.is_valid({"phase_id": 2, "image_indices": [3], "observation": "Future image."})
    assert not validator.is_valid({"phase_id": 2, "image_indices": [2], "observation": "Scene.", "ivt": [11]})
    if wire["response_format"]["type"] == "json_schema":
        assert wire["response_format"]["json_schema"] == {
            "name": "single_phase_choice_v1", "strict": True, "schema": schema,
        }


@pytest.mark.parametrize("count", (1, 2, 3))
@pytest.mark.parametrize("seat", SEATS)
def test_short_real_history_uses_actual_last_index_without_padding(seat, count, selected):
    selected["images"] = selected["images"][-count:]
    selected["causal_frame_ids"] = selected["causal_frame_ids"][-count:]
    wire = phase_wire(seat, selected)
    blocks = wire["messages"][0]["content"]
    packet = json.loads(blocks[0]["text"])
    assert len(blocks) == count + 1
    assert packet["current_image_index"] == count - 1
    assert packet["response_schema"] == phase_schema(count)
    assert [r["index"] for r in packet["images"]] == list(range(count))
    assert packet["images"][-1]["frame_id"] == selected["frame_id"]
    assert packet["images"][-1]["seconds_relative_to_target"] == 0


def test_build_inputs_uses_only_runtime_training_images_and_stops_at_actual_gap(tmp_path):
    # There is an old -30s frame, but a real gap before the final 15s segment.
    frame_ids = [251, *range(626, 1002, 25)]
    samples = []
    for frame in frame_ids:
        path = tmp_path / f"{frame}.png"
        Image.new("RGB", (2, 2), (20, 30, 40)).save(path)
        samples.append(SimpleNamespace(target_frame_id=frame, media_refs=(str(path),)))

    class RuntimeOnlyAdapter:
        def __init__(self):
            self.entries = {"VID_FIXTURE": SimpleNamespace(split=DatasetSplit.TRAINING)}

        def iter_inference_video(self, video):
            assert video == "VID_FIXTURE"
            return iter(samples)

        def iter_video(self, *args, **kwargs):
            raise AssertionError("Inference must never access GT labels")

    selection = [{"key": "VID_FIXTURE_1001", "video_id": "VID_FIXTURE", "frame_id": 1001,
                  "causal_frame_ids": [951, 976, 1001]}]
    inputs = build_inputs(RuntimeOnlyAdapter(), selection)["VID_FIXTURE_1001"]
    assert inputs["phase_short"]["causal_frame_ids"] == [951, 976, 1001]
    assert inputs["phase_context"]["causal_frame_ids"] == [751, 1001]
    assert [im["frame_id"] for im in inputs["phase_context"]["images"]] == [751, 1001]
    assert all(len(im["sha256"]) == 64 for arm in inputs.values() for im in arm["images"])


def test_build_inputs_rejects_testing_even_if_images_are_present():
    adapter = SimpleNamespace(
        entries={"VID_TEST": SimpleNamespace(split=DatasetSplit.TESTING)},
        iter_inference_video=lambda video: iter(()),
    )
    selected = [{"key": "VID_TEST_1", "video_id": "VID_TEST", "frame_id": 1,
                 "causal_frame_ids": [1]}]
    with pytest.raises(ValueError, match="Training only"):
        build_inputs(adapter, selected)
