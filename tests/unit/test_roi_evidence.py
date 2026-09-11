import base64
import io
import json
from copy import deepcopy

from PIL import Image

from surgical_agent.research.verification.roi_evidence import (
    augment_proposal,
    augment_review,
    create_views,
    regions,
)


def test_regions_clip_and_ignore_weak_or_degenerate_boxes():
    tracks = [{"score": .9, "track_id": "a", "bbox_tlwh": [0, 0, .1, .1]},
              {"score": .4, "track_id": "b", "bbox_tlwh": [.3, .3, .1, .1]},
              {"score": .8, "track_id": "c", "bbox_tlwh": [.3, .3, 0, .1]}]
    result = regions(tracks)
    assert len(result) == 1
    assert result[0]["rect_xyxy_normalized"] == [0, 0, .16, .16]


def test_auxiliary_views_preserve_original_images_schema_and_time_semantics(tmp_path):
    images = []
    for frame, color in zip([1, 26, 51], ["red", "green", "blue"], strict=True):
        path = tmp_path / f"{frame}.png"
        Image.new("RGB", (200, 100), color).save(path)
        images.append({"frame_id": frame, "path": str(path)})
    selected = {"frame_id": 51, "causal_frame_ids": [1, 26, 51], "images": images}
    views = create_views(selected, [{"score": .9, "track_id": "a", "bbox_tlwh": [.2, .2, .3, .3]}], tmp_path / "views")
    packet = {"academic_context": "medical research", "response_schema": {"type": "object"}}
    body = {"messages": [{"content": [{"type": "text", "text": json.dumps(packet)}] +
              [{"type": "image_url", "image_url": {"url": str(i)}} for i in range(3)]}],
            "max_tokens": 4096, "response_format": {"type": "json_schema"}}
    before = deepcopy(body)
    out = augment_proposal(body, views["roi_temporal"])
    assert body == before
    assert out["messages"][0]["content"][1:4] == body["messages"][0]["content"][1:]
    assert out["response_format"] == body["response_format"]
    new_packet = json.loads(out["messages"][0]["content"][0]["text"])
    assert next(iter(new_packet)) == "academic_context"
    assert new_packet["response_schema"] == packet["response_schema"]
    assert new_packet["auxiliary_visual_evidence"]["views"][0]["source_frame_ids"] == [1, 26, 51]
    assert "track_id" not in json.dumps(new_packet)
    assert Image.open(views["roi_temporal"][0]["path"]).size == (1152, 312)


def test_empty_regions_preserve_request_exactly():
    body = {"messages": [{"content": [{"type": "text", "text": "{}"}]}]}
    assert augment_proposal(body, []) == body


def test_review_gallery_keeps_original_target_pixels_and_three_image_references(tmp_path):
    original = Image.new("RGB", (320, 200), (20, 40, 60))
    buffer = io.BytesIO()
    original.save(buffer, format="PNG")
    block = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(), "detail": "high"}}
    crop_path = tmp_path / "crop.png"
    original.crop((20, 20, 80, 80)).save(crop_path)
    body = {"messages": [{"content": [{"type": "text", "text": '{"academic_context":"medical research"}'}] + [deepcopy(block) for _ in range(3)]}]}
    before = deepcopy(body)
    out = augment_review(body, [{"path": str(crop_path), "region": {"rect_xyxy_normalized": [.1, .1, .3, .3]}}])
    assert body == before
    content = out["messages"][0]["content"]
    assert len(content) == 4
    assert content[1:3] == body["messages"][0]["content"][1:3]
    image = Image.open(io.BytesIO(base64.b64decode(content[3]["image_url"]["url"].split(",", 1)[1])))
    assert image.crop((0, 0, 320, 200)).tobytes() == original.tobytes()
    assert json.loads(content[0]["text"])["target_view_layout"]["image_index"] == 2
