import json
from types import SimpleNamespace

from scripts.run_candidate_panel_trial import envelope
from scripts.run_reference_reflection_diagnostic import reference_body
from scripts.run_reflection_diagnostic import reflection_body
from surgical_agent.research.verification.candidate_coordinator import make_pool
from surgical_agent.research.verification.semantic_coordinator import item_error


def test_reference_images_are_separate_from_causal_query_and_costed(tmp_path):
    h0 = {"instrument": [2], "verb": [2], "target": [1], "ivt": [59], "phase": [1]}
    pool = make_pool(h0)
    base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=b"query")]*3,
                           payload={"image_details": ["low", "low", "high"]})
    selected = {"frame_id": 51, "causal_frame_ids": [1, 26, 51]}
    path = tmp_path / "ref.png"
    path.write_bytes(b"reference")
    gallery = [{"image_path": str(path), "video_id": "OTHER_TRAINING", "frame_id": 101,
                "triplet_id": 59, "components": {"instrument": 2, "verb": 2, "target": 1},
                "tool_bbox": [0.1, 0.1, 0.2, 0.2]}]
    baseline = reflection_body(base, selected, h0, pool)
    body = reference_body(base, selected, h0, pool, gallery)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    assert [r["image_index"] for r in packet["query_images"]] == [1, 2, 3]
    assert [r["seconds_relative_to_target"] for r in packet["query_images"]] == [-2, -1, 0]
    assert packet["reference_images"][0]["source_video"] == "OTHER_TRAINING"
    assert "query_gt" not in packet and "gt" not in packet
    assert "Cite current image index 3" in packet["instructions"]
    assert body["messages"][0]["content"][2:] == baseline["messages"][0]["content"][1:]
    assert envelope("base", body) > envelope("base", baseline)
    only_reference = {"rating": 5, "finding": "MATCH", "scope": "LOCAL_REGION",
                      "image_indices": [0], "observation": "Reference evidence alone."}
    assert item_error(only_reference, "target", 4) == "NO_CURRENT_FRAME_EVIDENCE"
