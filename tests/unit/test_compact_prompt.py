"""The wording candidate must leave the real provider contracts intact."""
import json
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.run_phase_extension_trial import phase_wire
from scripts.run_recent_mean_panel_trial import review_wire
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.compact_prompt import compact_review_wire


@pytest.mark.parametrize("seat", SEATS)
@pytest.mark.parametrize("branch", ("graph", "phase"))
@pytest.mark.parametrize("count", (1, 2, 3))
def test_real_wire_changes_only_instruction_text(seat, branch, count, tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"offline-image-fixture")
    ids = [951, 976, 1001][-count:]
    selected = {"frame_id": 1001, "causal_frame_ids": ids, "images": [{"path": str(path)} for _ in ids]}
    if branch == "phase":
        body = phase_wire(seat, selected)
    else:
        base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=path.read_bytes()) for _ in ids],
                               payload={"image_details": ["low"] * (count - 1) + ["high"]})
        pool = make_pool({"instrument": [0], "verb": [0], "target": [0], "ivt": [0], "phase": [1]})
        body = review_wire(seat, base, selected, pool)
    snapshot = deepcopy(body)
    result = compact_review_wire(body, branch, count_tokens=len)
    assert body == snapshot
    before = json.loads(body["messages"][0]["content"][0]["text"])
    after = json.loads(result["messages"][0]["content"][0]["text"])
    allowed = {"instructions", "proposition_semantics", "output_contract_clarification"}
    assert {k: v for k, v in before.items() if k not in allowed} == {
        k: v for k, v in after.items() if k not in allowed}
    assert str(count - 1) in after["instructions"]
    assert "{current_image_index}" not in after["instructions"]
    if seat == "qwen" and result["response_format"]["type"] == "json_object":
        assert re.search(r"\bjson\b", json.dumps(after).lower())
    result["messages"][0]["content"][0]["text"] = body["messages"][0]["content"][0]["text"]
    assert result == body
    with pytest.raises(ValueError, match="token budget"):
        compact_review_wire(body, branch, count_tokens=lambda s: -len(s))
