import base64
from copy import deepcopy

import pytest

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_compact_verifier_trial import ARMS, BRANCHES, hydrate, run_arm
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool


def test_hydration_rejects_different_image():
    body = {"messages": [{"content": [{"type": "text", "text": "{}"},
            {"type": "image_url", "image_url": {"detail": "high",
             "url": "data:image/png;base64," + base64.b64encode(b"original").decode()}}]}]}
    archive = redact_images(body)
    assert hydrate(archive, [b"original"]) == body
    with pytest.raises(ValueError, match="mismatch"):
        hydrate(archive, [b"different"])


@pytest.mark.parametrize("arm", ARMS)
def test_missing_reviews_keep_h0_and_remain_invalid(arm):
    class MissingCalls:
        def call(self, *_):
            return None

    h0 = {"instrument": [0], "verb": [0], "target": [0], "ivt": [0], "phase": [1]}
    initial = {"h0": h0, "pool": make_pool(h0)}
    before = deepcopy(initial)
    wire = {b: {s: {"messages": [{"content": [None] * 4}]} for s in SEATS} for b in BRANCHES}
    result = run_arm(MissingCalls(), "fixture", initial, wire, arm)
    assert initial == before
    assert result["final"] == h0
    assert all(v is None for v in result["means"].values())
    assert all(result["phase_errors"].values())
    assert result["phase_decision"]["reason"] == "INVALID_PANEL"
