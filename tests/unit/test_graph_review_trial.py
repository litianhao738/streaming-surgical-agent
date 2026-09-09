"""The retrieval arm must not change the baseline wire or selection policy."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import run_graph_review_trial as trial
from surgical_agent.research.verification.candidate_coordinator import make_pool


def body():
    return {"model": "fixed", "max_tokens": 8192, "reasoning": {"effort": "none"},
            "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": [
                {"type": "text", "text": json.dumps({"academic_context": "Academic surgical video analysis",
                 "candidate_pool": {"unchanged": True}, "response_schema": {"type": "object"}})},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA", "detail": "high"}}]}]}


def test_none_preserves_complete_original_body_and_does_not_mutate():
    original = body()
    copied = trial.add_reference(original, None)
    assert copied == original and copied is not original
    treatment = trial.add_reference(original, {"items": [{"text": "A definition is not visual evidence."}]})
    assert treatment["messages"][0]["content"][1:] == original["messages"][0]["content"][1:]
    assert {k: v for k, v in treatment.items() if k != "messages"} == {k: v for k, v in original.items() if k != "messages"}
    packet = json.loads(treatment["messages"][0]["content"][0]["text"])
    packet.pop("reference_knowledge")
    assert packet == json.loads(original["messages"][0]["content"][0]["text"])
    assert original == body()


def test_reference_cap_includes_serialized_metadata():
    with pytest.raises(ValueError, match="character limit"):
        trial.add_reference(body(), {"text": "x" * 2500})


def test_exact_duplicate_arms_reuse_failures_without_more_calls(monkeypatch):
    h0 = {"instrument": [0], "verb": [0], "target": [0], "ivt": [0], "phase": [0]}
    pool = make_pool(h0)
    original_pool = deepcopy(pool)
    monkeypatch.setattr(trial, "review_wire", lambda *args: body())
    monkeypatch.setattr(trial, "retrieve", lambda *args, **kwargs: {"reference_knowledge": None, "stats": {}, "audit": {}})

    class FailedCalls:
        def __init__(self):
            self.rows = []

        def call(self, target, stage, seat, wire):
            self.rows.append({"target": target, "stage": stage, "seat": seat})

    calls, reuse = FailedCalls(), []
    base, selected = SimpleNamespace(images=[1, 2, 3]), {"key": "VID103_18601"}
    first = trial.run_panel(calls, base, selected, h0, pool, "none", reuse)
    second = trial.run_panel(calls, base, selected, h0, pool, "graph", reuse)
    assert len(calls.rows) == 5
    assert first["prediction"] == second["prediction"] == h0
    assert first["status"] == second["status"] == "UNRESOLVED"
    assert second["shared_from"] == "none" and second["panel_seconds"] is None
    assert second["call_count"] == 0 and second["raw"] == {s: None for s in trial.SEATS}
    assert pool == original_pool
