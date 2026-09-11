import json
from copy import deepcopy

from scripts import run_phase_mechanism_comparison as trial


def test_revised_template_changes_only_instruction_and_untrusted_hypothesis():
    packet = {"academic_context": "Academic medical research.", "task": "single phase", "instructions": "old",
        "current_image_index": 2, "phase_definitions": trial.phase.PHASE_GUIDE,
        "response_schema": trial.phase.phase_schema(3)}
    body = {"model": "fixture", "max_tokens": 4096, "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": json.dumps(packet)},
            *[{"type": "image_url", "image_url": {"url": f"fixture:{i}"}} for i in range(3)]]}]}
    saved = deepcopy(body)
    result = trial.revised_wire(body, {"phase": [3]})
    revised = json.loads(result["messages"][0]["content"][0]["text"])
    assert body == saved
    assert revised.pop("reference_phase_hypothesis")["phase_id"] == 3
    assert "positive current-image evidence" in revised["instructions"]
    revised["instructions"] = packet["instructions"]
    assert revised == packet
    result["messages"][0]["content"][0]["text"] = saved["messages"][0]["content"][0]["text"]
    assert result == saved


def test_invalid_or_tied_phase_review_keeps_reference():
    current = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}
    raw = {s: {"phase_id": p, "image_indices": [2], "observation": "Visible evidence."}
        for s, p in zip(trial.old.SEATS, [1, 1, 3, 3, None], strict=True)}
    result, _ = trial.phase.apply_phase_choices(current, raw, 3)
    assert result == current
    raw[trial.old.SEATS[-1]] = None
    result, decision = trial.phase.apply_phase_choices(current, raw, 3)
    assert result == current and decision["reason"] == "INVALID_PANEL"


def test_stopped_run_has_no_dispatch_and_preserves_all_arm_fallbacks():
    class Calls:
        stopped = True
        def call(self, *args):
            raise AssertionError("no dispatch after stop")
    current = {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}
    initial = {"h0": current, "cached_default": current, "graph_r1": {"prediction": current}}
    result = trial.run_case(Calls(), None, {"key": "fixture"}, initial, 0)
    assert set(result["skipped_arms"]) == set(trial.ARMS)
    assert all(value == current for value in result["predictions"].values())
