"""Prompt-only isolation, unchanged candidate admission and paired reuse."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import run_contact_first_candidate_trial as trial
from scripts import run_prior_candidate_trial as old
from scripts.check_candidate_panel_providers import ACADEMIC
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.contact_first_candidate import (
    FIELD,
    TEMPLATE,
    append_contact_audit,
)


def body():
    return {"model": "fixed-model", "temperature": 0, "max_tokens": 4096,
            "reasoning": {"effort": "low"}, "provider": {"only": ["fixed"], "allow_fallbacks": False},
            "response_format": {"type": "json_schema", "json_schema": {"strict": True}},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": json.dumps({"academic_context": ACADEMIC,
                    "current_prediction": {"phase": [1]}, "candidate_pool": {"propositions": []},
                    "candidate_relation_hints": {"relations": []}, "response_schema": {"type": "object"}})},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{i}",
                    "detail": "high" if i == 2 else "low"}} for i in range(3)]]}]}


def h0():
    return {"instrument": [0], "verb": [0], "target": [0], "ivt": [7], "phase": [1]}


def test_only_one_input_field_changes_no_new_output_or_model_settings():
    original = body()
    snapshot = deepcopy(original)
    updated = append_contact_audit(original)
    packet = json.loads(updated["messages"][0]["content"][0]["text"])
    assert next(iter(packet)) == "academic_context"
    assert packet.pop(FIELD) == json.loads(TEMPLATE.read_text(encoding="utf-8"))
    updated["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    assert updated == original == snapshot


def test_duplicate_or_nonacademic_prefix_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        append_contact_audit(append_contact_audit(body()))
    invalid = body()
    invalid["messages"][0]["content"][0]["text"] = '{"task":"not prefixed"}'
    with pytest.raises(ValueError, match="academic"):
        append_contact_audit(invalid)


def test_control_wire_keeps_original_and_treatment_does_not_mutate_it(monkeypatch):
    original = body()
    monkeypatch.setattr(trial, "proposal_wire", lambda *args: original)
    assert trial.wire_for(None, None, h0(), None, None, "control") is original
    changed = trial.wire_for(None, None, h0(), None, None, "contact_first")
    assert FIELD in json.loads(changed["messages"][0]["content"][0]["text"])
    assert FIELD not in json.loads(original["messages"][0]["content"][0]["text"])
    with pytest.raises(ValueError):
        trial.wire_for(None, None, h0(), None, None, "second_round")


class FakeCalls:
    def __init__(self, proposal=None, fail_reviews=False):
        self.proposal, self.fail_reviews, self.rows = proposal, fail_reviews, []

    def call(self, target, stage, seat, wire):
        self.rows.append({"target": target, "stage": stage, "seat": seat})
        if seat == "base":
            return deepcopy(self.proposal)
        if self.fail_reviews:
            return None
        pool = wire["test_pool"]
        return {"judgments": {p["id"]: {"rating": 5, "finding": "MATCH", "scope": "LOCAL_REGION",
                "image_indices": [2], "observation": "The same tool acts on this tissue in the current frame."}
                for p in pool["propositions"]}}


@pytest.fixture
def fake_wires(monkeypatch):
    monkeypatch.setattr(trial, "wire_for", lambda *args: body())
    monkeypatch.setattr(old, "review_wire", lambda seat, base, selected, pool: {
        **body(), "seat": seat, "test_pool": deepcopy(pool)})
    return SimpleNamespace(images=[0, 1, 2]), {"key": "VID00_100", "frame_id": 100}


@pytest.mark.parametrize("proposal", [None, {"instrument": [], "verb": [999], "target": [], "ivt": []},
    {"instrument": [], "verb": [], "target": [], "ivt": [0, 1, 2, 3, 4]}])
def test_failed_or_excess_proposal_falls_back_without_panel(fake_wires, proposal):
    base, selected = fake_wires
    calls = FakeCalls(proposal)
    result = trial.run_arm(calls, base, selected, h0(), {}, "control", [])
    assert result["status"] == "PROPOSAL_FAILED"
    assert result["prediction"] == h0() and len(calls.rows) == 1


@pytest.mark.parametrize("fail_reviews", [False, True])
def test_identical_pool_shares_all_five_original_replies_including_failure(fake_wires, fail_reviews):
    base, selected = fake_wires
    calls = FakeCalls({"instrument": [], "verb": [], "target": [], "ivt": []}, fail_reviews)
    reuse = []
    first = trial.run_arm(calls, base, selected, h0(), {}, "control", reuse)
    second = trial.run_arm(calls, base, selected, h0(), {}, "contact_first", reuse)
    assert len(calls.rows) == 7  # Two proposals and one five-seat panel.
    assert second["shared_from"] == "control" and second["call_count"] == 0
    assert second["raw"] == first["raw"] and second["prediction"] == first["prediction"]
    assert second["prediction"]["phase"] == h0()["phase"]
    if fail_reviews:
        assert set(first["means"].values()) == {None}


def test_changed_pool_never_reuses_partial_votes(fake_wires):
    base, selected = fake_wires
    calls = FakeCalls({"instrument": [], "verb": [], "target": [], "ivt": []})
    reuse = []
    trial.run_arm(calls, base, selected, h0(), {}, "control", reuse)
    calls.proposal["ivt"] = [17]
    second = trial.run_arm(calls, base, selected, h0(), {}, "contact_first", reuse)
    assert len(calls.rows) == 12 and "shared_from" not in second
    assert second["prediction"] == panel.select(h0(), second["pool"], second["means"], threshold=4)
    assert second["prediction"]["phase"] == h0()["phase"]
    assert all(c["stage"] in {"control_proposal", "control_review", "contact_first_proposal", "contact_first_review"}
               for c in calls.rows)


def test_empty_h0_and_proposal_stays_scored_empty_without_review(fake_wires):
    base, selected = fake_wires
    initial = {t: [] for t in ("instrument", "verb", "target", "ivt")} | {"phase": [1]}
    calls = FakeCalls({t: [] for t in ("instrument", "verb", "target", "ivt")})
    result = trial.run_arm(calls, base, selected, initial, {}, "contact_first", [])
    assert result["status"] == "EMPTY_POOL_UNVERIFIED" and result["prediction"] == initial
    assert len(calls.rows) == 1


def test_partial_task_masks_exclude_unknown_truth_from_metrics_and_coverage():
    rows, truth, records = [], [], {}
    for i in range(2):
        key = f"VID00_{i}"
        rows.append({"key": key, "video_id": "VID00", "frame_id": i,
                     **{v: h0() for v in trial.VERSIONS}, "statuses": dict.fromkeys(trial.ARMS, "MODEL_PASS")})
        mask = {t: i == 0 or t in ("instrument", "phase") for t in trial.TASKS}
        truth.append({"video_id": "VID00", "frame_id": i, "mask": mask,
                      "gt": {t: h0()[t] if mask[t] else None for t in trial.TASKS}})
        records[key] = {a: {"pool": make_pool(h0())} for a in trial.ARMS}
    report, _, _, _ = trial.summarize({"limits": trial.LIMITS}, rows, records, truth, {"calls": []},
        {"elapsed_seconds": 0, "transport_statuses": {}, "arm_statuses": {}})
    assert report["metrics"]["contact_first"]["tasks"]["ivt"]["valid_targets"] == 1
    assert report["candidate_coverage"]["control"]["ivt"]["gt_positive"] == 1
    assert report["metrics"]["h0"]["tasks"]["instrument"]["valid_targets"] == 2
    assert report["predeclared_success"]["confirmed"] is False


def test_dispatch_rejects_h0_and_duplicate_without_network(tmp_path, monkeypatch):
    calls = trial.BoundCalls(tmp_path, {"selection": [{"key": "VID00_1"}], "limits": trial.LIMITS,
                                      "rates": trial.RATES_V2})
    monkeypatch.setattr(trial.TimedCalls, "call", lambda *args: None)
    with pytest.raises(ValueError, match="undeclared"):
        calls.call("VID00_1", "h0", "base", body())
    calls.call("VID00_1", "control_proposal", "base", body())
    with pytest.raises(ValueError, match="duplicate"):
        calls.call("VID00_1", "control_proposal", "base", body())
    assert SEATS == ("grok", "qwen", "gpt", "gemini", "deepseek")


def test_saved_hint_order_is_restored_without_changing_information(tmp_path, monkeypatch):
    original = {"packet": {"source": "frozen prior", "use": "hypotheses", "relations": []}, "audit": {}}
    initial = {"h0": h0(), "hints": json.loads(json.dumps(original, sort_keys=True))}
    assert list(initial["hints"]["packet"]) != list(original["packet"])
    trial.save(tmp_path / "prior_VID00.json", {"excluded_video": "VID00"})
    monkeypatch.setattr(trial, "retrieve_candidate_hints", lambda *a, **k: deepcopy(original))
    restored = trial.ordered_hints(tmp_path, {"video_id": "VID00"}, initial)
    assert json.dumps(restored) == json.dumps(original["packet"])
    initial["hints"]["packet"]["source"] = "modified"
    with pytest.raises(ValueError):
        trial.ordered_hints(tmp_path, {"video_id": "VID00"}, initial)
