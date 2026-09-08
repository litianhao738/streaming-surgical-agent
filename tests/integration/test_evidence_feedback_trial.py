"""No-network controls for the shared-first-round evidence feedback experiment."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import run_evidence_feedback_trial as trial
from scripts.check_candidate_panel_providers import ACADEMIC
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.verification.candidate_coordinator import make_pool


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("tests must never call API endpoints or load credentials")
    monkeypatch.setattr(trial.requests, "post", forbidden)
    monkeypatch.setattr(trial.requests, "get", forbidden)
    monkeypatch.setattr("scripts.run_candidate_panel_trial.key_for", forbidden)


def body(packet):
    return {"model": "mock-model", "messages": [{"role": "user", "content": [
        {"type": "text", "text": json.dumps(packet)}]}]}


def fake_run(monkeypatch, tmp_path, *, different_pool=False, first_pass=False, h0_failed=False, failed_review=False):
    output, previous = tmp_path / "run", tmp_path / "previous"
    selected = [{"key": f"{v}_51", "video_id": v, "frame_id": 51, "causal_frame_ids": [1, 26, 51],
                 "images": [], "request_metadata": {}} for v in ("VID103", "VID23")]
    prior = {"stopped": True, "occupied": {k: "0" for k in trial.ALLOWANCE}}
    trial.save(previous / "budget.json", prior)
    selection_source = tmp_path / "selection.json"
    trial.save(selection_source, {"selection": selected})
    trial.save(output / "plan.json", {"profile": trial.PROFILE, "selection": selected, "source_sha256": {},
               "selection_source": str(selection_source), "selection_sha256": trial.sha(selection_source),
               "earlier_selection_sha256": {}, "previous_budget": str(previous),
               "previous_budget_sha256": trial.sha(previous / "budget.json"), "carried_occupied": prior["occupied"],
               "limits": {k: "10" for k in trial.ALLOWANCE}, "rates": trial.RATES_V2, "max_calls": len(selected) * 19})
    dispatched, scored, call_instances, paid_count_at_score = [], [], [], []

    class FakeCalls:
        def __init__(self, *args, **kwargs):
            self.rows, self.stopped = [], False
            call_instances.append(self)

        def call(self, target, stage, seat, wire):
            packet = json.loads(wire["messages"][0]["content"][0]["text"])
            dispatched.append((target, stage, seat, deepcopy(packet)))
            row = {"index": len(self.rows), "target": target, "stage": stage, "seat": seat,
                   "charge": "0.01", "charge_kind": "native", "account": "openrouter_usd", "status": "JSON_PARSED"}
            self.rows.append(row)
            if stage == "h0":
                if h0_failed:
                    return None
                return {"schema_version": "joint_perception_final_only_v1", "instrument": {"selected_ids": [0]},
                        "verb": {"selected_ids": []}, "target": {"selected_ids": []}, "ivt": {"selected_ids": []},
                        "phase": {"selected_id": 2}}
            if "_proposal_" in stage:
                proposal = {t: [] for t in trial.TASKS[:4]}
                if different_pool and "review_evidence_feedback" in packet:
                    proposal["ivt"] = [7]
                return proposal
            if failed_review and stage.endswith("_review_2") and seat == "qwen":
                row["status"] = "FAILED"
                return None
            rating = 5 if first_pass or stage.endswith("_review_2") else 3
            items = {p["id"]: {"rating": rating, "finding": "MATCH" if rating == 5 else "UNCLEAR",
                              "scope": "LOCAL_REGION", "image_indices": [2], "observation": "Synthetic evidence."}
                     for p in packet["propositions"]}
            return ({"rows": [{"candidate_id": pid, **item} for pid, item in items.items()]}
                    if seat == "gemini" else {"judgments": items})

        def persist(self):
            pass

    monkeypatch.setattr(trial, "RevisionCalls", FakeCalls)
    monkeypatch.setattr(trial, "build_gemini_base", lambda *args: SimpleNamespace(images=[1, 2, 3]))
    monkeypatch.setattr(trial, "canonical_request_metadata", lambda *args: SimpleNamespace(to_mapping=dict))
    monkeypatch.setattr(trial, "gemini_h0_wire", lambda *args: body({"h0_request": True}))
    monkeypatch.setattr(trial, "gemini_proposal", lambda base, selected, current, pool, issues:
                        body({"academic_context": ACADEMIC, "issues": deepcopy(issues), "pool": deepcopy(pool)}))
    monkeypatch.setattr(trial, "review_wire", lambda seat, base, selected, pool:
                        body({"academic_context": ACADEMIC, "propositions": deepcopy(pool["propositions"])}))
    monkeypatch.setattr(trial, "build_feedback", lambda pool, reviews, issues, image_count:
                        {"schema_version": "candidate_review_feedback_v1", "instruction": "Fallible model observations; verify against images.",
                         "candidates": [{"candidate_id": issues[0]["candidate_id"]}]})

    def subprocess(command, **kwargs):
        number = int(command[-1])
        snapshot = trial.read(output / f"round_{number}_predictions.json")
        assert len(snapshot) == len(selected)
        assert call_instances[0].stopped is True
        paid_count_at_score.append(len(dispatched))
        assert all(count == len(dispatched) for count in paid_count_at_score)
        scored.append(number)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(trial.subprocess, "run", subprocess)
    return output, selected, dispatched, scored


def test_treatment_only_adds_proposal_feedback_field_without_mutating_issues_or_images():
    base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=b"fixture")] * 3,
                           payload={"image_details": ["low", "low", "high"]})
    selected = {"frame_id": 51, "causal_frame_ids": [1, 26, 51]}
    current = {"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [2]}
    pool = make_pool(current)
    issues = [{"candidate_id": "instrument_0", "currently_selected": True, "mean": 3, "scores": [3] * 5, "invalid": {}}]
    feedback = {"instruction": "Not ground truth; verify against images.", "candidates": []}
    control = trial.proposal_wire(base, selected, current, pool, issues)
    treatment = trial.proposal_wire(base, selected, current, pool, issues, feedback)
    before = deepcopy(treatment)
    packet = json.loads(treatment["messages"][0]["content"][0]["text"])
    assert next(iter(packet)) == "academic_context"
    assert packet.pop("review_evidence_feedback") == feedback
    treatment["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    assert treatment == control
    assert before["messages"][0]["content"][1:] == control["messages"][0]["content"][1:]
    assert packet["issues"] == issues


@pytest.mark.parametrize("failed_review", [False, True])
def test_identical_round2_pools_share_all_five_same_round_replies_including_failures(monkeypatch, tmp_path, failed_review):
    output, selected, dispatched, scored = fake_run(monkeypatch, tmp_path, failed_review=failed_review)
    trial.execute(output, None)
    assert scored == [1, 2]
    assert len(dispatched) == 28  # 2*(H0 + R1 proposer/panel + R2 two proposers/one shared panel).
    states = trial.read(output / "inference_states.json")
    for index, row in enumerate(selected):
        state = states[row["key"]]
        first_name, second_name = trial.ARMS if index == 0 else tuple(reversed(trial.ARMS))
        first, second = [state["arms"][name]["history"][0] for name in (first_name, second_name)]
        assert first["before"] == second["before"] == state["shared"]["current"]
        assert first["raw_reviews"] == second["raw_reviews"]
        assert second["shared_review"]["full_request_bodies_exactly_equal"] is True
        assert second["shared_review"]["source_record_sha256"] == trial.sha(output / "targets" / row["key"] / f"{first_name}_round_2.json")
        assert second["new_review_calls"] == 0 and first["new_review_calls"] == 5
        if failed_review:
            assert second["raw_reviews"]["qwen"] is None
        panels = [stage for target, stage, seat, _ in dispatched if target == row["key"] and seat == "qwen" and "_review_" in stage]
        assert len(panels) == 2  # No reuse across R1/R2, no retry after a shared failure.
    completion = trial.read(output / "completion.json")
    assert completion["shared_second_round_panels"] == 2
    assert completion["cost_groups"]["shared_round2_review"]["calls"] == 10
    assert sum(group["calls"] for group in completion["cost_groups"].values()) == len(dispatched)
    with pytest.raises(ValueError, match="replay"):
        trial.execute(output, None)


def test_different_pools_run_separate_panels_and_do_not_cross_contaminate_arms(monkeypatch, tmp_path):
    output, selected, dispatched, _ = fake_run(monkeypatch, tmp_path, different_pool=True)
    trial.execute(output, None)
    assert len(dispatched) == 38
    states = trial.read(output / "inference_states.json")
    for row in selected:
        state = states[row["key"]]
        assert state["arms"]["issues_only"]["current"]["ivt"] == []
        assert state["arms"]["evidence_feedback"]["current"]["ivt"] == [7]
        for arm in state["arms"].values():
            assert "shared_review" not in arm["history"][0]
            assert arm["current"]["phase"] == [2]
    for _, stage, _, packet in dispatched:
        if "_review_" in stage:
            assert "review_evidence_feedback" not in packet
            assert not {"h0", "gt", "ground_truth", "issues", "other_reviews"} & set(packet)


def test_true_shared_model_pass_stops_both_arms_without_forcing_round2(monkeypatch, tmp_path):
    output, _, dispatched, _ = fake_run(monkeypatch, tmp_path, first_pass=True)
    trial.execute(output, None)
    assert len(dispatched) == 14
    states = trial.read(output / "inference_states.json")
    for state in states.values():
        for arm in state["arms"].values():
            assert arm["status"] == "SHARED_MODEL_PASS" and not arm["round2_attempted"]
    assert trial.read(output / "completion.json")["completed"] is True


def test_h0_failure_keeps_every_target_and_marks_partial_not_success(monkeypatch, tmp_path):
    output, selected, dispatched, scored = fake_run(monkeypatch, tmp_path, h0_failed=True)
    trial.execute(output, None)
    assert len(dispatched) == 1 and scored == [1, 2]
    final = trial.read(output / "round_2_predictions.json")
    assert len(final) == len(selected) and all(row["h0"] is None for row in final)
    assert final[0]["shared_first_round"]["status"] == "H0_FAILED"
    assert final[1]["shared_first_round"]["status"] == "NOT_ATTEMPTED"
    assert trial.read(output / "completion.json")["completed"] is False


def test_bad_feedback_keeps_shared_prediction_and_does_not_silently_run_a_control_clone(monkeypatch, tmp_path):
    output, _, dispatched, _ = fake_run(monkeypatch, tmp_path)
    def invalid_feedback(*args):
        raise ValueError("fixture inconsistent feedback provenance")
    monkeypatch.setattr(trial, "build_feedback", invalid_feedback)
    trial.execute(output, None)
    states = trial.read(output / "inference_states.json")
    assert not any(stage.startswith("evidence_feedback_") for _, stage, _, _ in dispatched)
    for state in states.values():
        treated = state["arms"]["evidence_feedback"]
        assert treated["current"] == state["shared"]["current"]
        assert treated["status"] == "FEEDBACK_BUILD_FAILED" and not treated["round2_attempted"]
        assert state["arms"]["issues_only"]["status"] == "MODEL_PASS"
    assert trial.read(output / "completion.json")["completed"] is False


def test_selection_uses_fixed_index_two_and_rejects_overlap_without_reading_gt(monkeypatch, tmp_path):
    selection_path = tmp_path / "selection.json"
    entries, samples, selected = {}, {}, []
    for video in trial.VIDEOS:
        entries[video] = SimpleNamespace(split=DatasetSplit.TRAINING)
        for index in range(10):
            frame = 51 + 100 * index
            selected.append({"video_id": video, "frame_id": frame, "causal_frame_ids": [frame - 50, frame - 25, frame],
                             "images": [], "gt_availability_only": dict.fromkeys(trial.TASKS, True)})
            samples.setdefault(video, []).append(SimpleNamespace(target_frame_id=frame,
                causal_frame_ids=[frame - 50, frame - 25, frame], media_refs=[]))
    adapter = SimpleNamespace(entries=entries, iter_inference_video=lambda v: iter(samples[v]))
    trial.save(selection_path, {"no_gt_label_values_used_for_selection": True, "selection": selected})
    previous = tmp_path / "previous_plan.json"
    trial.save(previous, {"selection": [r for r in selected if r["frame_id"] in (151, 351, 651, 851)]})
    monkeypatch.setattr(trial, "build_gemini_base", lambda *args: None)
    monkeypatch.setattr(trial, "canonical_request_metadata", lambda *args: SimpleNamespace(to_mapping=dict))
    result = trial.select_targets(adapter, selection_path, [previous])
    assert [(r["video_id"], r["frame_id"]) for r in result] == [(v, 251) for v in trial.VIDEOS]
    trial.save(previous, {"selection": [result[0]]})
    with pytest.raises(ValueError, match="new Training"):
        trial.select_targets(adapter, selection_path, [previous])


def test_score_keeps_failed_and_unattempted_rows_in_all_three_outputs(monkeypatch, tmp_path):
    states = {"VID103_51": {"video_id": "VID103", "frame_id": 51, "h0": None,
              "shared": {"current": None, "status": "H0_FAILED", "reviewed": False},
              "arms": {a: {"current": None, "status": "H0_FAILED", "round2_attempted": False, "round2_reviewed": False}
                       for a in trial.ARMS}}}
    trial.save(tmp_path / "round_2_predictions.json", trial.snapshot(states))
    trial.save(tmp_path / "budget.json", {"stopped": True})
    calls = []
    def score_saved(adapter, rows):
        calls.append(deepcopy(rows))
        return {"arms": {"final": {"tasks": {t: {"micro_f1": 0} for t in trial.TASKS}}}}, rows
    monkeypatch.setattr(trial, "score_saved", score_saved)
    trial.score(tmp_path, None, 2)
    assert len(calls) == 3 and all(len(rows) == 1 and rows[0]["final"] is None for rows in calls)
    report = trial.read(tmp_path / "scores/round_2.json")
    assert report["coverage"]["h0_available"] == 0 and report["coverage"]["selected"] == 1


def test_scoring_cannot_read_gt_while_paid_inference_is_active(monkeypatch, tmp_path):
    trial.save(tmp_path / "budget.json", {"stopped": False})
    def forbidden(*args):
        raise AssertionError("GT scorer was reached before inference closed")
    monkeypatch.setattr(trial, "score_saved", forbidden)
    with pytest.raises(ValueError, match="ledger to be closed"):
        trial.score(tmp_path, None, 1)
