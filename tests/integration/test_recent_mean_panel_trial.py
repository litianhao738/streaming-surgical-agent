import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from scripts import run_recent_mean_panel_trial as trial
from scripts.check_candidate_panel_providers import ACADEMIC
from scripts.run_candidate_panel_trial import usage_charge
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool


def test_recent_review_wire_keeps_images_context_and_records_real_thinking_modes():
    base = SimpleNamespace(images=[SimpleNamespace(mime_type="image/png", content=b"test")]*3,
                           payload={"image_details": ["low", "low", "high"]})
    selected = {"frame_id": 51, "causal_frame_ids": [1, 26, 51]}
    pool = make_pool({"instrument": [0], "verb": [], "target": [], "ivt": [], "phase": [0]})
    bodies = {s: trial.review_wire(s, base, selected, pool) for s in SEATS}
    for seat, body in bodies.items():
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        assert body["model"] == trial.MODELS[seat]
        assert next(iter(packet)) == "academic_context" and packet["academic_context"] == ACADEMIC
        assert not {"h0", "gt", "ground_truth", "current_prediction"} & set(packet)
        assert len(body["messages"][0]["content"]) == 4
    assert bodies["gpt"]["reasoning"] == {"effort": "none"}
    assert "temperature" not in bodies["gpt"]
    assert bodies["gemini"]["reasoning"] == {"effort": "minimal"}
    assert bodies["grok"]["reasoning_effort"] == "low"
    assert bodies["qwen"]["enable_thinking"] is False
    qwen_packet = json.loads(bodies["qwen"]["messages"][0]["content"][0]["text"])
    assert "JSON" in qwen_packet["output_format"]
    assert bodies["deepseek"]["reasoning"] == {"enabled": False}


def test_custom_qwen_billing_rates_not_silently_old_rates():
    cost, kind = usage_charge("qwen", {"prompt_tokens": 100, "completion_tokens": 20}, Decimal(10),
                              {"qwen": ("0.02", "0.03")})
    assert cost == Decimal("2.60") and kind == "conservative_estimate"


def test_exact_wire_cache_has_provenance_and_never_repeats_paid_call(tmp_path):
    source, output = tmp_path / "source", tmp_path / "new"
    folder = source / "calls/000_VID103_51_h0_base"
    body = {"model": "model", "messages": [{"role": "user", "content": [{"type": "text", "text": "fixed"}]}]}
    trial.save(source / "budget.json", {"calls": [{"index": 0, "target": "VID103_51", "stage": "h0",
                                                "seat": "base", "status": "JSON_PARSED"}]})
    trial.save(folder / "request.json", body)
    trial.save(folder / "response.json", {"body": {"choices": [{"message": {"content": '{"ok":true}'}}]}})
    calls = trial.CachedCalls(output, reuse_source=source)
    assert calls.call("VID103_51", "h0", "base", body) == {"ok": True}
    assert calls.rows == [] and len(calls.cache_hits) == 1
    with pytest.raises(ValueError, match="differs"):
        calls.call("VID103_51", "h0", "base", {**body, "model": "different"})


@pytest.mark.parametrize("rating,rounds,empty", [(3, 3, False), (5, 1, False), (3, 0, True)])
def test_real_round_count_empty_refills_and_model_pass(monkeypatch, tmp_path, rating, rounds, empty):
    output, previous = tmp_path / "run", tmp_path / "previous"
    prior = {"stopped": True, "occupied": {k: "0" for k in trial.ALLOWANCE}}
    trial.save(previous / "budget.json", prior)
    selected = {"key": "VID103_51", "video_id": "VID103", "frame_id": 51, "request_metadata": {}}
    trial.save(output / "plan.json", {"profile": trial.PROFILE, "source_sha256": {}, "selection": [selected],
               "previous_budget": str(previous), "previous_budget_sha256": trial.sha(previous / "budget.json"),
               "carried_occupied": prior["occupied"], "limits": {k: "10" for k in trial.ALLOWANCE},
               "rates": trial.RATES_V2, "max_calls": 19})
    bodies, stages, score_files = [], [], []
    class FakeCalls:
        def __init__(self, *args, **kwargs):
            self.rows, self.stopped = [], False

        def call(self, target, stage, seat, body):
            stages.append(stage)
            bodies.append(body)
            self.rows.append({"charge": "0", "charge_kind": "native", "account": "openrouter_usd"})
            if stage == "h0":
                return {"schema_version": "joint_perception_final_only_v1", "instrument": {"selected_ids": [] if empty else [0]},
                        "verb": {"selected_ids": []}, "target": {"selected_ids": []}, "ivt": {"selected_ids": []},
                        "phase": {"selected_id": 0}}
            if stage.startswith("proposal"):
                return {q: [] for q in ("instrument", "verb", "target", "ivt")}
            return {"judgments": {"instrument_0": {"rating": rating,
                "finding": "MATCH" if rating == 5 else "UNCLEAR", "scope": "WHOLE_FRAME",
                "image_indices": [2], "observation": "Tool visible."}}}

        def persist(self):
            pass

    monkeypatch.setattr(trial, "Calls", FakeCalls)
    monkeypatch.setattr(trial, "build_gemini_base", lambda *a: SimpleNamespace(images=[1, 2, 3]))
    monkeypatch.setattr(trial, "canonical_request_metadata", lambda *a: SimpleNamespace(to_mapping=dict))
    monkeypatch.setattr(trial, "gemini_h0_wire", lambda *a: {"stage": "h0"})
    monkeypatch.setattr(trial, "gemini_proposal", lambda *a: {"issues": a[-1]})
    monkeypatch.setattr(trial, "review_wire", lambda *a: {"pool": a[-1]})
    monkeypatch.setattr(trial, "normalize_review_wire", lambda seat, raw, pool: raw)
    def score_subprocess(command, **kwargs):
        number = int(command[-1])
        rows = trial.read(output / f"round_{number}_predictions.json")
        score_files.append(number)
        assert rows[0]["reviewed_this_round"] is not empty
        return SimpleNamespace(returncode=0)  # No GT or model score returned to caller.
    monkeypatch.setattr(trial.subprocess, "run", score_subprocess)
    trial.execute(output, None)
    assert stages.count("h0") == 1 and len(stages) == (2 if empty else 1 + 6 * rounds)
    assert score_files == list(range(1, max(1, rounds)+1))
    assert trial.read(output / "completion.json")["actual_rounds"][selected["key"]] == rounds
    if empty:
        assert trial.read(output / "completion.json")["statuses"][selected["key"]] == "EMPTY_POOL_UNVERIFIED"
    assert "ground_truth" not in json.dumps(bodies)
    with pytest.raises(ValueError, match="replay"):
        trial.execute(output, None)
