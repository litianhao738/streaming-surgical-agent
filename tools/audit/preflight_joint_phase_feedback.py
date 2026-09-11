"""Actual image/request construction with mock responses; zero API and GT reads."""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from jsonschema import Draft202012Validator

from scripts import run_joint_phase_feedback_trial as trial


class MockCalls:
    def __init__(self, current):
        self.stopped = False
        self.rows = []
        self.current = current

    def call(self, target, stage, seat, body):
        packet = trial.packet_of(body)
        schema = packet["response_schema"]
        props = schema["properties"]
        assert body["model"] == (trial.PROPOSER if seat == "base" else trial.roster.MODELS[seat])
        assert len(body["messages"][0]["content"]) == 4
        assert len({b["image_url"]["url"] for b in body["messages"][0]["content"][1:]}) == 3
        if "phase_id" in props:
            answer = {"phase_id": None, "image_indices": [], "observation": "Insufficient distinguishing visual evidence."}
        elif "prediction" in props:
            answer = {"prediction": deepcopy(self.current),
                "observations": {t: "Uncertain visual interpretation; retain supported existing hypothesis." for t in trial.TASKS}}
        else:
            candidates = packet["propositions"]
            if isinstance(candidates, dict):
                candidates = list(candidates.values())
            items = {p["id"]: {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [2],
                "observation": "The visible evidence cannot distinguish the candidate."} for p in candidates}
            answer = {"rows": [{"candidate_id": k, **v} for k, v in items.items()]} if "rows" in props else {"judgments": items}
        Draft202012Validator(schema).validate(answer)
        if body.get("response_format", {}).get("type") == "json_schema":
            Draft202012Validator(body["response_format"]["json_schema"]["schema"]).validate(answer)
        self.rows.append({"target": target, "stage": stage, "seat": seat})
        return answer


def run(output):
    plan = trial.verify(output)
    initials = trial.read(output / "initials.json")
    adapter = trial.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    bases = trial.bases_for(adapter, plan)
    count = 0
    with patch("requests.post", side_effect=AssertionError("No API allowed")), \
         patch.object(adapter, "iter_video", side_effect=AssertionError("No GT allowed")), \
         trial.roster.lightweight_protocol():
        for index, selected in enumerate(plan["selection"]):
            initial = initials[selected["key"]]
            calls = MockCalls(initial["h0"])
            result = trial.run_case(calls, bases[selected["key"]], selected, initial, plan["variant"], index)
            assert all(result["predictions"][a] == initial["h0"] for a in trial.ARMS)
            assert len(calls.rows) == 22
            assert result["stop_reason"] == "STATE_REPEATED"
            assert result["joint_r1"]["phase_decision"]["valid_panel"]
            assert all(v == 3 for v in result["joint_r2"]["means"].values())
            count += len(calls.rows)
    result = {"actual_wire_mock_calls": count, "api_calls": 0, "gt_reads": 0,
        "five_heads_validated": True, "second_round_exercised": True, "plan_sha256": trial.sha(output / "plan.json")}
    trial.save(output / "offline_preflight.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
