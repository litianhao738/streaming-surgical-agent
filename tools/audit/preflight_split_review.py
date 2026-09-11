"""Real archived images/wires, synthetic answers, zero API and zero GT reads."""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_split_review_trial as trial


class MockCalls:
    def __init__(self):
        self.count = 0

    def call(self, target, stage, seat, body):
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        assert next(iter(packet)) == "academic_context"
        assert "academic" in packet["academic_context"]
        schema = packet["response_schema"]
        ids = [p["id"] for p in packet["propositions"]]
        tasks = {p["task"] for p in packet["propositions"]}
        if stage == "components":
            assert tasks <= {"instrument", "verb", "target"}
        elif stage == "relations":
            assert tasks == {"ivt"}
        self.count += 1
        item = {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [], "observation": "Offline fixture."}
        raw = ({"rows": [{"candidate_id": pid, **item} for pid in ids]} if "rows" in schema["properties"]
               else {"judgments": {pid: item for pid in ids}})
        Draft202012Validator(schema).validate(raw)
        if body["response_format"]["type"] == "json_schema":
            assert body["response_format"]["json_schema"]["schema"] == schema
        return raw


def main(output):
    plan = trial.verify(output)
    adapter = trial.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    initials, calls = trial.read(output / "initial_state.json"), MockCalls()
    for selected in plan["selection"]:
        wire = trial.bodies(output, selected, adapter)
        for seat in trial.SEATS:
            original = trial.read(trial.SOURCE / "requests" / selected["key"] / f"compact_graph_{seat}.json")
            assert trial.redact_images(wire["control"][seat]) == original
            for group in trial.split.GROUPS:
                assert wire[group][seat]["messages"][0]["content"][1:] == wire["control"][seat]["messages"][0]["content"][1:]
        results = {a: trial.run_arm(calls, selected["key"], initials[selected["key"]], wire, a) for a in trial.ARMS}
        assert results["control"]["final"] == results["split"]["final"]
        assert results["control"]["means"] == results["split"]["means"]
    report = {"api_calls": 0, "gt_reads": 0, "mock_requests": calls.count,
        "default_control_exact_wire": True, "schemas_and_disjoint_ids_verified": True,
        "same_images_and_shared_phase": True, "plan_sha256": trial.sha(output / "plan.json")}
    trial.save(output / "offline_preflight.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
