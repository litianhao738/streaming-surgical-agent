"""Actual-image request/schema mock; no inference API or query GT access."""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_phase_mechanism_comparison as trial


class MockCalls:
    stopped = False

    def __init__(self, initial):
        self.initial = initial
        self.rows = []

    def call(self, target, stage, seat, body):
        self.rows.append({"target": target, "stage": stage, "seat": seat})
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        assert next(iter(packet)) == "academic_context"
        before = self.initial[target]["graph_r1"]["prediction"]
        if stage in trial.PHASE_ARMS:
            value = {"phase_id": before["phase"][0], "image_indices": [2], "observation": "Offline fixture."}
        elif stage == trial.legacy.REPAIR_STAGE:
            value = {"prediction": before, "observations": {t: "Offline fixture." for t in before}}
        else:
            ids = [p["id"] for p in packet["propositions"]]
            item = {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [], "observation": "Offline fixture."}
            value = ({"rows": [{"candidate_id": pid, **item} for pid in ids]} if "rows" in packet["response_schema"]["properties"]
                else {"judgments": {pid: item for pid in ids}})
        Draft202012Validator(packet["response_schema"]).validate(value)
        return value


def main(output):
    plan = trial.verify(output)
    initial = trial.old.read(output / "initial_state.json")
    adapter = trial.old.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    calls = MockCalls(initial)
    for index, row in enumerate(plan["selection"]):
        base = trial.old.build_gemini_base(trial.old.InferenceOnlyAdapter(adapter), row)
        for arm in trial.PHASE_ARMS:
            for seat in trial.old.SEATS:
                assert trial.old.redact_images(trial.phase_body(arm, seat, row, initial[row["key"]])) == trial.old.read(
                    output / "preflight_requests" / row["key"] / f"{arm}_{seat}.json")
        record = trial.run_case(calls, base, row, initial[row["key"]], index)
        assert record["branches"]["legacy_panel"]["status"] == "REVIEWED"
        before = initial[row["key"]]["graph_r1"]["prediction"]
        assert all(record["predictions"][a] == before for a in trial.ARMS)
    assert len(calls.rows) == len({(r["target"], r["stage"], r["seat"]) for r in calls.rows}) == 672
    report = {"mock_calls": 672, "api_calls": 0, "query_gt_reads": 0, "schemas_and_frozen_requests_verified": True,
        "plan_sha256": trial.old.sha(output / "plan.json")}
    trial.old.save(output / "offline_preflight.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
