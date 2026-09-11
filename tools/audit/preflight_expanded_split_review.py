"""Exercise fresh H0/proposal/Phase and both panels with real images, no API/GT."""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_expanded_split_review as fresh


class MockCalls:
    def __init__(self):
        self.ids = []

    def call(self, target, stage, seat, body):
        self.ids.append((target, stage, seat))
        if stage == "h0":
            raw = {"schema_version": "joint_perception_final_only_v1", "instrument": {"selected_ids": [0]},
                "verb": {"selected_ids": [0]}, "target": {"selected_ids": [0]}, "ivt": {"selected_ids": [7]},
                "phase": {"selected_id": 1}}
            fresh.validate_final_only(raw)
            return raw
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        assert next(iter(packet)) == "academic_context"
        if stage == "proposal":
            raw = {t: [] for t in ("instrument", "verb", "target", "ivt")}
        elif stage == "phase":
            raw = {"phase_id": 1, "image_indices": [2], "observation": "Offline fixture."}
        else:
            ids = [p["id"] for p in packet["propositions"]]
            item = {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [], "observation": "Offline fixture."}
            raw = ({"rows": [{"candidate_id": pid, **item} for pid in ids]} if "rows" in packet["response_schema"]["properties"]
                else {"judgments": {pid: item for pid in ids}})
        Draft202012Validator(packet["response_schema"]).validate(raw)
        return raw


def main(output):
    plan = fresh.verify(output)
    adapter = fresh.old.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    calls = MockCalls()
    for index, selected in enumerate(plan["selection"]):
        base = fresh.old.build_gemini_base(fresh.old.InferenceOnlyAdapter(adapter), selected)
        for seat in fresh.old.SEATS:
            assert fresh.fingerprint(fresh.compact.phase_wire(seat, selected)) == selected["phase_fingerprints"][seat]
        record = fresh.run_case(calls, base, selected, fresh.old.read(output / "priors" / f"{selected['video_id']}.json"), index)
        assert record["status"] == "READY"
        assert record["predictions"]["control"] == record["predictions"]["split"] == record["h0"]
    assert len(calls.ids) == len(set(calls.ids)) == 704
    result = {"mock_calls": len(calls.ids), "api_calls": 0, "gt_label_reads": 0,
        "source_frozen": True, "shared_inputs_and_schemas_verified": True, "plan_sha256": fresh.old.sha(output / "plan.json")}
    fresh.old.save(output / "offline_preflight.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
