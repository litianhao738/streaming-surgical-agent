"""Reference fidelity and Phase replacement without candidate/panel machinery."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import run_original_blind_audit_trial as trial
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION


def sample():
    return {"video_id": "VID31", "frame_id": 100, "causal_frame_ids": [50, 75, 100],
            "images": [{"path": "not-read"}] * 3}


def payload():
    return {"schema_version": FINAL_ONLY_SCHEMA_VERSION,
            **{t: {"selected_ids": [0]} for t in trial.TASKS[:-1]}, "phase": {"selected_id": 1}}


def test_original_appendix_is_verbatim_and_only_system_prompt_differs(monkeypatch):
    native = trial.reference_functions()
    native["jpeg_b64"] = lambda p: "AAAA"
    monkeypatch.setattr(trial, "reference_functions", lambda: native)
    control = trial.wire(sample(), trial.ARMS[0])
    audit = trial.wire(sample(), trial.ARMS[1])
    assert audit["messages"][0]["content"] == control["messages"][0]["content"] + "\n\n" + native["AUDIT_APPENDIX"]
    audit["messages"][0] = deepcopy(control["messages"][0])
    assert audit == control
    assert "reasoning" not in control
    assert all("detail" not in p.get("image_url", {}) for p in control["messages"][1]["content"])
    assert "current_prediction" not in json.loads(control["messages"][1]["content"][0]["text"])


def test_transport_mapping_preserves_native_text_image_bytes_and_schema():
    native = trial.reference_functions()
    native["jpeg_b64"] = lambda p: "AAAA"
    original = native["wire"](sample(), arm=trial.ARMS[1])
    snapshot = deepcopy(original)
    converted = trial.convert_wire(original)
    assert converted["messages"][0]["content"] == original["systemInstruction"]["parts"][0]["text"]
    assert converted["messages"][1]["content"][0]["text"] == original["contents"][0]["parts"][0]["text"]
    assert len(converted["messages"][1]["content"]) == 4
    assert converted["messages"][1]["content"][1]["image_url"]["url"].endswith("AAAA")
    assert converted["response_format"]["json_schema"]["schema"] == original["generationConfig"]["responseJsonSchema"]
    assert converted["temperature"] == 0 and converted["max_tokens"] == 4096
    assert original == snapshot


@pytest.mark.parametrize("bad", [999, True, -1, "0"])
def test_invalid_ids_are_failed_predictions_not_uncaught_errors(bad):
    raw = payload()
    raw["verb"]["selected_ids"] = [bad]
    assert trial.decode(raw) == {"status": "SCHEMA_FAILURE", "labels": None}


def test_missing_and_duplicate_labels_fail():
    raw = payload()
    del raw["target"]
    assert trial.decode(raw)["labels"] is None
    raw = payload()
    raw["verb"]["selected_ids"] = [0, 0]
    assert trial.decode(raw)["status"] == "SCHEMA_FAILURE"
    assert trial.decode(None)["status"] == "API_UNAVAILABLE"


def test_phase_replay_replaces_only_phase_and_does_not_mutate_sources():
    control = trial.decode(payload())["labels"]
    audit = deepcopy(control)
    audit.update(phase=[4], verb=[1, 2], ivt=[17, 58])
    snap = deepcopy((control, audit))
    final = trial.freeze_phase(control, audit)
    assert final["phase"] == control["phase"]
    assert all(final[t] == audit[t] for t in trial.TASKS[:-1])
    final["phase"].append(2)
    assert (control, audit) == snap
    assert trial.freeze_phase(None, audit) is None and trial.freeze_phase(control, None) is None


def test_reference_only_loads_pure_functions():
    namespace = trial.reference_functions()
    assert "run" not in namespace and "call" not in namespace and "select" not in namespace
    assert isinstance(namespace["AUDIT_APPENDIX"], str)
    assert Path(trial.REFERENCE).is_file()
