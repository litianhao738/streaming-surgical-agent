"""Independent audit must preserve inference/GT separation and input identity."""

import json
from copy import deepcopy

import pytest

from tools.audit import audit_evidence_feedback as audit_module


@pytest.mark.parametrize("completion,closed", [(False, False), (True, False)])
def test_no_dataset_or_truth_read_before_closed(tmp_path, monkeypatch, completion, closed):
    if completion:
        (tmp_path / "completion.json").write_text("{}", encoding="utf-8")
        (tmp_path / "budget.json").write_text(json.dumps({"stopped": closed}), encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("dataset must remain inaccessible before paid inference closes")

    monkeypatch.setattr(audit_module, "CholecTrack20DatasetAdapter", forbidden)
    with pytest.raises(ValueError, match="GT audit deferred"):
        audit_module.audit(tmp_path)


def test_reconstructed_packet_allows_only_object_order_loss():
    before = {"model": "test", "messages": [{"content": [{"text": '{"b":[1,2],"a":1}'}]}]}
    reordered = deepcopy(before)
    reordered["messages"][0]["content"][0]["text"] = '{"a":1,"b":[1,2]}'
    assert audit_module.same_reconstructed_input(before, reordered)
    assert audit_module.fingerprint(before) != audit_module.fingerprint(reordered)
    reordered["messages"][0]["content"][0]["text"] = '{"a":1,"b":[2,1]}'
    assert not audit_module.same_reconstructed_input(before, reordered)
    reordered["messages"][0]["content"][0]["text"] = '{"a":1,"b":[1,2],"unknown":null}'
    assert not audit_module.same_reconstructed_input(before, reordered)
    reordered = deepcopy(before)
    reordered["model"] = "changed"
    assert not audit_module.same_reconstructed_input(before, reordered)
