"""Published raw answers must reproduce the frozen coordinator without API use."""
import json
import socket
from copy import deepcopy

import pytest

from scripts.replay_prior_graph_candidate import DEFAULT, ROOT, digest, replay


@pytest.fixture
def bundle():
    return json.loads(DEFAULT.read_text(encoding="utf-8"))


def resign(bundle):
    bundle["payload_sha256"] = digest(bundle["payload"])
    return bundle


def test_published_bundle_replays_without_network(bundle, monkeypatch):
    def no_connection(*args, **kwargs):
        pytest.fail("offline replay attempted network access")

    monkeypatch.setattr(socket, "create_connection", no_connection)
    monkeypatch.setattr(socket.socket, "connect", no_connection)
    result = replay(bundle, root=ROOT)
    assert result["verified"] is True
    assert result["api_calls"] == 0
    assert result["targets"] == 8
    assert result["priors_rebuilt"] == 4
    assert result["hints_rebuilt"] == 8
    assert result["replayed_panels"] == 16
    assert result["shared_panels"] == 4
    assert result["gt_independently_rescored"] is False


def test_payload_tamper_without_resigning_is_rejected(bundle):
    bundle["payload"]["release"] = "changed"
    with pytest.raises(ValueError, match="payload checksum"):
        replay(bundle)


def test_resigned_score_arithmetic_tamper_is_rejected(bundle):
    bundle["payload"]["scores"]["prior_graph"]["ivt"]["metrics"]["micro_f1"] = 0.99
    with pytest.raises(ValueError, match="metric arithmetic"):
        replay(resign(bundle))


def test_resigned_prediction_tamper_is_rejected(bundle):
    record = bundle["payload"]["targets"][0]["arms"]["prior_graph"]
    prediction = record["prediction"]["ivt"]
    record["prediction"]["ivt"] = prediction[:-1] if prediction else [0]
    with pytest.raises(ValueError, match="repaired prediction"):
        replay(resign(bundle))


def test_resigned_wrong_query_prior_is_rejected(bundle):
    priors = bundle["payload"]["priors"]
    first, second = list(priors)[:2]
    priors[first] = deepcopy(priors[second])
    with pytest.raises(ValueError, match="leave-video-out prior"):
        replay(resign(bundle))


def test_resigned_sharing_different_pool_is_rejected(bundle):
    target = next(t for t in bundle["payload"]["targets"]
                  if t["arms"]["control"]["pool"] != t["arms"]["prior_graph"]["pool"])
    target["arms"]["prior_graph"]["shared_from"] = "control"
    target["arms"]["control"]["shared_from"] = None
    with pytest.raises(ValueError, match="shared panel pool"):
        replay(resign(bundle))


def test_resigned_source_hash_removal_is_rejected(bundle):
    bundle["payload"]["replay_source_sha256_lf"].pop("scripts/replay_prior_graph_candidate.py")
    with pytest.raises(ValueError, match="missing required replay source hashes"):
        replay(resign(bundle))
