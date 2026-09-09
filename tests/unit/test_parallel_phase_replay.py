"""The published evidence must replay without images, labels or paid services."""
import json
import socket
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.replay_parallel_phase_repair import (
    DEFAULT,
    IDENTITIES,
    REQUIRED_SOURCES,
    ROOT,
    SEATS,
    TASKS,
    answer,
    digest,
    phase_merge,
    replay,
    score_counts,
)
from scripts.replay_verified_repair_candidate import source_digest
from surgical_agent.research.verification.phase_extension import apply_phase_choices


@pytest.fixture
def bundle():
    return json.loads(DEFAULT.read_text(encoding="utf-8"))


def rehash(bundle):
    bundle["payload_sha256"] = digest(bundle["payload"])
    return bundle


def test_published_eight_target_replay_is_offline_and_keeps_the_old_best_comparison(bundle, monkeypatch):
    def deny_network(*args, **kwargs):
        pytest.fail("offline replay attempted network access")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    original_read = Path.read_bytes

    def confined_read(path):
        assert path.resolve().is_relative_to(ROOT)
        assert "artifacts" not in path.parts or "src" in path.parts
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", confined_read)
    result = replay(bundle)
    assert result["verified"] is True
    assert result["api_calls"] == 0
    assert result["targets"] == 8
    assert result["graph_panels_replayed"] == 8
    assert result["phase_panels_replayed"] == 16
    assert result["production_parallel_merges_replayed"] == 8
    assert result["gt_independently_rescored"] is False
    assert result["h0_regenerated"] is False
    assert result["priors_refitted"] is False
    assert [row["key"] for row in result["outputs"]] == list(IDENTITIES)
    scores = result["metrics_from_archived_counts"]
    assert scores["final"]["phase"]["micro_f1"] == 0.75
    assert scores["previous_final"]["ivt"]["micro_f1"] == 0.4
    assert scores["final"]["ivt"]["micro_f1"] == pytest.approx(12 / 31)
    for task in TASKS[:-1]:
        assert scores["final"][task] == scores["graph_only"][task]
        assert scores["previous_final"][task] == scores["previous_graph"][task]


def test_payload_checksum_detects_an_edited_saved_answer(bundle):
    bundle["payload"]["targets"][0]["phase"]["decision"]["after"] = 6
    with pytest.raises(ValueError, match="payload checksum"):
        replay(bundle)


@pytest.mark.parametrize("change", ["reorder", "duplicate", "different_frame", "fewer"])
def test_fixed_target_set_cannot_be_replaced_after_rehashing(bundle, change):
    rows = bundle["payload"]["targets"]
    if change == "reorder":
        rows.reverse()
    elif change == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif change == "different_frame":
        rows[0]["key"] = "VID103_18351"
    else:
        rows.pop()
    with pytest.raises(ValueError, match="target"):
        replay(rehash(bundle))


def test_testing_or_future_frame_cannot_be_relabelled_as_this_training_protocol(bundle):
    bundle["payload"]["targets"][0]["split"] = "Testing"
    with pytest.raises(ValueError, match="split"):
        replay(rehash(bundle))
    bundle["payload"]["targets"][0]["split"] = "Training"
    bundle["payload"]["targets"][0]["causal_frame_ids"][-1] += 25
    with pytest.raises(ValueError, match="causal"):
        replay(rehash(bundle))


@pytest.mark.parametrize("section", ["graph", "phase", "previous_phase"])
def test_each_panel_has_five_distinct_named_model_seats(bundle, section):
    responses = bundle["payload"]["targets"][0][section]["responses"]
    responses["second_gpt"] = responses.pop("deepseek")
    with pytest.raises(ValueError, match="five distinct"):
        replay(rehash(bundle))


def test_model_identity_cannot_be_swapped_behind_a_named_seat(bundle):
    bundle["payload"]["targets"][0]["phase"]["responses"]["gpt"]["model"] = "grok-4.6"
    with pytest.raises(ValueError, match="model/seat"):
        replay(rehash(bundle))


@pytest.mark.parametrize("section", ["phase", "previous_phase"])
def test_phase_merge_cannot_silently_change_interaction_heads(bundle, section):
    bundle["payload"]["targets"][0][section]["prediction"]["ivt"].append(99)
    with pytest.raises(ValueError, match="Phase merged prediction"):
        replay(rehash(bundle))


def test_graph_phase_must_stay_h0_until_the_separate_phase_merge(bundle):
    bundle["payload"]["targets"][0]["graph"]["prediction"]["phase"] = [6]
    with pytest.raises(ValueError, match="graph local repair"):
        replay(rehash(bundle))


def test_replay_rebuilds_candidate_means_instead_of_trusting_saved_numbers(bundle):
    bundle["payload"]["targets"][0]["graph"]["means"]["ivt_60"] = 5
    with pytest.raises(ValueError, match="candidate means"):
        replay(rehash(bundle))


def test_replay_rebuilds_proposal_pool_before_scoring(bundle):
    bundle["payload"]["targets"][0]["graph"]["pool"]["propositions"].pop()
    with pytest.raises(ValueError, match="candidate pool"):
        replay(rehash(bundle))


def test_query_video_cannot_enter_the_archived_prior_source_audit(bundle):
    row = bundle["payload"]["targets"][0]
    row["graph"]["hints"]["audit"]["fit_videos"].append(row["video_id"])
    with pytest.raises(ValueError, match="query video"):
        replay(rehash(bundle))


def test_aggregate_arithmetic_and_prediction_totals_are_both_checked(bundle):
    item = bundle["payload"]["scores"]["final"]["ivt"]
    item["counts"]["fp"] += 1
    item["metrics"] = score_counts(item["counts"])
    with pytest.raises(ValueError, match="prediction total"):
        replay(rehash(bundle))


def test_each_task_mask_count_is_checked_instead_of_assuming_a_full_split(bundle):
    item = bundle["payload"]["scores"]["final"]["phase"]
    item["counts"]["valid_targets"] = 7
    item["metrics"] = score_counts(item["counts"])
    with pytest.raises(ValueError, match="mask count"):
        replay(rehash(bundle))


def test_per_frame_gt_is_not_an_allowed_evidence_field(bundle):
    bundle["payload"]["targets"][0]["gt"] = {"phase": [1]}
    with pytest.raises(ValueError, match="no per-frame GT"):
        replay(rehash(bundle))


def test_required_sources_include_the_ontology_and_final_label_schema(bundle):
    hashes = bundle["payload"]["replay_source_sha256_lf"]
    assert REQUIRED_SOURCES <= set(hashes)
    assert any(name.endswith("ivt_components_v1.csv") for name in REQUIRED_SOURCES)
    assert any(name.endswith("perception_schema_gate_owned_compact.json") for name in REQUIRED_SOURCES)
    del hashes["src/surgical_agent/research/verification/phase_extension.py"]
    with pytest.raises(ValueError, match="required source hashes"):
        replay(rehash(bundle))


def test_source_fingerprints_accept_crlf_but_reject_a_code_change(bundle, tmp_path):
    hashes = bundle["payload"]["replay_source_sha256_lf"]
    for name in hashes:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert replay(bundle, root=tmp_path)["verified"] is True
    altered = tmp_path / "src/surgical_agent/research/verification/phase_extension.py"
    with altered.open("ab") as stream:
        stream.write(b"\n# altered rule\n")
    with pytest.raises(ValueError, match="source changed"):
        replay(bundle, root=tmp_path)


def test_failed_phase_request_remains_an_invalid_panel_even_with_four_matching_votes(bundle):
    row = bundle["payload"]["targets"][0]
    current, record = row["graph"]["prediction"], deepcopy(row["phase"])
    response = record["responses"]["gpt"]
    response.update(http_status=403, finish_reason=None, content=None)
    raw = {seat: answer(record["responses"][seat], seat) for seat in SEATS}
    assert raw["gpt"] is None
    record["errors"]["gpt"] = "INVALID_FIELDS"
    expected, decision = apply_phase_choices(current, raw, 3)
    record.update(prediction=expected, decision=decision)
    result, actual_decision = phase_merge(current, record)
    assert result == current
    assert actual_decision["reason"] == "INVALID_PANEL"


def test_failure_envelope_cannot_smuggle_a_vote_from_its_content(bundle):
    response = deepcopy(bundle["payload"]["targets"][0]["phase"]["responses"]["gpt"])
    response["http_status"] = 403
    with pytest.raises(ValueError, match="failed request"):
        answer(response, "gpt")


def test_lf_source_digest_is_stable_across_checkout_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"one\ntwo\n")
    crlf.write_bytes(b"one\r\ntwo\r\n")
    assert source_digest(lf) == source_digest(crlf)
