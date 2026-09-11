import pytest

from scripts.write_review_diagnostics import read, save, sha, write_report


def test_closed_report_is_idempotent_and_preserves_predictions(tmp_path):
    path = tmp_path / "targets/fixture/pipeline.json"
    save(path, {"graph": {"diagnostics": {"verb_0": {"invalid": {"qwen": "SCHEMA_INVALID"}}},
        "format_diagnostics": {"qwen": {"errors": {"verb_0": ["LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL"]}}}},
        "final": {"verb": [0]}})
    original_hash = sha(path)
    save(tmp_path / "completion.json", {"fatal_error": None,
        "inference_artifact_sha256": {"targets/fixture/pipeline.json": original_hash}})
    report_path, result = write_report(tmp_path)
    assert result["reason_counts"] == {"LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL": 1}
    assert result["api_calls"] == 0 and not result["votes_changed"] and not result["predictions_changed"]
    assert sha(path) == original_hash
    assert write_report(tmp_path) == (report_path, read(report_path))
    save(path, {"graph": None})
    with pytest.raises(ValueError, match="changed source"):
        write_report(tmp_path)


def test_expected_early_fallback_can_have_no_review(tmp_path):
    path = tmp_path / "targets/fixture/pipeline.json"
    save(path, {"graph": {"diagnostics": None, "format_diagnostics": None}})
    save(tmp_path / "completion.json", {"fatal_error": None,
        "inference_artifact_sha256": {"targets/fixture/pipeline.json": sha(path)}})
    _, result = write_report(tmp_path)
    assert result["reason_counts"] == {}
