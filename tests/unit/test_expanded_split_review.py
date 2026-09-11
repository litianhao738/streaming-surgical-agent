from types import SimpleNamespace

from scripts import run_expanded_split_review as fresh


def test_invalid_h0_remains_failed_and_does_not_dispatch_repair():
    class Calls:
        def __init__(self):
            self.count = 0
        def call(self, *args):
            self.count += 1
    from unittest.mock import patch
    calls = Calls()
    with patch.object(fresh, "gemini_h0_wire", return_value={}):
        result = fresh.run_case(calls, None, {"key": "fixture"}, {}, 0)
    assert calls.count == 1
    assert result["status"] == "H0_FAILED"
    assert result["predictions"] == {"h0": None, "control": None, "split": None}


def test_history_exclusion_applies_to_every_image_not_just_target(monkeypatch, tmp_path):
    samples = [SimpleNamespace(target_frame_id=t, causal_frame_ids=(t-50, t-25, t), media_refs=(tmp_path,)*3)
               for t in range(1001, 10002, 25)]
    class Adapter:
        def __init__(self):
            self.entries = {v: SimpleNamespace(split=fresh.old.DatasetSplit.TRAINING) for v in fresh.VIDEOS}
        def iter_inference_video(self, video):
            return iter(samples)
        def iter_video(self, video):
            # Media iteration can contain frames without any annotation row.
            return iter(SimpleNamespace(inference=SimpleNamespace(target_frame_id=s.target_frame_id)) for s in samples[::2])
    monkeypatch.setattr(fresh, "masks_only", lambda r: {t: True for t in fresh.TASKS})
    monkeypatch.setattr(fresh.old, "sha", lambda p: "fixture")
    excluded = {v: {3901, 6876} for v in fresh.VIDEOS}
    rows = fresh.choose(Adapter(), excluded)
    assert len(rows) == 32
    for r in rows:
        assert r["frame_id"] in {s.target_frame_id for s in samples[::2]}
        assert all(abs(f-g) > fresh.GAP for f in r["causal_frame_ids"] for g in excluded[r["video_id"]])
    for video in fresh.VIDEOS:
        selected = [r for r in rows if r["video_id"] == video]
        assert len(selected) == 8
        for i, left in enumerate(selected):
            for right in selected[i+1:]:
                assert all(abs(f-g) > fresh.GAP for f in left["causal_frame_ids"] for g in right["causal_frame_ids"])
