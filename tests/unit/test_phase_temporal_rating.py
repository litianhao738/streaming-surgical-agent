from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_phase_temporal_rating_trial import context_input, trial


def sample(frames):
    return {f: SimpleNamespace(media_refs=[Path(f"frame_{f}.png")]) for f in frames}


def row(target):
    return {"key": f"V_{target}", "video_id": "V", "frame_id": target,
        "images": [{"frame_id": target, "path": f"frame_{target}.png", "sha256": f"frame_{target}.png"}]}


def test_exact_history_is_causal(monkeypatch):
    monkeypatch.setattr(trial.old, "sha", lambda p: str(p))
    result = context_input(row(1000), sample(range(0, 1050, 25)))
    assert result["causal_frame_ids"] == [250, 750, 1000]


def test_no_cross_gap_uses_earliest_available(monkeypatch):
    monkeypatch.setattr(trial.old, "sha", lambda p: str(p))
    result = context_input(row(1000), sample([*range(0, 300, 25), *range(600, 1025, 25)]))
    assert result["causal_frame_ids"] == [600, 750, 1000]


def test_inadequate_history_stops_without_padding(monkeypatch):
    monkeypatch.setattr(trial.old, "sha", lambda p: str(p))
    with pytest.raises(ValueError, match="three distinct"):
        context_input(row(1000), sample([975, 1000]))


def test_ten_second_segment_uses_distinct_midpoint(monkeypatch):
    monkeypatch.setattr(trial.old, "sha", lambda p: str(p))
    result = context_input(row(1000), sample(range(750, 1025, 25)))
    assert result["causal_frame_ids"] == [750, 875, 1000]
