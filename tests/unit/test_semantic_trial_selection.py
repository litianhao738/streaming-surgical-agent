from types import SimpleNamespace

from scripts import run_complete_gt_semantic_trial as trial
from surgical_agent.data.schemas import DatasetSplit


def test_sampling_checks_all_task_masks_and_never_reads_label_answers(monkeypatch):
    tasks = ("instrument", "verb", "target", "ivt", "phase")
    class Resolved:
        def __init__(self, frame):
            self.inference = SimpleNamespace(target_frame_id=frame)
            self.availability = {q: not (frame == 76 and q == "target") for q in tasks}

        @property
        def labels(self):
            raise AssertionError("GT labels must not influence sample selection")

    class Adapter:
        def iter_inference_video(self, video):
            return [SimpleNamespace(target_frame_id=f, causal_frame_ids=[f], media_refs=["image"],
                                    source_split=DatasetSplit.TRAINING) for f in (1, 26, 51, 76)]

        def iter_video(self, video):
            return [Resolved(f) for f in (26, 51, 76)]  # Frame 1 has no evaluation row.

    monkeypatch.setattr(trial, "_mask_only", lambda r: r.availability)
    monkeypatch.setattr(trial, "sha", lambda _: "image-hash")
    monkeypatch.setattr(trial, "build_base", lambda *_: None)
    monkeypatch.setattr(trial, "canonical_request_metadata", lambda _: SimpleNamespace(to_mapping=dict))
    rows = trial.choose_complete_samples(Adapter())
    assert [r["frame_id"] for r in rows] == [26, 51, 26, 51]
    assert [r["anchor_frame_id"] for r in rows] == [26, 76, 26, 76]
    assert all(all(r["gt_availability_only"].values()) for r in rows)
