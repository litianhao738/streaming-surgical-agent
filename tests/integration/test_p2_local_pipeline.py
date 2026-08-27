"""Real local P2 routing and end-to-end smoke checks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from surgical_agent.data.dataset import (
    MP4_ALIGNMENT_VERSION,
    CholecTrack20DatasetAdapter,
    CurrentFrameTensorDataset,
    collate_smoke_batch,
)
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.frame_result_writer import FrameResultWriter
from surgical_agent.models.baseline import LocalSmokeModel
from surgical_agent.systems.baseline_system import P2BaselineSystem
from surgical_agent.training.trainer import LocalSmokeTrainer

DATASET_ROOT = Path(
    os.environ.get("CHOLECTRACK20_ROOT", "__external_dataset_not_configured__")
)
pytestmark = pytest.mark.skipif(
    not DATASET_ROOT.is_dir(),
    reason="CholecTrack20 local-data integration fixture is unavailable",
)


def test_api_inference_iterator_returns_only_gold_free_samples() -> None:
    adapter = CholecTrack20DatasetAdapter(DATASET_ROOT)
    validation = tuple(adapter.iter_inference_video("VID30", max_samples=2))
    testing = tuple(adapter.iter_inference_video("VID01", max_samples=2))

    assert all(isinstance(sample, InferenceSample) for sample in validation + testing)
    assert all(len(sample.causal_frame_ids) <= 3 for sample in validation + testing)
    assert not hasattr(validation[0], "evaluation")
    assert testing[0].alignment_version == MP4_ALIGNMENT_VERSION


def test_vid02_vid31_vid30_routes_and_masks_are_explicit() -> None:
    adapter = CholecTrack20DatasetAdapter(DATASET_ROOT)
    vid02 = next(adapter.iter_video("VID02", max_samples=1))
    vid31 = next(adapter.iter_video("VID31", max_samples=1))
    vid30 = next(adapter.iter_video("VID30", max_samples=1))

    assert vid02.provenance.route == "official_raw"
    assert vid02.frame_supervision is not None
    assert vid02.frame_supervision.mask == vid02.frame_supervision.mask.__class__(
        False, False, False, False, True
    )
    assert vid31.provenance.route == "derived_vid31_frame_multilabel"
    assert vid31.frame_supervision is not None
    assert all(
        getattr(vid31.frame_supervision.mask, task)
        for task in ("instrument", "verb", "target", "ivt", "phase")
    )
    assert vid31.evaluation is not None
    assert not vid31.evaluation.instance_supervision_available
    assert not vid31.evaluation.instances
    assert vid30.provenance.route == "derived_vid30"
    assert vid30.inference.source_split is DatasetSplit.VALIDATION
    assert vid30.provenance.annotation_source is not None
    assert vid30.provenance.annotation_source.endswith("vid30_repaired.json")


def test_three_video_local_train_and_canonical_inference(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)
    adapter = CholecTrack20DatasetAdapter(DATASET_ROOT)
    training = adapter.collect(("VID02", "VID31"), samples_per_video=1)
    tensor_dataset = CurrentFrameTensorDataset(training, image_size=48)
    batch = collate_smoke_batch(
        [tensor_dataset[index] for index in range(len(tensor_dataset))]
    )
    model = LocalSmokeModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_result = LocalSmokeTrainer(
        model,
        optimizer,
        device=torch.device("cpu"),
    ).train_step(batch)
    assert train_result.valid_counts == {
        "instrument": 1,
        "verb": 1,
        "target": 1,
        "ivt": 1,
        "phase": 2,
    }

    records = training + tuple(adapter.iter_video("VID30", max_samples=2))
    writer = FrameResultWriter(tmp_path, run_id="integration")
    result = P2BaselineSystem(
        model,
        device=torch.device("cpu"),
        writer=writer,
        image_size=48,
    ).run(
        records,
        run_id="integration",
        manifest_metadata={"paper_metric_eligible": False},
    )

    assert [(item.video_id, item.frame_id) for item in result.predictions] == [
        (record.inference.video_id, record.inference.target_frame_id)
        for record in records
    ]
    assert result.metric_summary["sample_count"] == 4
    assert result.metric_summary["metrics"]["phase_accuracy"]["support"] == 4
    assert result.manifest_path.is_file()
    assert (tmp_path / "predictions/VID02.jsonl").is_file()
    assert (tmp_path / "evidence/VID02.jsonl").is_file()
    assert (tmp_path / "predictions/VID31.jsonl").is_file()
    assert (tmp_path / "evidence/VID31.jsonl").is_file()
    assert (tmp_path / "predictions/VID30.jsonl").is_file()
    assert (tmp_path / "evidence/VID30.jsonl").is_file()
