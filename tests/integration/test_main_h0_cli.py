"""Main H0 entry point: actual causal requests, hard labels and cache replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from PIL import Image

from scripts import run_dataset_api_pipeline as cli
from surgical_agent.api.errors import ApiContractError
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_rollout_selection import RolloutSelection
from surgical_agent.data.dataset import PNG_ALIGNMENT_VERSION
from surgical_agent.data.schemas import DatasetSplit, InferenceSample

ROOT = Path(__file__).resolve().parents[2]
MOCK = ROOT / "configs/perception/joint_mock_h0.yaml"
REAL = ROOT / "configs/perception/joint_openrouter_h0.yaml"


@pytest.fixture
def causal_selection(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    media = dataset / "media"
    media.mkdir(parents=True)
    (dataset / "repair_manifest.json").write_text("{}", encoding="utf-8")
    for frame in (1, 26, 51, 76):
        Image.new("RGB", (16, 12), (frame, 20, 30)).save(media / f"{frame}.png")
    samples = tuple(
        InferenceSample(
            video_id="VID110", target_frame_id=target,
            causal_frame_ids=(target - 50, target - 25, target),
            media_refs=tuple(str(media / f"{f}.png") for f in (target - 50, target - 25, target)),
            source_split=DatasetSplit.VALIDATION, alignment_version=PNG_ALIGNMENT_VERSION,
        ) for target in (51, 76)
    )
    selection = RolloutSelection(
        mode="engineering", split=None, video_ids=("VID110",), samples=samples,
        frame_counts=MappingProxyType({"VID110": 2}),
    )

    class Adapter:
        def __init__(self, *args, **kwargs):
            self.entries = {"VID110": SimpleNamespace(split=DatasetSplit.VALIDATION)}

        def iter_inference_video(self, video_id, *, max_samples=None):
            yield from samples if max_samples is None else samples[:max_samples]

    monkeypatch.setattr(cli, "CholecTrack20DatasetAdapter", Adapter)
    return dataset, selection


def test_default_is_main_real_h0_and_real_upload_remains_explicit():
    args = cli.build_parser().parse_args([
        "--mode", "engineering", "--video-id", "VID110", "--max-frames", "2",
    ])
    assert args.config == REAL
    config = load_api_config(args.config)
    cli._require_exact_dataset_config(
        config, has_credential=True, authorize_data_upload=True,
    )
    with pytest.raises(ApiContractError, match="credential"):
        cli._require_exact_dataset_config(
            config, has_credential=False, authorize_data_upload=True,
        )


def test_main_h0_rollout_uses_one_call_per_target_and_cache_replay(causal_selection, tmp_path):
    dataset, selection = causal_selection
    config = load_api_config(MOCK)
    first_transport = MockProviderTransport()
    first = cli.run_dataset_api_rollout(
        config=config, selection=selection, dataset_root=dataset,
        output_dir=tmp_path / "first", cache_root=tmp_path / "cache",
        run_id="main-first", max_provider_calls=2, transport=first_transport,
    )
    assert first_transport.provider_call_count == 2
    assert first["response_schema_version"] == "joint_perception_final_only_v1"
    assert first["context_profile"] == "frames_only"
    assert first["event_memory_enabled"] is False
    assert first["causal_window"]["schema_version"] == "fixed_causal_window_v1"
    again = MockProviderTransport()
    cli.run_dataset_api_rollout(
        config=config, selection=selection, dataset_root=dataset,
        output_dir=tmp_path / "second", cache_root=tmp_path / "cache",
        run_id="main-second", max_provider_calls=2, transport=again,
    )
    assert again.provider_call_count == 0


def test_preflight_builds_real_requests_without_credentials(causal_selection, tmp_path):
    dataset, _selection = causal_selection
    args = cli.build_parser().parse_args([
        "--mode", "engineering", "--video-id", "VID110", "--max-frames", "2",
        "--dataset-root", str(dataset), "--output-root", str(tmp_path / "out"),
        "--cache-root", str(tmp_path / "cache"), "--run-id", "ready", "--preflight",
    ])
    path = cli.run(args)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["provider_calls"] == 0
    assert artifact["planned_calls"] == 2
    request = json.loads((path.parent / "requests/VID110_51.json").read_text(encoding="utf-8"))
    body = json.loads(request["payload"]["input_text"])
    assert body["causal_frame_ids"] == [1, 26, 51]
    assert body["relative_seconds"] == [-2, -1, 0]
    assert request["payload"]["image_details"] == ["low", "low", "high"]
    assert "base64" not in json.dumps(request)


def test_main_h0_rejects_old_verification_profile_before_a_call(causal_selection, tmp_path):
    dataset, selection = causal_selection
    transport = MockProviderTransport()
    with pytest.raises(ApiContractError, match="single_pass"):
        cli.run_dataset_api_rollout(
            config=load_api_config(MOCK), selection=selection, dataset_root=dataset,
            output_dir=tmp_path / "out", cache_root=tmp_path / "cache",
            run_id="reject", max_provider_calls=4, transport=transport,
            pipeline_profile="always_verify",
        )
    assert transport.provider_call_count == 0
