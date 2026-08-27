"""Dataset API rollout through the canonical causal pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

from PIL import Image

from surgical_agent.api.budget import ProviderCallBudget
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.api.registry import build_validator
from surgical_agent.api.usage import UsageLedger
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.api_rollout_selection import RolloutSelection
from surgical_agent.data.dataset import PNG_ALIGNMENT_VERSION
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.frame_result_writer import FrameResultWriter
from surgical_agent.systems.api_dataset_system import DatasetApiPipelineSystem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOCK_CONFIG = PROJECT_ROOT / "configs/perception/joint_mock.yaml"


def _selection_with_pngs(
    tmp_path: Path,
    samples: tuple[tuple[str, int], ...],
) -> RolloutSelection:
    tmp_path.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[InferenceSample]] = {}
    for video_id, frame_id in samples:
        image_path = tmp_path / f"{video_id}_{frame_id}.png"
        Image.new("RGB", (8, 6), (frame_id, frame_id, frame_id)).save(image_path)
        prior = by_video.setdefault(video_id, [])
        causal = (*prior[-1].causal_frame_ids, frame_id)[-3:] if prior else (frame_id,)
        media_refs = tuple(str(tmp_path / f"{video_id}_{value}.png") for value in causal)
        prior.append(
            InferenceSample(
                video_id=video_id,
                target_frame_id=frame_id,
                causal_frame_ids=causal,
                media_refs=media_refs,
                source_split=DatasetSplit.VALIDATION,
                alignment_version=PNG_ALIGNMENT_VERSION,
            )
        )
    video_ids = tuple(by_video)
    selected = tuple(sample for video_id in video_ids for sample in by_video[video_id])
    return RolloutSelection(
        mode="paper",
        split=DatasetSplit.VALIDATION,
        video_ids=video_ids,
        samples=selected,
        frame_counts=MappingProxyType(
            {video_id: len(by_video[video_id]) for video_id in video_ids}
        ),
    )


def _mock_system(
    tmp_path: Path,
    *,
    provider_call_limit: int,
    cache: FileApiCache | None = None,
) -> tuple[DatasetApiPipelineSystem, MockProviderTransport]:
    config = replace(
        load_api_config(MOCK_CONFIG),
        synthetic_input_required=False,
        data_upload_authorized=True,
    )
    transport = MockProviderTransport()
    client = CachedMultimodalApiClient(
        transport=transport,
        cache=cache or FileApiCache(tmp_path / "api_cache"),
        usage=UsageLedger(tmp_path / "api_usage.jsonl"),
        validator=build_validator(config),
        provider_call_budget=ProviderCallBudget(provider_call_limit),
    )
    return (
        DatasetApiPipelineSystem(
            client=client,
            config=config,
            writer=FrameResultWriter(tmp_path / "run", run_id="dataset-mock"),
            media_loader=CausalApiMediaLoader(),
        ),
        transport,
    )


def test_dataset_api_system_runs_ordered_samples_and_resets_each_video(
    tmp_path: Path,
) -> None:
    """Changing the one canonical pipeline into a per-frame instance loses resets."""

    selection = _selection_with_pngs(
        tmp_path,
        (("VID11", 1), ("VID11", 2), ("VID30", 1)),
    )
    system, transport = _mock_system(tmp_path, provider_call_limit=3)

    result = system.run(selection, run_id="dataset-mock")

    assert [(p.video_id, p.frame_id) for p in result.predictions] == [
        ("VID11", 1),
        ("VID11", 2),
        ("VID30", 1),
    ]
    assert "01_video_boundary_reset" in result.predictions[0].trace
    assert "01_video_boundary_continue" in result.predictions[1].trace
    assert "01_video_boundary_reset" in result.predictions[2].trace
    assert transport.provider_call_count == 3
    assert result.manifest_path.is_file()


def test_dataset_api_system_replays_persistent_cache_without_provider_calls(
    tmp_path: Path,
) -> None:
    """Removing persistent cache reuse would issue duplicate provider calls."""

    selection = _selection_with_pngs(
        tmp_path / "media",
        (("VID11", 1), ("VID11", 2), ("VID30", 1)),
    )
    shared_cache = FileApiCache(tmp_path / "api_cache")
    first, first_transport = _mock_system(
        tmp_path / "first",
        provider_call_limit=3,
        cache=shared_cache,
    )
    first.run(selection, run_id="dataset-mock")
    replay, replay_transport = _mock_system(
        tmp_path / "replay",
        provider_call_limit=3,
        cache=shared_cache,
    )

    result = replay.run(selection, run_id="dataset-mock")

    assert first_transport.provider_call_count == 3
    assert replay_transport.provider_call_count == 0
    assert result.usage_summary["cache_hits"] == 3
