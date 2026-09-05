"""Exercise the main label-only API path without any network request."""

import json
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
from surgical_agent.perception.main_h0 import (
    MAIN_H0_PROMPT_VERSION,
    MAIN_H0_SCHEMA_VERSION,
)
from surgical_agent.systems.api_dataset_system import DatasetApiPipelineSystem
from surgical_agent.systems.pipeline import NoOpCausalStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _CaptureMockTransport(MockProviderTransport):
    def __init__(self):
        super().__init__()
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        return super().send(request)


def test_main_h0_runs_once_per_target_without_state_or_repair(tmp_path):
    config = replace(
        load_api_config(PROJECT_ROOT / "configs/perception/joint_mock_final_fixed3.yaml"),
        prompt_version=MAIN_H0_PROMPT_VERSION,
        response_schema_version=MAIN_H0_SCHEMA_VERSION,
        generation_parameters={
            "max_output_tokens": 4096,
            "temperature": 0,
            "reasoning": {"effort": "low"},
        },
        provider_options={
            "frame_selection_strategy": "fixed_all",
            "history_image_detail": "low",
            "target_image_detail": "high",
            "initial_prompt_profile": "fixed_visual_only",
        },
        synthetic_input_required=False,
        data_upload_authorized=True,
    )
    transport = _CaptureMockTransport()
    client = CachedMultimodalApiClient(
        transport=transport,
        cache=FileApiCache(tmp_path / "cache"),
        usage=UsageLedger(tmp_path / "usage.jsonl"),
        validator=build_validator(config),
        provider_call_budget=ProviderCallBudget(2),
    )
    samples = []
    for frame_id in (1, 26, 51, 76):
        Image.new("RGB", (8, 6), (frame_id, 20, 30)).save(tmp_path / f"{frame_id}.png")
    for frame_ids in ((1, 26, 51), (26, 51, 76)):
        samples.append(
            InferenceSample(
                video_id="VID11",
                target_frame_id=frame_ids[-1],
                causal_frame_ids=frame_ids,
                media_refs=tuple(str(tmp_path / f"{frame_id}.png") for frame_id in frame_ids),
                source_split=DatasetSplit.VALIDATION,
                alignment_version=PNG_ALIGNMENT_VERSION,
            )
        )
    selection = RolloutSelection(
        mode="paper",
        split=DatasetSplit.VALIDATION,
        video_ids=("VID11",),
        samples=tuple(samples),
        frame_counts=MappingProxyType({"VID11": 2}),
    )
    system = DatasetApiPipelineSystem(
        client=client,
        config=config,
        writer=FrameResultWriter(tmp_path / "run", run_id="main-mock"),
        media_loader=CausalApiMediaLoader(),
    )

    result = system.run(selection, run_id="main-mock")

    assert system.context_profile == "frames_only"
    assert system.event_memory_enabled is False
    assert isinstance(system.pipeline.components.workflow_store, NoOpCausalStore)
    assert isinstance(system.pipeline.components.event_memory, NoOpCausalStore)
    assert transport.provider_call_count == 2
    assert len(result.predictions) == 2
    assert result.manifest_path.is_file()
    assert all(record.score_semantics == "hard_label_v1" for record in result.predictions)
    assert all(record.verification_status == "NOT_REQUESTED" for record in result.predictions)
    assert all(record.gate_action == "ACCEPT" for record in result.predictions)
    for request, sample in zip(transport.requests, samples, strict=True):
        payload = json.loads(request.payload["input_text"])
        assert payload["causal_frame_ids"] == list(sample.causal_frame_ids)
        assert payload["relative_seconds"] == [-2, -1, 0]
        assert payload["prior_finalized_prediction"] is None
        assert payload["workflow_summary"]["recent_finalized_phases"] == []
        assert payload["track_summary"]["frames"] == []
        assert len(request.images) == 3
    evidence = [
        json.loads(line)
        for line in (tmp_path / "run/evidence/VID11.jsonl").read_text().splitlines()
    ]
    assert len(evidence) == 2
