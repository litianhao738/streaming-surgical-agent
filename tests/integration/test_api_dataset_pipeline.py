"""Dataset API rollout through the canonical causal pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import MappingProxyType

import pytest
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
from surgical_agent.research.gate.policy import EvidenceThresholdPolicy
from surgical_agent.research.gate.reliability import ReliabilityGatePolicy
from surgical_agent.research.signals.contracts import PhaseTransitionGraph
from surgical_agent.systems.api_dataset_system import DatasetApiPipelineSystem
from surgical_agent.tracking.contracts import PredictedTrackFrame, PredictedTrackVideo
from surgical_agent.tracking.predicted_provider import (
    PrecomputedPredictedTrackProvider,
)

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
    pipeline_profile: str = "single_pass",
    gate_artifact: Path | None = None,
    context_profile: str | None = None,
    phase_transition_graph: PhaseTransitionGraph | None = None,
    track_provider: PrecomputedPredictedTrackProvider | None = None,
    event_memory_enabled: bool | None = None,
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
            pipeline_profile=pipeline_profile,
            gate_artifact=gate_artifact,
            context_profile=context_profile,
            phase_transition_graph=phase_transition_graph,
            track_provider=track_provider,
            event_memory_enabled=event_memory_enabled,
        ),
        transport,
    )


def _write_final_gate(
    path: Path,
    *,
    threshold: float,
    scope: str = "joint",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "benefit_gate_linear_v1",
                "gate_stage": "final_g1",
                "source_split": "training",
                "feature_order": ["instrument.candidate_ambiguity.value"],
                "weights_by_scope": {scope: [1.0]},
                "bias_by_scope": {scope: 0.0},
                "scope_order": [scope],
                "threshold": threshold,
                "training_dataset_ids": ["cholectrack20_train_v1"],
                "training_recipe_id": "gate_train_v1",
                "normalization_version": "evidence_frame_v1_raw",
                "rollout_policy_id": "g0_policy_v1",
            }
        ),
        encoding="utf-8",
    )
    return path


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


def test_dataset_api_system_flushes_event_reports_before_frame_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection_with_pngs(
        tmp_path / "media",
        (("VID11", 1), ("VID11", 2)),
    )
    system, _transport = _mock_system(tmp_path, provider_call_limit=2)
    real_finalize = system.writer.finalize

    def assert_report_manifest_first(*args: object, **kwargs: object) -> Path:
        assert (tmp_path / "run/event_report_manifest.json").is_file()
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(system.writer, "finalize", assert_report_manifest_first)

    result = system.run(selection, run_id="dataset-mock")

    assert result.report_manifest_path == tmp_path / "run/event_report_manifest.json"
    assert result.report_count == 1
    assert result.report_mode == "template_report"
    report = json.loads(
        (tmp_path / "run/event_reports.jsonl").read_text(encoding="utf-8")
    )
    assert report["video_id"] == "VID11"
    assert report["trigger_reasons"] == ["video_end"]
    assert report["included_final_statuses"] == ["Accepted"]


def test_dataset_api_system_reports_frame_progress_on_one_line(tmp_path: Path) -> None:
    """Removing frame progress would make long API rollouts appear stalled."""

    selection = _selection_with_pngs(
        tmp_path / "media",
        (("VID11", 1), ("VID11", 2), ("VID30", 1)),
    )
    system, _transport = _mock_system(tmp_path, provider_call_limit=3)
    sink = StringIO()

    system.run(
        selection,
        run_id="dataset-mock",
        progress_enabled=True,
        progress_file=sink,
    )

    rendered = sink.getvalue()
    assert "API inference" in rendered
    assert "3/3" in rendered
    assert "VID30:1" in rendered
    assert rendered.count("\n") == 1


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


def test_workflow_context_activates_train_graph_only_after_first_commit(
    tmp_path: Path,
) -> None:
    selection = _selection_with_pngs(
        tmp_path / "media",
        (("VID11", 1), ("VID11", 2)),
    )
    graph = PhaseTransitionGraph(
        transitions=((1, 1),),
        source_video_ids=("VID02",),
        version="phase_transition_train_v1",
        sha256="a" * 64,
    )
    system, _transport = _mock_system(
        tmp_path,
        provider_call_limit=2,
        context_profile="workflow",
        phase_transition_graph=graph,
    )

    system.run(selection, run_id="dataset-mock")

    records = [
        json.loads(line)
        for line in (tmp_path / "run/evidence/VID11.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    first = records[0]["task_values"]["phase"]["phase_change_anomaly"]
    second = records[1]["task_values"]["phase"]["phase_change_anomaly"]
    assert first["available"] is False
    assert second == {
        "available": True,
        "source": "frozen_phase_transition_graph",
        "source_max_frame_id": 2,
        "value": 1.0,
    }
    snapshot = system.pipeline.components.workflow_store.snapshot()
    assert snapshot["recent_finalized_phases"] == ("0", "0")
    assert snapshot["source_max_frame_id"] == 2


def test_context_profiles_are_real_runtime_variants(tmp_path: Path) -> None:
    frames, _ = _mock_system(
        tmp_path / "frames",
        provider_call_limit=1,
        context_profile="frames_only",
    )
    workflow, _ = _mock_system(
        tmp_path / "workflow",
        provider_call_limit=1,
        context_profile="workflow",
    )

    assert frames.context_profile == "frames_only"
    assert frames.pipeline.components.workflow_store.snapshot()["status"] == (
        "NOOP_TRACEABLE"
    )
    assert workflow.context_profile == "workflow"
    assert workflow.pipeline.components.workflow_store.snapshot()["status"] == (
        "FINALIZED_ONLY"
    )
    track_provider = PrecomputedPredictedTrackProvider(
        provider_name="test_tracker",
        source_model_identifier="test_model",
        checkpoint_sha256="a" * 64,
        inference_mode="online_forward_only",
        producer_version="predicted_track_producer_v1",
        dataset_repair_manifest_sha256="b" * 64,
        inference_config_sha256="c" * 64,
        artifact_sha256="d" * 64,
        videos={
            "VID01": PredictedTrackVideo(
                video_id="VID01",
                source_split=DatasetSplit.VALIDATION,
                frames=(PredictedTrackFrame(frame_id=1, tracks=()),),
            )
        },
    )
    track, _ = _mock_system(
        tmp_path / "track-only",
        provider_call_limit=1,
        context_profile="track_only",
        track_provider=track_provider,
    )
    assert track.pipeline.components.workflow_store.snapshot()["status"] == (
        "NOOP_TRACEABLE"
    )
    assert track.pipeline.components.track_provider is track_provider
    with pytest.raises(ValueError, match="predicted-track"):
        _mock_system(
            tmp_path / "track",
            provider_call_limit=1,
            context_profile="track_workflow",
        )


def test_event_memory_switch_selects_noop_or_bounded_store(tmp_path: Path) -> None:
    disabled, _ = _mock_system(
        tmp_path / "disabled",
        provider_call_limit=1,
        pipeline_profile="always_verify",
        event_memory_enabled=False,
    )
    enabled, _ = _mock_system(
        tmp_path / "enabled",
        provider_call_limit=1,
        pipeline_profile="always_verify",
        event_memory_enabled=True,
    )

    assert disabled.pipeline.components.event_memory.snapshot()["status"] == (
        "NOOP_TRACEABLE"
    )
    assert enabled.pipeline.components.event_memory.snapshot()["status"] == (
        "FINALIZED_ONLY"
    )


def test_single_pass_uses_the_same_default_memory_input_as_main_profiles(
    tmp_path: Path,
) -> None:
    system, _ = _mock_system(tmp_path, provider_call_limit=1)

    assert system.pipeline.components.event_memory.snapshot()["status"] == (
        "FINALIZED_ONLY"
    )


def test_always_verify_profile_uses_one_extra_cached_call_per_frame(
    tmp_path: Path,
) -> None:
    selection = _selection_with_pngs(
        tmp_path / "media",
        (("VID11", 1), ("VID11", 2)),
    )
    system, transport = _mock_system(
        tmp_path,
        provider_call_limit=4,
        pipeline_profile="always_verify",
    )

    result = system.run(selection, run_id="dataset-mock")

    assert transport.provider_call_count == 4
    assert [record.gate_action for record in result.predictions] == [
        "VERIFY",
        "VERIFY",
    ]
    assert [record.verification_status for record in result.predictions] == [
        "VERIFIED_KEEP",
        "VERIFIED_KEEP",
    ]
    assert result.verification_summary["gate_verify_count"] == 2
    assert result.verification_summary["fallback_keep_count"] == 0
    assert result.verification_summary["verification_success_rate"] == 1.0
    assert result.verification_summary["verification_contract_satisfied"] is True
    assert system.pipeline.components.workflow_store.snapshot()["status"] == (
        "FINALIZED_ONLY"
    )
    assert system.pipeline.components.event_memory.snapshot()["status"] == (
        "FINALIZED_ONLY"
    )


def test_rule_gate_and_selective_verify_route_to_distinct_policies(
    tmp_path: Path,
) -> None:
    rule, _ = _mock_system(
        tmp_path / "rule",
        provider_call_limit=2,
        pipeline_profile="rule_gate",
    )
    selective, _ = _mock_system(
        tmp_path / "selective",
        provider_call_limit=2,
        pipeline_profile="selective_verify",
    )

    assert isinstance(rule.pipeline.components.gate_policy, EvidenceThresholdPolicy)
    assert isinstance(
        selective.pipeline.components.gate_policy,
        ReliabilityGatePolicy,
    )


def test_shared_and_cascade_profiles_publish_backbone_metadata(
    tmp_path: Path,
) -> None:
    shared, _ = _mock_system(
        tmp_path / "shared",
        provider_call_limit=1,
        pipeline_profile="single_pass",
    )
    shared_result = shared.run(
        _selection_with_pngs(tmp_path / "shared-media", (("VID11", 1),)),
        run_id="dataset-mock",
    )
    base_config = replace(
        load_api_config(MOCK_CONFIG),
        requested_model_identifier="openai/gpt-5.6-luna",
        synthetic_input_required=False,
        data_upload_authorized=True,
    )
    verification_config = replace(
        base_config,
        requested_model_identifier="openai/gpt-5.6-sol",
    )
    transport = MockProviderTransport()
    cascade = DatasetApiPipelineSystem(
        client=CachedMultimodalApiClient(
            transport=transport,
            cache=FileApiCache(tmp_path / "cascade-cache"),
            usage=UsageLedger(tmp_path / "cascade-usage.jsonl"),
            validator=build_validator(base_config),
            provider_call_budget=ProviderCallBudget(2),
        ),
        config=base_config,
        verification_config=verification_config,
        writer=FrameResultWriter(tmp_path / "cascade-run", run_id="cascade"),
        media_loader=CausalApiMediaLoader(),
        pipeline_profile="cascade_verify",
        context_profile="workflow",
    )
    cascade_result = cascade.run(
        _selection_with_pngs(tmp_path / "cascade-media", (("VID11", 1),)),
        run_id="cascade",
    )

    assert shared_result.backbone_policy == "shared"
    assert shared_result.initial_model_requested == "mock-joint-perception-v1"
    assert shared_result.verification_model_requested == "mock-joint-perception-v1"
    assert shared_result.main_profile_backbone_match is True
    assert transport.provider_call_count == 2
    assert cascade_result.backbone_policy == "cascade_efficiency"
    assert cascade_result.initial_model_requested == "openai/gpt-5.6-luna"
    assert cascade_result.verification_model_requested == "openai/gpt-5.6-sol"
    assert cascade_result.main_profile_backbone_match is False


def test_cascade_rejects_non_shared_gateway_or_unapproved_model_pair(
    tmp_path: Path,
) -> None:
    initial = replace(
        load_api_config(MOCK_CONFIG),
        requested_model_identifier="openai/gpt-5.6-luna",
    )
    client = CachedMultimodalApiClient(
        transport=MockProviderTransport(),
        cache=FileApiCache(tmp_path / "cache"),
        usage=UsageLedger(tmp_path / "usage.jsonl"),
        validator=build_validator(initial),
    )
    wrong_model = replace(initial, requested_model_identifier="other/model")
    wrong_endpoint = replace(
        initial,
        requested_model_identifier="openai/gpt-5.6-sol",
        endpoint_identifier="mock://different",
    )

    for verification_config, message in (
        (wrong_model, "Luna.*Sol"),
        (wrong_endpoint, "provider and endpoint"),
    ):
        with pytest.raises(ValueError, match=message):
            DatasetApiPipelineSystem(
                client=client,
                config=initial,
                verification_config=verification_config,
                writer=FrameResultWriter(tmp_path / message, run_id="cascade"),
                media_loader=CausalApiMediaLoader(),
                pipeline_profile="cascade_verify",
            )


def test_learned_gate_profile_requires_a_frozen_final_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gate_artifact"):
        _mock_system(
            tmp_path,
            provider_call_limit=2,
            pipeline_profile="learned_gate",
        )


def test_learned_gate_profile_loads_artifact_and_routes_verification(
    tmp_path: Path,
) -> None:
    selection = _selection_with_pngs(
        tmp_path / "media",
        (("VID11", 1),),
    )
    artifact = _write_final_gate(tmp_path / "gate.json", threshold=-1.0)
    system, transport = _mock_system(
        tmp_path,
        provider_call_limit=2,
        pipeline_profile="learned_gate",
        gate_artifact=artifact,
    )

    result = system.run(selection, run_id="dataset-mock")

    assert transport.provider_call_count == 2
    assert result.predictions[0].gate_action == "VERIFY"
    assert result.predictions[0].verification_status == "VERIFIED_KEEP"


def test_learned_gate_profile_rejects_artifact_without_matching_specialist(
    tmp_path: Path,
) -> None:
    artifact = _write_final_gate(
        tmp_path / "gate.json",
        threshold=-1.0,
        scope="interaction",
    )

    with pytest.raises(ValueError, match="joint verifier"):
        _mock_system(
            tmp_path,
            provider_call_limit=2,
            pipeline_profile="learned_gate",
            gate_artifact=artifact,
        )
