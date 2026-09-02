"""Strict CLI and artifact contracts for CholecTrack20 API rollouts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from PIL import Image

import scripts.run_dataset_api_pipeline as dataset_cli
import surgical_agent.inference.frame_result_writer as frame_writer_module
from scripts.run_dataset_api_pipeline import (
    build_parser,
    run,
    run_dataset_api_rollout,
    validate_cli_selection,
)
from surgical_agent.api.contracts import (
    ApiRequest,
    CompletionTokenDetails,
    ProviderResponse,
)
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError, ApiTransportError
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.config.loader import load_api_config, load_yaml
from surgical_agent.data.api_rollout_selection import RolloutSelection
from surgical_agent.data.dataset import PNG_ALIGNMENT_VERSION, DatasetContractError
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.inference.writer import ArtifactWriteError
from surgical_agent.perception.schema import (
    COMPACT_TASK_LAYOUT,
    GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
    RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOCK_DATASET_CONFIG = PROJECT_ROOT / "configs/perception/joint_mock_dataset.yaml"
REAL_DATASET_CONFIG = PROJECT_ROOT / "configs/perception/joint_openrouter_dataset.yaml"
LATENCY_DATASET_CONFIG = (
    PROJECT_ROOT / "configs/perception/joint_openrouter_latency_dataset.yaml"
)


class InjectedOpenRouterTransport:
    provider = "openrouter"
    endpoint_identifier = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        self.call_count = 0
        self.generation_parameters: list[dict[str, object]] = []

    def send(self, request: ApiRequest) -> ProviderResponse:
        self.call_count += 1
        self.generation_parameters.append(dict(request.generation_parameters))
        input_text = request.payload.get("input_text")
        decoded = json.loads(input_text) if isinstance(input_text, str) else {}
        frame_id = int(decoded["target_frame_id"])
        reliability_v2 = (
            request.response_schema_version
            == RELIABILITY_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
        )
        gate_owned = (
            request.response_schema_version
            == GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
        )
        payload: dict[str, object] = {"schema_version": request.response_schema_version}
        for task, count in COMPACT_TASK_LAYOUT:
            topk = [
                {
                    "id": index,
                    "confidence" if reliability_v2 else "score": (
                        1.0 - index / (count + 1)
                    ),
                }
                for index in range(count)
            ]
            payload[task] = (
                {"selected_id": 0, "topk": topk}
                if task == "phase"
                else {"selected_ids": [0], "topk": topk}
            )
        if reliability_v2:
            payload["uncertainty"] = []
        elif not gate_owned:
            payload["evidence_refs"] = [
                {"frame_id": frame_id, "code": "CURRENT_VISUAL_SUPPORT"}
            ]
            payload["self_reported_confidence"] = {
                task: 0.8 for task, _count in COMPACT_TASK_LAYOUT
            }
        return ProviderResponse(
            provider=self.provider,
            returned_model_identifier="openai/gpt-5.6-sol:injected",
            parsed_payload=payload,
            input_tokens=101,
            output_tokens=37,
            total_tokens=138,
            completion_tokens_details=CompletionTokenDetails(reasoning_tokens=0),
            visible_output_tokens=37,
            time_to_first_token_ms=1.0,
            total_latency_ms=2.0,
            image_count=len(request.images),
            provider_request_id=f"injected-{self.call_count}",
            provider_cost=0.00125,
        )


class LeakingFailureTransport:
    provider = "openrouter"
    endpoint_identifier = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, leak_path: Path, secret_text: str) -> None:
        self.leak_path = leak_path
        self.secret_text = secret_text
        self.call_count = 0

    def send(self, request: ApiRequest) -> ProviderResponse:
        del request
        self.call_count += 1
        self.leak_path.parent.mkdir(parents=True, exist_ok=True)
        self.leak_path.write_text(self.secret_text, encoding="utf-8")
        raise ApiTransportError(
            "injected failure after persistence",
            code="injected_failure",
            retryable=False,
        )


def _selection_with_two_pngs(media_root: Path) -> RolloutSelection:
    media_root.mkdir(parents=True)
    samples: list[InferenceSample] = []
    for frame_id in (1, 2):
        path = media_root / f"frame_{frame_id}.png"
        Image.new("RGB", (8, 6), (frame_id, frame_id, frame_id)).save(path)
        causal_ids = tuple(range(1, frame_id + 1))
        samples.append(
            InferenceSample(
                video_id="VID30",
                target_frame_id=frame_id,
                causal_frame_ids=causal_ids,
                media_refs=tuple(
                    str(media_root / f"frame_{value}.png") for value in causal_ids
                ),
                source_split=DatasetSplit.VALIDATION,
                alignment_version=PNG_ALIGNMENT_VERSION,
            )
        )
    return RolloutSelection(
        mode="engineering",
        split=None,
        video_ids=("VID30",),
        samples=tuple(samples),
        frame_counts=MappingProxyType({"VID30": 2}),
    )


def _persisted_text(*roots: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ).lower()


def test_cli_exposes_explicit_no_training_verification_profiles() -> None:
    args = build_parser().parse_args(
        [
            "--mode",
            "engineering",
            "--video-id",
            "VID30",
            "--max-frames",
            "2",
            "--pipeline-profile",
            "always_verify",
        ]
    )

    assert args.pipeline_profile == "always_verify"
    assert args.evidence_threshold == 0.75
    assert (
        dataset_cli._resolve_call_limit(
            "exact-selection",
            selection_count=2,
            calls_per_state=2,
        )
        == 4
    )


def test_cli_exposes_selective_cascade_and_event_report_controls() -> None:
    base = ["--mode", "engineering", "--video-id", "VID30", "--max-frames", "1"]
    parser = build_parser()

    selective = parser.parse_args(
        [
            *base,
            "--pipeline-profile",
            "selective_verify",
            "--report-mode",
            "template_report",
            "--report-window-frames",
            "12",
        ]
    )
    cascade = parser.parse_args([*base, "--pipeline-profile", "cascade_verify"])

    assert selective.report_mode == "template_report"
    assert selective.report_window_frames == 12
    assert cascade.pipeline_profile == "cascade_verify"
    assert cascade.report_mode == "template_report"
    assert cascade.report_window_frames == 30


def test_profile_call_budget_is_the_exact_n_or_two_n_reservation() -> None:
    assert (
        dataset_cli._resolve_profile_call_limit(
            "exact-selection", pipeline_profile="single_pass", selection_count=3
        )
        == 3
    )
    for profile in ("always_verify", "selective_verify", "cascade_verify"):
        assert (
            dataset_cli._resolve_profile_call_limit(
                "exact-selection", pipeline_profile=profile, selection_count=3
            )
            == 6
        )
    with pytest.raises(ApiContractError, match="exact profile reservation"):
        dataset_cli._resolve_profile_call_limit(
            3, pipeline_profile="always_verify", selection_count=2
        )


def test_dataset_cli_enables_progress_by_default_and_can_disable_it() -> None:
    parser = build_parser()

    base = ["--mode", "engineering", "--video-id", "VID30", "--max-frames", "1"]
    assert parser.parse_args(base).no_progress is False
    assert parser.parse_args([*base, "--no-progress"]).no_progress is True


def test_cli_exposes_explicit_model_key_and_causal_parallelism_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "engineering",
            "--video-id",
            "VID30",
            "--max-frames",
            "1",
            "--api-key",
            "temporary-test-key",
            "--model",
            "openai/gpt-5.6-sol",
            "--parallelism",
            "1",
        ]
    )

    assert args.api_key == "temporary-test-key"
    assert args.api_key_file is None
    assert args.model == "openai/gpt-5.6-sol"
    assert args.parallelism == 1
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--mode",
                "engineering",
                "--video-id",
                "VID30",
                "--max-frames",
                "1",
                "--api-key",
                "temporary-test-key",
                "--api-key-file",
                "docs/API.txt",
            ]
        )


def test_cli_model_override_preserves_the_approved_experiment_backbone() -> None:
    config = load_api_config(REAL_DATASET_CONFIG)

    overridden = dataset_cli._apply_cli_model_override(
        config,
        mode="engineering",
        model=" openai/gpt-5.6-sol ",
    )

    assert overridden.requested_model_identifier == "openai/gpt-5.6-sol"
    with pytest.raises(ApiContractError, match="paper mode forbids"):
        dataset_cli._apply_cli_model_override(
            config,
            mode="paper",
            model="openai/gpt-5.6-sol",
        )
    with pytest.raises(ApiContractError, match="approves only"):
        dataset_cli._apply_cli_model_override(
            config,
            mode="engineering",
            model="openai/gpt-5.6-luna",
        )


def test_parallelism_rejects_state_races_inside_one_causal_video() -> None:
    dataset_cli._validate_parallelism(1)

    with pytest.raises(ApiContractError, match="positive integer"):
        dataset_cli._validate_parallelism(0)
    with pytest.raises(ApiContractError, match="causal video stream"):
        dataset_cli._validate_parallelism(2)


def test_experiment_yaml_resolves_real_context_and_memory_switches() -> None:
    source = PROJECT_ROOT / "configs/ablations/workflow_only.yaml"

    resolved = dataset_cli._resolve_context_runtime(
        context_profile="auto",
        event_memory="auto",
        phase_transition_graph=None,
        predicted_track_artifact=None,
        experiment_config=source,
    )

    assert resolved[:2] == ("workflow", False)
    assert (
        resolved[2]
        == (PROJECT_ROOT / "artifacts/training/phase_transition_graph.json").resolve()
    )
    assert resolved[3] is None
    assert resolved[4] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_cli_rejects_switch_that_conflicts_with_experiment_yaml() -> None:
    with pytest.raises(ApiContractError, match="context-profile conflicts"):
        dataset_cli._resolve_context_runtime(
            context_profile="frames_only",
            event_memory="auto",
            phase_transition_graph=None,
            predicted_track_artifact=None,
            experiment_config=(PROJECT_ROOT / "configs/ablations/workflow_only.yaml"),
        )


def test_context_profiles_are_explicit_and_track_mode_requires_predictions(
    tmp_path: Path,
) -> None:
    selection = _selection_with_two_pngs(tmp_path / "media")
    resolved, graph, provider = dataset_cli._load_context_components(
        pipeline_profile="single_pass",
        context_profile="auto",
        phase_transition_graph=None,
        predicted_track_artifact=None,
        dataset_root=tmp_path,
        selection=selection,
    )
    assert (resolved, graph, provider) == ("workflow", None, None)

    with pytest.raises(ApiContractError, match="requires predicted-track"):
        dataset_cli._load_context_components(
            pipeline_profile="single_pass",
            context_profile="track_workflow",
            phase_transition_graph=None,
            predicted_track_artifact=None,
            dataset_root=tmp_path,
            selection=selection,
        )
    with pytest.raises(ApiContractError, match="requires predicted-track"):
        dataset_cli._load_context_components(
            pipeline_profile="single_pass",
            context_profile="track_only",
            phase_transition_graph=None,
            predicted_track_artifact=None,
            dataset_root=tmp_path,
            selection=selection,
        )


def test_phase_graph_must_equal_graph_rebuilt_from_current_training_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection_with_two_pngs(tmp_path / "media")
    declared = dataset_cli.PhaseTransitionGraph(
        transitions=((0, 0),),
        source_video_ids=("VID01",),
        version="phase_transition_train_v1",
        sha256="a" * 64,
    )
    rebuilt = dataset_cli.PhaseTransitionGraph(
        transitions=((0, 0), (0, 1), (1, 1)),
        source_video_ids=("VID01",),
        version="phase_transition_train_v1",
        sha256="b" * 64,
    )
    monkeypatch.setattr(
        dataset_cli,
        "load_phase_transition_graph",
        lambda _path: declared,
    )
    monkeypatch.setattr(
        dataset_cli,
        "CholecTrack20DatasetAdapter",
        lambda _root: object(),
    )
    monkeypatch.setattr(
        dataset_cli,
        "build_phase_transition_graph_from_training_adapter",
        lambda _adapter: rebuilt,
    )

    with pytest.raises(ApiContractError, match="does not match current Training"):
        dataset_cli._load_context_components(
            pipeline_profile="rule_gate",
            context_profile="workflow",
            phase_transition_graph=tmp_path / "declared.json",
            predicted_track_artifact=None,
            dataset_root=tmp_path,
            selection=selection,
        )


def test_track_artifact_must_match_current_dataset_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    provider = SimpleNamespace(dataset_repair_manifest_sha256="a" * 64)
    monkeypatch.setattr(
        dataset_cli,
        "PrecomputedPredictedTrackProvider",
        SimpleNamespace(from_json=lambda _path: provider),
    )

    with pytest.raises(ApiContractError, match="invalid predicted-track"):
        dataset_cli._load_context_components(
            pipeline_profile="rule_gate",
            context_profile="track_workflow",
            phase_transition_graph=None,
            predicted_track_artifact=tmp_path / "tracks.json",
            dataset_root=dataset_root,
            selection=selection,
        )


def test_cli_requires_an_artifact_for_the_learned_gate_profile(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--mode",
            "engineering",
            "--video-id",
            "VID30",
            "--max-frames",
            "1",
            "--pipeline-profile",
            "learned_gate",
        ]
    )

    with pytest.raises(ApiContractError, match="gate-artifact"):
        dataset_cli._validate_runtime_profile(
            args.pipeline_profile,
            evidence_threshold=args.evidence_threshold,
            gate_artifact=args.gate_artifact,
        )

    artifact_path = tmp_path / "gate.json"
    artifact_path.write_text("{}", encoding="utf-8")
    parsed = build_parser().parse_args(
        [
            "--mode",
            "engineering",
            "--video-id",
            "VID30",
            "--max-frames",
            "1",
            "--pipeline-profile",
            "learned_gate",
            "--gate-artifact",
            str(artifact_path),
        ]
    )
    dataset_cli._validate_runtime_profile(
        parsed.pipeline_profile,
        evidence_threshold=parsed.evidence_threshold,
        gate_artifact=parsed.gate_artifact,
    )


def _install_canonical_adapter(
    monkeypatch: pytest.MonkeyPatch,
    selection: RolloutSelection,
) -> None:
    samples_by_video = {
        video_id: tuple(
            sample for sample in selection.samples if sample.video_id == video_id
        )
        for video_id in selection.video_ids
    }
    splits = {
        video_id: samples_by_video[video_id][0].source_split
        for video_id in selection.video_ids
    }

    class CanonicalAdapter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.entries = {
                video_id: SimpleNamespace(split=split)
                for video_id, split in splits.items()
            }

        def iter_inference_video(
            self,
            video_id: str,
            *,
            max_samples: int | None = None,
        ) -> object:
            samples = samples_by_video[video_id]
            yield from samples if max_samples is None else samples[:max_samples]

    monkeypatch.setattr(dataset_cli, "CholecTrack20DatasetAdapter", CanonicalAdapter)


def test_real_dataset_cli_requires_double_upload_authorization(
    tmp_path: Path,
) -> None:
    """Removing the CLI consent gate could upload dataset frames accidentally."""

    args = build_parser().parse_args(
        [
            "--mode",
            "engineering",
            "--video-id",
            "VID30",
            "--max-frames",
            "1",
            "--dataset-root",
            str(tmp_path),
            "--config",
            str(REAL_DATASET_CONFIG),
            "--api-key-file",
            str(tmp_path / "API.txt"),
            "--output-root",
            str(tmp_path / "out"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--run-id",
            "real-one",
            "--max-provider-calls",
            "exact-selection",
        ]
    )

    with pytest.raises(ApiContractError, match="authorize-data-upload"):
        run(args)

    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "cache").exists()


def test_paper_cli_rejects_max_frames() -> None:
    """Allowing paper truncation would silently produce incomplete evaluation."""

    args = build_parser().parse_args(
        ["--mode", "paper", "--split", "validation", "--max-frames", "1"]
    )

    with pytest.raises(ApiContractError, match="paper mode forbids truncation"):
        validate_cli_selection(args)


def test_injected_mock_rollout_persists_safe_pairs_and_replays_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing shared-cache replay or safe persistence would duplicate calls or leak data."""

    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    repair_manifest = dataset_root / "repair_manifest.json"
    repair_manifest.write_text('{"private": "manifest-content"}\n', encoding="utf-8")
    expected_digest = hashlib.sha256(repair_manifest.read_bytes()).hexdigest()
    cache_root = tmp_path / "cache"
    config = load_api_config(MOCK_DATASET_CONFIG)
    _install_canonical_adapter(monkeypatch, selection)

    first_transport = MockProviderTransport()
    first_output = tmp_path / "first"
    first = run_dataset_api_rollout(
        config=config,
        selection=selection,
        dataset_root=dataset_root,
        output_dir=first_output,
        cache_root=cache_root,
        run_id="mock-first",
        max_provider_calls=2,
        transport=first_transport,
    )

    second_transport = MockProviderTransport()
    second_output = tmp_path / "second"
    second = run_dataset_api_rollout(
        config=config,
        selection=selection,
        dataset_root=dataset_root,
        output_dir=second_output,
        cache_root=cache_root,
        run_id="mock-second",
        max_provider_calls=2,
        transport=second_transport,
    )

    expected_keys = {
        "schema_version",
        "status",
        "run_id",
        "mode",
        "split",
        "video_ids",
        "expected_frame_counts",
        "completed_frame_counts",
        "provider",
        "provider_routing_profile",
        "model_requested",
        "models_returned",
        "prompt_version",
        "response_schema_version",
        "causal_window",
        "pipeline_profile",
        "backbone_policy",
        "initial_model_requested",
        "verification_model_requested",
        "main_profile_backbone_match",
        "context_profile",
        "event_memory_enabled",
        "context_experiment_sha256",
        "phase_transition_graph",
        "predicted_track_artifact_sha256",
        "predicted_track_provenance",
        "evidence_threshold",
        "gate_artifact_sha256",
        "verification_summary",
        "final_status_counts",
        "memory_action_counts",
        "report_manifest_path",
        "causal_window_audit_path",
        "report_count",
        "report_mode",
        "repair_manifest_sha256",
        "alignment_versions",
        "usage",
        "telemetry_summary",
        "cache_entry_count",
        "track20_image_uploaded",
        "paper_metric_eligible",
        "manifest_file",
    }
    assert set(first) == expected_keys
    assert first["expected_frame_counts"] == {"VID30": 2}
    assert first["completed_frame_counts"] == {"VID30": 2}
    assert first["repair_manifest_sha256"] == expected_digest
    assert first["track20_image_uploaded"] is False
    assert first["provider_routing_profile"] is None
    assert first["paper_metric_eligible"] is False
    assert first["context_profile"] == "workflow"
    assert first["event_memory_enabled"] is True
    assert first["context_experiment_sha256"] is None
    assert first["phase_transition_graph"] is None
    assert first["predicted_track_artifact_sha256"] is None
    assert first["predicted_track_provenance"] is None
    assert first["report_manifest_path"] == "event_report_manifest.json"
    assert first["report_count"] == 1
    assert first["report_mode"] == "template_report"
    assert first["backbone_policy"] == "shared"
    assert first["initial_model_requested"] == "mock-joint-perception-v1"
    assert first["verification_model_requested"] == "mock-joint-perception-v1"
    assert first["main_profile_backbone_match"] is True
    assert first["final_status_counts"] == {"Accepted": 2}
    assert first["memory_action_counts"] == {"WRITE_SHORT_TERM": 2}
    telemetry = first["telemetry_summary"]
    assert telemetry["schema_version"] == "api_rollout_telemetry_v1"
    assert telemetry["prompt_tokens"] == 24
    assert telemetry["completion_tokens"] == 14
    assert telemetry["reasoning_tokens"] == 0
    assert telemetry["visible_output_tokens"] == 14
    assert telemetry["time_to_first_token_ms"]["count"] == 2
    assert telemetry["total_latency_ms"]["count"] == 2
    assert first["usage"]["logical_calls"] == 2
    assert first["usage"]["provider_calls"] == 2
    assert first_transport.provider_call_count == 2
    assert second["usage"]["logical_calls"] == 2
    assert second["usage"]["provider_calls"] == 0
    assert second["usage"]["cache_hits"] == 2
    assert second["telemetry_summary"]["time_to_first_token_ms"]["count"] == 0
    assert second["telemetry_summary"]["total_latency_ms"]["count"] == 0
    assert second_transport.provider_call_count == 0
    assert first["cache_entry_count"] == second["cache_entry_count"] == 2

    for output_dir in (first_output, second_output):
        prediction_lines = (
            (output_dir / "predictions/VID30.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        evidence_lines = (
            (output_dir / "evidence/VID30.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        usage_lines = (
            (output_dir / "api_usage.jsonl").read_text(encoding="utf-8").splitlines()
        )
        assert len(prediction_lines) == len(evidence_lines) == len(usage_lines) == 2

    persisted = _persisted_text(first_output, second_output, cache_root)
    assert str(dataset_root.resolve()).lower() not in persisted
    assert "manifest-content" not in persisted
    for forbidden in (
        "api_key",
        "raw_response",
        "system_text",
        "input_text",
        "data:image",
        "base64",
        "image_bytes",
        "ground_truth",
        "evaluation_target",
    ):
        assert forbidden not in persisted

    assert (
        json.loads(
            (first_output / "dataset_rollout_artifact.json").read_text(encoding="utf-8")
        )
        == first
    )


def test_injected_real_rollout_marks_dataset_image_upload_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marking injected real data as mock would conceal an authorized upload path."""

    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    _install_canonical_adapter(monkeypatch, selection)
    transport = InjectedOpenRouterTransport()
    secret_text = "-".join(  # noqa: FLY002 - keep scan fixture non-contiguous
        ("injected", "dataset", "credential")
    )
    real_scan_secret = dataset_cli._scan_secret

    def observe_incomplete_scan_state(
        secret: SecretValue,
        *,
        output_dir: Path,
        cache_root: Path,
    ) -> None:
        status = json.loads((output_dir / "run_status.json").read_text())
        assert status["status"] == "INCOMPLETE"
        real_scan_secret(
            secret,
            output_dir=output_dir,
            cache_root=cache_root,
        )

    monkeypatch.setattr(dataset_cli, "_scan_secret", observe_incomplete_scan_state)

    artifact = run_dataset_api_rollout(
        config=load_api_config(REAL_DATASET_CONFIG),
        selection=selection,
        dataset_root=dataset_root,
        output_dir=tmp_path / "real-output",
        cache_root=tmp_path / "real-cache",
        run_id="real-injected",
        max_provider_calls=2,
        api_key=SecretValue(secret_text),
        authorize_data_upload=True,
        transport=transport,
    )

    assert transport.call_count == 2
    assert transport.generation_parameters == [
        {"max_output_tokens": 1536, "reasoning": {"effort": "none"}},
        {"max_output_tokens": 1536, "reasoning": {"effort": "none"}},
    ]
    assert artifact["status"] == "REAL_RESPONSE_RECEIVED"
    assert artifact["track20_image_uploaded"] is True
    assert artifact["paper_metric_eligible"] is False
    assert secret_text not in _persisted_text(
        tmp_path / "real-output", tmp_path / "real-cache"
    )


def test_dataset_rollout_freezes_compact_json_budget_at_lowest_reasoning_effort() -> (
    None
):
    """The real dataset configuration must preserve the compact lowest-effort budget."""

    real = load_api_config(REAL_DATASET_CONFIG)
    latency = load_api_config(LATENCY_DATASET_CONFIG)
    mock = load_api_config(MOCK_DATASET_CONFIG)

    assert real.prompt_version == GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    assert (
        real.response_schema_version
        == GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    )
    assert mock.prompt_version == GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    assert (
        mock.response_schema_version
        == GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    )
    assert dict(real.generation_parameters) == {
        "max_output_tokens": 1536,
        "reasoning": {"effort": "none"},
    }
    assert real.provider_options["routing_profile"] == "strict_openai"
    assert latency.provider_options["routing_profile"] == "latency_fallback"
    assert latency.generation_parameters == real.generation_parameters
    assert latency.max_causal_frames == real.max_causal_frames == 6
    assert latency.max_api_images == real.max_api_images == 3
    assert dict(mock.generation_parameters) == {"max_output_tokens": 4096}
    dataset_cli._require_exact_dataset_config(
        real,
        has_credential=True,
        authorize_data_upload=True,
        selection_mode="paper",
    )
    dataset_cli._require_exact_dataset_config(
        latency,
        has_credential=True,
        authorize_data_upload=True,
        selection_mode="engineering",
    )
    dataset_cli._require_exact_dataset_config(
        mock,
        has_credential=False,
        authorize_data_upload=True,
    )
    with pytest.raises(ApiContractError, match="strict OpenAI routing profile"):
        dataset_cli._require_exact_dataset_config(
            latency,
            has_credential=True,
            authorize_data_upload=True,
            selection_mode="paper",
        )

    reasoning_enabled = replace(
        real,
        generation_parameters={
            "max_output_tokens": 1536,
            "reasoning": {"effort": "low"},
        },
    )
    with pytest.raises(ApiContractError, match="exact generation settings"):
        dataset_cli._require_exact_dataset_config(
            reasoning_enabled,
            has_credential=True,
            authorize_data_upload=True,
        )


def test_luna_dataset_config_and_fair_experiment_profiles_are_explicit() -> None:
    luna = load_api_config(
        PROJECT_ROOT / "configs/perception/joint_openrouter_luna_dataset.yaml"
    )
    assert luna.requested_model_identifier == "openai/gpt-5.6-luna"
    assert luna.prompt_version == GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
    assert dict(luna.generation_parameters) == {
        "max_output_tokens": 1536,
        "reasoning": {"effort": "none"},
    }

    paths = (
        "configs/experiments/api_single_pass.yaml",
        "configs/experiments/v3_always_verify.yaml",
        "configs/experiments/v3_selective_verify.yaml",
    )
    configs = [load_yaml(PROJECT_ROOT / path) for path in paths]
    frozen = {
        (
            config["perception_config"],
            config["runtime"]["context_profile"],
            config["runtime"]["event_memory_enabled"],
            config["runtime"]["phase_transition_graph"],
            config["runtime"]["predicted_track_artifact"],
            tuple(sorted(config["report"].items())),
        )
        for config in configs
    }
    assert len(frozen) == 1
    assert {config["backbone_policy"] for config in configs} == {"shared"}
    assert {config["initial_model_requested"] for config in configs} == {
        "openai/gpt-5.6-sol"
    }
    cascade = load_yaml(PROJECT_ROOT / "configs/experiments/v3_cascade_efficiency.yaml")
    assert cascade["comparison_group"] == "efficiency_only"
    assert cascade["backbone_policy"] == "cascade_efficiency"


def test_llm_report_without_python_generator_fails_before_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    _install_canonical_adapter(monkeypatch, selection)
    transport = MockProviderTransport()

    with pytest.raises(ApiContractError, match="injected event-level generator"):
        run_dataset_api_rollout(
            config=load_api_config(MOCK_DATASET_CONFIG),
            selection=selection,
            dataset_root=dataset_root,
            output_dir=tmp_path / "output",
            cache_root=tmp_path / "cache",
            run_id="llm-rejected",
            max_provider_calls=2,
            transport=transport,
            report_mode="llm_report",
        )
    assert transport.provider_call_count == 0


def test_mock_always_verify_smoke_uses_two_calls_per_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    _install_canonical_adapter(monkeypatch, selection)
    transport = MockProviderTransport()

    artifact = run_dataset_api_rollout(
        config=load_api_config(MOCK_DATASET_CONFIG),
        selection=selection,
        dataset_root=dataset_root,
        output_dir=tmp_path / "output",
        cache_root=tmp_path / "cache",
        run_id="always-smoke",
        max_provider_calls=4,
        transport=transport,
        pipeline_profile="always_verify",
        context_profile="workflow",
    )

    assert transport.provider_call_count == 4
    assert artifact["usage"]["provider_calls"] == 4


def test_first_frame_failure_persists_sanitized_incomplete_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure before the first pair must still leave a categorized run record."""

    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    _install_canonical_adapter(monkeypatch, selection)

    class FailingLoader:
        def load(self, _sample: InferenceSample) -> object:
            raise DatasetContractError("sensitive first-frame detail")

    monkeypatch.setattr(dataset_cli, "CausalApiMediaLoader", FailingLoader)
    output_dir = tmp_path / "output"

    with pytest.raises(DatasetContractError, match="first-frame"):
        run_dataset_api_rollout(
            config=load_api_config(MOCK_DATASET_CONFIG),
            selection=selection,
            dataset_root=dataset_root,
            output_dir=output_dir,
            cache_root=tmp_path / "cache",
            run_id="first-frame-failure",
            max_provider_calls=2,
            transport=MockProviderTransport(),
        )

    assert json.loads((output_dir / "run_status.json").read_text()) == {
        "failure_category": "dataset_error",
        "run_id": "first-frame-failure",
        "schema_version": "frame_result_run_status_v1",
        "status": "INCOMPLETE",
    }
    assert "sensitive first-frame detail" not in _persisted_text(output_dir)


def test_mid_run_failure_persists_category_and_partial_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stopping after one pair must categorize the run without losing that pair."""

    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    _install_canonical_adapter(monkeypatch, selection)
    real_loader = dataset_cli.CausalApiMediaLoader

    class SecondFrameFailureLoader:
        def __init__(self) -> None:
            self.delegate = real_loader()

        def load(self, sample: InferenceSample) -> object:
            if sample.target_frame_id == 2:
                raise DatasetContractError("sensitive mid-run detail")
            return self.delegate.load(sample)

    monkeypatch.setattr(dataset_cli, "CausalApiMediaLoader", SecondFrameFailureLoader)
    output_dir = tmp_path / "output"

    with pytest.raises(DatasetContractError, match="mid-run"):
        run_dataset_api_rollout(
            config=load_api_config(MOCK_DATASET_CONFIG),
            selection=selection,
            dataset_root=dataset_root,
            output_dir=output_dir,
            cache_root=tmp_path / "cache",
            run_id="mid-run-failure",
            max_provider_calls=2,
            transport=MockProviderTransport(),
        )

    status = json.loads((output_dir / "run_status.json").read_text())
    assert status["status"] == "INCOMPLETE"
    assert status["failure_category"] == "dataset_error"
    assert len((output_dir / "predictions/VID30.jsonl").read_text().splitlines()) == 1
    assert len((output_dir / "evidence/VID30.jsonl").read_text().splitlines()) == 1
    assert "sensitive mid-run detail" not in _persisted_text(output_dir)


def test_dataset_artifact_failure_replaces_premature_complete_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frame manifest alone must not make the overall rollout COMPLETE."""

    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    _install_canonical_adapter(monkeypatch, selection)
    output_dir = tmp_path / "output"

    def fail_dataset_artifact(_path: Path, _payload: object) -> None:
        raise ArtifactWriteError("injected dataset artifact failure")

    monkeypatch.setattr(dataset_cli, "atomic_write_json", fail_dataset_artifact)

    with pytest.raises(ArtifactWriteError, match="dataset artifact"):
        run_dataset_api_rollout(
            config=load_api_config(MOCK_DATASET_CONFIG),
            selection=selection,
            dataset_root=dataset_root,
            output_dir=output_dir,
            cache_root=tmp_path / "cache",
            run_id="artifact-failure",
            max_provider_calls=2,
            transport=MockProviderTransport(),
        )

    status = json.loads((output_dir / "run_status.json").read_text())
    assert status["status"] == "INCOMPLETE"
    assert status["failure_category"] == "artifact_error"
    assert (output_dir / "manifest.json").is_file()
    assert not (output_dir / "dataset_rollout_artifact.json").exists()


def test_final_complete_status_failure_leaves_durable_incomplete_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure publishing the last state must preserve a truthful prior state."""

    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    _install_canonical_adapter(monkeypatch, selection)
    output_dir = tmp_path / "output"
    real_atomic_write = frame_writer_module.atomic_write_text

    def fail_complete_status(path: Path, content: str) -> None:
        if path.name == "run_status.json":
            payload = json.loads(content)
            if payload.get("status") == "COMPLETE":
                raise ArtifactWriteError("injected final status failure")
        real_atomic_write(path, content)

    monkeypatch.setattr(
        frame_writer_module,
        "atomic_write_text",
        fail_complete_status,
    )

    with pytest.raises(ArtifactWriteError, match="final status"):
        run_dataset_api_rollout(
            config=load_api_config(MOCK_DATASET_CONFIG),
            selection=selection,
            dataset_root=dataset_root,
            output_dir=output_dir,
            cache_root=tmp_path / "cache",
            run_id="completion-failure",
            max_provider_calls=2,
            transport=MockProviderTransport(),
        )

    status = json.loads((output_dir / "run_status.json").read_text())
    assert status["status"] == "INCOMPLETE"
    assert status["failure_category"] == "artifact_error"
    assert (output_dir / "dataset_rollout_artifact.json").is_file()


def test_programmatic_paper_selection_must_match_canonical_dataset_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trusting a caller's truncated paper selection could authorize a real upload."""

    dataset_root = tmp_path / "dataset"
    engineering = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    canonical_samples = engineering.samples

    class CanonicalAdapter:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.entries = {"VID30": SimpleNamespace(split=DatasetSplit.VALIDATION)}

        def iter_inference_video(
            self,
            video_id: str,
            *,
            max_samples: int | None = None,
        ) -> object:
            assert video_id == "VID30"
            selected = (
                canonical_samples
                if max_samples is None
                else canonical_samples[:max_samples]
            )
            yield from selected

    decode_calls: list[int] = []
    real_loader = dataset_cli.CausalApiMediaLoader

    class TrackingLoader:
        def __init__(self) -> None:
            self.delegate = real_loader()

        def load(self, sample: InferenceSample) -> object:
            decode_calls.append(sample.target_frame_id)
            return self.delegate.load(sample)

    monkeypatch.setattr(dataset_cli, "CholecTrack20DatasetAdapter", CanonicalAdapter)
    monkeypatch.setattr(dataset_cli, "CausalApiMediaLoader", TrackingLoader)
    malformed = RolloutSelection(
        mode="paper",
        split=DatasetSplit.VALIDATION,
        video_ids=("VID30",),
        samples=canonical_samples[:1],
        frame_counts=MappingProxyType({"VID30": 1}),
    )
    transport = InjectedOpenRouterTransport()

    with pytest.raises(ApiContractError, match="canonical dataset selection"):
        run_dataset_api_rollout(
            config=load_api_config(REAL_DATASET_CONFIG),
            selection=malformed,
            dataset_root=dataset_root,
            output_dir=tmp_path / "output",
            cache_root=tmp_path / "cache",
            run_id="malformed-paper",
            max_provider_calls=1,
            api_key=SecretValue(
                "-".join(  # noqa: FLY002 - keep fixture non-contiguous
                    ("malformed", "paper", "credential")
                )
            ),
            authorize_data_upload=True,
            transport=transport,
        )

    assert decode_calls == []
    assert transport.call_count == 0
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "cache").exists()


def test_failed_real_rollout_scans_and_removes_generated_secret_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scanning only successful runs would leave failed-call secret leaks on disk."""

    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    _install_canonical_adapter(monkeypatch, selection)
    cache_root = tmp_path / "cache"
    leak_path = cache_root / "generated-secret.txt"
    secret_text = "-".join(  # noqa: FLY002 - keep fixture non-contiguous
        ("failed", "rollout", "credential")
    )
    transport = LeakingFailureTransport(leak_path, secret_text)

    with pytest.raises(ApiContractError, match="credential scan"):
        run_dataset_api_rollout(
            config=load_api_config(REAL_DATASET_CONFIG),
            selection=selection,
            dataset_root=dataset_root,
            output_dir=tmp_path / "output",
            cache_root=cache_root,
            run_id="failed-real",
            max_provider_calls=2,
            api_key=SecretValue(secret_text),
            authorize_data_upload=True,
            transport=transport,
        )

    assert transport.call_count == 1
    assert not leak_path.exists()
    assert secret_text not in _persisted_text(tmp_path / "output", cache_root)


def test_programmatic_engineering_selection_must_match_canonical_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structurally valid non-prefix sample could upload noncanonical media."""

    dataset_root = tmp_path / "dataset"
    canonical = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    _install_canonical_adapter(monkeypatch, canonical)
    noncanonical = RolloutSelection(
        mode="engineering",
        split=None,
        video_ids=("VID30",),
        samples=canonical.samples[1:],
        frame_counts=MappingProxyType({"VID30": 1}),
    )
    transport = MockProviderTransport()

    with pytest.raises(ApiContractError, match="canonical dataset selection"):
        run_dataset_api_rollout(
            config=load_api_config(MOCK_DATASET_CONFIG),
            selection=noncanonical,
            dataset_root=dataset_root,
            output_dir=tmp_path / "output",
            cache_root=tmp_path / "cache",
            run_id="noncanonical-engineering",
            max_provider_calls=1,
            transport=transport,
        )

    assert transport.provider_call_count == 0


def test_output_directory_cannot_be_nested_inside_cache_root(tmp_path: Path) -> None:
    """An output child would contaminate the reusable external cache tree."""

    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    transport = MockProviderTransport()

    with pytest.raises(ApiContractError, match="disjoint"):
        run_dataset_api_rollout(
            config=load_api_config(MOCK_DATASET_CONFIG),
            selection=selection,
            dataset_root=dataset_root,
            output_dir=cache_root / "run",
            cache_root=cache_root,
            run_id="nested-output",
            max_provider_calls=2,
            transport=transport,
        )

    assert transport.provider_call_count == 0


def test_cache_root_cannot_be_an_ancestor_of_dataset_root(tmp_path: Path) -> None:
    """A cache ancestor could place generated envelopes in the dataset tree."""

    cache_root = tmp_path / "cache"
    dataset_root = cache_root / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
    transport = MockProviderTransport()

    with pytest.raises(ApiContractError, match="disjoint"):
        run_dataset_api_rollout(
            config=load_api_config(MOCK_DATASET_CONFIG),
            selection=selection,
            dataset_root=dataset_root,
            output_dir=tmp_path / "output",
            cache_root=cache_root,
            run_id="ancestor-cache",
            max_provider_calls=2,
            transport=transport,
        )

    assert transport.provider_call_count == 0
