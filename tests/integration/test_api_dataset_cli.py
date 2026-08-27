"""Strict CLI and artifact contracts for CholecTrack20 API rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from PIL import Image

import scripts.run_dataset_api_pipeline as dataset_cli
from scripts.run_dataset_api_pipeline import (
    build_parser,
    run,
    run_dataset_api_rollout,
    validate_cli_selection,
)
from surgical_agent.api.contracts import ApiRequest, ProviderResponse
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError, ApiTransportError
from surgical_agent.api.providers.mock import MockProviderTransport
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_rollout_selection import RolloutSelection
from surgical_agent.data.dataset import PNG_ALIGNMENT_VERSION
from surgical_agent.data.schemas import DatasetSplit, InferenceSample
from surgical_agent.perception.schema import (
    JOINT_PERCEPTION_SCHEMA_VERSION,
    TASK_LAYOUT,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOCK_DATASET_CONFIG = (
    PROJECT_ROOT / "configs/perception/joint_mock_dataset.yaml"
)
REAL_DATASET_CONFIG = (
    PROJECT_ROOT / "configs/perception/joint_openrouter_dataset.yaml"
)


class InjectedOpenRouterTransport:
    provider = "openrouter"
    endpoint_identifier = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        self.call_count = 0

    def send(self, request: ApiRequest) -> ProviderResponse:
        self.call_count += 1
        input_text = request.payload.get("input_text")
        decoded = json.loads(input_text) if isinstance(input_text, str) else {}
        frame_id = int(decoded["target_frame_id"])
        payload: dict[str, object] = {
            "schema_version": JOINT_PERCEPTION_SCHEMA_VERSION
        }
        for task, count in TASK_LAYOUT:
            topk = [
                {"id": index, "score": 1.0 - index / (count + 1)}
                for index in range(count)
            ]
            payload[task] = (
                {"selected_id": 0, "topk": topk}
                if task == "phase"
                else {"selected_ids": [0], "topk": topk}
            )
        payload["evidence_refs"] = [
            {"frame_id": frame_id, "code": "CURRENT_VISUAL_SUPPORT"}
        ]
        payload["self_reported_confidence"] = {
            task: 0.8 for task, _count in TASK_LAYOUT
        }
        return ProviderResponse(
            provider=self.provider,
            returned_model_identifier="openai/gpt-5.6-sol:injected",
            parsed_payload=payload,
            input_tokens=101,
            output_tokens=37,
            total_tokens=138,
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
                    str(media_root / f"frame_{value}.png")
                    for value in causal_ids
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
        "model_requested",
        "models_returned",
        "prompt_version",
        "response_schema_version",
        "repair_manifest_sha256",
        "alignment_versions",
        "usage",
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
    assert first["paper_metric_eligible"] is False
    assert first["usage"]["logical_calls"] == 2
    assert first["usage"]["provider_calls"] == 2
    assert first_transport.provider_call_count == 2
    assert second["usage"]["logical_calls"] == 2
    assert second["usage"]["provider_calls"] == 0
    assert second["usage"]["cache_hits"] == 2
    assert second_transport.provider_call_count == 0
    assert first["cache_entry_count"] == second["cache_entry_count"] == 2

    for output_dir in (first_output, second_output):
        prediction_lines = (output_dir / "predictions/VID30.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        evidence_lines = (output_dir / "evidence/VID30.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        usage_lines = (output_dir / "api_usage.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
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

    assert json.loads(
        (first_output / "dataset_rollout_artifact.json").read_text(encoding="utf-8")
    ) == first


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
    assert artifact["status"] == "REAL_RESPONSE_RECEIVED"
    assert artifact["track20_image_uploaded"] is True
    assert artifact["paper_metric_eligible"] is False
    assert secret_text not in _persisted_text(
        tmp_path / "real-output", tmp_path / "real-cache"
    )


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
            self.entries = {
                "VID30": SimpleNamespace(split=DatasetSplit.VALIDATION)
            }

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
