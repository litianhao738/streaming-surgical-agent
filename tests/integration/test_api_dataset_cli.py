"""Strict CLI and artifact contracts for CholecTrack20 API rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest
from PIL import Image

from scripts.run_dataset_api_pipeline import (
    build_parser,
    run,
    run_dataset_api_rollout,
    validate_cli_selection,
)
from surgical_agent.api.contracts import ApiRequest, ProviderResponse
from surgical_agent.api.credentials import SecretValue
from surgical_agent.api.errors import ApiContractError
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
) -> None:
    """Losing shared-cache replay or safe persistence would duplicate calls or leak data."""

    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    repair_manifest = dataset_root / "repair_manifest.json"
    repair_manifest.write_text('{"private": "manifest-content"}\n', encoding="utf-8")
    expected_digest = hashlib.sha256(repair_manifest.read_bytes()).hexdigest()
    cache_root = tmp_path / "cache"
    config = load_api_config(MOCK_DATASET_CONFIG)

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
) -> None:
    """Marking injected real data as mock would conceal an authorized upload path."""

    dataset_root = tmp_path / "dataset"
    selection = _selection_with_two_pngs(dataset_root / "media")
    (dataset_root / "repair_manifest.json").write_text("{}\n", encoding="utf-8")
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
