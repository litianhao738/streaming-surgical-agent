"""Run a bounded CholecTrack20 selection through the joint API pipeline."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.api.accounting import CompleteAccountingTransport
from surgical_agent.api.budget import ProviderCallBudget
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import ProviderTransport
from surgical_agent.api.credentials import (
    SecretValue,
    assert_secret_absent,
    load_api_key_file,
)
from surgical_agent.api.errors import ApiCallFailure, ApiContractError, ApiError
from surgical_agent.api.registry import build_transport, build_validator
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.config.loader import load_api_config
from surgical_agent.config.schema import ApiConfig
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.api_rollout_selection import (
    RolloutSelection,
    resolve_rollout_selection,
)
from surgical_agent.data.dataset import (
    CholecTrack20DatasetAdapter,
    DatasetContractError,
)
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.inference.frame_result_writer import FrameResultWriter
from surgical_agent.inference.writer import ArtifactWriteError
from surgical_agent.systems.api_dataset_system import DatasetApiPipelineSystem

_JOINT_VERSION = "joint_perception_frame_v1"
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_REAL_MODEL = "openai/gpt-5.6-sol"
_MOCK_MODEL = "mock-joint-perception-v1"
_MOCK_ENDPOINT = "mock://local/p3"
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset rollout CLI without credential-bearing defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("engineering", "paper"), required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--split", choices=("validation", "testing"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data/CholecTrack20",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/perception/joint_mock_dataset.yaml",
    )
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/api_dataset",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/api_dataset_cache/mock_validation",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--max-provider-calls", default="exact-selection")
    parser.add_argument("--authorize-data-upload", action="store_true")
    return parser


def validate_cli_selection(args: argparse.Namespace) -> None:
    """Reject incomplete or paper-invalid selectors without touching the dataset."""

    mode = getattr(args, "mode", None)
    video_id = getattr(args, "video_id", None)
    max_frames = getattr(args, "max_frames", None)
    split = getattr(args, "split", None)
    if mode == "engineering":
        if not isinstance(video_id, str) or not video_id.strip():
            raise ApiContractError("engineering mode requires one video-id")
        if split is not None:
            raise ApiContractError("engineering mode does not accept split")
        if (
            not isinstance(max_frames, int)
            or isinstance(max_frames, bool)
            or max_frames <= 0
        ):
            raise ApiContractError("engineering mode requires positive max-frames")
        return
    if mode == "paper":
        if video_id is not None:
            raise ApiContractError("paper mode does not accept video-id")
        if max_frames is not None:
            raise ApiContractError("paper mode forbids truncation")
        if split not in {"validation", "testing"}:
            raise ApiContractError("paper mode requires validation or testing split")
        return
    raise ApiContractError("mode must be engineering or paper")


def _require_exact_dataset_config(
    config: ApiConfig,
    *,
    has_credential: bool,
    authorize_data_upload: bool,
) -> None:
    if not isinstance(config, ApiConfig):
        raise TypeError("config must be an ApiConfig")
    if type(has_credential) is not bool or type(authorize_data_upload) is not bool:
        raise TypeError("credential and upload authorization state must be boolean")
    config.require_enabled()
    config.validate()
    if config.prompt_version != _JOINT_VERSION:
        raise ApiContractError("dataset rollout requires the exact prompt version")
    if config.response_schema_version != _JOINT_VERSION:
        raise ApiContractError("dataset rollout requires the exact response schema")
    generation = dict(config.generation_parameters)
    reasoning = generation.get("reasoning")
    if (
        set(generation) != {"max_output_tokens", "reasoning"}
        or type(generation.get("max_output_tokens")) is not int
        or generation["max_output_tokens"] != 4096
        or not isinstance(reasoning, Mapping)
        or dict(reasoning) != {"effort": "low"}
    ):
        raise ApiContractError("dataset rollout requires exact generation settings")
    if config.synthetic_input_required is not False:
        raise ApiContractError("dataset rollout requires real dataset input")
    if config.cache_required is not True:
        raise ApiContractError("dataset rollout requires the request cache")
    if config.data_upload_authorized is not True:
        raise ApiContractError("dataset config must authorize data upload")
    if config.max_causal_frames != 3:
        raise ApiContractError("dataset rollout requires three causal frames")

    if config.mode == "mock" and config.provider == "mock":
        if config.endpoint_identifier != _MOCK_ENDPOINT:
            raise ApiContractError("mock dataset rollout requires the exact endpoint")
        if config.requested_model_identifier != _MOCK_MODEL:
            raise ApiContractError("mock dataset rollout requires the exact model")
        if dict(config.provider_options):
            raise ApiContractError("mock dataset rollout rejects provider options")
        if has_credential:
            raise ApiContractError("mock dataset rollout rejects credential inputs")
        return

    if config.mode == "real" and config.provider == "openrouter":
        if config.endpoint_identifier != _OPENROUTER_ENDPOINT:
            raise ApiContractError("real dataset rollout requires OpenRouter")
        if config.requested_model_identifier != _REAL_MODEL:
            raise ApiContractError("real dataset rollout requires openai/gpt-5.6-sol")
        options = dict(config.provider_options)
        if (
            set(options) != {"timeout_seconds"}
            or type(options["timeout_seconds"]) is not float
            or options["timeout_seconds"] != 120.0
        ):
            raise ApiContractError("real dataset rollout requires exact timeout")
        if not authorize_data_upload:
            raise ApiContractError(
                "real dataset rollout requires --authorize-data-upload"
            )
        if not has_credential:
            raise ApiContractError("real dataset rollout requires exactly one credential")
        return

    raise ApiContractError("dataset rollout requires an exact mock or real config")


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _validate_paths(
    *,
    dataset_root: Path,
    output_dir: Path,
    cache_root: Path,
    output_root: Path | None = None,
) -> None:
    if not dataset_root.is_dir():
        raise ApiContractError("dataset root must be an existing directory")
    if output_dir.exists():
        raise ApiContractError("dataset rollout output directory must be fresh")
    if cache_root.exists() and not cache_root.is_dir():
        raise ApiContractError("cache root must be a directory")
    effective_output_root = output_dir if output_root is None else output_root
    if (
        _paths_overlap(dataset_root, effective_output_root)
        or _paths_overlap(dataset_root, cache_root)
        or _paths_overlap(effective_output_root, cache_root)
    ):
        raise ApiContractError(
            "dataset, output, and cache roots must be pairwise disjoint"
        )
    if not (dataset_root / "repair_manifest.json").is_file():
        raise ApiContractError("dataset repair manifest is missing")


def _require_safe_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ApiContractError("run-id is not a safe artifact identifier")


def _resolve_call_limit(value: object, *, selection_count: int) -> int:
    if selection_count <= 0:
        raise ApiContractError("rollout selection must not be empty")
    if value == "exact-selection":
        return selection_count
    if isinstance(value, int) and not isinstance(value, bool):
        limit = value
    elif isinstance(value, str) and _POSITIVE_INTEGER.fullmatch(value) is not None:
        limit = int(value)
    else:
        raise ApiContractError(
            "max-provider-calls must be a positive integer or exact-selection"
        )
    if limit <= 0:
        raise ApiContractError("max-provider-calls must be positive")
    return limit


def _validate_selection_semantics(
    selection: RolloutSelection,
    *,
    dataset_root: Path,
    max_causal_frames: int,
) -> None:
    """Reject caller-built selections that bypass canonical rollout policy."""

    if selection.mode == "engineering":
        if selection.split is not None or len(selection.video_ids) != 1:
            raise ApiContractError(
                "engineering selection requires exactly one video and no split"
            )
        video_id = selection.video_ids[0]
        samples = selection.samples
        if not samples or dict(selection.frame_counts) != {video_id: len(samples)}:
            raise ApiContractError(
                "engineering selection requires a positive bounded frame set"
            )
        frame_ids = tuple(sample.target_frame_id for sample in samples)
        if (
            any(sample.video_id != video_id for sample in samples)
            or any(
                frame_ids[index] >= frame_ids[index + 1]
                for index in range(len(frame_ids) - 1)
            )
        ):
            raise ApiContractError(
                "engineering selection samples must match one ordered video"
            )
        canonical_video_id = video_id
        canonical_max_frames: int | None = len(samples)
        canonical_split: str | None = None
    elif selection.mode == "paper":
        if selection.split not in {DatasetSplit.VALIDATION, DatasetSplit.TESTING}:
            raise ApiContractError(
                "paper selection requires validation or testing split"
            )
        canonical_video_id = None
        canonical_max_frames = None
        canonical_split = selection.split.value
    else:
        raise ApiContractError("selection mode must be engineering or paper")

    adapter = CholecTrack20DatasetAdapter(
        dataset_root,
        causal_window_size=max_causal_frames,
    )
    try:
        canonical = resolve_rollout_selection(
            adapter,
            mode=selection.mode,
            video_id=canonical_video_id,
            max_frames=canonical_max_frames,
            split=canonical_split,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiContractError(str(exc)) from None
    if (
        selection.video_ids != canonical.video_ids
        or selection.samples != canonical.samples
        or dict(selection.frame_counts) != dict(canonical.frame_counts)
    ):
        raise ApiContractError("selection must match canonical dataset selection")


def _require_transport_identity(
    config: ApiConfig,
    transport: ProviderTransport,
) -> None:
    if not hasattr(transport, "send") or not callable(transport.send):
        raise ApiContractError("transport must provide send(request)")
    if transport.provider != config.provider:
        raise ApiContractError("transport provider does not match config")
    if transport.endpoint_identifier != config.endpoint_identifier:
        raise ApiContractError("transport endpoint does not match config")


def _scan_secret(
    secret: SecretValue,
    *,
    output_dir: Path,
    cache_root: Path,
) -> None:
    paths = [
        *(path for path in output_dir.rglob("*") if path.is_file()),
        *(path for path in cache_root.rglob("*") if path.is_file()),
    ]
    try:
        assert_secret_absent(secret, paths)
    except RuntimeError:
        needle = secret.reveal().encode("utf-8")
        leaked = [
            path for path in paths if path.is_file() and needle in path.read_bytes()
        ]
        for path in leaked:
            path.unlink(missing_ok=True)
        raise ApiContractError("credential scan detected persisted credential") from None


def run_dataset_api_rollout(
    *,
    config: ApiConfig,
    selection: RolloutSelection,
    dataset_root: str | Path,
    output_dir: str | Path,
    cache_root: str | Path,
    run_id: str,
    max_provider_calls: int,
    api_key: SecretValue | None = None,
    authorize_data_upload: bool = False,
    transport: ProviderTransport | None = None,
) -> dict[str, object]:
    """Run one preselected rollout with an optional injected transport."""

    if api_key is not None and not isinstance(api_key, SecretValue):
        raise TypeError("api_key must be SecretValue or None")
    if not isinstance(selection, RolloutSelection):
        raise TypeError("selection must be a RolloutSelection")
    _require_exact_dataset_config(
        config,
        has_credential=api_key is not None,
        authorize_data_upload=authorize_data_upload,
    )
    _require_safe_run_id(run_id)
    resolved_dataset_root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    resolved_cache_root = Path(cache_root).expanduser().resolve()
    _validate_paths(
        dataset_root=resolved_dataset_root,
        output_dir=destination,
        cache_root=resolved_cache_root,
    )
    _validate_selection_semantics(
        selection,
        dataset_root=resolved_dataset_root,
        max_causal_frames=config.max_causal_frames,
    )
    call_limit = _resolve_call_limit(
        max_provider_calls,
        selection_count=selection.expected_provider_calls,
    )
    if transport is not None:
        _require_transport_identity(config, transport)

    repair_manifest_sha256 = sha256_file(
        resolved_dataset_root / "repair_manifest.json"
    )
    selected_transport = transport or build_transport(config, api_key=api_key)
    _require_transport_identity(config, selected_transport)
    client_transport = (
        CompleteAccountingTransport(selected_transport)
        if config.mode == "real"
        else selected_transport
    )
    usage = UsageLedger(destination / "api_usage.jsonl")
    cache = FileApiCache(resolved_cache_root)
    client = CachedMultimodalApiClient(
        transport=client_transport,
        cache=cache,
        usage=usage,
        validator=build_validator(config),
        retry_policy=RetryPolicy(
            max_attempts=1,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
        ),
        sleep=lambda _seconds: None,
        provider_call_budget=ProviderCallBudget(call_limit),
    )
    writer = FrameResultWriter(destination, run_id=run_id)
    system = DatasetApiPipelineSystem(
        client=client,
        config=config,
        writer=writer,
        media_loader=CausalApiMediaLoader(),
    )
    writer.begin()
    try:
        try:
            result = system.run(
                selection,
                run_id=run_id,
                defer_completion=True,
            )
            artifact: dict[str, object] = {
                "schema_version": "cholectrack20_api_rollout_v1",
                "status": (
                    "REAL_RESPONSE_RECEIVED"
                    if config.mode == "real"
                    else "MOCK_COMPLETE"
                ),
                "run_id": run_id,
                "mode": selection.mode,
                "split": selection.split.value if selection.split else None,
                "video_ids": list(selection.video_ids),
                "expected_frame_counts": dict(selection.frame_counts),
                "completed_frame_counts": dict(result.frame_counts),
                "provider": config.provider,
                "model_requested": config.requested_model_identifier,
                "models_returned": sorted(
                    {
                        str(row["returned_model_identifier"])
                        for row in usage.records()
                        if row.get("returned_model_identifier") is not None
                    }
                ),
                "prompt_version": config.prompt_version,
                "response_schema_version": config.response_schema_version,
                "repair_manifest_sha256": repair_manifest_sha256,
                "alignment_versions": sorted(
                    {sample.alignment_version for sample in selection.samples}
                ),
                "usage": result.usage_summary,
                "cache_entry_count": len(tuple(resolved_cache_root.glob("*.json"))),
                "track20_image_uploaded": config.mode == "real",
                "paper_metric_eligible": False,
                "manifest_file": "manifest.json",
            }
            atomic_write_json(destination / "dataset_rollout_artifact.json", artifact)
        finally:
            if config.mode == "real":
                assert api_key is not None
                _scan_secret(
                    api_key,
                    output_dir=destination,
                    cache_root=resolved_cache_root,
                )
        writer.complete()
    except Exception as exc:
        writer.mark_failed(_safe_error_category(exc))
        raise
    return artifact


def run(args: argparse.Namespace) -> Path:
    """Complete all preflight checks before credentials, media, or real transport."""

    validate_cli_selection(args)
    config = load_api_config(args.config)
    has_credential = args.api_key_file is not None
    _require_exact_dataset_config(
        config,
        has_credential=has_credential,
        authorize_data_upload=args.authorize_data_upload,
    )
    run_id = args.run_id or datetime.now(UTC).strftime(
        "dataset_api_%Y%m%dT%H%M%SZ"
    )
    _require_safe_run_id(run_id)
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / run_id
    cache_root = args.cache_root.expanduser().resolve()
    _validate_paths(
        dataset_root=dataset_root,
        output_dir=output_dir,
        cache_root=cache_root,
        output_root=args.output_root.expanduser().resolve(),
    )
    adapter = CholecTrack20DatasetAdapter(
        dataset_root,
        causal_window_size=config.max_causal_frames,
    )
    try:
        selection = resolve_rollout_selection(
            adapter,
            mode=args.mode,
            video_id=args.video_id,
            max_frames=args.max_frames,
            split=args.split,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiContractError(str(exc)) from None
    call_limit = _resolve_call_limit(
        args.max_provider_calls,
        selection_count=selection.expected_provider_calls,
    )
    api_key = (
        load_api_key_file(args.api_key_file)
        if args.api_key_file is not None
        else None
    )
    artifact = run_dataset_api_rollout(
        config=config,
        selection=selection,
        dataset_root=dataset_root,
        output_dir=output_dir,
        cache_root=cache_root,
        run_id=run_id,
        max_provider_calls=call_limit,
        api_key=api_key,
        authorize_data_upload=args.authorize_data_upload,
    )
    artifact_path = output_dir / "dataset_rollout_artifact.json"
    print(f"DATASET_API_ROLLOUT_{artifact['status']} artifact={artifact_path}")
    return artifact_path


def _safe_error_category(exc: BaseException) -> str:
    if isinstance(exc, ApiCallFailure):
        return exc.cause.code
    if isinstance(exc, ApiError):
        return exc.code
    if isinstance(exc, DatasetContractError):
        return "dataset_error"
    if isinstance(exc, ArtifactWriteError):
        return "artifact_error"
    if isinstance(exc, UnicodeError):
        return "invalid_text"
    if isinstance(exc, OSError):
        return "io_error"
    if isinstance(exc, (TypeError, ValueError, KeyError)):
        return "invalid_input"
    return "internal_error"


def main() -> None:
    try:
        run(build_parser().parse_args())
    except Exception as exc:  # noqa: BLE001 - CLI emits only a safe category
        print(
            f"DATASET_API_ROLLOUT_FAILED category={_safe_error_category(exc)}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
