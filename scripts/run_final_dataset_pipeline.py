"""Run one formal Tracker x Gate cell through the uploaded final Pipeline."""

from __future__ import annotations

import argparse
import re
import sys
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
from surgical_agent.api.credentials import assert_secret_absent, resolve_api_key
from surgical_agent.api.registry import build_transport, build_validator
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.usage import UsageLedger
from surgical_agent.config.final_experiment import load_tracker_gate_cell
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.api_rollout_selection import resolve_rollout_selection
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.signals.phase_graph import load_phase_transition_graph
from surgical_agent.runtime.final_artifacts import FinalPipelineArtifactWriter
from surgical_agent.systems.final_dataset_system import FinalDatasetPipelineSystem
from surgical_agent.systems.final_pipeline_factory import (
    build_final_api_pipeline,
    validate_verifier_pair,
)
from surgical_agent.tracking.predicted_provider import PrecomputedPredictedTrackProvider

_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("engineering", "paper"), required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--target-frame-id", type=int)
    parser.add_argument("--split", choices=("validation", "testing"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "CholecTrack20",
    )
    parser.add_argument(
        "--api-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "perception" / "joint_mock_final_fixed6.yaml",
    )
    parser.add_argument(
        "--verification-api-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "perception"
            / "targeted_mock_final_fixed6.yaml"
        ),
    )
    parser.add_argument(
        "--cell-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ablations" / "a_base.yaml",
    )
    parser.add_argument("--tracker-artifact", type=Path)
    parser.add_argument("--phase-transition-graph", type=Path)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--verification-api-key-file", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "final_pipeline_runs",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "final_pipeline_cache",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--max-provider-calls", type=int, required=True)
    return parser


def _validate_paths(*, dataset: Path, output: Path, cache: Path) -> None:
    if not dataset.is_dir():
        raise ValueError("dataset root is missing")
    if output == dataset or output.is_relative_to(dataset):
        raise ValueError("output directory cannot be inside the read-only dataset")
    if cache == dataset or cache.is_relative_to(dataset):
        raise ValueError("cache directory cannot be inside the read-only dataset")
    if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError("output and cache directories cannot overlap")


def _failure_category(exc: BaseException) -> str:
    if isinstance(exc, (TypeError, ValueError, KeyError)):
        return "invalid_input"
    if isinstance(exc, OSError):
        return "io_error"
    return "runtime_error"


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now(UTC).strftime("final_%Y%m%dT%H%M%SZ")
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id contains unsafe characters")
    if args.max_provider_calls <= 0:
        raise ValueError("max-provider-calls must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / run_id
    cache_root = args.cache_root.expanduser().resolve()
    _validate_paths(dataset=dataset_root, output=output_dir, cache=cache_root)

    api_config = load_api_config(args.api_config)
    verification_api_config = load_api_config(args.verification_api_config)
    validate_verifier_pair(api_config, verification_api_config)
    cell = load_tracker_gate_cell(args.cell_config)
    if cell.tracker_enabled != (args.tracker_artifact is not None):
        raise ValueError(
            "tracker-artifact presence must exactly match the formal cell Tracker switch"
        )
    track_provider = (
        None
        if args.tracker_artifact is None
        else PrecomputedPredictedTrackProvider.from_json(args.tracker_artifact)
    )
    phase_graph = (
        None
        if args.phase_transition_graph is None
        else load_phase_transition_graph(args.phase_transition_graph)
    )
    adapter = CholecTrack20DatasetAdapter(dataset_root, causal_window_size=6)
    selection = resolve_rollout_selection(
        adapter,
        mode=args.mode,
        video_id=args.video_id,
        max_frames=args.max_frames,
        split=args.split,
        target_frame_id=args.target_frame_id,
    )
    initial_secret = (
        None
        if api_config.provider == "mock"
        else resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    )
    same_transport_boundary = (
        api_config.provider == verification_api_config.provider
        and api_config.endpoint_identifier
        == verification_api_config.endpoint_identifier
    )
    verification_key_file = getattr(args, "verification_api_key_file", None)
    if verification_key_file is None and same_transport_boundary:
        verification_key_file = args.api_key_file
    verification_secret = (
        None
        if verification_api_config.provider == "mock"
        else resolve_api_key(api_key=None, api_key_file=verification_key_file)
    )
    initial_transport = build_transport(api_config, api_key=initial_secret)
    verification_transport = build_transport(
        verification_api_config,
        api_key=verification_secret,
    )
    if api_config.mode == "real":
        initial_transport = CompleteAccountingTransport(initial_transport)
    if verification_api_config.mode == "real":
        verification_transport = CompleteAccountingTransport(verification_transport)
    usage = UsageLedger(output_dir / "api_usage.jsonl")
    provider_budget = ProviderCallBudget(args.max_provider_calls)
    cache = FileApiCache(cache_root)
    initial_client = CachedMultimodalApiClient(
        transport=initial_transport,
        cache=cache,
        usage=usage,
        validator=build_validator(api_config),
        retry_policy=RetryPolicy(max_attempts=4),
        provider_call_budget=provider_budget,
    )
    verification_client = CachedMultimodalApiClient(
        transport=verification_transport,
        cache=cache,
        usage=usage,
        validator=build_validator(verification_api_config),
        retry_policy=RetryPolicy(max_attempts=4),
        provider_call_budget=provider_budget,
    )
    pipeline = build_final_api_pipeline(
        client=initial_client,
        verification_client=verification_client,
        api_config=api_config,
        verification_api_config=verification_api_config,
        cell=cell,
        state_dir=output_dir / "finalization_state",
        track_provider=track_provider,
        phase_transition_graph=phase_graph,
    )
    writer = FinalPipelineArtifactWriter(
        output_dir,
        run_id=run_id,
        cell=cell.cell,
        initial_model_requested=api_config.requested_model_identifier,
        verification_model_requested=(
            verification_api_config.requested_model_identifier
        ),
    )
    system = FinalDatasetPipelineSystem(
        pipeline=pipeline,
        media_loader=CausalApiMediaLoader(),
        writer=writer,
    )
    try:
        result = system.run(selection)
        secrets = tuple(
            secret
            for secret in (initial_secret, verification_secret)
            if secret is not None
        )
        if secrets:
            paths = tuple(path for path in (output_dir, cache_root) if path.exists())
            files = tuple(
                file for path in paths for file in path.rglob("*") if file.is_file()
            )
            for secret in secrets:
                assert_secret_absent(secret, files)
    except Exception as exc:
        writer.mark_failed(_failure_category(exc))
        raise
    print(
        f"FINAL_PIPELINE_COMPLETE cell={cell.cell} "
        f"processed={result.processed_count} resumed={result.resumed_count} "
        f"manifest={result.manifest_path}"
    )
    return result.manifest_path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 - CLI reports a sanitized category only
        print(
            f"FINAL_PIPELINE_FAILED category={_failure_category(exc)}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
