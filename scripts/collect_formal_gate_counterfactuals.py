"""Collect Training-only, video-OOF counterfactuals for the formal Gate."""

from __future__ import annotations

import argparse
import json
import sys
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
from surgical_agent.api.errors import ApiCallFailure
from surgical_agent.api.registry import build_transport, build_validator
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import (
    atomic_write_json,
    sha256_file,
    sha256_mapping,
)
from surgical_agent.config.final_experiment import load_tracker_gate_cell
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.gate.counterfactual import (
    FormalGateCounterfactualCollector,
    label_counterfactual,
)
from surgical_agent.research.gate.oof_dataset import (
    CounterfactualCollectionStore,
    write_counterfactual_records,
)
from surgical_agent.research.gate.supervision import (
    deterministic_timeline_sample,
    gate_training_target,
)
from surgical_agent.research.signals.phase_graph import load_phase_transition_graph
from surgical_agent.systems.final_pipeline_factory import (
    build_final_api_pipeline,
    validate_verifier_pair,
    verifier_pair_is_heterogeneous,
)
from surgical_agent.tracking.oof_index import load_tracker_oof_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tracker-oof-index", type=Path, required=True)
    parser.add_argument(
        "--api-config",
        type=Path,
        default=PROJECT_ROOT / "configs/perception/joint_openrouter_final_fixed6.yaml",
    )
    parser.add_argument(
        "--verification-api-config",
        type=Path,
        required=True,
        help=(
            "Explicit Verifier config. Same-model verification requires the "
            "conservative V7 contract; heterogeneous V6 remains pilot-only."
        ),
    )
    parser.add_argument("--phase-transition-graph", type=Path)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument(
        "--verification-api-key-file",
        type=Path,
        help=(
            "Verifier credential file. Required when its provider/endpoint differs "
            "from H0; otherwise --api-key-file is reused."
        ),
    )
    parser.add_argument("--video-id", action="append")
    parser.add_argument("--max-frames-per-video", type=int)
    parser.add_argument("--max-provider-calls", type=int, required=True)
    parser.add_argument(
        "--allow-api-failures",
        action="store_true",
        help=(
            "Pilot-only: record terminal provider failures and continue with later "
            "frames. Formal collection remains fail-fast by default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/gate/formal/collection",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/final_pipeline_cache/gate_collection",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.max_provider_calls <= 0:
        raise ValueError("max-provider-calls must be positive")
    if args.max_frames_per_video is not None and args.max_frames_per_video <= 0:
        raise ValueError("max-frames-per-video must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    if output_dir == dataset_root or output_dir.is_relative_to(dataset_root):
        raise ValueError("counterfactual output cannot be inside the dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    api_config = load_api_config(args.api_config)
    verification_api_config = load_api_config(args.verification_api_config)
    validate_verifier_pair(api_config, verification_api_config)
    heterogeneous_verifier = verifier_pair_is_heterogeneous(
        api_config, verification_api_config
    )
    index = load_tracker_oof_index(args.tracker_oof_index)
    selected_video_ids = tuple(
        sorted(
            set(index.video_to_artifact)
            if args.video_id is None
            else {value.upper() for value in args.video_id}
        )
    )
    if not selected_video_ids or set(selected_video_ids) - set(index.video_to_artifact):
        raise ValueError("selected videos are not fully covered by the Tracker OOF index")
    adapter = CholecTrack20DatasetAdapter(dataset_root, causal_window_size=6)
    if any(
        video_id not in adapter.entries
        or adapter.entries[video_id].split is not DatasetSplit.TRAINING
        for video_id in selected_video_ids
    ):
        raise ValueError("formal Gate collection accepts Training videos only")
    samples_by_video = {}
    selected_frames_by_video = {}
    for video_id in selected_video_ids:
        label_frame_ids = {
            item.inference.target_frame_id for item in adapter.iter_video(video_id)
        }
        eligible_samples = tuple(
            sample
            for sample in adapter.iter_inference_video(video_id)
            if sample.target_frame_id in label_frame_ids
        )
        samples = deterministic_timeline_sample(
            eligible_samples, args.max_frames_per_video
        )
        samples_by_video[video_id] = samples
        selected_frames_by_video[video_id] = [
            sample.target_frame_id for sample in samples
        ]
    collection_contract = {
        "schema_version": "formal_gate_collection_contract_v3",
        "initial_api_config_sha256": sha256_file(args.api_config),
        "verification_api_config_sha256": sha256_file(
            args.verification_api_config
        ),
        "initial_model_requested": api_config.requested_model_identifier,
        "verification_model_requested": (
            verification_api_config.requested_model_identifier
        ),
        "heterogeneous_verifier": heterogeneous_verifier,
        "verifier_architecture": (
            "HETEROGENEOUS_V6"
            if heterogeneous_verifier
            else "CONSERVATIVE_SAME_MODEL_V7"
        ),
        "tracker_oof_index_sha256": sha256_file(args.tracker_oof_index),
        "sampling_version": "deterministic_timeline_spread_v1",
        "scope_execution": "PARALLEL_SAME_PREDECISION_SNAPSHOT_V1",
        "selected_frames_by_video": selected_frames_by_video,
    }
    collection_key = sha256_mapping(collection_contract)
    store = CounterfactualCollectionStore(
        output_dir / "observations" / collection_key
    )
    state_path = output_dir / "collection_state.json"
    atomic_write_json(
        state_path,
        {
            **collection_contract,
            "collection_key": collection_key,
            "status": "RUNNING_OR_RESUMABLE",
        },
    )
    graph = (
        None
        if args.phase_transition_graph is None
        else load_phase_transition_graph(args.phase_transition_graph)
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
    cell = load_tracker_gate_cell(
        PROJECT_ROOT / "configs" / "ablations" / "b_tracker.yaml"
    )
    media_loader = CausalApiMediaLoader()
    api_failures: list[dict[str, object]] = []
    for video_id in selected_video_ids:
        artifact = index.video_to_artifact[video_id]
        pipeline = build_final_api_pipeline(
            client=initial_client,
            verification_client=verification_client,
            api_config=api_config,
            verification_api_config=verification_api_config,
            cell=cell,
            track_provider=index.provider_for(video_id),
            phase_transition_graph=graph,
        )
        samples = samples_by_video[video_id]
        selected_frame_ids = tuple(sample.target_frame_id for sample in samples)
        collector = FormalGateCounterfactualCollector(
            pipeline=pipeline,
            tracker_artifact_sha256=index.artifact_sha256[artifact],
            max_provider_attempts=(
                len(samples)
                * 3
                * pipeline.components.max_verify_attempts
            ),
        )
        # No GT object exists in this loop: all provider calls finish first.
        unlabeled = []
        for sample in samples:
            try:
                unlabeled.append(
                    collector.collect(sample, media_loader.load(sample).frames)
                )
            except ApiCallFailure as exc:
                if not args.allow_api_failures:
                    raise
                api_failures.append(
                    {
                        "sample_id": f"{sample.video_id}:{sample.target_frame_id}",
                        "video_id": sample.video_id,
                        "frame_id": sample.target_frame_id,
                        "error_code": exc.cause.code,
                        "status_code": exc.cause.status_code,
                        "provider_call_count": exc.provider_call_count,
                        "retry_count": exc.retry_count,
                    }
                )
                atomic_write_json(
                    output_dir / "api_failures.json",
                    {
                        "schema_version": "formal_gate_api_failures_v1",
                        "failures": api_failures,
                    },
                )
        targets = {
            item.inference.target_frame_id: gate_training_target(item)
            for item in adapter.iter_video(video_id, frame_ids=selected_frame_ids)
        }
        for item in unlabeled:
            target = targets.get(item.sample.target_frame_id)
            if target is None:
                raise RuntimeError("Training sample is missing frame supervision")
            labeled = label_counterfactual(item, target)
            store.write_observation(
                video_id=item.sample.video_id,
                frame_id=item.sample.target_frame_id,
                safety_class=labeled.safety_class,
                record=labeled.record,
            )
    observations = store.observations()
    records = store.records()
    hard_invalid = sum(
        item["safety_class"] == "HARD_INVALID" for item in observations
    )
    label_counts = {
        scope: {"0": 0, "1": 0, "null": 0}
        for scope in ("instrument_presence", "interaction", "workflow")
    }
    for record in records:
        for scope, label in record.benefit_by_scope.items():
            label_counts[scope]["null" if label is None else str(label)] += 1
    output_path = write_counterfactual_records(
        output_dir / "counterfactuals.jsonl",
        records,
    )
    manifest = {
        "schema_version": "formal_gate_counterfactual_manifest_v3",
        "source_split": "Training",
        "ground_truth_access": "AFTER_ALL_PROVIDER_CALLS_PER_VIDEO",
        "joint_h0_tracker_independent": True,
        "specialist_tracker_independent": True,
        "heterogeneous_verifier": heterogeneous_verifier,
        "verifier_architecture": (
            "HETEROGENEOUS_V6"
            if heterogeneous_verifier
            else "CONSERVATIVE_SAME_MODEL_V7"
        ),
        "initial_model_requested": api_config.requested_model_identifier,
        "verification_model_requested": (
            verification_api_config.requested_model_identifier
        ),
        "initial_api_config_sha256": sha256_file(args.api_config),
        "verification_api_config_sha256": sha256_file(
            args.verification_api_config
        ),
        "tracker_source": "VIDEO_OOF_INDEX",
        "tracker_oof_index_sha256": sha256_file(args.tracker_oof_index),
        "video_ids": list(selected_video_ids),
        "selected_frames_by_video": selected_frames_by_video,
        "sampling_version": "deterministic_timeline_spread_v1",
        "scope_execution": "PARALLEL_SAME_PREDECISION_SNAPSHOT_V1",
        "collection_key": collection_key,
        "record_count": len(records),
        "selected_frame_count": sum(
            len(frame_ids) for frame_ids in selected_frames_by_video.values()
        ),
        "completed_observation_count": len(observations),
        "api_failure_count": len(api_failures),
        "api_failures": api_failures,
        "partial_collection_allowed": args.allow_api_failures,
        "hard_invalid_excluded": hard_invalid,
        "label_counts": label_counts,
        "truncated_engineering_collection": args.max_frames_per_video is not None,
        "provider_attempt_limit": args.max_provider_calls,
        "provider_attempts_used": provider_budget.used,
        "provider_attempts_remaining": provider_budget.remaining,
        "api_usage_summary": usage.summarize(),
        "counterfactuals": output_path.name,
        "counterfactuals_sha256": sha256_file(output_path),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(
        state_path,
        {
            **collection_contract,
            "collection_key": collection_key,
            "status": (
                "COMPLETE_WITH_API_FAILURES" if api_failures else "COMPLETE"
            ),
            "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        },
    )
    secrets = tuple(
        secret
        for secret in (initial_secret, verification_secret)
        if secret is not None
    )
    if secrets:
        files = tuple(
            file
            for root in (output_dir, cache_root)
            for file in root.rglob("*")
            if file.is_file()
        )
        for secret in secrets:
            assert_secret_absent(secret, files)
    print("FORMAL_GATE_COUNTERFACTUALS_COMPLETE " + json.dumps(manifest, sort_keys=True))
    return output_path


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
