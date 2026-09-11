"""Collect Training-only counterfactuals and fit both demo Learned Gates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_dataset_api_pipeline import run_dataset_api_rollout

from surgical_agent.api.credentials import load_api_key_file
from surgical_agent.artifacts.manifest import sha256_file
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_rollout_selection import RolloutSelection
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter, ResolvedSample
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.inference.schemas import InitialPrediction
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import JointPerceptionResult
from surgical_agent.research.gate.features import (
    GateFeatureVector,
    extract_gate_features,
)
from surgical_agent.research.gate.learned import (
    LearnedGateExample,
    build_example,
    fit_learned_gate_pair,
    write_examples,
)
from surgical_agent.research.signals.contracts import EvidenceProfile
from surgical_agent.research.signals.phase_graph import (
    build_phase_ivt_compatibility_from_training_adapter,
)
from surgical_agent.research.verification.contracts import CoordinationResult
from surgical_agent.systems.pipeline import GateDecision


@dataclass(frozen=True)
class _ObservedCounterfactual:
    initial_prediction: InitialPrediction
    no_tracker_features: GateFeatureVector
    with_tracker_features: GateFeatureVector
    decision: GateDecision
    coordinated: CoordinationResult


class _CounterfactualObserver:
    """In-memory observer; it cannot receive or represent GT."""

    def __init__(self, phase_allowed_ivt: dict[int, tuple[int, ...]]) -> None:
        self.records: dict[tuple[str, int], _ObservedCounterfactual] = {}
        self.phase_allowed_ivt = phase_allowed_ivt

    def observe(
        self,
        *,
        context: PerceptionContext,
        perception_result: JointPerceptionResult,
        evidence: EvidenceProfile,
        decision: GateDecision,
        coordinated: CoordinationResult,
    ) -> None:
        identity = (context.sample.video_id, context.sample.target_frame_id)
        if identity in self.records:
            raise ValueError("duplicate counterfactual observation")
        if decision.action != "VERIFY":
            raise ValueError("training collection must force verification")
        no_tracker = extract_gate_features(
            evidence,
            context=context,
            perception_result=perception_result,
            include_tracker=False,
            phase_allowed_ivt=self.phase_allowed_ivt,
        )
        with_tracker = extract_gate_features(
            evidence,
            context=context,
            perception_result=perception_result,
            include_tracker=True,
            phase_allowed_ivt=self.phase_allowed_ivt,
        )
        self.records[identity] = _ObservedCounterfactual(
            initial_prediction=perception_result.prediction,
            no_tracker_features=no_tracker,
            with_tracker_features=with_tracker,
            decision=decision,
            coordinated=coordinated,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/perception/joint_openai_fixed3_dataset.yaml",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=PROJECT_ROOT / "docs/openaiAPI.txt",
    )
    parser.add_argument(
        "--proxy-url",
        help="optional loopback HTTP proxy, for example http://127.0.0.1:7897",
    )
    parser.add_argument(
        "--predicted-track-artifact",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/training/tracker_clip_v2_oof5_20260906/predicted_tracks.json"
        ),
    )
    parser.add_argument(
        "--phase-transition-graph",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/phase_transition_graph.json",
    )
    parser.add_argument("--video-id", default="VID31")
    parser.add_argument("--target-frame-id", type=int)
    parser.add_argument("--train-count", type=int, default=160)
    parser.add_argument("--gap-count", type=int, default=20)
    parser.add_argument("--dev-count", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/learned_gate_fixed3",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/api_dataset_cache/openai_fixed3_shared",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--authorize-data-upload", action="store_true")
    return parser


def _resolved_targets(
    adapter: CholecTrack20DatasetAdapter,
    video_id: str,
) -> dict[int, ResolvedSample]:
    return {
        item.inference.target_frame_id: item for item in adapter.iter_video(video_id)
    }


def _first_full_window_frame(
    adapter: CholecTrack20DatasetAdapter, video_id: str
) -> int:
    for resolved in adapter.iter_video(video_id):
        if len(resolved.inference.causal_frame_ids) == adapter.causal_window_size:
            return resolved.inference.target_frame_id
    raise ValueError("selected video has no complete three-frame causal window")


def _partition_for_position(
    position: int,
    *,
    train_count: int,
    gap_count: int,
    dev_count: int,
) -> str | None:
    if position < train_count:
        return "train"
    if position < train_count + gap_count:
        return None
    if position < train_count + gap_count + dev_count:
        return "dev"
    raise IndexError(position)


def _evenly_spaced(
    sequence: tuple[ResolvedSample, ...], count: int
) -> tuple[ResolvedSample, ...]:
    """Sample a full temporal span deterministically while preserving order."""

    if count <= 0 or count > len(sequence):
        raise ValueError("evenly spaced count must fit the source sequence")
    if count == 1:
        return (sequence[0],)
    indices = tuple(
        (position * (len(sequence) - 1)) // (count - 1) for position in range(count)
    )
    if len(set(indices)) != count:
        raise RuntimeError("evenly spaced selection produced duplicate positions")
    return tuple(sequence[index] for index in indices)


def main() -> None:
    args = _parser().parse_args()
    if not args.authorize_data_upload:
        raise SystemExit("Refusing image upload without --authorize-data-upload")
    for name in ("train_count", "gap_count", "dev_count"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    total = args.train_count + args.gap_count + args.dev_count
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    config = load_api_config(args.config)
    if (
        config.provider != "openai"
        or config.requested_model_identifier != "gpt-5.6-sol"
    ):
        raise SystemExit("training collector requires the fixed official OpenAI config")
    if config.max_causal_frames != 3 or config.max_api_images != 3:
        raise SystemExit("training collector requires the fixed three-frame config")

    dataset_root = args.dataset_root.expanduser().resolve()
    adapter = CholecTrack20DatasetAdapter(
        dataset_root,
        causal_window_size=config.max_causal_frames,
    )
    video_id = args.video_id.upper()
    entry = adapter.entries.get(video_id)
    if entry is None or entry.split is not DatasetSplit.TRAINING:
        raise SystemExit("Learned Gate collection is restricted to Training videos")
    first_target = args.target_frame_id or _first_full_window_frame(adapter, video_id)
    resolved_sequence = tuple(
        item
        for item in adapter.iter_video(video_id)
        if item.inference.target_frame_id >= first_target
    )
    selected_resolved = _evenly_spaced(resolved_sequence, total)
    if len(selected_resolved) != total:
        raise SystemExit("selected Training video does not contain enough samples")
    selected_samples = tuple(item.inference for item in selected_resolved)
    if selected_samples[0].target_frame_id != first_target:
        raise SystemExit("--target-frame-id is not an available Training GT point")
    selection = RolloutSelection(
        mode="engineering",
        split=None,
        video_ids=(video_id,),
        samples=selected_samples,
        frame_counts={video_id: len(selected_samples)},
    )

    phase_allowed_ivt = dict(
        build_phase_ivt_compatibility_from_training_adapter(adapter)
    )
    observer = _CounterfactualObserver(phase_allowed_ivt)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"learned_gate_training_{video_id}_{timestamp}"
    api_output = destination / "collection_runs" / run_id
    api_output.parent.mkdir(parents=True, exist_ok=True)
    artifact = run_dataset_api_rollout(
        config=config,
        selection=selection,
        dataset_root=dataset_root,
        output_dir=api_output,
        cache_root=args.cache_root,
        run_id=run_id,
        max_provider_calls=2 * total,
        api_key=load_api_key_file(args.api_key_file),
        authorize_data_upload=True,
        pipeline_profile="counterfactual_verify",
        evidence_threshold=0.60,
        context_profile="track_workflow",
        phase_transition_graph=args.phase_transition_graph,
        predicted_track_artifact=args.predicted_track_artifact,
        event_memory_enabled=True,
        progress_enabled=not args.no_progress,
        report_mode="template_report",
        report_window_frames=30,
        engineering_target_frame_id=first_target,
        gate_observer=observer,
        supervised_point_selection=True,
        proxy_url=args.proxy_url,
        phase_allowed_ivt=phase_allowed_ivt,
    )
    if len(observer.records) != total:
        raise RuntimeError(
            "counterfactual observer did not receive every selected frame"
        )

    targets = _resolved_targets(adapter, video_id)
    examples: list[LearnedGateExample] = []
    excluded_frames: list[int] = []
    for position, sample in enumerate(selection.samples):
        partition = _partition_for_position(
            position,
            train_count=args.train_count,
            gap_count=args.gap_count,
            dev_count=args.dev_count,
        )
        if partition is None:
            excluded_frames.append(sample.target_frame_id)
            continue
        resolved = targets[sample.target_frame_id]
        target = resolved.frame_supervision
        if target is None:
            raise RuntimeError("Training counterfactual has no frame-level GT")
        observed = observer.records[(video_id, sample.target_frame_id)]
        no_tracker = observed.no_tracker_features
        with_tracker = observed.with_tracker_features
        examples.append(
            build_example(
                sample_id=f"{video_id}:{sample.target_frame_id}",
                video_id=video_id,
                frame_id=sample.target_frame_id,
                partition=partition,
                no_tracker_features=no_tracker.as_mapping(),
                with_tracker_features=with_tracker.as_mapping(),
                initial_prediction=observed.initial_prediction,
                verified_prediction=observed.coordinated.prediction,
                target=target,
            )
        )

    examples_path = write_examples(destination / "training_examples.jsonl", examples)
    split_manifest = {
        "schema_version": "learned_gate_training_split_v1",
        "source_split": "Training",
        "video_id": video_id,
        "fixed_causal_frames": 3,
        "initial_prompt_uses_tracker": False,
        "train_frame_ids": [
            item.frame_id for item in examples if item.partition == "train"
        ],
        "gap_frame_ids": excluded_frames,
        "dev_frame_ids": [
            item.frame_id for item in examples if item.partition == "dev"
        ],
        "train_positive": sum(
            item.benefit_label for item in examples if item.partition == "train"
        ),
        "dev_positive": sum(
            item.benefit_label for item in examples if item.partition == "dev"
        ),
        "config_sha256": sha256_file(args.config),
        "track_artifact_sha256": sha256_file(args.predicted_track_artifact),
        "phase_graph_sha256": sha256_file(args.phase_transition_graph),
        "api_rollout_artifact": str(
            (api_output / "dataset_rollout_artifact.json").resolve()
        ),
        "provider": artifact["provider"],
        "model_requested": artifact["model_requested"],
    }
    (destination / "training_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.collect_only:
        print(f"LEARNED_GATE_COLLECTION_COMPLETE examples={examples_path}")
        return
    summary = fit_learned_gate_pair(examples, output_dir=destination, seed=args.seed)
    print(
        "LEARNED_GATE_END_TO_END_COMPLETE "
        + json.dumps(
            {
                "examples": str(examples_path),
                "train_positive": split_manifest["train_positive"],
                "dev_positive": split_manifest["dev_positive"],
                "training_summary": str(destination / "training_summary.json"),
                "variants": ["gate_no_tracker", "gate_with_tracker"],
                "example_count": summary["example_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
