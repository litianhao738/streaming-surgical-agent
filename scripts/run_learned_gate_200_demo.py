"""Run the fixed-prefix Validation pilot for the Learned Gate teacher demo."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
from surgical_agent.data.schemas import DatasetSplit, FrameSupervisionTarget
from surgical_agent.perception.context_builder import PerceptionContext
from surgical_agent.perception.contracts import JointPerceptionResult
from surgical_agent.research.gate.benefit_model import FrozenLinearBenefitArtifact
from surgical_agent.research.gate.demo import (
    DemoObservation,
    evaluate_demo_group,
    frame_target_from_evaluation,
    load_demo_observations,
    oracle_benefit_labels,
    write_demo_observations,
)
from surgical_agent.research.signals.contracts import EvidenceProfile
from surgical_agent.research.signals.phase_graph import (
    build_phase_ivt_compatibility_from_training_adapter,
)
from surgical_agent.research.verification.contracts import CoordinationResult
from surgical_agent.systems.pipeline import GateDecision

_VIDEOS = ("VID110", "VID30")
_PARENT_POINTS_PER_VIDEO = 100
_PILOT_COUNT = 80
_PILOT_LABEL = "DEMO / PILOT ONLY — 80-frame Validation pilot."
_GROUPS = {
    "A_base": {
        "label": "A_base",
        "tracker": False,
        "learned": False,
        "profile": "selective_verify",
        "artifact": None,
    },
    "B_tracker": {
        "label": "B_tracker",
        "tracker": True,
        "learned": False,
        "profile": "selective_verify",
        "artifact": None,
    },
    "C_gate": {
        "label": "C_gate",
        "tracker": False,
        "learned": True,
        "profile": "learned_gate",
        "artifact": "gate_no_tracker.json",
    },
    "D_full": {
        "label": "D_full",
        "tracker": True,
        "learned": True,
        "profile": "learned_gate",
        "artifact": "gate_with_tracker.json",
    },
}


class _Observer:
    def __init__(self) -> None:
        self.records: dict[tuple[str, int], DemoObservation] = {}

    def observe(
        self,
        *,
        context: PerceptionContext,
        perception_result: JointPerceptionResult,
        evidence: EvidenceProfile,
        decision: GateDecision,
        coordinated: CoordinationResult,
    ) -> None:
        del evidence
        item = DemoObservation(
            video_id=context.sample.video_id,
            frame_id=context.sample.target_frame_id,
            initial_prediction=perception_result.prediction,
            final_prediction=coordinated.prediction,
            gate_action=decision.action,
            verification_status=coordinated.verification_status,
            flagged_fields=decision.flagged_fields,
            benefit_probability=decision.benefit_probability,
        )
        if item.identity in self.records:
            raise ValueError("duplicate demo observation")
        self.records[item.identity] = item


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("manifest", "oracle", "groups", "report", "all"),
        default="all",
    )
    parser.add_argument(
        "--group",
        choices=("all", *_GROUPS),
        default="all",
        help="limit --stage groups to one ablation",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/perception/joint_openai_fixed6_dataset.yaml",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=PROJECT_ROOT / "docs/openaiAPI.txt",
    )
    parser.add_argument("--proxy-url")
    parser.add_argument(
        "--gate-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/gate/demo",
    )
    parser.add_argument(
        "--predicted-track-artifact",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/tracker/predicted_tracks.json",
    )
    parser.add_argument(
        "--phase-transition-graph",
        type=Path,
        default=PROJECT_ROOT / "artifacts/training/phase_transition_graph.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluation/learned_gate_200_demo",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/api_dataset_cache/openai_fixed6_shared",
    )
    parser.add_argument("--points-per-video", type=int, default=100)
    parser.add_argument(
        "--pilot-count",
        type=int,
        default=_PILOT_COUNT,
        help="fixed prefix length taken from the existing 200-point manifest",
    )
    parser.add_argument("--evidence-threshold", type=float, default=0.60)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--authorize-data-upload", action="store_true")
    return parser


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "validation_200_manifest.json"


def _pilot_manifest_path(output_dir: Path) -> Path:
    return output_dir / "validation_80_pilot_manifest.json"


def _eligible_by_video(
    adapter: CholecTrack20DatasetAdapter,
    *,
    points_per_video: int,
) -> dict[str, tuple[ResolvedSample, ...]]:
    selected: dict[str, tuple[ResolvedSample, ...]] = {}
    for video_id in _VIDEOS:
        entry = adapter.entries[video_id]
        if entry.split is not DatasetSplit.VALIDATION:
            raise ValueError("demo videos must belong to Validation")
        values = tuple(
            item
            for item in adapter.iter_video(video_id)
            if len(item.inference.causal_frame_ids) == 6 and item.evaluation is not None
        )[:points_per_video]
        if len(values) != points_per_video:
            raise ValueError(f"{video_id} has too few eligible Validation points")
        selected[video_id] = values
    return selected


def freeze_or_load_manifest(
    adapter: CholecTrack20DatasetAdapter,
    *,
    output_dir: Path,
    points_per_video: int,
) -> tuple[dict[str, tuple[ResolvedSample, ...]], Path]:
    path = _manifest_path(output_dir)
    selected = _eligible_by_video(adapter, points_per_video=points_per_video)
    payload = {
        "schema_version": "learned_gate_validation_manifest_v1",
        "source_split": "Validation",
        "demo_only": True,
        "paper_final": False,
        "fixed_six_frames": True,
        "points_per_video": points_per_video,
        "sample_count": points_per_video * len(_VIDEOS),
        "video_ids": list(_VIDEOS),
        "samples": [
            {
                "video_id": item.inference.video_id,
                "target_frame_id": item.inference.target_frame_id,
                "causal_frame_ids": list(item.inference.causal_frame_ids),
            }
            for video_id in _VIDEOS
            for item in selected[video_id]
        ],
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                "existing Validation manifest differs from canonical selection"
            )
    else:
        _json_write(path, payload)
    return selected, path


def select_frozen_manifest_prefix(
    selected: dict[str, tuple[ResolvedSample, ...]],
    *,
    parent_manifest_path: Path,
    output_dir: Path,
    pilot_count: int,
) -> tuple[dict[str, tuple[ResolvedSample, ...]], Path]:
    """Take an exact ordered prefix from the already frozen parent manifest."""

    if pilot_count != _PILOT_COUNT:
        raise ValueError(f"this teacher pilot requires exactly {_PILOT_COUNT} points")
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    rows = parent.get("samples")
    if not isinstance(rows, list) or len(rows) < pilot_count:
        raise ValueError("parent Validation manifest has too few samples")
    canonical = {
        (item.inference.video_id, item.inference.target_frame_id): item
        for video_id in _VIDEOS
        for item in selected[video_id]
    }
    prefix_rows = rows[:pilot_count]
    grouped: dict[str, list[ResolvedSample]] = {video_id: [] for video_id in _VIDEOS}
    normalized_rows: list[dict[str, object]] = []
    for index, row in enumerate(prefix_rows):
        if not isinstance(row, dict):
            raise TypeError(f"parent manifest sample {index} is not an object")
        video_id = row.get("video_id")
        frame_id = row.get("target_frame_id")
        causal_ids = row.get("causal_frame_ids")
        if (
            not isinstance(video_id, str)
            or video_id not in grouped
            or not isinstance(frame_id, int)
            or isinstance(frame_id, bool)
            or not isinstance(causal_ids, list)
        ):
            raise ValueError(f"parent manifest sample {index} is invalid")
        item = canonical.get((video_id, frame_id))
        if item is None or list(item.inference.causal_frame_ids) != causal_ids:
            raise ValueError(
                f"parent manifest sample {video_id}:{frame_id} is not canonical"
            )
        grouped[video_id].append(item)
        normalized_rows.append(
            {
                "video_id": video_id,
                "target_frame_id": frame_id,
                "causal_frame_ids": list(causal_ids),
            }
        )
    prefix = {video_id: tuple(grouped[video_id]) for video_id in _VIDEOS}
    active_video_ids = [video_id for video_id in _VIDEOS if prefix[video_id]]
    payload = {
        "schema_version": "learned_gate_validation_pilot_manifest_v1",
        "result_scope": _PILOT_LABEL,
        "source_split": "Validation",
        "demo_only": True,
        "paper_final": False,
        "selection_policy": "ordered_prefix_of_frozen_parent_manifest",
        "parent_manifest": parent_manifest_path.name,
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "prefix_count": pilot_count,
        "sample_count": pilot_count,
        "video_ids": active_video_ids,
        "samples": normalized_rows,
    }
    path = _pilot_manifest_path(output_dir)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing 80-frame pilot manifest differs from prefix")
    else:
        _json_write(path, payload)
    return prefix, path


def _selected_video_ids(
    selected: dict[str, tuple[ResolvedSample, ...]],
) -> tuple[str, ...]:
    return tuple(video_id for video_id in _VIDEOS if selected[video_id])


def _selection(video_id: str, values: tuple[ResolvedSample, ...]) -> RolloutSelection:
    samples = tuple(item.inference for item in values)
    return RolloutSelection(
        mode="engineering",
        split=None,
        video_ids=(video_id,),
        samples=samples,
        frame_counts={video_id: len(samples)},
    )


def _targets(
    selected: dict[str, tuple[ResolvedSample, ...]],
) -> dict[tuple[str, int], FrameSupervisionTarget]:
    targets: dict[tuple[str, int], FrameSupervisionTarget] = {}
    for video_id in _selected_video_ids(selected):
        for item in selected[video_id]:
            if item.evaluation is None:
                raise ValueError("Validation demo point has no evaluation target")
            target = frame_target_from_evaluation(item.evaluation)
            if not any(getattr(target.mask, task) for task in (*TASK_ORDER,)):
                raise ValueError("Validation demo point has no valid tasks")
            targets[(video_id, item.inference.target_frame_id)] = target
    return targets


TASK_ORDER = ("instrument", "verb", "target", "ivt", "phase")


def _run_profile(
    *,
    name: str,
    selected: dict[str, tuple[ResolvedSample, ...]],
    args: argparse.Namespace,
    pipeline_profile: str,
    tracker: bool,
    gate_artifact: Path | None,
) -> tuple[tuple[DemoObservation, ...], tuple[dict[str, object], ...]]:
    config = load_api_config(args.config)
    observer = _Observer()
    artifacts: list[dict[str, object]] = []
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    active_video_ids = _selected_video_ids(selected)
    for video_id in active_video_ids:
        values = selected[video_id]
        first_target = values[0].inference.target_frame_id
        run_id = f"demo_{name}_{video_id}_{timestamp}"
        output = args.output_dir / "runs" / run_id
        artifact = run_dataset_api_rollout(
            config=config,
            selection=_selection(video_id, values),
            dataset_root=args.dataset_root,
            output_dir=output,
            cache_root=args.cache_root,
            run_id=run_id,
            max_provider_calls=2 * len(values),
            api_key=load_api_key_file(args.api_key_file),
            authorize_data_upload=True,
            pipeline_profile=pipeline_profile,
            evidence_threshold=args.evidence_threshold,
            gate_artifact=gate_artifact,
            context_profile="track_workflow" if tracker else "workflow",
            phase_transition_graph=args.phase_transition_graph,
            phase_allowed_ivt=args.phase_allowed_ivt,
            predicted_track_artifact=(
                args.predicted_track_artifact if tracker else None
            ),
            event_memory_enabled=True,
            progress_enabled=not args.no_progress,
            report_mode="template_report",
            report_window_frames=30,
            engineering_target_frame_id=first_target,
            gate_observer=observer,
            supervised_point_selection=True,
            proxy_url=args.proxy_url,
        )
        artifact["artifact_path"] = str(
            (output / "dataset_rollout_artifact.json").resolve()
        )
        artifacts.append(artifact)
    ordered = tuple(
        observer.records[(video_id, item.inference.target_frame_id)]
        for video_id in active_video_ids
        for item in selected[video_id]
    )
    expected = sum(len(selected[video_id]) for video_id in active_video_ids)
    if len(ordered) != expected:
        raise RuntimeError("demo observer did not receive every frame")
    return ordered, tuple(artifacts)


def _aggregate_api(
    artifacts: tuple[dict[str, object], ...],
    *,
    sample_count: int,
) -> dict[str, object]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    usage_fields = (
        "provider_calls",
        "logical_calls",
        "cache_hits",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "visible_output_tokens",
    )
    totals = {name: 0 for name in usage_fields}
    targeted_input_tokens = 0
    ttft_count = 0
    ttft_sum = 0.0
    latency_count = 0
    latency_sum = 0.0
    for artifact in artifacts:
        usage = artifact["usage"]
        for name in usage_fields:
            totals[name] += int(usage[name])
        telemetry = artifact["telemetry_summary"]
        ttft = telemetry["time_to_first_token_ms"]
        latency = telemetry["total_latency_ms"]
        ttft_count += int(ttft["count"])
        ttft_sum += float(ttft["sum"])
        latency_count += int(latency["count"])
        latency_sum += float(latency["sum"])
        artifact_path = Path(artifact["artifact_path"])
        usage_path = artifact_path.parent / "api_usage.jsonl"
        with usage_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                prompt_version = row["request"]["prompt_version"]
                if (
                    not row["cache_hit"]
                    and isinstance(prompt_version, str)
                    and prompt_version.startswith("targeted_verification")
                ):
                    targeted_input_tokens += int(row["prompt_tokens"])
    result = {
        **totals,
        "result_scope": _PILOT_LABEL,
        "sample_count": sample_count,
        "logical_api_calls_per_frame": totals["logical_calls"] / sample_count,
        "new_provider_calls_per_frame": totals["provider_calls"] / sample_count,
        "verification_input_tokens": targeted_input_tokens,
        "mean_ttft_ms_new_provider_calls": (
            ttft_sum / ttft_count if ttft_count else None
        ),
        "mean_total_latency_ms_new_provider_calls": (
            latency_sum / latency_count if latency_count else None
        ),
    }
    result["token_statistics"] = {
        "totals": {
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "reasoning_tokens": totals["reasoning_tokens"],
            "visible_output_tokens": totals["visible_output_tokens"],
            "verification_input_tokens": targeted_input_tokens,
        },
        "per_frame": {
            "prompt_tokens": totals["prompt_tokens"] / sample_count,
            "completion_tokens": totals["completion_tokens"] / sample_count,
            "reasoning_tokens": totals["reasoning_tokens"] / sample_count,
            "visible_output_tokens": totals["visible_output_tokens"] / sample_count,
            "verification_input_tokens": targeted_input_tokens / sample_count,
        },
    }
    return result


def _assert_shared_initial(
    oracle: tuple[DemoObservation, ...],
    group: tuple[DemoObservation, ...],
) -> None:
    oracle_by_id = {item.identity: item.initial_prediction for item in oracle}
    if tuple(item.identity for item in oracle) != tuple(
        item.identity for item in group
    ):
        raise ValueError("group identities differ from the frozen oracle")
    if any(item.initial_prediction != oracle_by_id[item.identity] for item in group):
        raise ValueError("group initial predictions are not shared exactly")


def _latest_complete_profile_artifacts(
    *,
    name: str,
    selected: dict[str, tuple[ResolvedSample, ...]],
    output_dir: Path,
) -> tuple[dict[str, object], ...]:
    artifacts: list[dict[str, object]] = []
    for video_id in _selected_video_ids(selected):
        expected = len(selected[video_id])
        candidates = sorted(
            (output_dir / "runs").glob(f"demo_{name}_{video_id}_*"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for run_dir in candidates:
            artifact_path = run_dir / "dataset_rollout_artifact.json"
            if not artifact_path.is_file():
                continue
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            completed = artifact.get("completed_frame_counts")
            if not isinstance(completed, dict) or completed.get(video_id) != expected:
                continue
            artifact["artifact_path"] = str(artifact_path.resolve())
            artifacts.append(artifact)
            break
        else:
            raise FileNotFoundError(
                f"no complete saved run for {name}:{video_id} ({expected} points)"
            )
    return tuple(artifacts)


def _refresh_saved_api_metrics(
    selected: dict[str, tuple[ResolvedSample, ...]],
    args: argparse.Namespace,
) -> None:
    sample_count = sum(len(values) for values in selected.values())
    profiles = {
        "oracle_counterfactual_verify": args.output_dir / "oracle" / "api_metrics.json",
        **{
            name: args.output_dir / "groups" / name / "api_metrics.json"
            for name in _GROUPS
        },
    }
    for name, destination in profiles.items():
        artifacts = _latest_complete_profile_artifacts(
            name=name,
            selected=selected,
            output_dir=args.output_dir,
        )
        _json_write(
            destination,
            _aggregate_api(artifacts, sample_count=sample_count),
        )


def _ensure_gate_artifacts(gate_dir: Path) -> None:
    for name in ("gate_no_tracker.json", "gate_with_tracker.json"):
        artifact = FrozenLinearBenefitArtifact.from_json(gate_dir / name)
        if artifact.source_split != "training":
            raise ValueError("Learned Gate artifact is not Training-derived")
        if set(artifact.training_dataset_ids) & set(_VIDEOS):
            raise ValueError("Learned Gate artifact leaks Validation video IDs")


def _run_oracle(
    selected: dict[str, tuple[ResolvedSample, ...]],
    args: argparse.Namespace,
) -> None:
    observations, artifacts = _run_profile(
        name="oracle_counterfactual_verify",
        selected=selected,
        args=args,
        pipeline_profile="counterfactual_verify",
        tracker=True,
        gate_artifact=None,
    )
    path = write_demo_observations(
        args.output_dir / "oracle" / "observations.jsonl", observations
    )
    _json_write(
        args.output_dir / "oracle" / "api_metrics.json",
        _aggregate_api(artifacts, sample_count=len(observations)),
    )
    print(f"ORACLE_80_PILOT_COMPLETE observations={path}")


def _run_groups(
    selected: dict[str, tuple[ResolvedSample, ...]],
    args: argparse.Namespace,
) -> None:
    oracle_path = args.output_dir / "oracle" / "observations.jsonl"
    oracle = load_demo_observations(oracle_path)
    _ensure_gate_artifacts(args.gate_dir)
    names = tuple(_GROUPS) if args.group == "all" else (args.group,)
    for name in names:
        spec = _GROUPS[name]
        artifact_name = spec["artifact"]
        gate_artifact = (
            None if artifact_name is None else args.gate_dir / str(artifact_name)
        )
        observations, artifacts = _run_profile(
            name=name,
            selected=selected,
            args=args,
            pipeline_profile=str(spec["profile"]),
            tracker=bool(spec["tracker"]),
            gate_artifact=gate_artifact,
        )
        _assert_shared_initial(oracle, observations)
        group_dir = args.output_dir / "groups" / name
        path = write_demo_observations(group_dir / "observations.jsonl", observations)
        _json_write(
            group_dir / "api_metrics.json",
            _aggregate_api(artifacts, sample_count=len(observations)),
        )
        print(f"DEMO_GROUP_COMPLETE group={name} observations={path}")


def _format(value: object, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _write_report(
    selected: dict[str, tuple[ResolvedSample, ...]],
    manifest_path: Path,
    args: argparse.Namespace,
) -> None:
    oracle = load_demo_observations(args.output_dir / "oracle" / "observations.jsonl")
    targets = _targets(selected)
    labels = oracle_benefit_labels(oracle, targets)
    results: dict[str, dict[str, object]] = {}
    for name in _GROUPS:
        observations = load_demo_observations(
            args.output_dir / "groups" / name / "observations.jsonl"
        )
        _assert_shared_initial(oracle, observations)
        metrics = evaluate_demo_group(
            observations,
            targets=targets,
            benefit_labels=labels,
        )
        api = json.loads(
            (args.output_dir / "groups" / name / "api_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        metrics["api"] = api
        metrics["result_scope"] = _PILOT_LABEL
        results[name] = metrics
    summary = {
        "schema_version": "learned_gate_validation_pilot_results_v1",
        "result_scope": _PILOT_LABEL,
        "demo_only": True,
        "paper_final": False,
        "pilot_manifest": manifest_path.name,
        "pilot_manifest_sha256": sha256_file(manifest_path),
        "parent_manifest": args.parent_manifest_path.name,
        "parent_manifest_sha256": sha256_file(args.parent_manifest_path),
        "selection_policy": "first_80_of_frozen_200_point_manifest",
        "sample_count": len(targets),
        "beneficial_count": sum(labels.values()),
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "fixed_six_frames": True,
        "groups": results,
    }
    _json_write(args.output_dir / "results.json", summary)

    columns = [
        "Result Scope",
        "Method",
        "Tracker",
        "Learned Gate",
        "Instrument F1",
        "Verb F1",
        "Target F1",
        "IVT F1",
        "Phase Accuracy",
        "Verification Rate",
        "Gate Precision",
        "Gate Recall",
        "Gate F1",
        "Repair Success Rate",
        "Logical API Calls / Frame",
        "New Provider Calls / Frame",
        "Mean TTFT ms (new calls)",
        "Mean Latency ms (new calls)",
        "Prompt Tokens",
        "Completion Tokens",
        "Reasoning Tokens",
        "Visible Output Tokens",
        "Verification Input Tokens",
    ]
    rows: list[list[object]] = []
    for name, spec in _GROUPS.items():
        metric = results[name]
        recognition = metric["recognition"]
        gate = metric["gate"]
        api = metric["api"]
        rows.append(
            [
                _PILOT_LABEL,
                spec["label"],
                "yes" if spec["tracker"] else "no",
                "yes" if spec["learned"] else "no",
                recognition["instrument_f1"],
                recognition["verb_f1"],
                recognition["target_f1"],
                recognition["ivt_f1"],
                recognition["phase_accuracy"],
                metric["verification_rate"],
                gate["precision"],
                gate["recall"],
                gate["f1"],
                metric["repair_success_rate"],
                metric["logical_calls_per_frame"],
                api["new_provider_calls_per_frame"],
                api["mean_ttft_ms_new_provider_calls"],
                api["mean_total_latency_ms_new_provider_calls"],
                api["prompt_tokens"],
                api["completion_tokens"],
                api["reasoning_tokens"],
                api["visible_output_tokens"],
                api["verification_input_tokens"],
            ]
        )
    with (args.output_dir / "teacher_demo_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    table_lines = [
        "| Method | Instrument F1 | Verb F1 | Target F1 | IVT F1 | Phase Accuracy | Verification Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    efficiency_lines = [
        "| Method | Gate P | Gate R | Gate F1 | Repair Success | API Calls/Frame | New Calls/Frame | TTFT ms | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, spec in _GROUPS.items():
        metric = results[name]
        recognition = metric["recognition"]
        gate = metric["gate"]
        api = metric["api"]
        table_lines.append(
            f"| {spec['label']} | {_format(recognition['instrument_f1'])} | "
            f"{_format(recognition['verb_f1'])} | {_format(recognition['target_f1'])} | "
            f"{_format(recognition['ivt_f1'])} | {_format(recognition['phase_accuracy'])} | "
            f"{_format(metric['verification_rate'])} |"
        )
        efficiency_lines.append(
            f"| {spec['label']} | {_format(gate['precision'])} | {_format(gate['recall'])} | "
            f"{_format(gate['f1'])} | {_format(metric['repair_success_rate'])} | "
            f"{_format(metric['logical_calls_per_frame'])} | "
            f"{_format(api['new_provider_calls_per_frame'])} | "
            f"{_format(api['mean_ttft_ms_new_provider_calls'], 1)} | "
            f"{_format(api['mean_total_latency_ms_new_provider_calls'], 1)} |"
        )
    detail_lines: list[str] = []
    for name, spec in _GROUPS.items():
        metric = results[name]
        gate = metric["gate"]
        api = metric["api"]
        detail_lines.extend(
            [
                f"### {spec['label']}",
                "",
                f"- Gate precision / recall / F1: {_format(gate['precision'])} / {_format(gate['recall'])} / {_format(gate['f1'])}",
                f"- Repair success rate: {_format(metric['repair_success_rate'])}",
                f"- Logical calls/frame / new provider calls/frame: {_format(metric['logical_calls_per_frame'])} / {_format(api['new_provider_calls_per_frame'])}",
                f"- New provider calls / cache hits: {api['provider_calls']} / {api['cache_hits']}",
                f"- Prompt / completion / reasoning / visible-output tokens: {api['prompt_tokens']} / {api['completion_tokens']} / {api['reasoning_tokens']} / {api['visible_output_tokens']}",
                f"- Verification input tokens: {api['verification_input_tokens']}",
                f"- Mean TTFT / total latency for new calls: {_format(api['mean_ttft_ms_new_provider_calls'], 1)} ms / {_format(api['mean_total_latency_ms_new_provider_calls'], 1)} ms",
                f"- Final states: `{json.dumps(metric['state_counts'], sort_keys=True)}`",
                "",
            ]
        )
    report = "\n".join(
        [
            "# Full Learned Gate — 80-frame Validation teacher demo",
            "",
            f"> **{_PILOT_LABEL}**",
            "",
            "## Fixed protocol",
            "",
            f"- Frozen parent manifest: `{args.parent_manifest_path.name}` (`{sha256_file(args.parent_manifest_path)}`)",
            f"- Pilot manifest: `{manifest_path.name}` (`{sha256_file(manifest_path)}`)",
            "- Selection: exact first 80 rows of the frozen 200-point Validation manifest; no reselection or random sampling",
            f"- Samples: {len(targets)} ({', '.join(_selected_video_ids(selected))})",
            "- Input: identical fixed continuous six-frame windows; history detail low, current detail auto",
            "- Backbone: official OpenAI `gpt-5.6-sol`; strict structured JSON; reasoning effort none",
            "- Tracker is excluded from the Initial GPT prompt and used only by the downstream Gate/Coordinator path",
            "- The four groups share the initial-request cache, specialist implementation, memory policy and template event reporter",
            "- Gate weights and threshold are Training-only; these Validation points are evaluation-only",
            "- Instrument/Verb/Target/IVT F1 is mean frame-level set-F1; Phase Accuracy is frame accuracy",
            f"- Oracle beneficial verification points: {sum(labels.values())}/{len(labels)}",
            "",
            "## Main table",
            "",
            *table_lines,
            "",
            "## Gate and efficiency table",
            "",
            *efficiency_lines,
            "",
            "`API Calls/Frame` is the logical pipeline count (one initial call plus routed verification), independent of cache replay. `New Calls/Frame` shows fresh provider traffic after shared-cache reuse.",
            "",
            "## Gate, efficiency and state details",
            "",
            *detail_lines,
        ]
    )
    (args.output_dir / "teacher_demo_report.md").write_text(
        report + "\n", encoding="utf-8"
    )
    print(
        f"TEACHER_DEMO_REPORT_COMPLETE report={args.output_dir / 'teacher_demo_report.md'}"
    )


def main() -> None:
    args = _parser().parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.cache_root = args.cache_root.expanduser().resolve()
    args.gate_dir = args.gate_dir.expanduser().resolve()
    if args.points_per_video != _PARENT_POINTS_PER_VIDEO:
        raise SystemExit(
            "the frozen teacher demo requires exactly 100 points per video"
        )
    if args.pilot_count != _PILOT_COUNT:
        raise SystemExit("the current teacher pilot requires exactly 80 points")
    api_stages = {"oracle", "groups", "all"}
    if args.stage in api_stages and not args.authorize_data_upload:
        raise SystemExit("Refusing image upload without --authorize-data-upload")
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=6)
    args.phase_allowed_ivt = build_phase_ivt_compatibility_from_training_adapter(
        adapter
    )
    parent_selected, parent_manifest_path = freeze_or_load_manifest(
        adapter,
        output_dir=args.output_dir,
        points_per_video=args.points_per_video,
    )
    selected, manifest_path = select_frozen_manifest_prefix(
        parent_selected,
        parent_manifest_path=parent_manifest_path,
        output_dir=args.output_dir,
        pilot_count=args.pilot_count,
    )
    args.parent_manifest_path = parent_manifest_path
    print(f"VALIDATION_200_PARENT_MANIFEST_READY path={parent_manifest_path}")
    print(f"VALIDATION_80_PILOT_MANIFEST_READY path={manifest_path}")
    if args.stage == "manifest":
        return
    if args.stage in {"oracle", "all"}:
        _run_oracle(selected, args)
        if args.stage == "oracle":
            return
    if args.stage in {"groups", "all"}:
        _run_groups(selected, args)
        if args.stage == "groups":
            return
    if args.stage in {"report", "all"}:
        _refresh_saved_api_metrics(selected, args)
        _write_report(selected, manifest_path, args)


if __name__ == "__main__":
    main()
