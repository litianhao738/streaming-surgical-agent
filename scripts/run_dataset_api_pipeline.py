"""Run a bounded CholecTrack20 selection through the joint API pipeline."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from surgical_agent.api.accounting import CompleteAccountingTransport
from surgical_agent.api.budget import ProviderCallBudget
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import ProviderTransport, thaw_json
from surgical_agent.api.credentials import (
    SecretValue,
    assert_secret_absent,
    resolve_api_key,
)
from surgical_agent.api.errors import ApiCallFailure, ApiContractError, ApiError
from surgical_agent.api.openrouter_routing import (
    STRICT_OPENAI_ROUTING_PROFILE,
    routing_profile_from_options,
)
from surgical_agent.api.proxy import configure_local_proxy
from surgical_agent.api.registry import build_transport, build_validator
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.config.context_experiment import load_context_experiment
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
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.joint_api_vlm import JointPerceptionRequestBuilder
from surgical_agent.perception.main_h0 import (
    is_main_h0_config,
    validate_main_h0_config,
)
from surgical_agent.perception.schema import (
    GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION,
)
from surgical_agent.research.reporting import EventReportGenerator
from surgical_agent.research.signals.contracts import PhaseTransitionGraph
from surgical_agent.research.signals.phase_graph import (
    build_phase_transition_graph_from_training_adapter,
    load_phase_transition_graph,
)
from surgical_agent.systems.api_dataset_system import DatasetApiPipelineSystem
from surgical_agent.tracking.predicted_provider import (
    PrecomputedPredictedTrackProvider,
)

_MOCK_JOINT_VERSION = GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
_REAL_JOINT_VERSION = GATE_OWNED_COMPACT_JOINT_PERCEPTION_SCHEMA_VERSION
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"
_REAL_MODEL = "openai/gpt-5.6-sol"
_OPENAI_REAL_MODEL = "gpt-5.6-sol"
_LUNA_MODEL = "openai/gpt-5.6-luna"
_MOCK_MODEL = "mock-joint-perception-v1"
_MOCK_ENDPOINT = "mock://local/p3"
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*\Z")


def _apply_cli_model_override(
    config: ApiConfig,
    *,
    mode: str,
    model: str | None,
) -> ApiConfig:
    """Apply an explicit engineering model without editing YAML."""

    if model is None:
        return config
    normalized = model.strip()
    if not normalized:
        raise ApiContractError("model override must not be empty")
    if mode != "engineering":
        raise ApiContractError("paper mode forbids model overrides")
    if config.provider not in {"openrouter", "openai_compatible"} or (
        config.mode != "real"
    ):
        raise ApiContractError(
            "--model requires a real OpenRouter or OpenAI-compatible config"
        )
    if config.provider == "openrouter" and normalized != _REAL_MODEL:
        if is_main_h0_config(config) and normalized == config.requested_model_identifier:
            return config
        raise ApiContractError(
            f"the current pipeline protocol approves only {_REAL_MODEL}"
        )
    overridden = replace(config, requested_model_identifier=normalized)
    overridden.validate()
    return overridden


def _normalize_compatible_endpoint(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ApiContractError("base-url override must not be empty")
    parsed = urlsplit(normalized)
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        path = parsed.path.rstrip("/")
    else:
        path = parsed.path.rstrip("/") + "/chat/completions"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )


def _apply_cli_base_url_override(
    config: ApiConfig,
    *,
    mode: str,
    base_url: str | None,
) -> ApiConfig:
    if base_url is None:
        return config
    if mode != "engineering":
        raise ApiContractError("paper mode forbids base-url overrides")
    if config.provider != "openai_compatible" or config.mode != "real":
        raise ApiContractError(
            "--base-url requires a real OpenAI-compatible config"
        )
    overridden = replace(
        config,
        endpoint_identifier=_normalize_compatible_endpoint(base_url),
    )
    overridden.validate()
    return overridden


def _validate_parallelism(value: int) -> None:
    """Keep one video stream sequential until state-safe sharding is implemented."""

    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ApiContractError("parallelism must be a positive integer")
    if value != 1:
        raise ApiContractError(
            "one causal video stream requires --parallelism 1; "
            "parallelize independent video runs as separate processes"
        )


def build_parser(*, fixed_profile: str | None = None) -> argparse.ArgumentParser:
    """Build the dataset rollout CLI without credential-bearing defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("engineering", "paper"), required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--target-frame-id",
        type=int,
        help="engineering-mode first target frame; useful for one full-window probe",
    )
    parser.add_argument("--split", choices=("validation", "testing"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data/CholecTrack20",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/perception/joint_openrouter_h0.yaml",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate the main H0 config and prepare real requests without API calls",
    )
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument(
        "--api-key",
        help="API key supplied directly in CMD (engineering convenience)",
    )
    credentials.add_argument("--api-key-file", type=Path)
    parser.add_argument(
        "--model",
        help=(
            "engineering-only model override; compatible configs accept any model ID"
        ),
    )
    parser.add_argument(
        "--base-url",
        help=(
            "engineering-only OpenAI-compatible base URL or full chat endpoint"
        ),
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help=(
            "causal stream worker count; currently must be 1 because frames "
            "within one video commit state autoregressively"
        ),
    )
    parser.add_argument(
        "--proxy-url",
        help="optional loopback HTTP proxy, for example http://127.0.0.1:7897",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/api_dataset",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/api_dataset_cache/main_h0",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--max-provider-calls", default="exact-selection")
    if fixed_profile is None:
        parser.add_argument(
            "--pipeline-profile",
            choices=(
                "single_pass",
                "always_verify",
                "rule_gate",
                "selective_verify",
                "cascade_verify",
                "learned_gate",
                "counterfactual_verify",
            ),
            default="single_pass",
        )
    else:
        if fixed_profile not in {
            "single_pass",
            "always_verify",
            "rule_gate",
            "selective_verify",
            "cascade_verify",
            "learned_gate",
            "counterfactual_verify",
        }:
            raise ValueError("unsupported fixed pipeline profile")
        parser.set_defaults(pipeline_profile=fixed_profile)
    parser.add_argument("--evidence-threshold", type=float, default=0.75)
    parser.add_argument("--gate-artifact", type=Path)
    parser.add_argument("--verification-config", type=Path)
    parser.add_argument(
        "--report-mode",
        choices=("template_report", "llm_report"),
        default="template_report",
    )
    parser.add_argument("--report-window-frames", type=int, default=30)
    parser.add_argument(
        "--context-profile",
        choices=("auto", "frames_only", "track_only", "workflow", "track_workflow"),
        default="auto",
    )
    parser.add_argument(
        "--event-memory",
        choices=("auto", "enabled", "disabled"),
        default="auto",
    )
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--phase-transition-graph", type=Path)
    parser.add_argument("--predicted-track-artifact", type=Path)
    parser.add_argument("--authorize-data-upload", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable dynamic progress bars",
    )
    return parser


def _same_resolved_path(first: str | Path, second: str | Path) -> bool:
    return Path(first).expanduser().resolve() == Path(second).expanduser().resolve()


def _resolve_context_runtime(
    *,
    context_profile: str,
    event_memory: str,
    phase_transition_graph: str | Path | None,
    predicted_track_artifact: str | Path | None,
    experiment_config: str | Path | None,
) -> tuple[str, bool | None, Path | None, Path | None, str | None]:
    """Resolve CLI/config context switches without silently overriding either."""

    if event_memory not in {"auto", "enabled", "disabled"}:
        raise ApiContractError("unsupported event-memory switch")
    event_memory_enabled = None if event_memory == "auto" else event_memory == "enabled"
    phase_path = (
        None
        if phase_transition_graph is None
        else Path(phase_transition_graph).expanduser().resolve()
    )
    track_path = (
        None
        if predicted_track_artifact is None
        else Path(predicted_track_artifact).expanduser().resolve()
    )
    if experiment_config is None:
        return (
            context_profile,
            event_memory_enabled,
            phase_path,
            track_path,
            None,
        )

    source = Path(experiment_config).expanduser().resolve()
    try:
        experiment = load_context_experiment(source)
    except (OSError, TypeError, ValueError) as exc:
        raise ApiContractError("invalid context experiment config") from exc
    if context_profile != "auto" and context_profile != experiment.context_profile:
        raise ApiContractError("context-profile conflicts with experiment config")
    if (
        event_memory_enabled is not None
        and event_memory_enabled != experiment.event_memory_enabled
    ):
        raise ApiContractError("event-memory conflicts with experiment config")
    if phase_path is not None and (
        experiment.phase_transition_graph is None
        or not _same_resolved_path(phase_path, experiment.phase_transition_graph)
    ):
        raise ApiContractError(
            "phase-transition-graph conflicts with experiment config"
        )
    if track_path is not None and (
        experiment.predicted_track_artifact is None
        or not _same_resolved_path(track_path, experiment.predicted_track_artifact)
    ):
        raise ApiContractError(
            "predicted-track-artifact conflicts with experiment config"
        )
    return (
        experiment.context_profile,
        experiment.event_memory_enabled,
        experiment.phase_transition_graph,
        experiment.predicted_track_artifact,
        sha256_file(source),
    )


def _validate_runtime_profile(
    pipeline_profile: str,
    *,
    evidence_threshold: float,
    gate_artifact: str | Path | None,
) -> Path | None:
    allowed = {
        "single_pass",
        "always_verify",
        "rule_gate",
        "selective_verify",
        "cascade_verify",
        "learned_gate",
        "counterfactual_verify",
    }
    if pipeline_profile not in allowed:
        raise ApiContractError("unsupported pipeline-profile")
    if (
        not isinstance(evidence_threshold, (int, float))
        or isinstance(evidence_threshold, bool)
        or not 0.0 <= float(evidence_threshold) <= 1.0
    ):
        raise ApiContractError("evidence-threshold must be in [0, 1]")
    if pipeline_profile == "learned_gate":
        if gate_artifact is None:
            raise ApiContractError("learned_gate requires --gate-artifact")
        resolved = Path(gate_artifact).expanduser().resolve()
        if not resolved.is_file():
            raise ApiContractError("gate-artifact must be an existing file")
        return resolved
    if gate_artifact is not None:
        raise ApiContractError("--gate-artifact is only valid with learned_gate")
    return None


def _load_context_components(
    *,
    pipeline_profile: str,
    context_profile: str,
    phase_transition_graph: str | Path | None,
    predicted_track_artifact: str | Path | None,
    dataset_root: Path,
    selection: RolloutSelection,
) -> tuple[
    str,
    PhaseTransitionGraph | None,
    PrecomputedPredictedTrackProvider | None,
]:
    """Resolve an executable context ablation before API/output mutation."""

    if context_profile == "auto":
        resolved_profile = "workflow"
    elif context_profile in {
        "frames_only",
        "track_only",
        "workflow",
        "track_workflow",
    }:
        resolved_profile = context_profile
    else:
        raise ApiContractError("unsupported context-profile")
    if (
        resolved_profile in {"frames_only", "track_only"}
        and phase_transition_graph is not None
    ):
        raise ApiContractError(
            f"{resolved_profile} context forbids phase-transition-graph"
        )
    track_profiles = {"track_only", "track_workflow"}
    if resolved_profile not in track_profiles and predicted_track_artifact is not None:
        raise ApiContractError("predicted-track-artifact requires a track context")
    if resolved_profile in track_profiles and predicted_track_artifact is None:
        raise ApiContractError(
            f"{resolved_profile} context requires predicted-track-artifact"
        )

    graph: PhaseTransitionGraph | None = None
    if phase_transition_graph is not None:
        try:
            graph = load_phase_transition_graph(phase_transition_graph)
        except (OSError, TypeError, ValueError) as exc:
            raise ApiContractError("invalid phase-transition-graph") from exc
        adapter = CholecTrack20DatasetAdapter(dataset_root)
        try:
            expected_graph = build_phase_transition_graph_from_training_adapter(adapter)
        except (OSError, TypeError, ValueError) as exc:
            raise ApiContractError(
                "phase-transition-graph could not be verified against Training labels"
            ) from exc
        if graph != expected_graph:
            raise ApiContractError(
                "phase-transition-graph does not match current Training labels"
            )

    track_provider: PrecomputedPredictedTrackProvider | None = None
    if predicted_track_artifact is not None:
        try:
            track_provider = PrecomputedPredictedTrackProvider.from_json(
                predicted_track_artifact
            )
            repair_manifest_sha256 = sha256_file(dataset_root / "repair_manifest.json")
            if track_provider.dataset_repair_manifest_sha256 != repair_manifest_sha256:
                raise ValueError(
                    "predicted-track artifact targets a different dataset manifest"
                )
            active_video: str | None = None
            for sample in selection.samples:
                if sample.video_id != active_video:
                    track_provider.reset(sample.video_id)
                    active_video = sample.video_id
                track_provider.snapshot(sample)
        except (OSError, TypeError, ValueError) as exc:
            raise ApiContractError("invalid predicted-track-artifact") from exc
    return resolved_profile, graph, track_provider


def validate_cli_selection(args: argparse.Namespace) -> None:
    """Reject incomplete or paper-invalid selectors without touching the dataset."""

    mode = getattr(args, "mode", None)
    video_id = getattr(args, "video_id", None)
    max_frames = getattr(args, "max_frames", None)
    target_frame_id = getattr(args, "target_frame_id", None)
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
        if target_frame_id is not None and (
            not isinstance(target_frame_id, int)
            or isinstance(target_frame_id, bool)
            or target_frame_id <= 0
        ):
            raise ApiContractError("target-frame-id must be a positive integer")
        return
    if mode == "paper":
        if video_id is not None:
            raise ApiContractError("paper mode does not accept video-id")
        if max_frames is not None:
            raise ApiContractError("paper mode forbids truncation")
        if target_frame_id is not None:
            raise ApiContractError("paper mode does not accept target-frame-id")
        if split not in {"validation", "testing"}:
            raise ApiContractError("paper mode requires validation or testing split")
        return
    raise ApiContractError("mode must be engineering or paper")


def _require_exact_dataset_config(
    config: ApiConfig,
    *,
    has_credential: bool,
    authorize_data_upload: bool,
    expected_real_model: str = _REAL_MODEL,
    selection_mode: str | None = None,
) -> None:
    if not isinstance(config, ApiConfig):
        raise TypeError("config must be an ApiConfig")
    if type(has_credential) is not bool or type(authorize_data_upload) is not bool:
        raise TypeError("credential and upload authorization state must be boolean")
    if selection_mode not in {None, "engineering", "paper"}:
        raise ApiContractError("selection mode must be engineering or paper")
    config.require_enabled()
    config.validate()
    if is_main_h0_config(config):
        validate_main_h0_config(config)
        if config.provider == "mock":
            if has_credential:
                raise ApiContractError("mock H0 rejects credential inputs")
        elif not has_credential or not authorize_data_upload:
            raise ApiContractError(
                "real H0 requires a credential and --authorize-data-upload"
            )
        return
    expected_joint_version = (
        _REAL_JOINT_VERSION
        if config.mode == "real"
        and config.provider in {"openai", "openrouter", "openai_compatible"}
        else _MOCK_JOINT_VERSION
    )
    if config.prompt_version != expected_joint_version:
        raise ApiContractError("dataset rollout requires the exact prompt version")
    if config.response_schema_version != expected_joint_version:
        raise ApiContractError("dataset rollout requires the exact response schema")
    generation = dict(config.generation_parameters)
    expected_generation = (
        {"max_output_tokens": 1536, "temperature": 0.0, "enable_thinking": False}
        if config.mode == "real" and config.provider == "openai_compatible"
        else (
            {"max_output_tokens": 1536, "reasoning": {"effort": "none"}}
            if config.mode == "real" and config.provider in {"openai", "openrouter"}
            else {"max_output_tokens": 4096}
        )
    )
    if generation != expected_generation:
        raise ApiContractError("dataset rollout requires exact generation settings")
    if config.synthetic_input_required is not False:
        raise ApiContractError("dataset rollout requires real dataset input")
    if config.cache_required is not True:
        raise ApiContractError("dataset rollout requires the request cache")
    if config.data_upload_authorized is not True:
        raise ApiContractError("dataset config must authorize data upload")
    if config.max_causal_frames != 3:
        raise ApiContractError("dataset rollout requires a three-frame causal buffer")
    expected_images = 3
    if config.max_api_images != expected_images:
        raise ApiContractError(
            f"dataset rollout requires a {expected_images}-image upload budget"
        )

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
        if config.requested_model_identifier != expected_real_model:
            raise ApiContractError(
                "real dataset rollout requires the selected approved model"
            )
        options = dict(config.provider_options)
        if (
            set(options) != {"timeout_seconds", "routing_profile"}
            or type(options["timeout_seconds"]) is not float
            or options["timeout_seconds"] != 120.0
        ):
            raise ApiContractError(
                "real OpenRouter rollout requires exact timeout and routing profile"
            )
        routing_profile = routing_profile_from_options(options)
        if (
            selection_mode == "paper"
            and routing_profile != STRICT_OPENAI_ROUTING_PROFILE
        ):
            raise ApiContractError(
                "paper mode requires the strict OpenAI routing profile"
            )
        if not authorize_data_upload:
            raise ApiContractError(
                "real dataset rollout requires --authorize-data-upload"
            )
        if not has_credential:
            raise ApiContractError(
                "real dataset rollout requires exactly one credential"
            )
        return

    if config.mode == "real" and config.provider == "openai":
        if config.endpoint_identifier != _OPENAI_ENDPOINT:
            raise ApiContractError(
                "real OpenAI rollout requires the Responses endpoint"
            )
        if config.requested_model_identifier != _OPENAI_REAL_MODEL:
            raise ApiContractError("real OpenAI rollout requires gpt-5.6-sol")
        options = dict(config.provider_options)
        required_options = {
            "timeout_seconds": 120.0,
            "frame_selection_strategy": "fixed_all",
            "history_image_detail": "low",
            "target_image_detail": "auto",
            "initial_prompt_profile": "fixed_visual_only",
            "verification_prompt_profile": "delta_visual_only",
        }
        if any(options.get(key) != value for key, value in required_options.items()):
            raise ApiContractError("real OpenAI rollout requires exact demo options")
        if set(options) - (set(required_options) | {"service_tier"}):
            raise ApiContractError(
                "real OpenAI rollout has unsupported provider options"
            )
        if not authorize_data_upload:
            raise ApiContractError(
                "real dataset rollout requires --authorize-data-upload"
            )
        if not has_credential:
            raise ApiContractError(
                "real dataset rollout requires exactly one credential"
            )
        return

    if config.mode == "real" and config.provider == "openai_compatible":
        if selection_mode != "engineering":
            raise ApiContractError(
                "OpenAI-compatible gateways are engineering-only in this protocol"
            )
        options = dict(config.provider_options)
        if set(options) != {"timeout_seconds", "response_format"}:
            raise ApiContractError(
            "compatible rollout requires timeout_seconds and response_format"
            )
        if (
            type(options["timeout_seconds"]) is not float
            or options["timeout_seconds"] != 120.0
            or options["response_format"] != "json_schema"
        ):
            raise ApiContractError("compatible rollout options are invalid")
        if not authorize_data_upload:
            raise ApiContractError(
                "real dataset rollout requires --authorize-data-upload"
            )
        if not has_credential:
            raise ApiContractError(
                "real dataset rollout requires exactly one credential"
            )
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


def _resolve_call_limit(
    value: object,
    *,
    selection_count: int,
    calls_per_state: int = 1,
) -> int:
    if selection_count <= 0:
        raise ApiContractError("rollout selection must not be empty")
    if (
        not isinstance(calls_per_state, int)
        or isinstance(calls_per_state, bool)
        or calls_per_state <= 0
    ):
        raise ApiContractError("calls_per_state must be a positive integer")
    if value == "exact-selection":
        return selection_count * calls_per_state
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
    expected = selection_count * calls_per_state
    if limit != expected:
        raise ApiContractError(
            "max-provider-calls must equal the exact profile reservation"
        )
    return limit


def _resolve_profile_call_limit(
    value: object,
    *,
    pipeline_profile: str,
    selection_count: int,
) -> int:
    return _resolve_call_limit(
        value,
        selection_count=selection_count,
        calls_per_state=(1 if pipeline_profile == "single_pass" else 2),
    )


def _validate_profile_backbones(
    pipeline_profile: str,
    *,
    initial_config: ApiConfig,
    verification_config: ApiConfig | None,
) -> tuple[str, ApiConfig]:
    if pipeline_profile == "cascade_verify":
        if verification_config is None:
            raise ApiContractError("cascade_verify requires --verification-config")
        if (
            initial_config.provider != verification_config.provider
            or initial_config.endpoint_identifier
            != verification_config.endpoint_identifier
        ):
            raise ApiContractError(
                "cascade configs require the same provider and endpoint"
            )
        if (
            initial_config.requested_model_identifier != _LUNA_MODEL
            or verification_config.requested_model_identifier != _REAL_MODEL
        ):
            raise ApiContractError("cascade requires Luna initial and Sol verification")
        return "cascade_efficiency", verification_config
    if verification_config is not None and verification_config != initial_config:
        raise ApiContractError("main profiles require one shared ApiConfig")
    return "shared", initial_config


def _latency_aggregate(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "sum": 0.0}
    total = sum(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": total / len(values),
        "sum": total,
    }


def _telemetry_summary(
    usage_summary: dict[str, object],
    records: tuple[dict[str, object], ...],
) -> dict[str, object]:
    current = [row for row in records if int(row["provider_call_count"]) > 0]
    ttft = [
        float(row["time_to_first_token_ms"])
        for row in current
        if row["time_to_first_token_ms"] is not None
    ]
    total_latency = [
        float(row["total_latency_ms"])
        for row in current
        if row["total_latency_ms"] is not None
    ]
    image_counts = [
        len(row["request"]["images"])
        for row in current
        if isinstance(row.get("request"), dict)
        and isinstance(row["request"].get("images"), list)
    ]
    return {
        "schema_version": "api_rollout_telemetry_v1",
        "prompt_tokens": usage_summary["prompt_tokens"],
        "completion_tokens": usage_summary["completion_tokens"],
        "reasoning_tokens": usage_summary["reasoning_tokens"],
        "visible_output_tokens": usage_summary["visible_output_tokens"],
        "time_to_first_token_ms": _latency_aggregate(ttft),
        "total_latency_ms": _latency_aggregate(total_latency),
        "uploaded_images_per_call": _latency_aggregate(
            [float(value) for value in image_counts]
        ),
    }


def _validate_selection_semantics(
    selection: RolloutSelection,
    *,
    dataset_root: Path,
    max_causal_frames: int,
    engineering_target_frame_id: int | None = None,
    supervised_point_selection: bool = False,
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
        if any(sample.video_id != video_id for sample in samples) or any(
            frame_ids[index] >= frame_ids[index + 1]
            for index in range(len(frame_ids) - 1)
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
    if supervised_point_selection:
        if selection.mode != "engineering" or len(selection.video_ids) != 1:
            raise ApiContractError(
                "supervised-point selection requires one engineering video"
            )
        selected_video_id = selection.video_ids[0]
        entry = adapter.entries.get(selected_video_id)
        if entry is None or entry.split is DatasetSplit.TESTING:
            raise ApiContractError("supervised-point selection forbids Testing")
        resolved = tuple(adapter.iter_video(selected_video_id))
        if engineering_target_frame_id is not None:
            resolved = tuple(
                item
                for item in resolved
                if item.inference.target_frame_id >= engineering_target_frame_id
            )
        position_by_frame_id = {
            item.inference.target_frame_id: position
            for position, item in enumerate(resolved)
        }
        positions: list[int] = []
        canonical_samples: list[object] = []
        for sample in selection.samples:
            position = position_by_frame_id.get(sample.target_frame_id)
            if position is None or resolved[position].inference != sample:
                raise ApiContractError(
                    "selection contains a non-canonical supervised split point"
                )
            positions.append(position)
            canonical_samples.append(resolved[position].inference)
        if (
            any(left >= right for left, right in pairwise(positions))
            or tuple(canonical_samples) != selection.samples
            or dict(selection.frame_counts)
            != {selected_video_id: len(canonical_samples)}
        ):
            raise ApiContractError(
                "selection must match ordered supervised split points"
            )
        return
    try:
        canonical = resolve_rollout_selection(
            adapter,
            mode=selection.mode,
            video_id=canonical_video_id,
            max_frames=canonical_max_frames,
            split=canonical_split,
            target_frame_id=engineering_target_frame_id,
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
        raise ApiContractError(
            "credential scan detected persisted credential"
        ) from None


def run_dataset_api_rollout(
    *,
    config: ApiConfig,
    verification_config: ApiConfig | None = None,
    selection: RolloutSelection,
    dataset_root: str | Path,
    output_dir: str | Path,
    cache_root: str | Path,
    run_id: str,
    max_provider_calls: int,
    api_key: SecretValue | None = None,
    authorize_data_upload: bool = False,
    transport: ProviderTransport | None = None,
    pipeline_profile: str = "single_pass",
    evidence_threshold: float = 0.75,
    gate_artifact: str | Path | None = None,
    context_profile: str = "auto",
    phase_transition_graph: str | Path | None = None,
    phase_allowed_ivt: Mapping[int, tuple[int, ...]] | None = None,
    predicted_track_artifact: str | Path | None = None,
    event_memory_enabled: bool | None = None,
    context_experiment_sha256: str | None = None,
    progress_enabled: bool = False,
    report_mode: str = "template_report",
    report_window_frames: int = 30,
    report_generator: EventReportGenerator | None = None,
    engineering_target_frame_id: int | None = None,
    gate_observer: object | None = None,
    supervised_point_selection: bool = False,
    proxy_url: str | None = None,
) -> dict[str, object]:
    """Run one preselected rollout with an optional injected transport."""

    configure_local_proxy(proxy_url)
    if is_main_h0_config(config):
        validate_main_h0_config(config)
        if pipeline_profile != "single_pass":
            raise ApiContractError("main H0 requires the single_pass profile")
        if context_profile not in {"auto", "frames_only"}:
            raise ApiContractError("main H0 requires frames_only context")
        if event_memory_enabled is True:
            raise ApiContractError("main H0 does not consume event memory")
        context_profile = "frames_only"
        event_memory_enabled = False
    if report_mode == "llm_report" and report_generator is None:
        raise ApiContractError("llm_report requires an injected event-level generator")
    if report_mode == "template_report" and report_generator is not None:
        raise ApiContractError("template_report does not accept a generator")
    if (
        not isinstance(report_window_frames, int)
        or isinstance(report_window_frames, bool)
        or report_window_frames <= 0
    ):
        raise ApiContractError("report-window-frames must be a positive integer")
    if api_key is not None and not isinstance(api_key, SecretValue):
        raise TypeError("api_key must be SecretValue or None")
    if not isinstance(selection, RolloutSelection):
        raise TypeError("selection must be a RolloutSelection")
    if event_memory_enabled is not None and type(event_memory_enabled) is not bool:
        raise TypeError("event_memory_enabled must be boolean or None")
    if context_experiment_sha256 is not None and (
        not isinstance(context_experiment_sha256, str)
        or len(context_experiment_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in context_experiment_sha256
        )
    ):
        raise ValueError("context_experiment_sha256 must be a lowercase SHA-256")
    resolved_gate_artifact = _validate_runtime_profile(
        pipeline_profile,
        evidence_threshold=evidence_threshold,
        gate_artifact=gate_artifact,
    )
    backbone_policy, resolved_verification_config = _validate_profile_backbones(
        pipeline_profile,
        initial_config=config,
        verification_config=verification_config,
    )
    _require_exact_dataset_config(
        config,
        has_credential=api_key is not None,
        authorize_data_upload=authorize_data_upload,
        selection_mode=selection.mode,
        expected_real_model=(
            _LUNA_MODEL if backbone_policy == "cascade_efficiency" else _REAL_MODEL
        ),
    )
    if backbone_policy == "cascade_efficiency":
        _require_exact_dataset_config(
            resolved_verification_config,
            has_credential=api_key is not None,
            authorize_data_upload=authorize_data_upload,
            selection_mode=selection.mode,
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
        engineering_target_frame_id=engineering_target_frame_id,
        supervised_point_selection=supervised_point_selection,
    )
    resolved_context_profile, graph, track_provider = _load_context_components(
        pipeline_profile=pipeline_profile,
        context_profile=context_profile,
        phase_transition_graph=phase_transition_graph,
        predicted_track_artifact=predicted_track_artifact,
        dataset_root=resolved_dataset_root,
        selection=selection,
    )
    call_limit = _resolve_profile_call_limit(
        max_provider_calls,
        pipeline_profile=pipeline_profile,
        selection_count=selection.expected_provider_calls,
    )
    if transport is not None:
        _require_transport_identity(config, transport)

    repair_manifest_sha256 = sha256_file(resolved_dataset_root / "repair_manifest.json")
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
            max_attempts=1 if is_main_h0_config(config) else 4,
            base_delay_seconds=0.25,
            max_delay_seconds=0.25,
        ),
        sleep=lambda _seconds: None,
        provider_call_budget=ProviderCallBudget(call_limit),
    )
    writer = FrameResultWriter(destination, run_id=run_id)
    system = DatasetApiPipelineSystem(
        client=client,
        config=config,
        verification_config=resolved_verification_config,
        writer=writer,
        media_loader=CausalApiMediaLoader(),
        pipeline_profile=pipeline_profile,
        evidence_threshold=evidence_threshold,
        gate_artifact=resolved_gate_artifact,
        context_profile=resolved_context_profile,
        phase_transition_graph=graph,
        phase_allowed_ivt=phase_allowed_ivt,
        track_provider=track_provider,
        event_memory_enabled=event_memory_enabled,
        report_mode=report_mode,
        report_window_size=report_window_frames,
        report_generator=report_generator,
        gate_observer=gate_observer,
    )
    writer.begin()
    try:
        try:
            result = system.run(
                selection,
                run_id=run_id,
                defer_completion=True,
                progress_enabled=progress_enabled,
            )
            base_status = (
                "REAL_RESPONSE_RECEIVED" if config.mode == "real" else "MOCK_COMPLETE"
            )
            if not result.verification_summary["verification_contract_satisfied"]:
                base_status += "_WITH_VERIFICATION_FALLBACK"
            artifact: dict[str, object] = {
                "schema_version": "cholectrack20_api_rollout_v1",
                "status": base_status,
                "run_id": run_id,
                "mode": selection.mode,
                "split": selection.split.value if selection.split else None,
                "video_ids": list(selection.video_ids),
                "expected_frame_counts": dict(selection.frame_counts),
                "completed_frame_counts": dict(result.frame_counts),
                "provider": config.provider,
                "provider_routing_profile": (
                    routing_profile_from_options(config.provider_options)
                    if config.provider == "openrouter"
                    else None
                ),
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
                "causal_window": {
                    "schema_version": (
                        "fixed_causal_window_v1"
                        if is_main_h0_config(config)
                        else "adaptive_causal_window_v1"
                    ),
                    "max_frames": config.max_causal_frames,
                    "max_uploaded_images": config.max_api_images,
                    "target_frame_always_uploaded": True,
                },
                "pipeline_profile": pipeline_profile,
                "backbone_policy": result.backbone_policy,
                "initial_model_requested": result.initial_model_requested,
                "verification_model_requested": result.verification_model_requested,
                "main_profile_backbone_match": result.main_profile_backbone_match,
                "context_profile": resolved_context_profile,
                "event_memory_enabled": system.event_memory_enabled,
                "context_experiment_sha256": context_experiment_sha256,
                "phase_transition_graph": (
                    None
                    if graph is None
                    else {
                        "version": graph.version,
                        "sha256": graph.sha256,
                        "source_video_ids": list(graph.source_video_ids),
                    }
                ),
                "predicted_track_artifact_sha256": (
                    None if track_provider is None else track_provider.artifact_sha256
                ),
                "predicted_track_provenance": (
                    None
                    if track_provider is None
                    else {
                        "provider": track_provider.provider_name,
                        "source_model_identifier": (
                            track_provider.source_model_identifier
                        ),
                        "checkpoint_sha256": track_provider.checkpoint_sha256,
                        "inference_mode": track_provider.inference_mode,
                        "producer_version": track_provider.producer_version,
                        "dataset_repair_manifest_sha256": (
                            track_provider.dataset_repair_manifest_sha256
                        ),
                        "inference_config_sha256": (
                            track_provider.inference_config_sha256
                        ),
                    }
                ),
                "evidence_threshold": (
                    evidence_threshold if pipeline_profile == "rule_gate" else None
                ),
                "gate_artifact_sha256": (
                    sha256_file(resolved_gate_artifact)
                    if resolved_gate_artifact is not None
                    else None
                ),
                "verification_summary": dict(result.verification_summary),
                "final_status_counts": dict(
                    Counter(item.final_status for item in result.predictions)
                ),
                "memory_action_counts": dict(
                    Counter(item.memory_action for item in result.predictions)
                ),
                "report_manifest_path": result.report_manifest_path.relative_to(
                    destination
                ).as_posix(),
                "causal_window_audit_path": (
                    result.causal_window_audit_path.relative_to(destination).as_posix()
                ),
                "report_count": result.report_count,
                "report_mode": result.report_mode,
                "repair_manifest_sha256": repair_manifest_sha256,
                "alignment_versions": sorted(
                    {sample.alignment_version for sample in selection.samples}
                ),
                "usage": result.usage_summary,
                "telemetry_summary": _telemetry_summary(
                    dict(result.usage_summary), usage.records()
                ),
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


def _preflight_main_h0(args: argparse.Namespace, config: ApiConfig) -> Path:
    """Prepare the same production requests, without credentials or a transport."""
    if not is_main_h0_config(config):
        raise ApiContractError("--preflight is available for the main H0 config")
    validate_main_h0_config(config)
    if args.pipeline_profile != "single_pass" or args.context_profile not in {
        "auto", "frames_only"
    }:
        raise ApiContractError("main H0 preflight requires single_pass / frames_only")
    if any((args.gate_artifact, args.verification_config, args.phase_transition_graph,
            args.predicted_track_artifact, args.experiment_config)):
        raise ApiContractError("main H0 preflight does not accept research components")
    if args.event_memory not in {"auto", "disabled"}:
        raise ApiContractError("main H0 preflight does not accept event memory")
    run_id = args.run_id or datetime.now(UTC).strftime("h0_preflight_%Y%m%dT%H%M%SZ")
    _require_safe_run_id(run_id)
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / run_id
    _validate_paths(
        dataset_root=dataset_root, output_dir=output_dir,
        cache_root=args.cache_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
    )
    adapter = CholecTrack20DatasetAdapter(dataset_root, causal_window_size=3)
    selection = resolve_rollout_selection(
        adapter, mode=args.mode, video_id=args.video_id, max_frames=args.max_frames,
        split=args.split, target_frame_id=args.target_frame_id,
    )
    _resolve_profile_call_limit(
        args.max_provider_calls, pipeline_profile="single_pass",
        selection_count=selection.expected_provider_calls,
    )
    builder = JointPerceptionRequestBuilder(config=config)
    contexts = CausalPerceptionContextBuilder(
        max_frames=3, max_images=3, selection_strategy="fixed_all",
        history_image_detail="low", target_image_detail="high",
    )
    loader = CausalApiMediaLoader()
    rows = []
    for sample in selection.samples:
        loaded = loader.load(sample)
        context = contexts.build(
            loaded.runtime_sample, loaded.frames, workflow_snapshot={},
            memory_snapshot={}, prior_finalized_prediction=None,
        )
        request = builder.build(context)
        metadata = canonical_request_metadata(request)
        name = f"{sample.video_id}_{sample.target_frame_id}"
        # Payload has schema/text/detail settings; image bytes stay out of artifacts.
        atomic_write_json(output_dir / "requests" / f"{name}.json", {
            "metadata": metadata.to_mapping(), "payload": thaw_json(request.payload),
        })
        rows.append({
            "video_id": sample.video_id, "target_frame_id": sample.target_frame_id,
            "causal_frame_ids": list(sample.causal_frame_ids),
            "image_count": len(request.images), "request_hash": metadata.request_hash,
        })
    artifact_path = output_dir / "preflight.json"
    atomic_write_json(artifact_path, {
        "schema_version": "main_h0_preflight_v1", "status": "REQUESTS_READY",
        "model_requested": config.requested_model_identifier,
        "prompt_version": config.prompt_version,
        "response_schema_version": config.response_schema_version,
        "planned_calls": len(rows), "provider_calls": 0, "rows": rows,
    })
    print(f"H0_PREFLIGHT_READY artifact={artifact_path} planned_calls={len(rows)}")
    return artifact_path


def run(args: argparse.Namespace) -> Path:
    """Complete all preflight checks before credentials, media, or real transport."""

    validate_cli_selection(args)
    _validate_parallelism(args.parallelism)
    if args.report_mode == "llm_report":
        raise ApiContractError("llm_report requires an injected event-level generator")
    if args.report_window_frames <= 0:
        raise ApiContractError("report-window-frames must be a positive integer")
    gate_artifact = _validate_runtime_profile(
        args.pipeline_profile,
        evidence_threshold=args.evidence_threshold,
        gate_artifact=args.gate_artifact,
    )
    (
        context_profile,
        event_memory_enabled,
        phase_transition_graph,
        predicted_track_artifact,
        context_experiment_sha256,
    ) = _resolve_context_runtime(
        context_profile=args.context_profile,
        event_memory=args.event_memory,
        phase_transition_graph=args.phase_transition_graph,
        predicted_track_artifact=args.predicted_track_artifact,
        experiment_config=args.experiment_config,
    )
    config = _apply_cli_base_url_override(
        _apply_cli_model_override(
            load_api_config(args.config),
            mode=args.mode,
            model=args.model,
        ),
        mode=args.mode,
        base_url=args.base_url,
    )
    if getattr(args, "preflight", False):
        return _preflight_main_h0(args, config)
    verification_config = (
        None
        if args.verification_config is None
        else _apply_cli_base_url_override(
            _apply_cli_model_override(
                load_api_config(args.verification_config),
                mode=args.mode,
                model=args.model,
            ),
            mode=args.mode,
            base_url=args.base_url,
        )
    )
    backbone_policy, resolved_verification_config = _validate_profile_backbones(
        args.pipeline_profile,
        initial_config=config,
        verification_config=verification_config,
    )
    direct_api_key = args.api_key.strip() if args.api_key is not None else None
    if args.api_key is not None and not direct_api_key:
        raise ApiContractError("api-key must not be empty")
    has_credential = direct_api_key is not None or args.api_key_file is not None
    expected_initial_model = (
        config.requested_model_identifier
        if args.model is not None or config.provider == "openai_compatible"
        else (_LUNA_MODEL if backbone_policy == "cascade_efficiency" else _REAL_MODEL)
    )
    _require_exact_dataset_config(
        config,
        has_credential=has_credential,
        authorize_data_upload=args.authorize_data_upload,
        expected_real_model=expected_initial_model,
        selection_mode=args.mode,
    )
    if backbone_policy == "cascade_efficiency":
        _require_exact_dataset_config(
            resolved_verification_config,
            has_credential=has_credential,
            authorize_data_upload=args.authorize_data_upload,
            selection_mode=args.mode,
        )
    run_id = args.run_id or datetime.now(UTC).strftime("dataset_api_%Y%m%dT%H%M%SZ")
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
            target_frame_id=args.target_frame_id,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ApiContractError(str(exc)) from None
    call_limit = _resolve_profile_call_limit(
        args.max_provider_calls,
        pipeline_profile=args.pipeline_profile,
        selection_count=selection.expected_provider_calls,
    )
    api_key = (
        resolve_api_key(
            api_key=direct_api_key,
            api_key_file=args.api_key_file,
        )
        if has_credential
        else None
    )
    artifact = run_dataset_api_rollout(
        config=config,
        verification_config=resolved_verification_config,
        selection=selection,
        dataset_root=dataset_root,
        output_dir=output_dir,
        cache_root=cache_root,
        run_id=run_id,
        max_provider_calls=call_limit,
        api_key=api_key,
        authorize_data_upload=args.authorize_data_upload,
        pipeline_profile=args.pipeline_profile,
        evidence_threshold=args.evidence_threshold,
        gate_artifact=gate_artifact,
        context_profile=context_profile,
        phase_transition_graph=phase_transition_graph,
        predicted_track_artifact=predicted_track_artifact,
        event_memory_enabled=event_memory_enabled,
        context_experiment_sha256=context_experiment_sha256,
        progress_enabled=not args.no_progress,
        report_mode=args.report_mode,
        report_window_frames=args.report_window_frames,
        engineering_target_frame_id=args.target_frame_id,
        proxy_url=args.proxy_url,
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


def main(
    argv: Sequence[str] | None = None,
    *,
    fixed_profile: str | None = None,
    learned_when_artifact_present: bool = False,
) -> None:
    parser = build_parser(fixed_profile=fixed_profile)
    args = parser.parse_args(argv)
    if learned_when_artifact_present and args.gate_artifact is not None:
        args.pipeline_profile = "learned_gate"
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001 - CLI emits only a safe category
        print(
            f"DATASET_API_ROLLOUT_FAILED category={_safe_error_category(exc)}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
