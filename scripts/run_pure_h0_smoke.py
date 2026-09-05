"""Small pure-H0 smoke: one first response per causal window, then masked scoring.

No Tracker, Gate, Verifier, Repair, or prediction-memory object is constructed.
Inference selection reads only media identities. GT is loaded after predictions
have been persisted, in a separate offline evaluation step.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from itertools import islice, pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgical_agent.api.accounting import CompleteAccountingTransport
from surgical_agent.api.budget import ProviderCallBudget
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.credentials import resolve_api_key
from surgical_agent.api.errors import ApiCallFailure, ApiError, ApiSchemaError
from surgical_agent.api.registry import build_transport, build_validator
from surgical_agent.api.request_hash import canonical_request_hash
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.final_only import FINAL_ONLY_SCHEMA_VERSION, validate_final_only
from surgical_agent.perception.joint_api_vlm import (
    JointApiVlm,
    JointPerceptionRequestBuilder,
)

TASKS = {
    "instrument": "instrument_ids",
    "verb": "verb_ids",
    "target": "target_ids",
    "ivt": "triplet_ids",
    "phase": "phase_id",
}


class RawResponseAuditTransport:
    """Persist provider JSON before local validation; never correct its labels."""

    def __init__(self, transport, output):
        self.transport = transport
        self.output = output
        self.provider = transport.provider
        self.endpoint_identifier = transport.endpoint_identifier

    def send(self, request):
        response = self.transport.send(request)
        request_hash = canonical_request_hash(request)
        atomic_write_json(
            self.output / "raw_responses" / f"{request_hash}.json",
            {
                "request_hash": request_hash,
                "requested_model": request.model_identifier,
                "returned_model": response.returned_model_identifier,
                "parsed_payload": thaw_json(response.parsed_payload),
            },
        )
        return response


class SchemaExplicitRequestBuilder(JointPerceptionRequestBuilder):
    """Clarify a wire-version literal without changing labels or model answers."""

    def build(self, context):
        request = super().build(context)
        payload = thaw_json(request.payload)
        payload["system_text"] += (
            f'\nThe output JSON schema_version must be exactly "{request.response_schema_version}". '
            "Do not use the input ontology_version as the output schema_version.\n"
        )
        return replace(request, payload=payload)


class FinalOnlyRequestBuilder(JointPerceptionRequestBuilder):
    """Remove only the top-k instructions; retain visual and ontology instructions."""

    def build(self, context):
        request = super().build(context)
        payload = thaw_json(request.payload)
        original = (
            "selected_ids are sparse final predictions. They may be empty and must never be\n"
            "padded to the top-k size. Select only labels with direct support in the target\n"
            "frame or causal motion ending at the target. The exact counts below apply only\n"
            "to topk: return 3 instrument, 4 verb, 5 target, 8 IVT, and 3 phase candidates in\n"
            "descending finite score order in [0, 1]. Every selected ID must occur in its\n"
            "corresponding topk. The scores are uncalibrated ranking signals for a separate\n"
            "local Gate; do not decide whether verification is required."
        )
        if payload["system_text"].count(original) != 1:
            raise ValueError("final-only ablation requires the expected gate-owned prompt")
        payload["system_text"] = payload["system_text"].replace(original, (
            "selected_ids are sparse final predictions. They may be empty and must never be\n"
            "padded. Select only labels with direct support in the target\n"
            "frame or causal motion ending at the target. Return only final selected labels;\n"
            "do not return topk candidates or scores."
        ))
        payload["system_text"] += (
            f'\nThe output JSON schema_version must be exactly "{FINAL_ONLY_SCHEMA_VERSION}". '
            "Do not use the input ontology_version as the output schema_version.\n"
        )
        return replace(request, payload=payload, prompt_version="joint_final_only_ablation_v1",
                       response_schema_version=FINAL_ONLY_SCHEMA_VERSION)


def selected_labels(prediction):
    return {
        task: (
            [getattr(prediction, attr)]
            if task == "phase"
            else list(getattr(prediction, attr))
        )
        for task, attr in TASKS.items()
    }


def evaluate_offline(adapter, video, predictions):
    from surgical_agent.evaluation.frame_ground_truth import aggregate_evaluation_target

    targets = {}
    for record in adapter.iter_video(
        video, frame_ids=[row["frame_id"] for row in predictions]
    ):
        target = record.frame_supervision
        if (
            record.evaluation is not None
            and record.evaluation.instance_supervision_available
        ):
            target = aggregate_evaluation_target(
                record.evaluation, source="offline_pure_h0_smoke"
            )
        targets[record.inference.target_frame_id] = target
    metrics = {
        task: {"valid_gt": 0, "scored": 0, "correct": 0, "failed_with_gt": 0}
        for task in TASKS
    }
    rows = []
    for prediction in predictions:
        target = targets.get(prediction["frame_id"])
        mask = {
            task: target is not None and getattr(target.mask, task) for task in TASKS
        }
        truth = selected_labels(target) if target is not None else {}
        exact = {}
        for task in TASKS:
            if not mask[task]:
                exact[task] = None
                continue
            metrics[task]["valid_gt"] += 1
            if prediction["status"] != "OK":
                metrics[task]["failed_with_gt"] += 1
                exact[task] = None
                continue
            good = set(prediction["selected_ids"][task]) == set(truth[task])
            exact[task] = good
            metrics[task]["scored"] += 1
            metrics[task]["correct"] += good
        rows.append(
            {
                "frame_id": prediction["frame_id"],
                "mask": mask,
                "gt": {task: truth.get(task) if mask[task] else None for task in TASKS},
                "h0": prediction.get("selected_ids"),
                "exact": exact,
                "status": prediction["status"],
            }
        )
    for values in metrics.values():
        values["exact_accuracy_conditional_on_response"] = (
            values["correct"] / values["scored"] if values["scored"] else None
        )
    return {"metric": "masked_set_exact_NOT_mAP", "tasks": metrics, "frames": rows}


def run(args):
    config = load_api_config(args.config)
    if config.provider_options.get("initial_prompt_profile") != "fixed_visual_only":
        raise ValueError("pure H0 requires fixed_visual_only")
    if not 1 <= args.max_frames <= 20:
        raise ValueError("this bounded smoke supports 1..20 timestamps")
    if args.output.exists():
        raise ValueError("use a fresh output directory; never mix smoke attempts")
    adapter = CholecTrack20DatasetAdapter(
        args.dataset, causal_window_size=config.max_causal_frames
    )
    samples = tuple(
        islice(
            (
                sample
                for sample in adapter.iter_inference_video(args.video)
                if sample.target_frame_id >= args.start
            ),
            args.max_frames,
        )
    )
    if len(samples) != args.max_frames:
        raise ValueError("not enough available target timestamps")
    if samples[0].target_frame_id != args.start:
        raise ValueError("requested first target image is missing")
    if any(
        b.target_frame_id - a.target_frame_id != adapter.expected_frame_id_step
        for a, b in pairwise(samples)
    ):
        raise ValueError("selected targets are not continuous")
    args.output.mkdir(parents=True)
    atomic_write_json(
        args.output / "run_status.json",
        {
            "status": "RUNNING",
            "profile": "pure_h0",
            "target_frame_ids": [sample.target_frame_id for sample in samples],
            "model": config.requested_model_identifier,
            "config_sha256": sha256_file(args.config),
        },
    )
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    client = CachedMultimodalApiClient(
        transport=RawResponseAuditTransport(
            CompleteAccountingTransport(build_transport(config, api_key=secret)),
            args.output,
        ),
        cache=FileApiCache(args.cache),
        usage=UsageLedger(args.output / "api_usage.jsonl"),
        validator=validate_final_only if getattr(args, "final_only", False) else build_validator(config),
        retry_policy=RetryPolicy(max_attempts=1),
        provider_call_budget=ProviderCallBudget(len(samples)),
    )
    model = JointApiVlm(
        client=client,
        request_builder=(
            SchemaExplicitRequestBuilder
            if getattr(args, "explicit_schema_version", False)
            else JointPerceptionRequestBuilder
        )(config=config),
        data_upload_authorized=config.data_upload_authorized,
    )
    context_builder = CausalPerceptionContextBuilder(
        max_frames=config.max_causal_frames,
        max_images=config.max_api_images,
        selection_strategy="fixed_all",
        history_image_detail="low",
        target_image_detail="auto",
    )
    loader = CausalApiMediaLoader()
    predictions = []
    terminal_service_error = None
    for sample in samples:
        row = {
            "video_id": sample.video_id,
            "frame_id": sample.target_frame_id,
            "causal_frame_ids": list(sample.causal_frame_ids),
        }
        try:
            if terminal_service_error is not None:
                row.update(status="NOT_ATTEMPTED", error=terminal_service_error)
            else:
                loaded = loader.load(sample)
                context = context_builder.build(
                    loaded.runtime_sample,
                    loaded.frames,
                    workflow_snapshot={},
                    memory_snapshot={},
                    prior_finalized_prediction=None,
                )
                if getattr(args, "final_only", False):
                    request = FinalOnlyRequestBuilder(config=config).build(context)
                    if not config.data_upload_authorized:
                        raise ValueError("data upload must be authorized")
                    atomic_write_json(args.output / "requests" / f"{sample.target_frame_id}.json", {
                        "request_hash": canonical_request_hash(request),
                        "prompt_version": request.prompt_version,
                        "response_schema_version": request.response_schema_version,
                        "payload": thaw_json(request.payload),
                    })
                    response = client.call(request)
                    payload = response.parsed_payload
                    row.update(status="OK", selected_ids={
                        task: [payload[task]["selected_id"]] if task == "phase"
                        else list(payload[task]["selected_ids"]) for task in TASKS
                    }, provenance={"request_hash": canonical_request_hash(request),
                                   "returned_model_identifier": response.returned_model_identifier,
                                   "cache_hit": response.cache_hit}, topk=None)
                else:
                    result = model.predict(context)
                    row.update(
                        status="OK",
                        selected_ids=selected_labels(result.prediction),
                        provenance=asdict(result.api_provenance),
                        topk={
                            task: [asdict(item) for item in candidates]
                            for task, candidates in result.raw_evidence.ranked_candidates.items()
                        },
                    )
        except ApiCallFailure as exc:
            row.update(
                status="API_FAILURE",
                error=exc.cause.code,
                http_status=exc.cause.status_code,
            )
            if exc.cause.status_code in {401, 402}:
                terminal_service_error = exc.cause.code
        except ApiSchemaError as exc:
            row.update(status="API_FAILURE", error=exc.code, error_detail=str(exc))
        except ApiError as exc:
            row.update(status="API_FAILURE", error=exc.code)
        predictions.append(row)
        atomic_write_json(args.output / "predictions.json", predictions)
        print(
            f"{sample.video_id}:{sample.target_frame_id} {row['status']} {row.get('selected_ids', row.get('error'))}",
            flush=True,
        )

    # Only now load GT, after every inference result is frozen on disk.
    evaluation = evaluate_offline(adapter, args.video, predictions)
    usage = [
        json.loads(line)
        for line in (args.output / "api_usage.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    summary = {
        "profile": "pure_h0",
        "model": config.requested_model_identifier,
        "explicit_schema_version": getattr(args, "explicit_schema_version", False),
        "output_contract": "final_only" if getattr(args, "final_only", False) else "selected_plus_topk",
        "tracker": False,
        "gate": False,
        "verifier": False,
        "repair": False,
        "predictions_sha256": sha256_file(args.output / "predictions.json"),
        "attempted_timestamps": len(samples),
        "successful_predictions": sum(row["status"] == "OK" for row in predictions),
        "provider_calls": sum(row["provider_call_count"] for row in usage),
        "cache_hits": sum(row["cache_hit"] for row in usage),
        "reported_cost_usd": round(
            sum(row.get("provider_cost") or 0 for row in usage), 6
        ),
        "unpriced_calls": sum(
            row["provider_call_count"]
            for row in usage
            if row.get("provider_cost") is None
        ),
        "evaluation": evaluation,
    }
    atomic_write_json(args.output / "summary.json", summary)
    atomic_write_json(
        args.output / "run_status.json",
        {
            "status": "COMPLETE"
            if summary["successful_predictions"] == len(samples)
            else "COMPLETE_WITH_FAILURES",
            **{key: value for key, value in summary.items() if key != "evaluation"},
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", default="VID110")
    parser.add_argument("--start", type=int, default=51)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--final-only", action="store_true", help="Ablation: final five-task labels without top-k or scores")
    parser.add_argument(
        "--explicit-schema-version",
        action="store_true",
        help="Append only the expected wire schema_version literal; preserves original responses",
    )
    run(parser.parse_args())
