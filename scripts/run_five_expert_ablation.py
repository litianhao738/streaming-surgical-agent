"""Frozen six-target A/B/C pilot; independent calls, no repair, GT offline only."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_pure_h0_smoke import RawResponseAuditTransport, SchemaExplicitRequestBuilder, evaluate_offline
from surgical_agent.api.accounting import CompleteAccountingTransport
from surgical_agent.api.budget import ProviderCallBudget
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.credentials import resolve_api_key
from surgical_agent.api.errors import ApiError
from surgical_agent.api.registry import build_transport
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.schema import validator_for
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.expert_ablation import expert_version, TASK_COUNTS
from surgical_agent.perception.ontology_prompt import _ivt_rows

TARGETS = (6251, 6276, 6301, 8501, 8526, 8551)
BOUNDARY = (
    "\nDataset label boundary: A visible instrument does not by itself establish a named "
    "verb or anatomical target. Classify the actual tool-contact relationship, not nearby "
    "anatomy alone. The 100 IVT classes are a closed vocabulary, not a list covering every "
    "possible surgical interaction. For a valid instrument whose actual verb, target, or "
    "joint combination is outside these classes, retain its instrument identity and use "
    "the corresponding null-verb/null-target IVT. Null does not necessarily mean no "
    "physical motion or contact. Do not replace an out-of-vocabulary interaction with "
    "the nearest familiar valid tuple. Conversely, do not use null merely because "
    "prediction is difficult; use a non-null tuple when that defined interaction is "
    "visually supported. These definitions apply per instrument; a frame may contain "
    "both null and non-null interactions.\n"
)


def build_variant(base, arm, task=None):
    payload = thaw_json(base.payload)
    if arm in ("B", "C"):
        payload["system_text"] += BOUNDARY
    version = base.response_schema_version
    if arm == "C":
        version = expert_version(task)
        payload["system_text"] = payload["system_text"].replace(base.response_schema_version, version)
        payload["system_text"] += (
            f"\nYour sole output task is {task}. The joint instructions above describe the "
            "shared scene and ontology; other task outputs are not requested in this call. "
            f"Return only schema_version and the {task} object, with its selected label(s) "
            f"and exactly {TASK_COUNTS[task]} topk candidates. Use the shared ontology to "
            "interpret relationships internally, but output no other task fields. "
            "You receive no predictions from other experts.\n"
        )
    elif arm not in ("A", "B"):
        raise ValueError("unknown experiment arm")
    return replace(base, payload=payload, response_schema_version=version,
                   prompt_version=base.prompt_version if arm == "A" else f"five_expert_ablation_{arm}_v1")


def closure(labels):
    components = {row[0]: row[1:] for row in _ivt_rows()}
    projected = {task: set() for task in ("instrument", "verb", "target")}
    for value in labels["ivt"]:
        for task, component in zip(projected, components[value]):
            projected[task].add(component)
    return {
        "ivt_components_in_selected": all(projected[t] <= set(labels[t]) for t in projected),
        "exact_projection": all(projected[t] == set(labels[t]) for t in projected),
    }


def run(args):
    if args.output.exists():
        raise ValueError("fresh output directory required")
    config = load_api_config(args.config)
    if config.requested_model_identifier != "qwen/qwen3.8-max-0902":
        raise ValueError("this frozen pilot uses Qwen0902 only")
    if not config.data_upload_authorized or config.max_causal_frames != 3:
        raise ValueError("authorized fixed-three-frame configuration required")
    if config.provider_options.get("initial_prompt_profile") != "fixed_visual_only":
        raise ValueError("visual-only required")
    adapter = CholecTrack20DatasetAdapter(args.dataset, causal_window_size=3)
    contexts = {}
    builder = CausalPerceptionContextBuilder(max_frames=3, max_images=3,
        selection_strategy="fixed_all", history_image_detail="low", target_image_detail="auto")
    for sample in adapter.iter_inference_video("VID110"):
        if sample.target_frame_id in TARGETS:
            if sample.causal_frame_ids != (sample.target_frame_id - 50, sample.target_frame_id - 25, sample.target_frame_id):
                raise ValueError("missing three-frame causal window")
            loaded = CausalApiMediaLoader().load(sample)
            contexts[sample.target_frame_id] = builder.build(loaded.runtime_sample, loaded.frames,
                workflow_snapshot={}, memory_snapshot={}, prior_finalized_prediction=None)
    if set(contexts) != set(TARGETS):
        raise ValueError("missing frozen targets")
    jobs = []
    for index, fid in enumerate(TARGETS):
        base = SchemaExplicitRequestBuilder(config=config).build(contexts[fid])
        # Rotate arm order by timestamp to reduce systematic run-order imbalance.
        order = ("A", "B", "C")
        order = order[index % 3:] + order[:index % 3]
        for arm in order:
            for task in TASK_COUNTS if arm == "C" else (None,):
                jobs.append((fid, arm, task, build_variant(base, arm, task)))
    args.output.mkdir(parents=True)
    atomic_write_json(args.output / "plan.json", {
        "status": "FROZEN_BEFORE_CALLS", "model": config.requested_model_identifier,
        "target_frames": TARGETS, "video": "VID110", "max_calls": len(jobs),
        "workers": 4, "retries": 0, "gt_in_requests": False,
        "selection": "GT-stratified diagnostic clips selected before predictions; not a representative benchmark",
        "boundary_text": BOUNDARY,
        "jobs": [{"frame": fid, "arm": arm, "task": task,
                  "metadata": canonical_request_metadata(request).to_mapping()}
                 for fid, arm, task, request in jobs]})
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)

    def call(job):
        fid, arm, task, request = job
        out = args.output / "calls" / f"{fid}_{arm}_{task or 'joint'}"
        atomic_write_json(out / "request.json", {"payload": thaw_json(request.payload),
            "metadata": canonical_request_metadata(request).to_mapping()})
        client = CachedMultimodalApiClient(
            transport=RawResponseAuditTransport(CompleteAccountingTransport(build_transport(config, api_key=secret)), out),
            cache=FileApiCache(args.cache), usage=UsageLedger(out / "api_usage.jsonl"),
            validator=validator_for(request.response_schema_version),
            retry_policy=RetryPolicy(max_attempts=1), provider_call_budget=ProviderCallBudget(1))
        row = {"frame": fid, "arm": arm, "task": task}
        try:
            response = client.call(request)
            row.update(status="OK", payload=thaw_json(response.parsed_payload),
                       returned_model=response.returned_model_identifier)
        except ApiError as exc:
            row.update(status="API_FAILURE", error=exc.code)
        atomic_write_json(out / "result.json", row)
        return row

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(call, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            atomic_write_json(args.output / "results.json", results)
            print(f"{len(results)}/{len(jobs)} {row['frame']} {row['arm']} {row['task']} {row['status']}", flush=True)
    # GT is read only after every raw call result has been frozen.
    gt_report = evaluate_offline(adapter, "VID110", [{"frame_id": fid, "status": "API_FAILURE"} for fid in TARGETS])
    truths = {row["frame_id"]: row for row in gt_report["frames"]}
    frames = []
    for arm in ("A", "B", "C"):
        for fid in TARGETS:
            labels = {}
            for result in results:
                if result["arm"] != arm or result["frame"] != fid or result["status"] != "OK":
                    continue
                for task in ([result["task"]] if arm == "C" else TASK_COUNTS):
                    field = result["payload"][task]
                    labels[task] = [field["selected_id"]] if task == "phase" else field["selected_ids"]
            truth = truths[fid]
            exact = {task: set(labels[task]) == set(truth["gt"][task])
                     if task in labels and truth["mask"][task] else None for task in TASK_COUNTS}
            frames.append({"arm": arm, "frame": fid, "labels": labels, "gt": truth["gt"],
                           "mask": truth["mask"], "exact": exact,
                           "closure": closure(labels) if all(t in labels for t in ("instrument", "verb", "target", "ivt")) else None})
    metrics = {}
    for arm in ("A", "B", "C"):
        rows = [r for r in frames if r["arm"] == arm]
        metrics[arm] = {t: {"valid_gt": sum(r["mask"][t] for r in rows),
                            "scored": sum(r["exact"][t] is not None for r in rows),
                            "correct": sum(r["exact"][t] is True for r in rows)} for t in TASK_COUNTS}
    costs = {}
    for arm in ("A", "B", "C"):
        ledger = [json.loads(line) for path in (args.output / "calls").glob(f"*_{arm}_*/api_usage.jsonl")
                  for line in path.read_text().splitlines()]
        costs[arm] = {"calls": sum(r["provider_call_count"] for r in ledger),
                      "cache_hits": sum(r["cache_hit"] for r in ledger),
                      "reported_cost": round(sum(r.get("provider_cost") or 0 for r in ledger), 6),
                      "unpriced": sum(r["provider_call_count"] for r in ledger if r.get("provider_cost") is None)}
    atomic_write_json(args.output / "summary.json", {"metrics": metrics, "costs": costs, "frames": frames})
    print(json.dumps({"metrics": metrics, "costs": costs}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "config", "api-key-file", "cache", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    run(p.parse_args())
