"""Six-frame isolated H0/grounding/contrast-review smoke. No GT in requests."""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_pure_h0_smoke import FinalOnlyRequestBuilder, RawResponseAuditTransport, evaluate_offline
from scripts.run_five_expert_ablation import BOUNDARY, closure
from surgical_agent.api.accounting import CompleteAccountingTransport
from surgical_agent.api.budget import ProviderCallBudget
from surgical_agent.api.cache import FileApiCache
from surgical_agent.api.client import CachedMultimodalApiClient
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.credentials import resolve_api_key
from surgical_agent.api.errors import ApiError
from surgical_agent.api.providers.openrouter import OpenRouterTransport, urllib_send_json, _decode_sse_response
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.api.retry import RetryPolicy
from surgical_agent.api.schema import validator_for, schema_for
from surgical_agent.api.usage import UsageLedger
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.ontology_prompt import ACADEMIC_MEDICAL_CONTEXT, load_prompt_ontology_text
from surgical_agent.research.verification.grounded_repair import (
    LOCATOR_VERSION, PROPOSAL_VERSION, REVIEW_VERSION, make_contact_crops, proposed_labels, admit,
)

TARGETS = (8601, 8626, 8651, 8676, 8701, 8726)


class HttpAuditSender:
    """Save response before JSON/SSE/schema parsing, never headers or image bytes."""

    def __init__(self, output, secret, sender=urllib_send_json):
        self.output, self.secret, self.sender = output, secret, sender

    def __call__(self, url, headers, body, timeout):
        wire = json.loads(body)
        for message in wire["messages"]:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        value = item["image_url"]["url"]
                        item["image_url"]["url"] = "[image omitted; bound by request metadata]"
                        item["image_url"]["data_url_sha256"] = hashlib.sha256(value.encode()).hexdigest()
        atomic_write_json(self.output / "wire_request.json", wire)
        response = self.sender(url, headers, body, timeout)
        text = response.body.decode("utf-8", errors="replace").replace(self.secret.reveal(), "[REDACTED]")
        record = {"status_code": response.status_code, "body": text}
        try:
            decoded = _decode_sse_response(response.body) if text.lstrip().startswith("data:") else json.loads(text)
            record["reported_usage"] = decoded.get("usage")
            record["returned_model"] = decoded.get("model")
        except (ValueError, TypeError, ApiError):
            record["reported_usage"] = None
        atomic_write_json(self.output / "http_response.json", record)
        return response


def make_request(base, version, instruction, extra, crops=()):
    payload = thaw_json(base.payload)
    payload["system_text"] = (
        ACADEMIC_MEDICAL_CONTEXT + "\n" + instruction + "\n"
        + (load_prompt_ontology_text() + BOUNDARY if version != LOCATOR_VERSION else "")
        + f'\nReturn JSON only. schema_version must be "{version}". '
        + "Include every required field. No markdown. Observations must be brief visible facts, not reasoning traces.\n"
        + "Required output schema:\n" + json.dumps(schema_for(version), separators=(",", ":"))
    )
    data = json.loads(payload["input_text"])
    data.update(extra)
    payload["input_text"] = json.dumps(data, separators=(",", ":"))
    payload["image_details"] = ["low", "low", "high"] + ["high"] * len(crops)
    payload["openrouter_image_detail_mode"] = "explicit_v1"
    return replace(base, payload=payload, images=base.images + crops,
                   response_schema_version=version, prompt_version=version + "_contact_pilot_v1")


def run(args):
    if args.output.exists():
        raise ValueError("fresh output required")
    config = load_api_config(args.config)
    if config.requested_model_identifier != "qwen/qwen3.8-max-0902" or not config.data_upload_authorized:
        raise ValueError("approved Qwen0902 and authorized data upload required")
    if config.provider_options.get("initial_prompt_profile") != "fixed_visual_only":
        raise ValueError("isolated visual-only configuration required")
    adapter = CholecTrack20DatasetAdapter(args.dataset, causal_window_size=3)
    context_builder = CausalPerceptionContextBuilder(max_frames=3, max_images=3,
        selection_strategy="fixed_all", history_image_detail="low", target_image_detail="high")
    bases = {}
    for sample in adapter.iter_inference_video("VID110"):
        if sample.target_frame_id not in TARGETS:
            continue
        if sample.causal_frame_ids != (sample.target_frame_id-50, sample.target_frame_id-25, sample.target_frame_id):
            raise ValueError("missing causal window")
        loaded = CausalApiMediaLoader().load(sample)
        context = context_builder.build(loaded.runtime_sample, loaded.frames,
            workflow_snapshot={}, memory_snapshot={}, prior_finalized_prediction=None)
        base = FinalOnlyRequestBuilder(config=config).build(context)
        payload = thaw_json(base.payload)
        payload["openrouter_image_detail_mode"] = "explicit_v1"
        payload["image_details"] = ["low", "low", "high"]
        payload["system_text"] += BOUNDARY + (
            "\nAll six root keys are mandatory: schema_version, instrument, verb, target, ivt, phase. "
            "Do not stop after instrument. The exact output schema is:\n"
            + json.dumps(schema_for(base.response_schema_version), separators=(",", ":"))
        )
        bases[sample.target_frame_id] = replace(base, payload=payload, prompt_version="grounded_contact_h0_v1")
    if set(bases) != set(TARGETS):
        raise ValueError("missing targets")
    args.output.mkdir(parents=True)
    atomic_write_json(args.output / "plan.json", {"targets": TARGETS, "model": config.requested_model_identifier,
        "max_provider_calls": 30, "semantic_retries": 0, "format_retry_limit": 1,
        "crop_source": "blind_API_locator_on_target_not_GT", "max_instances": 3,
        "phase_policy": "keep_H0", "config_sha256": sha256_file(args.config),
        "admission": "experimental_uncalibrated_contrastive_visual_preference",
        "input_metadata": {str(fid): canonical_request_metadata(base).to_mapping() for fid, base in bases.items()}})
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)

    def process(fid):
        base = bases[fid]
        calls = []

        def call(stage, request):
            out = args.output / "calls" / f"{fid}_{stage}"
            atomic_write_json(out / "request.json", {"payload": thaw_json(request.payload),
                "metadata": canonical_request_metadata(request).to_mapping()})
            transport = OpenRouterTransport(api_key=secret, endpoint_identifier=config.endpoint_identifier,
                timeout_seconds=180, sender=HttpAuditSender(out, secret))
            client = CachedMultimodalApiClient(
                transport=RawResponseAuditTransport(CompleteAccountingTransport(transport), out),
                cache=FileApiCache(args.cache), usage=UsageLedger(out / "api_usage.jsonl"),
                validator=validator_for(request.response_schema_version), retry_policy=RetryPolicy(max_attempts=1),
                provider_call_budget=ProviderCallBudget(1))
            result = {"stage": stage, "request_hash": canonical_request_metadata(request).request_hash}
            try:
                response = client.call(request)
                result.update(status="OK", payload=thaw_json(response.parsed_payload))
            except ApiError as exc:
                cause = getattr(exc, "cause", exc)
                result.update(status="FAILURE", error=cause.code)
            atomic_write_json(out / "result.json", result)
            calls.append(result)
            print(f"{fid} {stage} {result['status']}", flush=True)
            return result.get("payload")

        h0_payload = call("h0", base)
        if h0_payload is None and calls[-1].get("error") in {"schema_error", "response_content_invalid", "completion_length"}:
            retry_payload = thaw_json(base.payload)
            retry_payload["system_text"] += "\nFormat retry: produce the complete required five-task JSON object, not a partial object."
            h0_payload = call("h0_format_retry", replace(base, payload=retry_payload, prompt_version="grounded_contact_h0_format_retry_v1"))
        if h0_payload is None:
            return {"frame_id": fid, "status": "H0_FAILURE", "calls": calls}
        h0 = {t: [h0_payload[t]["selected_id"]] if t == "phase" else h0_payload[t]["selected_ids"]
              for t in ("instrument", "verb", "target", "ivt", "phase")}
        row = {"frame_id": fid, "status": "OK", "h0": h0, "final": h0,
               "admission": {"decision": "KEEP", "reason": "NO_VALID_GROUNDING"}}
        locator = call("locator", make_request(base, LOCATOR_VERSION,
            "Locate each currently visible surgical instrument TIP and contact region in the THIRD (target) frame. "
            "Return normalized [left,top,right,bottom] boxes using the full target image dimensions. "
            "Assign distinct instance IDs 1..3, left-to-right. Do not classify actions, anatomy, phase or instrument type. "
            "If more than three tools exist, or visibility prevents full coverage, all_visible_tools_covered must be false. "
            "Boxes are proposals, not trusted detections; do not invent hidden tools.", {}))
        if locator and locator["instances"] and locator["all_visible_tools_covered"]:
            crops, manifest = make_contact_crops(base.images[-1], locator, target_frame_id=fid)
            row["locator"] = locator
            row["crop_manifest"] = manifest
            extra = {"predicted_instance_regions": locator, "crop_manifest": manifest,
                     "image_order": "first 3 full causal images; remaining target-frame crops in manifest order"}
            proposal = call("proposal", make_request(base, PROPOSAL_VERSION,
                "Recognize each localized instrument and its CURRENT interaction using full causal images plus target crops. "
                "The location proposals can be wrong: return INSUFFICIENT if a box is irrelevant or identity/contact is unclear. "
                "Return one instance record per supplied instance ID. Use the full 100-class ontology, not a ranked candidate pool. "
                "Select only supported tuples belonging to that instance. Describe a short factual contact observation. "
                "Nearby visible anatomy alone is not an interaction target. Do not predict the phase. "
                "For valid tools acting outside the ontology use the defined null tuple only when visually supported; "
                "do not use null as an uncertainty fallback.", extra, crops))
            row["proposal"] = proposal
            h1 = proposed_labels(h0, locator, proposal) if proposal else None
            row["h1"] = h1
            review = None
            slot = "FIRST" if TARGETS.index(fid) % 2 == 0 else "SECOND"
            if h1 is not None and h1 != h0:
                hypotheses = {"FIRST": h1 if slot == "FIRST" else h0, "SECOND": h0 if slot == "FIRST" else h1}
                review = call("review", make_request(base, REVIEW_VERSION,
                    "Compare two alternative scene hypotheses against the original causal frames and localized target crops. "
                    "Their order is arbitrary; neither is ground truth. Do not favor one for being more structured or more detailed. "
                    "Prefer a hypothesis only if visual contact, anatomy identity and action evidence distinguish it. "
                    "Check each supplied region is relevant and all visible tools are covered; no new answer is requested. "
                    "If the evidence cannot distinguish the alternatives, return TIE or INSUFFICIENT. "
                    "For null interactions require evidence of the ontology boundary, not just uncertainty. "
                    "Report brief observable distinctions, no chain of thought. Phase is held fixed and is not under review.",
                    {**extra, "hypotheses": hypotheses}, crops))
            decision = admit(h0, h1, locator, proposal, review, proposal_slot=slot) if proposal else row["admission"]
            row.update(review=review, proposal_slot=slot, admission=decision)
            if decision["decision"] == "ACCEPT":
                row["final"] = h1
        row["calls"] = calls
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(process, fid) for fid in TARGETS]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            atomic_write_json(args.output / "predictions.json", sorted(rows, key=lambda r: r["frame_id"]))
    reports = {}
    for name in ("h0", "final"):
        reports[name] = evaluate_offline(adapter, "VID110", [{"frame_id": r["frame_id"],
            "status": "OK" if r["status"] == "OK" else "API_FAILURE", "selected_ids": r.get(name)} for r in rows])
    before = {r["frame_id"]: r for r in reports["h0"]["frames"]}
    after = {r["frame_id"]: r for r in reports["final"]["frames"]}
    repairs = []
    for row in rows:
        if row["status"] != "OK":
            continue
        fid = row["frame_id"]
        improved, harmed = [], []
        for task in ("instrument", "verb", "target", "ivt", "phase"):
            if not before[fid]["mask"][task]:
                continue
            gt = set(before[fid]["gt"][task])
            old_loss = len(set(row["h0"][task]) ^ gt)
            new_loss = len(set(row["final"][task]) ^ gt)
            if new_loss < old_loss:
                improved.append(task)
            if new_loss > old_loss:
                harmed.append(task)
        repairs.append({"frame_id": fid, "decision": row["admission"], "improved_tasks": improved,
                        "harmed_tasks": harmed, "h0_closure": closure(row["h0"]), "final_closure": closure(row["final"])})
    usage = [json.loads(line) for path in (args.output / "calls").glob("*/api_usage.jsonl") for line in path.read_text().splitlines()]
    summary = {"evaluation": reports, "repairs": repairs,
        "provider_calls": sum(r["provider_call_count"] for r in usage), "cache_hits": sum(r["cache_hit"] for r in usage),
        "ledger_cost": round(sum(r.get("provider_cost") or 0 for r in usage), 6),
        "ledger_unpriced_calls": sum(r["provider_call_count"] for r in usage if r.get("provider_cost") is None)}
    atomic_write_json(args.output / "summary.json", summary)
    atomic_write_json(args.output / "run_status.json", {"status": "COMPLETE", "h0_successes": sum(r["status"] == "OK" for r in rows),
        "accepted_repairs": sum(r.get("admission", {}).get("decision") == "ACCEPT" for r in rows)})
    print(json.dumps({k: v for k, v in summary.items() if k != "evaluation"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("dataset", "config", "api-key-file", "cache", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    run(p.parse_args())
