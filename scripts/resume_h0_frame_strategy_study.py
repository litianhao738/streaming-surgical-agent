"""Continue the frozen study without repeating any previously dispatched request."""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from PIL import Image

from scripts import run_h0_frame_strategy_study as study
from surgical_agent.api.contracts import ApiImageInput, ApiRequest, canonical_json_bytes
from surgical_agent.api.providers.openrouter import HttpResponse, _sse_line_has_content


class DurableAuditSender:
    def __init__(self, directory, secret):
        self.directory, self.secret = directory, secret

    def __call__(self, url, headers, body, timeout):
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        # Exclusive marker prevents an accidental second dispatch to this call directory.
        with (directory / "dispatch.lock").open("x", encoding="utf-8") as marker:
            marker.write("one provider attempt; never retry\n")
        wire = json.loads(body)
        for message in wire["messages"]:
            if isinstance(message["content"], list):
                for part in message["content"]:
                    if part.get("type") == "image_url":
                        uri = part["image_url"]["url"]
                        part["image_url"]["url"] = "[image omitted; SHA-bound]"
                        part["image_url"]["data_url_sha256"] = hashlib.sha256(uri.encode()).hexdigest()
        study.atomic_write_json(directory / "wire_request.json", wire)
        started = perf_counter()
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        secret_bytes = self.secret.reveal().encode()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                chunks, first, generation = [], None, None
                study.atomic_write_json(directory / "http_status.json", {"status_code": response.status})
                with (directory / "response_stream.sse").open("xb") as stream:
                    for line in response:
                        chunks.append(line)
                        stream.write(line.replace(secret_bytes, b"[REDACTED]"))
                        stream.flush()
                        if first is None and _sse_line_has_content(line):
                            first = (perf_counter() - started) * 1000
                        if generation is None and line.startswith(b"data:"):
                            try:
                                event = json.loads(line[5:])
                            except (ValueError, UnicodeDecodeError):
                                continue
                            if isinstance(event, dict) and event.get("id"):
                                generation = event["id"]
                                study.atomic_write_json(directory / "generation.json", {"id": generation})
                result = HttpResponse(response.status, {k.lower(): v for k, v in response.headers.items()}, b"".join(chunks),
                                      first, (perf_counter() - started) * 1000)
        except urllib.error.HTTPError as exc:
            result = HttpResponse(exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read(),
                                  total_latency_ms=(perf_counter() - started) * 1000)
        study.atomic_write_json(directory / "http_response.json", {
            "status_code": result.status_code,
            "body": result.body.decode("utf-8", errors="replace").replace(self.secret.reveal(), "[REDACTED]"),
        })
        return result


def restore_requests(output, plan):
    """Reconstruct exact PNG bytes and canonical requests before any paid call."""
    frozen = dict(plan)
    digest = frozen.pop("plan_sha256")
    if hashlib.sha256(canonical_json_bytes(frozen)).hexdigest() != digest:
        raise ValueError("frozen plan digest mismatch")
    for path, expected in plan["source_sha256"].items():
        if study.sha256_file(ROOT / path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    def encode_image(pair):
        name, expected = pair
        path = Path(name)
        if study.sha256_file(path) != expected:
            raise ValueError("source image changed")
        encoded = io.BytesIO()
        with Image.open(path) as source:
            source.convert("RGB").save(encoded, format="PNG", compress_level=9)
        identifier = f"cholectrack20:{path.parent.parent.name}:frame:{int(path.stem)}"
        return identifier, ApiImageInput(identifier, "image/png", encoded.getvalue())
    with ThreadPoolExecutor(max_workers=8) as pool:
        images = dict(pool.map(encode_image, plan["source_png_sha256"].items()))
    requests = {}
    for key, meta in plan["requests"].items():
        saved = json.loads((output / "requests" / f"{key}.json").read_text(encoding="utf-8"))
        if saved["metadata"] != meta:
            raise ValueError("saved request metadata changed")
        request = ApiRequest(
            provider=meta["provider"], model_identifier=meta["requested_model_identifier"],
            endpoint_identifier=meta["endpoint_identifier"], prompt_version=meta["prompt_version"],
            response_schema_version=meta["response_schema_version"], payload=saved["payload"],
            images=tuple(images[item["identifier"]] for item in meta["images"]),
            generation_parameters=meta["generation_parameters"],
        )
        if study.canonical_request_metadata(request).to_mapping() != meta:
            raise ValueError("reconstructed request changed")
        requests[key] = request
    return requests


def recover_saved_response(directory, request, config, secret, row):
    saved = json.loads((directory / "http_response.json").read_text(encoding="utf-8"))
    records = study.read_call_usage(directory)
    if len(records) != 1 or records[0]["error"]["code"] != "response_envelope_invalid":
        raise ValueError("unexpected recovery ledger")
    old = records[0]

    def replay_sender(*_):
        return HttpResponse(saved["status_code"], {"content-type": "text/event-stream"}, saved["body"].encode(),
                            total_latency_ms=old["total_latency_ms"])

    transport = study.OpenRouterTransport(api_key=secret, endpoint_identifier=config.endpoint_identifier,
                                         sender=replay_sender)
    response = study.CompleteAccountingTransport(transport).send(request)
    study.validate_final_only(response.parsed_payload)
    if response.returned_model_identifier != config.requested_model_identifier:
        raise ValueError("recovered model mismatch")
    study.atomic_write_json(directory / "usage_before_header_recovery.json", old)
    corrected = {**old, "error": None, "provider_cost": response.provider_cost,
                 "origin_provider_cost": response.provider_cost, "provider_request_id": response.provider_request_id,
                 "returned_model_identifier": response.returned_model_identifier,
                 "input_tokens": response.input_tokens, "prompt_tokens": response.input_tokens,
                 "output_tokens": response.output_tokens, "completion_tokens": response.output_tokens,
                 "total_tokens": response.total_tokens, "visible_output_tokens": response.visible_output_tokens,
                 "completion_tokens_details": {"reasoning_tokens": response.completion_tokens_details.reasoning_tokens}}
    (directory / "api_usage.jsonl").write_text(json.dumps(corrected) + "\n", encoding="utf-8")
    study.atomic_write_json(directory / "offline_recovery.json", {
        "method": "same production parser; local saved SSE with normalized content-type; zero network calls",
        "http_response_sha256": study.sha256_file(directory / "http_response.json"),
        "original_ledger": "usage_before_header_recovery.json", "provider_request_id": response.provider_request_id,
        "provider_cost": response.provider_cost,
    })
    value = study.thaw_json(response.parsed_payload)
    recovered = {k: v for k, v in row.items() if k not in {"error", "http_status", "accounting_error"}}
    recovered.update(status="OK", returned_model=response.returned_model_identifier, cache_hit=False,
                     recovered_from_saved_response=True,
                     selected_ids={task: [value[task]["selected_id"]] if task == "phase" else value[task]["selected_ids"]
                                   for task in study.TASKS}, usage=study.usage_summary([corrected]))
    return recovered


def reconcile_accounting(row, directory, secret):
    """Look up billing for a known generation; never repeat or repair inference."""
    if not row.get("usage", {}).get("unpriced_calls"):
        return row
    generation_path = directory / "generation.json"
    if not generation_path.exists():
        return row
    generation = json.loads(generation_path.read_text(encoding="utf-8"))["id"]
    metadata_path = directory / "generation_accounting.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        url = "https://openrouter.ai/api/v1/generation?" + urllib.parse.urlencode({"id": generation})
        request = urllib.request.Request(url, headers={"Authorization": "Bearer " + secret.reveal()})
        with urllib.request.urlopen(request, timeout=30) as response:
            metadata = json.load(response)
        study.atomic_write_json(metadata_path, metadata)
    data = metadata["data"]
    cost = data.get("total_cost")
    if data.get("id") != generation or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        raise ValueError("invalid generation billing")
    records = study.read_call_usage(directory)
    if len(records) != 1:
        raise ValueError("unexpected ledger count")
    old = records[0]
    study.atomic_write_json(directory / "usage_before_billing_reconciliation.json", old)
    corrected = {**old, "provider_cost": cost, "origin_provider_cost": cost, "provider_request_id": generation}
    (directory / "api_usage.jsonl").write_text(json.dumps(corrected) + "\n", encoding="utf-8")
    row = {**row, "usage": study.usage_summary([corrected]), "billing_reconciled": True,
           "billing_source": "generation_accounting.json; inference status unchanged"}
    row.pop("accounting_error", None)
    return row


def call_accounted(item, request, config, args, secret, budget):
    row = study.call_one(item, request, config, args, secret, budget)
    try:
        return reconcile_accounting(row, args.output / "calls" / item["key"], secret)
    except Exception as exc:  # noqa: BLE001 - preserve failure and stop if billing lookup fails
        row["billing_lookup_error"] = type(exc).__name__
        return row


def run(args):
    output = args.output
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    config = study.load_api_config(args.config)
    if study.sha256_file(args.config) != plan["config_sha256"]:
        raise ValueError("config changed")
    if study.sha256_file(args.dataset / "repair_manifest.json") != plan["repair_manifest_sha256"]:
        raise ValueError("dataset repair manifest changed")
    with (output / "continuation_v4.lock").open("x", encoding="utf-8") as lock:
        lock.write("single continuation; never launch twice\n")
    previous = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    previous = [row for row in previous if row["status"] != "NOT_ATTEMPTED"]
    if len(previous) != 73 or previous[0]["key"] != "VID103_5351_A":
        raise ValueError("unexpected prior predictions")
    dispatched = {path.parent.name for path in (output / "calls").glob("*/wire_request.json")}
    if dispatched != {row["key"] for row in previous}:
        raise ValueError("unexpected prior dispatches")
    credits = json.loads((output / "credits_snapshot.json").read_text(encoding="utf-8"))["data"]
    available = credits["total_credits"] - credits["total_usage"]
    prior_known = sum(row["usage"]["reported_cost_usd"] for row in previous if "usage" in row)
    balance_cap = prior_known + available - 0.10
    study.atomic_write_json(output / "continuation_v4_plan.json", {
        "original_plan_sha256": plan["plan_sha256"], "continuation_script_sha256": study.sha256_file(__file__),
        "authorization": "User: 继续跑实验吧，快点; resume remaining requests without replaying prior dispatches",
        "supersedes": "Account balance insufficient for full study. Continue original order to first 32 targets (8 per video, 128 planned calls); no repeat dispatch; original full plan preserved",
        "unknown_call": "VID103_5351_B", "unknown_cost_reserve_usd": 3.0,
        "unknown_cost_is_actual_charge": False, "total_planning_budget_usd": 10.0,
        "new_call_limit": 55, "concurrency": 4, "retries": 0,
        "balance_at_resume_usd": available, "known_spend_admission_cap_usd": balance_cap,
        "selection_note": "original order prefix, chosen for available funds only; represents earlier portions of the four videos, not full temporal coverage",
        "spend_policy": "known spend + 3 USD old unknown reserve + 0.15 USD per outstanding new call <= 10 USD; reconcile missing billing via generation ID, stop if unreconciled; not provider-enforced cap",
    })
    print(json.dumps({"status": "VERIFYING_FROZEN_REQUESTS"}), flush=True)
    requests = restore_requests(output, plan)
    study.AuditSender = DurableAuditSender
    secret = study.resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    for index, row in enumerate(previous):
        previous[index] = reconcile_accounting(row, output / "calls" / row["key"], secret)
    budget = study.ProviderCallBudget(55)
    completed = {row["key"]: row for row in previous}
    unknown = next(item for item in plan["call_order"] if item["key"] == "VID103_5351_B")
    completed[unknown["key"]] = {**unknown, "status": "API_FAILURE", "error": "PRIOR_RESPONSE_LOST",
                                "accounting_error": "UNACCOUNTED_CALL", "reported_cost_usd": None}
    order = [item for item in plan["call_order"][:128] if item["key"] not in dispatched]
    spending = sum(row["usage"]["reported_cost_usd"] for row in previous if "usage" in row)
    cursor, stop = 0, None

    def checkpoint(status):
        study.atomic_write_json(output / "predictions.json", [completed[x["key"]] for x in plan["call_order"] if x["key"] in completed])
        study.atomic_write_json(output / "run_status.json", {
            "status": status, "completed": len(completed), "successful": sum(r["status"] == "OK" for r in completed.values()),
            "new_provider_attempts": budget.used, "known_cost_usd": round(spending, 8),
            "unknown_cost_reserve_usd": 3.0, "reason": stop, "plan_sha256": plan["plan_sha256"],
        })

    checkpoint("RUNNING_CONTINUATION")
    with ThreadPoolExecutor(max_workers=4) as executor:
        active = {}
        while cursor < len(order) or active:
            while not stop and cursor < len(order) and len(active) < 4:
                if spending + 3.0 + (len(active) + 1) * 0.15 > 10.0 or spending + (len(active) + 1) * 0.15 > balance_cap:
                    if not active:
                        stop = "PLANNING_SPEND_LIMIT"
                    break
                item = order[cursor]
                active[executor.submit(call_accounted, item, requests[item["key"]], config, args, secret, budget)] = item
                cursor += 1
            if not active:
                break
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                item = active.pop(future)
                row = future.result()
                completed[item["key"]] = row
                spending += row["usage"]["reported_cost_usd"]
                if row.get("accounting_error") or row["usage"]["unpriced_calls"]:
                    stop = row.get("accounting_error", "UNPRICED_CALL")
                if row.get("http_status") in {400, 401, 402, 403, 404} or row["status"] == "INTERNAL_FAILURE":
                    stop = row.get("error", "INTERNAL_FAILURE")
                if row["status"] == "OK" and row["returned_model"] != config.requested_model_identifier:
                    stop = "RETURNED_MODEL_MISMATCH"
                checkpoint("RUNNING_CONTINUATION")
                print(json.dumps({"key": item["key"], "status": row["status"], "completed": len(completed),
                                  "known_cost_usd": round(spending, 6), "stop": stop}), flush=True)
    stop = stop or "ACCOUNT_BALANCE_LIMITED_32_TARGET_PILOT"
    rows = [completed.get(item["key"], {**item, "status": "NOT_ATTEMPTED", "error": stop}) for item in plan["call_order"]]
    study.atomic_write_json(output / "predictions.json", rows)
    summary = study.score(args, plan, rows)
    summary["accounting_note"] = {"cost_is_confirmed_lower_bound": True, "prior_unknown_cost_calls": [unknown["key"]],
                                   "unknown_cost_reserve_usd": 3.0, "total_dispatched": budget.used + len(dispatched)}
    study.atomic_write_json(output / "summary.json", summary)
    checkpoint("STOPPED" if stop else "COMPLETE_WITH_PRIOR_LOST_RESPONSE")
    # Preserve final unattempted rows too if budget/provider stopped the study.
    study.atomic_write_json(output / "predictions.json", rows)
    print(json.dumps({"status": "FINISHED", "successful": summary["successful_predictions"], "known_cost_usd": spending}), flush=True)


if __name__ == "__main__":
    run(study.parser().parse_args())
