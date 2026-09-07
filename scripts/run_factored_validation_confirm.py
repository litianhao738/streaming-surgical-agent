"""One explicitly selected factored B Validation confirmation, no automatic selection.

Default preflight is offline. The separate lock action must precede every paid
Validation call. This driver never sends or reports a joint names1000 baseline.
"""

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_verifier_variant_trial as shared
from surgical_agent.api.contracts import canonical_json_bytes, thaw_json
from surgical_agent.api.credentials import assert_secret_absent
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.research.verification.factored_presence import (
    FACTORED_PRESENCE_PROMPT_VERSION,
    merge_factored_presence_reviews,
    split_factored_presence_request,
)
from surgical_agent.research.verification.presence_review import (
    PRESENCE_REVIEW_1000_VERSION,
    evaluate_presence_review,
)

VARIANT = "factored_names1000"
PRIMARY = "final_b"
TARGETS = 8
COLLECT_CAP = TARGETS * 3
POLICY = (
    "One preselected factored_names1000 confirmation; primary final_b, final_a descriptive only. "
    "Validation results, including positive results, cannot select another variant, change rules, "
    "or trigger tuning. Missing GT is masked per task. All eight targets remain in denominators. "
    "No joint names1000 API baseline is run or inferred."
)


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def check_signed(value, field):
    if value.get(field) != digest({k: v for k, v in value.items() if k != field}):
        raise ValueError(f"{field} mismatch")


def sources():
    return list(dict.fromkeys(shared._source_files() + [Path(__file__),
        ROOT / "src/surgical_agent/research/verification/factored_presence.py"]))


def check_manifest(path):
    manifest = shared.read(path)
    selected = manifest.get("selection", [])
    keys = [f"{s['video_id']}_{s['frame_id']}" for s in selected]
    if (manifest.get("source_split") != "Validation" or len(keys) != TARGETS
            or len(set(keys)) != TARGETS
            or any(s.get("source_split") != "Validation" for s in selected)):
        raise ValueError("exactly eight fixed Validation targets required; Training/Testing forbidden")
    return manifest, keys


def create_choice_lock(args):
    """Operator-only explicit selection; inspect completion metadata, never GT values."""
    if not args.select_factored_for_validation:
        raise ValueError("explicit --select-factored-for-validation required")
    if not 1 <= args.max_review_calls <= 320 or not 1 <= args.timeout_seconds <= 600:
        raise ValueError("review cap 1..320 and timeout 1..600 required")
    _, keys = check_manifest(args.targets_json)
    training = args.training_evidence.resolve()
    plan, summary = shared.read(training / "plan.json"), shared.read(training / "summary.json")
    check_signed(plan, "plan_sha256")
    if (plan.get("source_split") != "Training" or summary.get("source_split") != "Training"
            or plan.get("variants") != [VARIANT] or summary.get("variants") != [VARIANT]
            or summary.get("targets") != 16 or summary.get("plan_sha256") != plan["plan_sha256"]
            or summary.get("predictions_sha256") != shared.sha(training / "predictions.json")
            or not (training / "scored_predictions.json").is_file()):
        raise ValueError("completed sixteen-target Training factored evidence required")
    artifact_paths = [args.targets_json.resolve(), args.protocol.resolve()]
    artifact_paths += [training / name for name in
                       ("plan.json", "summary.json", "predictions.json", "scored_predictions.json")]
    lock = {"schema_version": "factored_validation_choice_lock_v1", "variant": VARIANT,
            "primary": PRIMARY, "secondary": "final_a", "source_split": "Validation",
            "targets_manifest": str(args.targets_json.resolve()), "keys": keys,
            "training_evidence": str(training), "protocol": str(args.protocol.resolve()),
            "goal_id": args.goal_id, "budget_ledger": str(args.budget_ledger.resolve()),
            "goal_budget_usd": str(shared.GOAL_CAP_USD),
            "development_budget_usd": str(shared.DEVELOPMENT_CAP_USD),
            "max_collection_calls": COLLECT_CAP, "max_review_calls": args.max_review_calls,
            "timeout_seconds": args.timeout_seconds, "selection_policy": POLICY,
            "source_artifact_sha256": {str(p): shared.sha(p) for p in artifact_paths},
            "source_sha256": {str(p.relative_to(ROOT)): shared.sha(p) for p in sources()},
            "gt_values_read_by_lock_writer": False,
            "selection_uses_training_results": True,
            "validation_gt_values_used_for_selection": False}
    lock["choice_sha256"] = digest(lock)
    args.choice_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.choice_lock.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, ensure_ascii=False, indent=2)
    return lock


def check_choice(args):
    lock = shared.read(args.choice_lock)
    check_signed(lock, "choice_sha256")
    if (lock.get("variant") != VARIANT or lock.get("primary") != PRIMARY
            or lock.get("source_split") != "Validation" or lock.get("selection_policy") != POLICY
            or lock.get("max_collection_calls") != COLLECT_CAP
            or lock.get("goal_id") != args.goal_id
            or lock.get("budget_ledger") != str(args.budget_ledger.resolve())
            or lock.get("targets_manifest") != str(args.targets_json.resolve())):
        raise ValueError("explicit Validation choice/budget/manifest binding mismatch")
    shared.assert_frozen(lock)
    _, keys = check_manifest(args.targets_json)
    if keys != lock["keys"]:
        raise ValueError("Validation target order changed after selection")
    return lock


def prepare(args):
    lock = check_choice(args)
    output = args.output.resolve()
    if (output / "execution.lock").exists():
        raise ValueError("already dispatched; never repeat this confirmation")
    existed = output.exists()
    if existed and not (output / "plan.json").is_file():
        raise ValueError("incomplete preflight; use a fresh output directory")
    nested = output / "collection_inputs"
    collection_args = SimpleNamespace(**vars(args))
    collection_args.output, collection_args.max_calls = nested, COLLECT_CAP
    collection_args.confirmation, collection_args.variants = True, [VARIANT]
    collection_args.timeout_seconds = lock["timeout_seconds"]
    _, collection, bases = shared.prepare_collect(collection_args)
    artifacts = {**lock["source_artifact_sha256"], **collection["source_artifact_sha256"],
                 str(args.choice_lock.resolve()): shared.sha(args.choice_lock),
                 str(nested / "plan.json"): shared.sha(nested / "plan.json")}
    for selected in collection["selection"]:
        name = f"{selected['key']}.json"
        request_hash = shared.sha(nested / "requests" / name)
        artifacts[str(nested / "requests" / name)] = request_hash
        artifacts[str(output / "requests" / name)] = request_hash
    plan = {"schema_version": "factored_validation_confirmation_plan_v1",
            "model": shared.MODEL, "source_split": "Validation", "confirmation": True,
            "variants": [VARIANT], "primary": PRIMARY, "secondary": "final_a",
            "choice_sha256": lock["choice_sha256"], "selection_policy": POLICY,
            "keys": lock["keys"], "selection": collection["selection"], "target_count": TARGETS,
            "max_provider_calls": COLLECT_CAP + lock["max_review_calls"],
            "max_collection_calls": COLLECT_CAP, "max_review_calls": lock["max_review_calls"],
            "goal_id": lock["goal_id"], "budget_ledger": lock["budget_ledger"],
            "goal_budget_usd": lock["goal_budget_usd"],
            "development_budget_usd": lock["development_budget_usd"],
            "timeout_seconds": lock["timeout_seconds"], "pricing": collection["pricing"],
            "reservation": collection["reservation"], "retries": 0,
            "review_manifest_policy": "freeze after all collection outputs, before first review POST",
            "source_artifact_sha256": artifacts, "source_sha256": lock["source_sha256"],
            "gt_values_used_in_requests": False,
            "selection_uses_training_results": True,
            "validation_gt_values_used_for_selection": False}
    plan["plan_sha256"] = digest(plan)
    if existed:
        if shared.read(output / "plan.json") != plan:
            raise ValueError("preflight changed; use a fresh output directory")
    else:
        atomic_write_json(output / "plan.json", plan)
        for path in sources():
            destination = output / "frozen_source" / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        for selected in plan["selection"]:
            key = selected["key"]
            atomic_write_json(output / "requests" / f"{key}.json", shared.read(nested / "requests" / f"{key}.json"))
    shared.assert_frozen(plan)
    return plan, bases


def keep(h0, status):
    return {"status": status, "final_a": deepcopy(h0), "final_b": deepcopy(h0),
            "decision_a": "KEEP", "decision_b": "KEEP", "review": None}


def merge_and_evaluate(row, names, reviews):
    body = json.loads(names.payload["input_text"])
    merged = merge_factored_presence_reviews(names, reviews)
    result = evaluate_presence_review(row["h0"], row["h1"], merged,
        allowed_evidence_refs=body["allowed_evidence_refs"], full_frame_ref=body["full_frame_ref"],
        schema_version=PRESENCE_REVIEW_1000_VERSION)
    return {**result, "review": merged, "status": "OK" if merged is not None else "REVIEW_FAILURE_KEEP"}


def saved_request(request):
    wire = shared.wire_body(request)
    return {"metadata": canonical_request_metadata(request).to_mapping(),
            "payload": thaw_json(request.payload), "wire": shared.safe_wire(wire),
            "wire_sha256": digest(wire)}


class StageCalls:
    """Two disjoint caps, with the same native-cost ledger boundary underneath."""

    def __init__(self, calls, *, cap, collection):
        self.calls, self.cap, self.collection = calls, cap, collection
        self.used, self.actual_requests = 0, {}

    def call(self, key, stage, request):
        if (stage in ("h0", "locator", "proposal")) != self.collection:
            raise ValueError("stage is forbidden in this confirmation phase")
        if not self.collection:
            body = json.loads(request.payload["input_text"])
            propositions = body.get("propositions", [])
            if (request.prompt_version != FACTORED_PRESENCE_PROMPT_VERSION
                    or request.response_schema_version != PRESENCE_REVIEW_1000_VERSION
                    or len(propositions) != 1 or propositions[0]["proposition_id"] != stage):
                raise ValueError("only a bound single-proposition factored request may dispatch")
        if self.calls.stopped or self.used >= self.cap:
            return None
        before = self.calls.used
        response = self.calls.call(key, stage, request)
        delta = self.calls.used - before
        self.used += delta
        if delta:
            self.actual_requests[(key, stage)] = request
        return response


def prepare_reviews(output, plan, rows):
    hashes = {str(output / "collected_predictions.json"): shared.sha(output / "collected_predictions.json")}
    requests, templates, selection = {}, {}, []
    for selected, row in zip(plan["selection"], rows, strict=True):
        key = selected["key"]
        _, rebuilt = shared.restore_target(output, selected, row, hashes)
        row["factored_parts"] = []
        row["variants"] = {VARIANT: keep(row["h0"], "NO_CHANGED_CANDIDATE")}
        if rebuilt is None:
            continue
        numeric, evidence, full_ref = rebuilt
        names = shared.variant_request(numeric, "names1000")
        templates[key] = names
        template_path = output / "unsent_joint_review_templates" / f"{key}.json"
        atomic_write_json(template_path, {**saved_request(names), "dispatch_status": "NEVER_SENT_TEMPLATE_ONLY"})
        hashes[str(template_path)] = shared.sha(template_path)
        part_records = []
        for part in split_factored_presence_request(names):
            shared.check_request_envelope(part)
            identity = json.loads(part.payload["input_text"])["propositions"][0]["proposition_id"]
            requests[(key, identity)] = part
            path = output / "review_requests" / key / f"{identity}.json"
            saved = saved_request(part)
            atomic_write_json(path, saved)
            hashes[str(path)] = shared.sha(path)
            part_records.append({"proposition_id": identity, "request_metadata": saved["metadata"],
                                 "wire_sha256": saved["wire_sha256"]})
        row["variants"][VARIANT] = keep(row["h0"], "NOT_ATTEMPTED")
        selection.append({"key": key, "parts": part_records,
                          "evidence_manifest": evidence, "full_frame_ref": full_ref})
    manifest = {"schema_version": "factored_validation_review_manifest_v1",
                "plan_sha256": plan["plan_sha256"], "choice_sha256": plan["choice_sha256"],
                "selection": selection, "expected_calls": len(requests),
                "max_review_calls": plan["max_review_calls"],
                "source_artifact_sha256": hashes, "source_sha256": plan["source_sha256"],
                "names_joint_api_calls": 0, "gt_values_read": False}
    manifest["manifest_sha256"] = digest(manifest)
    atomic_write_json(output / "review_manifest.json", manifest)
    return manifest, templates, requests


def audit_actual_receipts(output, actual_requests):
    """Bind every real dispatch's full wire and native response; no GT access."""
    hashes = {}
    for (key, stage), expected in actual_requests.items():
        directory = output / "calls" / key / stage
        request, result = (shared.read(directory / name) for name in ("request.json", "result.json"))
        saved = saved_request(expected)
        if (any(request[name] != saved[name] for name in saved)
                or result["request_hash"] != saved["metadata"]["request_hash"]
                or result["key"] != key or result["stage"] != stage):
            raise ValueError("actual dispatch request/wire/result binding mismatch")
        for name in ("request.json", "result.json", "dispatch.lock"):
            hashes[str(directory / name)] = shared.sha(directory / name)
        http_path = directory / "http_response.json"
        if http_path.is_file():
            http = shared.read(http_path)
            hashes[str(http_path)] = shared.sha(http_path)
            if http["status_code"] != result.get("http_status"):
                raise ValueError("saved native HTTP status differs from local result")
            # Native failures remain failures and stay in the denominator. A
            # malformed body or wrong route must not prevent reporting KEEP.
            if result["status"] == "OK":
                native = json.loads(http["body"])
                if (http["status_code"] != 200 or native.get("model") != shared.MODEL
                        or native.get("provider") != "Alibaba" or not native.get("id")
                        or native["id"] != result.get("provider_request_id")
                        or json.loads(native["choices"][0]["message"]["content"]) != result["payload"]):
                    raise ValueError("successful response differs from native route/content")
        elif result.get("http_status") is not None or result["status"] == "OK":
            raise ValueError("reported HTTP response lacks native evidence")
    return hashes


def run(args):
    plan, bases = prepare(args)
    if not args.execute:
        return plan
    if args.api_key_file is None:
        raise ValueError("execution requires an external key file")
    # Choice, sources, inputs and root plan are checked before even public pricing.
    check_choice(args)
    execution_args = SimpleNamespace(**vars(args))
    execution_args.max_calls = plan["max_provider_calls"]
    execution_args.timeout_seconds = plan["timeout_seconds"]
    secret, ledger, calls = shared.start_execution(execution_args, plan)
    collect = StageCalls(calls, cap=COLLECT_CAP, collection=True)
    rows = [{"video_id": s["video_id"], "frame_id": s["frame_id"], "status": "NOT_ATTEMPTED",
             "h0": None, "h0_payload": None, "h1": None, "final": None, "locator": None,
             "proposal": None, "request_hashes": {}, "crop_manifest": []} for s in plan["selection"]]
    atomic_write_json(args.output / "predictions.json", rows)
    for index, selected in enumerate(plan["selection"]):
        if calls.stopped:
            break
        rows[index] = shared.collect_target(bases[selected["key"]], collect, selected["key"])
        atomic_write_json(args.output / "predictions.json", rows)
    atomic_write_json(args.output / "collected_predictions.json", rows)
    manifest, templates, requests = prepare_reviews(args.output, plan, rows)
    atomic_write_json(args.output / "predictions.json", rows)
    shared.assert_frozen(plan)
    shared.assert_frozen(manifest)
    manifest_hash = shared.sha(args.output / "review_manifest.json")
    review = StageCalls(calls, cap=plan["max_review_calls"], collection=False)
    by_key = {f"{r['video_id']}_{r['frame_id']}": r for r in rows}
    for selected in manifest["selection"]:
        if calls.stopped or review.used >= review.cap:
            break
        key, responses = selected["key"], []
        row = by_key[key]
        for part in selected["parts"]:
            if calls.stopped or review.used >= review.cap:
                break
            identity = part["proposition_id"]
            response = review.call(key, identity, requests[(key, identity)])
            responses.append(response)
            row["factored_parts"].append({"proposition_id": identity,
                "request_hash": part["request_metadata"]["request_hash"], "review": response})
            atomic_write_json(args.output / "predictions.json", rows)
        row["variants"][VARIANT] = merge_and_evaluate(row, templates[key], responses)
        atomic_write_json(args.output / "predictions.json", rows)
    calls.stopped = calls.stopped or ("REVIEW_CALL_CAP" if review.used >= review.cap
                                     and review.used < manifest["expected_calls"] else "INFERENCE_FINISHED")
    prediction_hash = shared.sha(args.output / "predictions.json")
    shared.assert_frozen(plan)
    shared.assert_frozen(manifest)
    if shared.sha(args.output / "review_manifest.json") != manifest_hash:
        raise ValueError("review manifest changed during inference")
    receipts = audit_actual_receipts(args.output, {**collect.actual_requests, **review.actual_requests})
    binding = {"plan_sha256": plan["plan_sha256"], "choice_sha256": plan["choice_sha256"],
               "review_manifest_sha256": manifest_hash, "predictions_sha256": prediction_hash,
               "source_artifact_sha256": receipts, "gt_values_read": False}
    atomic_write_json(args.output / "inference_completion_binding.json", binding)
    # First access to GT for scoring is after all eight finals and bindings are durable.
    scored, comparisons = shared.score_saved_rows(args, rows, [VARIANT])
    atomic_write_json(args.output / "scored_predictions.json", scored)
    if shared.sha(args.output / "predictions.json") != prediction_hash:
        raise ValueError("predictions changed during scoring")
    summary = {"schema_version": "factored_validation_confirmation_result_v1",
        "source_split": "Validation", "model": shared.MODEL, "targets": len(rows),
        "variants": [VARIANT], "primary": PRIMARY, "secondary": "final_a", "selection_policy": POLICY,
        "plan_sha256": plan["plan_sha256"], "choice_sha256": plan["choice_sha256"],
        "predictions_sha256": prediction_hash,
        "scored_predictions_sha256": shared.sha(args.output / "scored_predictions.json"),
        "review_manifest_sha256": manifest_hash,
        "inference_completion_binding_sha256": shared.sha(args.output / "inference_completion_binding.json"),
        "collection_calls": collect.used, "review_calls": review.used,
        "expected_review_calls": manifest["expected_calls"], "names_joint_api_calls": 0,
        "stop_reason": calls.stopped, "comparisons": comparisons, "goal_budget": ledger.snapshot(),
        "successful_merged_reviews": sum(r["variants"][VARIANT]["status"] == "OK" for r in rows),
        "not_attempted_review_targets": sum(r["variants"][VARIANT]["status"] == "NOT_ATTEMPTED" for r in rows),
        **shared.accounting(calls.records)}
    atomic_write_json(args.output / "summary.json", summary)
    assert_secret_absent(secret, (p for p in args.output.rglob("*") if p.is_file()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lock", "run"), default="run")
    parser.add_argument("--choice-lock", type=Path, required=True)
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument("--goal-id", required=True)
    parser.add_argument("--training-evidence", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--max-review-calls", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--select-factored-for-validation", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--pricing-snapshot", type=Path)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    required = ("training_evidence", "protocol", "max_review_calls") if args.mode == "lock" else (
        "output", "dataset_root", "pricing_snapshot")
    if any(getattr(args, name) is None for name in required):
        parser.error("required for this mode: " + ", ".join(required))
    result = create_choice_lock(args) if args.mode == "lock" else run(args)
    print(json.dumps({k: v for k, v in result.items() if k not in
        ("comparisons", "selection", "source_artifact_sha256", "source_sha256")}), flush=True)


if __name__ == "__main__":
    main()
