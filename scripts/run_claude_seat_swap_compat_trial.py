"""Explicit v2 after a provider grammar-size rejection; never a silent retry.

Only the transport response_format field is removed. Original prompt-embedded
JSON Schema, pictures, local validators, routing and stage token limits remain.
The failed v1 request's unknown reserve stays inside the shared $2 envelope.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_claude_seat_swap_trial as original

PROFILE = "claude_for_grok_cached_four_seat_swap_v2_prompt_json"
DEFAULT = ROOT / "artifacts/preflight/claude_seat_swap_20260909_v2"
PREVIOUS = ROOT / "artifacts/preflight/claude_seat_swap_20260909_v1"
ORIGINAL_PROFILE = original.PROFILE
ORIGINAL_VERIFY = original.verify_plan
ORIGINAL_CALLS = original.ClaudeCalls


def compatibility_wire(body):
    wire = deepcopy(body)
    wire.update(model=original.MODEL, temperature=0,
                provider={"only": ["anthropic"], "order": ["anthropic"],
                          "allow_fallbacks": False, "require_parameters": True},
                reasoning={"enabled": False})
    wire.pop("reasoning_effort", None)
    wire.pop("response_format", None)
    original.same(wire["messages"], body["messages"], "unchanged full semantic packet and images")
    return wire


def previous_evidence():
    plan = original.read(PREVIOUS / "plan.json")
    original.same(plan["profile"], ORIGINAL_PROFILE, "original v1 profile")
    closed = original.read(PREVIOUS / "completion.json")
    if closed["fatal_error"] is not None or closed["post_calls"] != 1:
        raise ValueError("expected the closed one-request compatibility failure")
    for name, value in closed["inference_artifact_sha256"].items():
        original.same(original.sha(PREVIOUS / name), value, "unchanged v1 inference evidence")
    for name, value in plan["source_sha256"].items():
        original.same(original.sha(ROOT / name), value, "unchanged v1 execution source")
    ledger = original.read(PREVIOUS / "budget.json")
    if not ledger["stopped"] or len(ledger["calls"]) != 1 or ledger["calls"][0]["http_status"] != 400:
        raise ValueError("v1 must be a stopped one-request HTTP 400 run")
    original.same(ledger["limits"], original.LIMITS, "shared two-dollar budget")
    total = sum(Decimal(r["charge"]) for r in ledger["calls"])
    original.same(str(total), ledger["occupied"]["openrouter_usd"], "v1 occupied accounting")
    return plan, ledger


class CompatibilityCalls(ORIGINAL_CALLS):
    def __init__(self, output):
        plan = original.read(output / "plan.json")
        original.Calls.__init__(self, output, previous_budget=plan["carried_occupied"],
            limits={k: Decimal(v) for k, v in original.LIMITS.items()}, rates=original.RATES,
            providers={"claude": original.PROVIDER}, max_calls=16)


def verify_v2(output):
    plan = ORIGINAL_VERIFY(output)
    _, previous = previous_evidence()
    original.same(plan["carried_occupied"], previous["occupied"], "carried v1 charge/reserve")
    original.same(plan["transport_compatibility_change"], "remove_response_format_only", "explicit compatibility protocol")
    original.same(plan["combined_max_calls"], 17, "one v1 plus sixteen v2 calls")
    for name, value in plan["previous_evidence_sha256"].items():
        original.same(original.sha(PREVIOUS / name), value, "frozen compatibility failure " + name)
    if (output / "budget.json").exists():
        ledger = original.read(output / "budget.json")
        original.same(ledger["carried_occupied"], previous["occupied"], "budget retained old reserve")
        accounted = Decimal(previous["occupied"]["openrouter_usd"]) + sum(Decimal(r["charge"]) for r in ledger["calls"])
        if accounted != Decimal(ledger["occupied"]["openrouter_usd"]) or accounted > Decimal(2):
            raise ValueError("combined v1/v2 accounting exceeds or differs from the budget")
    return plan


@contextmanager
def v2_protocol():
    """Scoped reuse of v1 helpers; no v1 file, cache or provider identity changes."""
    updates = {"PROFILE": PROFILE, "DEFAULT": DEFAULT, "claude_wire": compatibility_wire,
               "ClaudeCalls": CompatibilityCalls, "verify_plan": verify_v2}
    before = {name: getattr(original, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(original, name, value)
        yield
    finally:
        for name, value in before.items():
            setattr(original, name, value)


def prepare(output, adapter):
    if output.exists():
        raise ValueError("new single-use v2 directory required")
    prior_plan, prior_ledger = previous_evidence()
    originals = original.read(PREVIOUS / "source_inputs.json")
    old_wires = original.read(PREVIOUS / "wire_preflight.json")
    wires, fingerprints, reservations = {}, {}, {}
    for initial, selected in zip(originals["targets"], prior_plan["selection"], strict=True):
        key = initial["key"]
        bodies = original.build_wires(initial, selected, prior_plan["phase_inputs"][key], adapter, original.SOURCE)
        wires[key], fingerprints[key], reservations[key] = {}, {}, {}
        for stage, body in bodies.items():
            expected = deepcopy(old_wires[key][stage])
            expected.pop("response_format")
            safe = original.redact_images(body)
            original.same(safe, expected, "v2 differs from v1 only by missing transport response_format")
            packet = json.loads(body["messages"][0]["content"][0]["text"])
            if not isinstance(packet.get("response_schema"), dict) or "response_format" in body:
                raise ValueError("full prompt schema must remain while transport grammar is absent")
            wires[key][stage] = safe
            fingerprints[key][stage] = original.fingerprint(body)
            reservations[key][stage] = str(original.envelope("claude", body, original.RATES))
    maximum = sum(Decimal(v) for row in reservations.values() for v in row.values())
    carried = Decimal(prior_ledger["occupied"]["openrouter_usd"])
    if carried + maximum > Decimal(2):
        raise ValueError("old unresolved reserve plus all v2 calls exceeds shared $2")
    output.mkdir(parents=True)
    shutil.copyfile(PREVIOUS / "source_inputs.json", output / "source_inputs.json")
    shutil.copyfile(PREVIOUS / "metadata.json", output / "metadata.json")
    original.save(output / "wire_preflight.json", wires)
    original.save(output / "previous_budget.json", prior_ledger)
    plan = deepcopy(prior_plan)
    plan.update(profile=PROFILE, created_utc=original.now(), previous_attempt=str(PREVIOUS),
                carried_occupied=prior_ledger["occupied"], combined_max_calls=17,
                transport_compatibility_change="remove_response_format_only",
                transport_schema_adaptations={"wire": "response_format omitted; full prompt schema unchanged"},
                wire_fingerprints=fingerprints, conservative_request_reservations_usd=reservations,
                all_requests_reservation_usd=str(maximum), combined_reservation_usd=str(carried + maximum),
                previous_evidence_sha256={name: original.sha(PREVIOUS / name)
                    for name in ["plan.json", "completion.json", "budget.json",
                                 "calls/000_VID103_18326_graph_swap_claude/response.json"]},
                input_sha256={name: original.sha(output / name)
                    for name in ("source_inputs.json", "metadata.json", "wire_preflight.json", "previous_budget.json")})
    plan["source_sha256"][Path(__file__).relative_to(ROOT).as_posix()] = original.sha(__file__)
    plan["policy"] = ("Explicit new compatibility experiment after one v1 HTTP400 grammar-size rejection. "
        "Only the top-level transport response_format is removed from v1 requests; complete semantic packet, "
        "embedded JSON Schema, original three images, model/provider/nonthinking settings, graph8192/Phase4096 "
        "limits and all strict local validators stay unchanged. H0, candidate pool and four other model responses "
        "remain cached. All five valid graph scores and all five valid Phase choices are still required. "
        "No automatic retry, fallback route, replacement vote or reduced quorum. Up to 16 new requests; "
        "one previous request and its unresolved occupied amount remain inside the shared $2 envelope.")
    plan["limitations"].append("V1 produced only a provider grammar compilation failure, not a model-quality score. Its unknown reserve is carried conservatively, not declared paid or waived.")
    original.save(output / "plan.json", plan)
    for name in plan["source_sha256"]:
        destination = output / "frozen_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    verify_v2(output)
    print(json.dumps({"prepared": str(output), "profile": PROFILE, "targets": 8, "new_requests_max": 16,
        "combined_requests_max": 17, "carried_unknown_reserve_usd": str(carried),
        "new_requests_reservation_usd": str(maximum), "combined_reservation_usd": str(carried + maximum),
        "shared_budget_usd": "2", "change": "remove_response_format_only", "new_paid_calls": 0}), flush=True)


def score(output, adapter):
    original.score(output, adapter)
    report = original.read(output / "metrics.json")
    ledger = original.read(output / "previous_budget.json")
    carried_native = sum(Decimal(r["charge"]) for r in ledger["calls"] if r["charge_kind"] == "native")
    carried_unknown = sum(Decimal(r["charge"]) for r in ledger["calls"] if r["charge_kind"] == "unknown_reserved")
    report["costs"].update(carried_v1_native_usd=str(carried_native), carried_v1_unknown_reserved_usd=str(carried_unknown),
        combined_native_usd=str(Decimal(report["costs"]["native_usd"]) + carried_native),
        combined_unknown_reserved_usd=str(Decimal(report["costs"]["unknown_reserved_usd"]) + carried_unknown))
    report["combined_post_calls"] = report["post_calls"] + len(ledger["calls"])
    original.save(output / "metrics.json", report)
    print(json.dumps({"profile": PROFILE, "costs": report["costs"],
                      "combined_post_calls": report["combined_post_calls"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "audit", "score"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    with v2_protocol():
        if args.command == "audit":
            _, rows, _, _ = original.audit(args.output)
            print(json.dumps({"verified": True, "profile": PROFILE, "targets": len(rows)}))
            return
        adapter = original.CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
        if args.command == "prepare":
            prepare(args.output, adapter)
        elif args.command == "execute":
            original.execute(args.output, adapter)
        else:
            score(args.output, adapter)


if __name__ == "__main__":
    main()
