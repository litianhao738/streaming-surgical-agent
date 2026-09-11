"""Isolated Claude-for-Grok reviewer swap; cached H0, candidates and four seats.

Prepare makes no model call. Execute is single-use (16 requests, OR <= $2).
Score is separate and reads GT only after immutable inference closure checks.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import requests

from scripts.check_candidate_panel_providers import key_for, redact_images
from scripts.run_candidate_panel_trial import Calls, envelope, sha, usage_charge
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import sources
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_phase_extension_trial import phase_wire
from scripts.run_prior_feedback_continuation import same
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import review_wire
from scripts.run_repair_revision_trial import normalize_five
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.api.credentials import assert_secret_absent
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.phase_extension import apply_phase_choices
from surgical_agent.research.verification.prior_panel import labels
from surgical_agent.research.verification.review_json_compat import (
    parse_review_json_compatible,
)

PROFILE = "claude_for_grok_cached_four_seat_swap_v1"
MODEL = "anthropic/claude-haiku-4.5"
PROVIDER = "Anthropic"
SOURCE = ROOT / "artifacts/preflight/parallel_phase_eight_20260909_v1"
DEFAULT = ROOT / "artifacts/preflight/claude_seat_swap_20260909_v1"
METADATA = ROOT / "artifacts/preflight/reviewer_options_metadata_20260909_v1/claude-haiku-4.5_endpoints.json"
IDENTITIES = ("VID103_18326", "VID103_33576", "VID23_13176", "VID23_28876",
              "VID31_40701", "VID31_73701", "VID96_15051", "VID96_26051")
LIMITS = {"openrouter_usd": "2"}
RATES = {"claude": ("0.000001", "0.000005")}
STAGES = ("graph_swap", "phase_swap")
STAGE_MAX_TOKENS = {"graph_swap": 8192, "phase_swap": 4096}
TASKS = ("instrument", "verb", "target", "ivt", "phase")
VERSIONS = ("h0", "original_graph", "original_final", "swap_graph_only",
            "swap_graph_original_phase", "swap_phase_only", "both_swap")
UNSUPPORTED_SCHEMA_KEYS = frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                                    "multipleOf", "minLength", "maxLength", "minItems", "maxItems", "uniqueItems"})


def transport_schema(schema):
    """Anthropic transport subset only; semantic packet/local validators stay full."""
    dropped = []

    def clean(value, path):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in UNSUPPORTED_SCHEMA_KEYS:
                    dropped.append({"path": path + "/" + key, "value": item})
                else:
                    result[key] = clean(item, path + "/" + key)
            return result
        if isinstance(value, list):
            return [clean(item, path + "/" + str(index)) for index, item in enumerate(value)]
        return deepcopy(value)

    return clean(schema, ""), dropped


def claude_wire(original):
    body = deepcopy(original)
    body.update(model=MODEL, temperature=0,
                provider={"only": ["anthropic"], "order": ["anthropic"],
                          "allow_fallbacks": False, "require_parameters": True},
                reasoning={"enabled": False})
    body.pop("reasoning_effort", None)
    if body.get("response_format", {}).get("type") != "json_schema":
        raise ValueError("original Grok JSON Schema required; no silent format substitution")
    same(body["messages"], original["messages"], "unchanged semantic packet and images")
    schema, _ = transport_schema(body["response_format"]["json_schema"]["schema"])
    body["response_format"]["json_schema"]["schema"] = schema
    return body


def parse_claude_response(raw):
    if not isinstance(raw, dict):
        raise TypeError("Claude response must be an object")
    if raw.get("model") != MODEL or raw.get("provider") != PROVIDER:
        raise ValueError("Claude model or Anthropic provider identity mismatch")
    choice = raw["choices"][0]
    message = choice["message"]
    if choice["finish_reason"] != "stop" or message.get("refusal"):
        raise ValueError("incomplete or refused Claude response")
    usage = raw.get("usage") or {}
    if ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            or message.get("reasoning_content") or message.get("reasoning") or message.get("reasoning_details")):
        raise ValueError("Claude emitted reasoning despite disabled thinking")
    return parse_review_json_compatible(message["content"])


class ClaudeCalls(Calls):
    """Reuse accounting helpers, with an explicit Claude/OR identity and endpoint."""

    def __init__(self, output):
        super().__init__(output, limits={k: Decimal(v) for k, v in LIMITS.items()},
                         rates=RATES, providers={"claude": PROVIDER}, max_calls=16)

    def call(self, target, stage, seat, body):
        if target not in IDENTITIES or stage not in STAGES or seat != "claude":
            raise ValueError("undeclared Claude request")
        if body.get("max_tokens") != STAGE_MAX_TOKENS[stage]:
            raise ValueError("preserve the original stage output-token limit")
        same(body, claude_wire(body), "frozen Claude transport parameters")
        secret = key_for("gpt")  # The existing assigned OpenRouter credential file.
        reserve = envelope(seat, body, self.rates)
        account = "openrouter_usd"
        with self.lock:
            if any((r["target"], r["stage"]) == (target, stage) for r in self.rows):
                raise ValueError("no automatic paid retries")
            if self.stopped or len(self.rows) >= 16 or self.occupied[account] + reserve > self.limits[account]:
                self.stopped = True
                return None
            self.occupied[account] += reserve
            row = {"index": len(self.rows), "target": target, "stage": stage, "seat": "claude",
                   "replaced_logical_slot": "grok", "account": account, "model": MODEL,
                   "expected_provider": PROVIDER, "status": "DISPATCHED", "started_utc": now(),
                   "reserve": str(reserve), "charge": str(reserve), "charge_kind": "unknown_reserved"}
            self.rows.append(row)
            folder = self.output / "calls" / f"{row['index']:03d}_{target}_{stage}_claude"
            save(folder / "request.json", redact_images(body))
            save(folder / "record.json", row)
            self.persist()
        started, parsed = perf_counter(), None
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                    headers={"Authorization": "Bearer " + secret.reveal()}, timeout=(15, 120))
            try:
                raw = response.json()
            except ValueError:
                raw = {"non_json_response": response.text}
            raw = json.loads(json.dumps(raw).replace(secret.reveal(), "[REDACTED]"))
            save(folder / "response.json", {"http_status": response.status_code, "body": raw})
            if not isinstance(raw, dict):
                raise TypeError("provider response is not an object")
            usage = raw.get("usage") or {}
            charge, kind = usage_charge("claude", usage, reserve, self.rates)
            row.update(http_status=response.status_code, usage=usage, charge=str(charge),
                       charge_kind=kind, returned_model=raw.get("model"), returned_provider=raw.get("provider"),
                       reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       reasoning_usage_status=("reported" if "reasoning_tokens" in
                           (usage.get("completion_tokens_details") or {}) else "not_reported"),
                       status="API_FAILED")
            if response.status_code in (400, 401, 402, 404):
                self.stopped = True
            if response.ok and not raw.get("error"):
                parsed, diagnostic = parse_claude_response(raw)
                save(folder / "parsed.json", {"parsed": parsed, "diagnostic": diagnostic})
                row["status"] = "JSON_PARSED" if parsed is not None else "SAFE_JSON_REJECTED"
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            row.update(status="FAILED", exception_type=type(exc).__name__)
            if not isinstance(exc, requests.RequestException):
                self.stopped = True
        finally:
            with self.lock:
                self.occupied[account] += Decimal(row["charge"]) - reserve
                row.update(finished_utc=now(), elapsed_seconds=perf_counter() - started)
                save(folder / "record.json", row)
                self.persist()
            assert_secret_absent(secret, folder.glob("*.json"))
        return parsed


def mix(initial, graph_claude, phase_claude):
    """Only the named slot changes; old other-family failures remain failures."""
    old = initial["source_record"]
    raw_graph, raw_phase = deepcopy(old["graph"]["raw_reviews"]), deepcopy(old["phase"]["raw_reviews"])
    raw_graph["grok"], raw_phase["grok"] = graph_claude, phase_claude
    for seat in SEATS:
        if seat != "grok":
            same(raw_graph[seat], old["graph"]["raw_reviews"][seat], "cached graph seat " + seat)
            same(raw_phase[seat], old["phase"]["raw_reviews"][seat], "cached Phase seat " + seat)
    reviews, formats = normalize_five(raw_graph, old["graph"]["pool"], 3)
    means, diagnostics = panel.aggregate(reviews, old["graph"]["pool"], image_count=3)
    selection_error = None
    try:
        graph = panel.select(initial["h0"], old["graph"]["pool"], means, threshold=4)
    except (ApiSchemaError, ValueError, TypeError, KeyError) as exc:
        graph = deepcopy(initial["h0"])
        selection_error = {"status": "SELECTION_FAILED", "exception_type": type(exc).__name__}
    graph_old_phase = deepcopy(graph)
    graph_old_phase["phase"] = deepcopy(initial["original_final"]["phase"])
    phase_only, phase_decision = apply_phase_choices(initial["original_graph"], raw_phase, 3)
    both, both_decision = apply_phase_choices(graph, raw_phase, 3)
    for task in TASKS[:4]:
        same(phase_only[task], initial["original_graph"][task], "Phase-only invariant")
        same(both[task], graph[task], "combined four-head invariant")
    return {"raw_claude": {"graph": graph_claude, "phase": phase_claude},
            "logical_slot_mapping": {"grok": {"actual_reviewer": "claude", "model": MODEL}},
            "selection_error": selection_error,
            "reviews": reviews, "format_diagnostics": formats, "means": means, "diagnostics": diagnostics,
            "phase_decision": phase_decision, "both_decision": both_decision,
            "predictions": {"swap_graph_only": graph, "swap_graph_original_phase": graph_old_phase,
                            "swap_phase_only": phase_only, "both_swap": both}}


def verify_source(source):
    done = read(source / "completion.json")
    if done.get("fatal_error") is not None:
        raise ValueError("source inference must have closed without fatal failure")
    for name in ("plan", "initial_state", "predictions", "budget"):
        same(sha(source / f"{name}.json"), done[f"{name}_sha256"], "closed source " + name)
    for name, value in done["inference_artifact_sha256"].items():
        same(sha(source / name), value, "closed source evidence")
    plan = read(source / "plan.json")
    for name, value in plan["source_sha256"].items():
        same(sha(ROOT / name), value, "original runtime source " + name)
    return plan


def build_wires(initial, selected, phase_selected, adapter, source):
    for item in [*selected["images"], *phase_selected["images"]]:
        same(sha(item["path"]), item["sha256"], "original image bytes")
    base = build_gemini_base(adapter, selected)
    same(canonical_request_metadata(base).to_mapping(), selected["request_metadata"], "H0 image protocol")
    old = initial["source_record"]
    # JSON archive sorting changes nested dict insertion order inside prompt
    # strings. Rebuild the exact original pool before serializing its packet.
    original_pool = make_pool(initial["h0"], old["graph"]["proposal"], make_pool(initial["h0"]))
    same(original_pool, old["graph"]["pool"], "unchanged original candidate pool")
    originals = {"graph_swap": review_wire("grok", base, selected, original_pool),
                 "phase_swap": phase_wire("grok", phase_selected)}
    for stage, body in originals.items():
        branch = "graph" if stage == "graph_swap" else "phase"
        folders = list((source / "branches" / branch / "calls").glob(f"*_{selected['key']}_*_grok"))
        if len(folders) != 1:
            raise ValueError("exactly one archived Grok wire required")
        same(redact_images(body), read(folders[0] / "request.json"), "exact original Grok request")
    return {stage: claude_wire(body) for stage, body in originals.items()}


def verify_plan(output):
    plan = read(output / "plan.json")
    for name, value in (("profile", PROFILE), ("model", MODEL), ("provider", PROVIDER), ("limits", LIMITS),
                        ("rates", RATES), ("max_calls", 16), ("automatic_retries", 0), ("stages", STAGES)):
        same(plan[name], value, "frozen " + name)
    same([r["key"] for r in plan["selection"]], IDENTITIES, "fixed eight Training identities")
    same(plan["stage_max_tokens"], STAGE_MAX_TOKENS, "unchanged stage output limits")
    for name, value in plan["source_sha256"].items():
        same(sha(ROOT / name), value, "frozen execution code " + name)
    for name, value in plan["input_sha256"].items():
        same(sha(output / name), value, "immutable experiment input " + name)
    for name, value in plan["archive_sha256"].items():
        same(sha(Path(plan["source_root"]) / name), value, "original archive " + name)
    return plan


def prepare(output, adapter):
    if output.exists():
        raise ValueError("new single-use directory required")
    old_plan = verify_source(SOURCE)
    rows = read(SOURCE / "predictions.json")["targets"]
    same([r["key"] for r in rows], IDENTITIES, "eight unchanged targets")
    metadata = read(METADATA)
    endpoint = [e for e in metadata["data"]["endpoints"] if e["provider_name"] == PROVIDER and e["tag"] == "anthropic"]
    if (len(endpoint) != 1 or endpoint[0]["status"] != 0 or "image" not in metadata["data"]["architecture"]["input_modalities"]
            or not {"reasoning", "temperature", "response_format", "structured_outputs"} <= set(endpoint[0]["supported_parameters"])):
        raise ValueError("Anthropic image/JSON/disabled-reasoning metadata not supported")
    same((endpoint[0]["pricing"]["prompt"], endpoint[0]["pricing"]["completion"]), RATES["claude"], "frozen Claude rates")
    initials, wires, fingerprints, reservations, adaptations = [], {}, {}, {}, {}
    for row, selected in zip(rows, old_plan["selection"], strict=True):
        same(selected["source_split"], "Training", "Training-only selection")
        initial = {k: deepcopy(row[k]) for k in ("key", "video_id", "frame_id", "h0")}
        initial.update(original_graph=row["graph_only"], original_final=row["final"],
                       source_record=read(SOURCE / "targets" / row["key"] / "pipeline.json"))
        same(initial["source_record"]["final"], initial["original_final"], "original final")
        old = initial["source_record"]
        rebuilt = mix(initial, old["graph"]["raw_reviews"]["grok"], old["phase"]["raw_reviews"]["grok"])
        same(rebuilt["predictions"]["swap_graph_only"], initial["original_graph"], "cached graph replay")
        same(rebuilt["predictions"]["both_swap"], initial["original_final"], "cached Phase merge replay")
        bodies = build_wires(initial, selected, old_plan["phase_inputs"][row["key"]], adapter, SOURCE)
        wires[row["key"]] = {stage: redact_images(body) for stage, body in bodies.items()}
        fingerprints[row["key"]] = {stage: fingerprint(body) for stage, body in bodies.items()}
        reservations[row["key"]] = {stage: str(envelope("claude", body, RATES)) for stage, body in bodies.items()}
        adaptations[row["key"]] = {}
        for stage, body in bodies.items():
            packet = json.loads(body["messages"][0]["content"][0]["text"])
            adapted, dropped = transport_schema(packet["response_schema"])
            same(adapted, body["response_format"]["json_schema"]["schema"], "transport-only schema adaptation")
            adaptations[row["key"]][stage] = {"dropped_constraints": dropped,
                "semantic_packet_schema_unchanged": True, "local_validation_unchanged": True}
        initials.append(initial)
    maximum = sum(Decimal(v) for row in reservations.values() for v in row.values())
    if maximum > Decimal(LIMITS["openrouter_usd"]):
        raise ValueError("all-request conservative reservation exceeds announced budget")
    save(output / "source_inputs.json", {"targets": initials})
    save(output / "wire_preflight.json", wires)
    save(output / "metadata.json", metadata)
    dependencies = sorted({*sources(), Path(__file__).resolve(), ROOT / "scripts/run_phase_extension_trial.py",
                           ROOT / "scripts/run_parallel_phase_trial.py", ROOT / "scripts/score_graph_review_trial.py"})
    archives = [SOURCE / f"{name}.json" for name in ("plan", "initial_state", "predictions", "budget", "completion")]
    archives += list((SOURCE / "targets").rglob("*.json"))
    archives += list((SOURCE / "branches").rglob("*.json"))
    plan = {"profile": PROFILE, "created_utc": now(), "model": MODEL, "provider": PROVIDER,
            "source_root": str(SOURCE), "selection": old_plan["selection"], "phase_inputs": old_plan["phase_inputs"],
            "limits": LIMITS, "rates": RATES, "max_calls": 16, "automatic_retries": 0, "stages": STAGES,
            "stage_max_tokens": STAGE_MAX_TOKENS,
            "logical_slot_mapping": {"grok": {"actual_reviewer": "claude", "model": MODEL}},
            "frozen_other_reviewers": [s for s in SEATS if s != "grok"], "wire_fingerprints": fingerprints,
            "transport_schema_adaptations": adaptations,
            "schema_adaptation_source": "https://platform.claude.com/docs/en/build-with-claude/structured-outputs",
            "conservative_request_reservations_usd": reservations, "all_requests_reservation_usd": str(maximum),
            "input_sha256": {name: sha(output / name) for name in ("source_inputs.json", "wire_preflight.json", "metadata.json")},
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
            "archive_sha256": {p.relative_to(SOURCE).as_posix(): sha(p) for p in sorted(set(archives))},
            "policy": "Only Grok is replaced by Claude. H0, proposal, pool and four other raw reviewer responses are immutable. Same semantic packet, images and original output limits (graph 8192 / Phase 4096). Unsupported numeric/string/array constraints are removed only from transport response_format schema; packet schema and local strict validation remain intact. Claude uses Anthropic-only OpenRouter, no thinking, temperature 0. Graph keeps original five-valid mean>=4 admission; Phase keeps five-valid >=3 single-label votes. Existing other-family failed responses remain invalid. No retries or fallback routes.",
            "limitations": ["Previously inspected eight Training targets; not independent generalization evidence.",
                "Cache-controlled replacement tests this one reviewer slot; it does not regenerate all five reviewers concurrently.",
                "Missing replacement response is invalid evidence, not a copied Grok vote or a reduced quorum.",
                "Only post-H0 Claude costs and latency are newly measured."]}
    save(output / "plan.json", plan)
    for path in dependencies:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    verify_plan(output)
    print(json.dumps({"prepared": str(output), "targets": 8, "requests": 16, "model": MODEL,
                      "provider": PROVIDER, "budget_usd": "2", "all_requests_reservation_usd": str(maximum),
                      "paid_calls": 0, "packet_and_images_match_original_grok": True}), flush=True)


def execute(output, adapter):
    plan = verify_plan(output)
    if (output / "execution.lock").exists():
        raise ValueError("single-use execution; no paid replay")
    initials = read(output / "source_inputs.json")["targets"]
    with (output / "execution.lock").open("x") as handle:
        handle.write(sha(output / "plan.json"))
    calls = ClaudeCalls(output)
    calls.persist()
    rows, fatal, started = [], None, perf_counter()
    try:
        for initial, selected in zip(initials, plan["selection"], strict=True):
            key = initial["key"]
            bodies = build_wires(initial, selected, plan["phase_inputs"][key], adapter, Path(plan["source_root"]))
            same({s: fingerprint(b) for s, b in bodies.items()}, plan["wire_fingerprints"][key], "frozen wire")
            raw = {stage: calls.call(key, stage, "claude", bodies[stage]) for stage in STAGES}
            result = mix(initial, raw["graph_swap"], raw["phase_swap"])
            save(output / "targets" / key / "result.json", result)
            rows.append({**{k: deepcopy(initial[k]) for k in ("key", "video_id", "frame_id", "h0", "original_graph", "original_final")},
                         **result["predictions"]})
            save(output / "predictions.json", {"targets": rows})
            print(json.dumps({"target": key, "calls": len(calls.rows), "occupied_usd": str(calls.occupied["openrouter_usd"])}), flush=True)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        verify_plan(output)
        files = [output / "plan.json", output / "source_inputs.json", output / "budget.json"]
        files += list((output / "calls").rglob("*.json")) + list((output / "targets").rglob("*.json"))
        if (output / "predictions.json").exists():
            files.append(output / "predictions.json")
        save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal, "targets": len(rows),
            "post_calls": len(calls.rows), "statuses": dict(Counter(r["status"] for r in calls.rows)),
            "elapsed_seconds": perf_counter() - started,
            "inference_artifact_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(files)}})


def audit(output):
    plan = verify_plan(output)
    done = read(output / "completion.json")
    if done["fatal_error"] is not None or done["targets"] != 8:
        raise ValueError("complete closed eight-target inference required before GT scoring")
    for name, value in done["inference_artifact_sha256"].items():
        same(sha(output / name), value, "closed inference artifact")
    ledger = read(output / "budget.json")
    if not ledger["stopped"] or len(ledger["calls"]) > 16 or any(r["status"] == "DISPATCHED" for r in ledger["calls"]):
        raise ValueError("open or oversized paid ledger")
    if len({(r["target"], r["stage"]) for r in ledger["calls"]}) != len(ledger["calls"]):
        raise ValueError("undeclared repeated API request")
    initials = read(output / "source_inputs.json")["targets"]
    rows = read(output / "predictions.json")["targets"]
    for initial, row in zip(initials, rows, strict=True):
        raw = {}
        for stage in STAGES:
            matches = [r for r in ledger["calls"] if (r["target"], r["stage"]) == (initial["key"], stage)]
            raw[stage] = None
            if not matches:
                continue
            call = matches[0]
            same((call["seat"], call["model"], call["replaced_logical_slot"]), ("claude", MODEL, "grok"), "actual reviewer identity")
            folder = output / "calls" / f"{call['index']:03d}_{initial['key']}_{stage}_claude"
            same(read(folder / "request.json"), read(output / "wire_preflight.json")[initial["key"]][stage], "actual dispatched wire")
            if call["status"] == "JSON_PARSED":
                response = read(folder / "response.json")
                if response["http_status"] != 200:
                    raise ValueError("parsed non-200 call")
                raw[stage] = parse_claude_response(response["body"])[0]
        rebuilt = mix(initial, raw["graph_swap"], raw["phase_swap"])
        same(rebuilt, read(output / "targets" / initial["key"] / "result.json"), "raw Claude replay")
        for name in VERSIONS:
            expected = initial[name] if name in initial else rebuilt["predictions"][name]
            same(row[name], expected, "saved arm " + name)
            labels(row[name])
    return plan, rows, ledger, done


def score(output, adapter):
    _, rows, ledger, done = audit(output)  # All immutable checks precede GT access.
    _, truth = score_saved(adapter, [{"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": r["h0"],
                                      "h1": None, "final": r["both_swap"]} for r in rows])
    truths = {(t["video_id"], t["frame_id"]): t for t in truth}
    metrics = {version: compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]],
                "h0": r["h0"], "h1": None, "final": r[version]} for r in rows])["arms"]["final"] for version in VERSIONS}
    comparisons, details = {}, {}
    for before, after in (("original_graph", "swap_graph_only"), ("original_final", "swap_graph_original_phase"),
                          ("original_final", "swap_phase_only"), ("original_final", "both_swap"), ("h0", "both_swap")):
        name = before + "_to_" + after
        details[name] = [{"key": r["key"], **frame_delta(r[before], r[after],
                          truths[r["video_id"], r["frame_id"]]["gt"],
                          truths[r["video_id"], r["frame_id"]]["mask"])} for r in rows]
        comparisons[name] = summarize_deltas(details[name])
    save(output / "scored_truth.json", truth)
    save(output / "frame_deltas.json", details)
    native = sum(Decimal(r["charge"]) for r in ledger["calls"] if r["charge_kind"] == "native")
    unknown = sum(Decimal(r["charge"]) for r in ledger["calls"] if r["charge_kind"] == "unknown_reserved")
    report = {"profile": PROFILE, "targets": 8, "metrics": metrics, "comparisons": comparisons,
              "costs": {"native_usd": str(native), "unknown_reserved_usd": str(unknown),
                        "ledger_accounted_usd": ledger["occupied"]["openrouter_usd"],
                        "note": "Unknown reserves are upper-bound accounting, not reported provider charges."},
              "reasoning_usage": [{"target": r["target"], "stage": r["stage"],
                                    "status": r.get("reasoning_usage_status", "not_reported"),
                                    "tokens": r.get("reasoning_tokens")} for r in ledger["calls"]],
              "post_calls": len(ledger["calls"]),
              "statuses": done["statuses"], "elapsed_seconds": done["elapsed_seconds"],
              "audit": {"gt_loaded_after_closed_inference": True, "raw_outputs_replayed": True,
                        "h0_pool_and_other_four_reviewers_cached": True}}
    save(output / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "audit", "score"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    if args.command == "audit":
        _, rows, _, _ = audit(args.output)
        print(json.dumps({"verified": True, "targets": len(rows), "paid_calls": 0}))
        return
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, adapter)
    elif args.command == "execute":
        execute(args.output, adapter)
    else:
        score(args.output, adapter)


if __name__ == "__main__":
    main()
