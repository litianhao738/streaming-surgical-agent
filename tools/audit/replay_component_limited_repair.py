"""Offline exact-request replay of distributed repair with limited component additions.

No transport or credentials are used. Replay never reads GT. The separate score
command consumes only previously saved complete replay predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import redact_images
from scripts.panel_proposal_wire import (
    review_and_propose_wire,
    split_review_and_proposals,
)
from scripts.run_candidate_panel_trial import sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_prior_panel_trial import read, save
from scripts.run_repair_revision_trial import (
    ARMS,
    PROVIDERS,
    merge_suggestions,
    normalize_five,
    verify_plan,
)
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import COMPONENTS, labels
from surgical_agent.research.verification.review_normalization import parse_review_json

ARM = ARMS[1]
PROFILE = "component_limited_exact_request_replay_v1"


class ReplayUnavailable(ValueError):
    """A counterfactual request has no verified matching saved response."""


def saved_json_equal(actual, saved):
    """Compare the persisted JSON representation, including integer object keys.

    JSON object keys are strings. Proposal counts use integer IVT keys while
    in memory, so compare after that serialization conversion. Array order and
    numeric/boolean values are not coerced or ignored.
    """
    normalized = json.loads(json.dumps(actual, allow_nan=False))
    return json.dumps(normalized, sort_keys=True, allow_nan=False) == json.dumps(
        saved, sort_keys=True, allow_nan=False)


def component_limited_select(current, pool, means):
    """Only newly accepted IVTs may justify newly added independent components."""
    current = labels(current)
    after = panel.select(current, pool, means)
    new_ivts = set(after["ivt"]) - set(current["ivt"])
    for task in ("instrument", "verb", "target"):
        after[task] = [value for value in after[task] if value in current[task]
                       or any(COMPONENTS[ivt][task] == value for ivt in new_ivts)]
    return labels(after)


def replay_policy(h0, get_round, *, limited, round_cap=3):
    """Run deterministic control from H0; get_round must verify exact requests.

    The callback receives only round number and reconstructed pool. It must
    raise ReplayUnavailable on missing or changed requests; missing future
    rounds are never fabricated from a previous response.
    """
    current, pool = labels(h0), make_pool(h0)
    history = []
    for number in range(1, round_cap + 1):
        try:
            payload = get_round(number, deepcopy(pool))
        except ReplayUnavailable as exc:
            return {"complete": False, "status": "CACHE_INCOMPLETE", "reason": str(exc),
                    "final": None, "last_valid_prediction": current, "history": history,
                    "review_calls_reused": 5 * len(history)}
        parts, suggestions, proposal_diagnostics = {}, {}, {}
        for seat in SEATS:
            parts[seat], suggestions[seat], proposal_diagnostics[seat] = split_review_and_proposals(
                payload["raw_reviews"][seat], seat, pool)
        reviews, format_diagnostics = normalize_five(parts, pool, payload["image_count"])
        before = deepcopy(current)
        if pool["propositions"]:
            means, diagnostics = panel.aggregate(reviews, pool, image_count=payload["image_count"])
            try:
                after = component_limited_select(current, pool, means) if limited else panel.select(current, pool, means)
                issues = panel.unresolved(after, pool, means, diagnostics)
            except (ApiSchemaError, TypeError, ValueError, KeyError) as exc:
                return {"complete": False, "status": "SELECTION_FAILED", "reason": type(exc).__name__,
                        "final": None, "last_valid_prediction": current, "history": history,
                        "review_calls_reused": 5 * (len(history) + 1)}
        else:
            means, diagnostics, after, issues = {}, {}, deepcopy(current), []
        next_pool, pending = merge_suggestions(after, pool, suggestions)
        proposal_incomplete = any(d.get("proposal_status") != "VALID" for d in proposal_diagnostics.values())
        unresolved = bool(issues or pending["queued_ivt"] or pending.get("dropped")
                          or not pool["propositions"] or proposal_incomplete)
        status = "UNRESOLVED" if unresolved else "MODEL_PASS"
        history.append({"round": number, "before": before, "pool": pool, "after": after,
                        "next_pool": next_pool, "means": means, "diagnostics": diagnostics,
                        "issues": issues, "pending": pending, "status": status,
                        "proposal_diagnostics": proposal_diagnostics,
                        "format_diagnostics": format_diagnostics, "source_calls": payload.get("source_calls", [])})
        current, pool = after, next_pool
        if not unresolved:
            break
    return {"complete": True, "status": status, "final": current, "history": history,
            "review_calls_reused": 5 * len(history), "stopped_before_round_cap": len(history) < round_cap}


class VerifiedResponses:
    """Resolve successful cache hits and original failed calls without new requests."""

    def __init__(self, source, budget, hits):
        self.source, self.rows, self.hits = source, budget["calls"], hits
        self.files = {}

    def load(self, path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ReplayUnavailable("SOURCE_HASH_CHANGED:" + str(path))
        value = read(path)
        if sha(path) != digest:
            raise ReplayUnavailable("SOURCE_CHANGED_DURING_READ:" + str(path))
        self.files[str(path)] = digest
        return value

    def get(self, target, stage, seat, request):
        key = target, stage, seat
        own = [r for r in self.rows if (r["target"], r["stage"], r["seat"]) == key]
        hits = [r for r in self.hits if (r["target"], r["stage"], r["seat"]) == key]
        if len(own) + len(hits) != 1:
            raise ReplayUnavailable("MISSING_OR_AMBIGUOUS_CALL:" + ":".join(key))
        if hits:
            hit = hits[0]
            folder = Path(hit["source"])
            saved_request = self.load(folder / "request.json", hit["request_sha256"])
            response = self.load(folder / "response.json", hit["response_sha256"])
            record = self.load(folder / "record.json")
        else:
            row = own[0]
            folder = self.source / "calls" / f"{row['index']:03d}_{target}_{stage}_{seat}"
            saved_request = self.load(folder / "request.json")
            response = self.load(folder / "response.json")
            record = self.load(folder / "record.json")
            if record != row:
                raise ReplayUnavailable("RECORD_LEDGER_MISMATCH:" + str(folder))
        if saved_request != redact_images(request):
            raise ReplayUnavailable("EXACT_REQUEST_MISMATCH:" + ":".join(key))
        parsed = None
        if record["status"] in ("JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"):
            raw = response["body"]
            expected_provider = PROVIDERS.get(request["model"], PROVIDERS.get(seat))
            try:
                choice = raw["choices"][0]
                if (response["http_status"] != 200 or raw.get("error")
                        or raw.get("model") != request["model"]
                        or (expected_provider is not None and raw.get("provider") != expected_provider)
                        or choice["finish_reason"] != "stop" or choice["message"].get("refusal")):
                    raise ReplayUnavailable("INVALID_CACHED_TRANSPORT:" + str(folder))
                if seat == "base":
                    parsed = json.loads(choice["message"]["content"])
                else:
                    parsed, diagnostic = parse_review_json(choice["message"]["content"])
                    if diagnostic["error"]:
                        raise ReplayUnavailable("INVALID_CACHED_JSON:" + str(folder))
            except (KeyError, TypeError, IndexError, ValueError) as exc:
                raise ReplayUnavailable("INVALID_CACHED_RESPONSE:" + str(folder)) from exc
        elif hits:
            raise ReplayUnavailable("CACHE_HIT_POINTS_TO_FAILURE:" + str(folder))
        return parsed, {"source": str(folder), "request_sha256": sha(folder / "request.json"),
                        "response_sha256": sha(folder / "response.json"),
                        "record_sha256": sha(folder / "record.json"), "original_status": record["status"],
                        "account": record["account"], "historical_charge": record["charge"]}


def replay(source, output, adapter, *, targets=None, allow_running=False):
    if output.exists():
        raise ValueError("new output directory required")
    plan, budget, states = read(source / "plan.json"), read(source / "budget.json"), read(source / "inference_states.json")
    if not budget["stopped"] and not allow_running:
        raise ValueError("source is running; use --allow-running only for a partial operational self-check")
    verify_plan(plan)
    selection = [r for r in plan["selection"] if targets is None or r["key"] in targets]
    if not selection or targets is not None and set(targets) != {r["key"] for r in selection}:
        raise ValueError("unknown or empty target selection")
    hits = read(source / "cache_hits.json") if (source / "cache_hits.json").exists() else []
    responses = VerifiedResponses(source, budget, hits)
    results, predictions = {}, []
    for selected in selection:
        key = selected["key"]
        if adapter.entries[selected["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        state = states[key]
        if state["h0"] is None:
            results[key] = {"complete": False, "status": "NO_H0", "final": None}
            continue
        base = build_gemini_base(adapter, selected)
        raw_h0, h0_source = responses.get(key, "h0", "base", gemini_h0_wire(base))
        validate_final_only(raw_h0)
        decoded_h0 = {t: [raw_h0[t]["selected_id"]] if t == "phase" else raw_h0[t]["selected_ids"] for t in state["h0"]}
        if decoded_h0 != state["h0"]:
            raise ValueError("saved H0 differs from verified response")
        original_history = {r["round"]: r for r in state["arms"][ARM]["history"]}

        def get_round(number, pool, key=key, base=base, selected=selected, original_history=original_history):
            record = original_history.get(number)
            if record is None:
                raise ReplayUnavailable(f"ROUND_NOT_AVAILABLE:{key}:{number}")
            if record["pool"] != pool:
                raise ReplayUnavailable(f"POOL_CHANGED:{key}:{number}")
            raw_reviews, source_calls = {}, []
            for seat in SEATS:
                body = review_and_propose_wire(seat, base, selected, pool)
                raw_reviews[seat], call = responses.get(key, f"{ARM}_review_{number}", seat, body)
                source_calls.append(call)
            if raw_reviews != record["raw_reviews"]:
                raise ReplayUnavailable(f"RAW_REVIEW_CHANGED:{key}:{number}")
            return {"raw_reviews": raw_reviews, "image_count": len(base.images), "source_calls": source_calls}

        original = replay_policy(state["h0"], get_round, limited=False, round_cap=plan["round_cap"])
        original_verified = (original["complete"]
                             and original["final"] == state["arms"][ARM]["current"]
                             and len(original["history"]) == len(original_history)
                             and original["status"] == state["arms"][ARM]["status"]
                             and not state["arms"][ARM]["active"])
        for rec in original["history"]:
            expected = original_history[rec["round"]]
            for field in ("before", "pool", "after", "next_pool", "means", "diagnostics", "issues", "pending", "status"):
                if not saved_json_equal(rec[field], expected[field]):
                    raise ValueError(f"original policy does not reproduce {key}:{rec['round']}:{field}")
        if not original_verified:
            results[key] = {"complete": False, "status": "ORIGINAL_NOT_FULLY_REPRODUCIBLE", "original": original}
            continue
        limited = replay_policy(state["h0"], get_round, limited=True, round_cap=plan["round_cap"])
        results[key] = {**limited, "original_verified": True, "original_final": original["final"],
                        "original_rounds": len(original["history"]), "h0_source": h0_source,
                        "unused_original_review_calls": max(0, original["review_calls_reused"] - limited["review_calls_reused"])}
        if limited["complete"]:
            predictions.append({"video_id": selected["video_id"], "frame_id": selected["frame_id"],
                                "h0": state["h0"], "h1": original["final"], "final": limited["final"]})
    for path, digest in responses.files.items():
        if sha(path) != digest:
            raise ValueError("source call changed during replay: " + path)
    save(output / "source_snapshot.json", {"plan": plan, "budget": budget, "states": {r["key"]: states[r["key"]] for r in selection}})
    save(output / "replay.json", results)
    save(output / "predictions.json", predictions)
    manifest = {"profile": PROFILE, "source": str(source), "source_was_running": not budget["stopped"],
                "selected": len(selection), "complete": len(predictions), "targets": [r["key"] for r in selection],
                "source_target_count": len(plan["selection"]),
                "operational_subset": len(selection) != len(plan["selection"]),
                "all_complete": len(predictions) == len(selection), "new_api_calls": 0, "new_api_cost": 0,
                "complete_policy_review_calls_reused": sum(r.get("review_calls_reused", 0)
                                                          for r in results.values() if r["complete"]),
                "source_call_sha256": responses.files, "prediction_sha256": sha(output / "predictions.json"),
                "source_snapshot_sha256": sha(output / "source_snapshot.json"), "replay_sha256": sha(output / "replay.json"),
                "audit_code_sha256": sha(Path(__file__)), "gt_read_during_replay": False,
                "claim": "Deterministic policy replay with identical saved model requests/responses; not independent model sampling.",
                "budget_boundary": "Model/control replay only; counterfactual global account scheduling is not simulated.",
                "incomplete_replays_are_not_negative_gate_labels": True}
    save(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "selected": len(selection), "complete": len(predictions), "new_api_calls": 0}))


def score(output, adapter):
    manifest = read(output / "manifest.json")
    if manifest["source_was_running"]:
        raise ValueError("operational partial self-check cannot be scored as a finished experiment")
    if manifest.get("operational_subset"):
        raise ValueError("operational partial target selection cannot be scored as a full experiment")
    if sha(output / "predictions.json") != manifest["prediction_sha256"]:
        raise ValueError("replay predictions changed before scoring")
    if not manifest["all_complete"]:
        raise ValueError("incomplete counterfactual trajectories; report coverage instead of selecting a favorable subset")
    report, truth = score_saved(adapter, read(output / "predictions.json"))
    save(output / "scores/truth.json", truth)
    save(output / "scores/report.json", {"manifest_sha256": sha(output / "manifest.json"), "comparison": report})
    print(json.dumps({a: {t: m["micro_f1"] for t, m in r["tasks"].items()} for a, r in report["arms"].items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "score"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--targets", nargs="*")
    parser.add_argument("--allow-running", action="store_true")
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "replay":
        replay(args.source, args.output, adapter, targets=args.targets, allow_running=args.allow_running)
    else:
        score(args.output, adapter)
