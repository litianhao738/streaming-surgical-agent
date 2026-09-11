"""Eight frozen Training targets, fresh paired four-head and Phase review.

Prepare freezes requests without GT; execute is single-use and bounded; score
replays closed responses before accessing GT. No retries or model substitution.
"""
import argparse
import base64
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_graph_review_trial import TimedCalls
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_new_training_verb_guard_trial import InferenceOnlyAdapter
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_prior_feedback_continuation import metadata_preflight
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2
from scripts.run_repair_revision_trial import frozen_sources, normalize_five
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.compact_prompt import (
    PROFILE,
    compact_review_wire,
)
from surgical_agent.research.verification.phase_extension import (
    apply_phase_choices,
    phase_choice_error,
)

SOURCE = ROOT / "artifacts/preflight/parallel_phase_eight_20260909_v1"
TOKEN_AUDIT = ROOT / "artifacts/preflight/verifier_compact_prompt_20260909_v2/audit.json"
PREVIOUS = ROOT / "artifacts/preflight/compact_verifier_paired_20260909_v1"
ARMS = ("control", "compact")
BRANCHES = ("graph", "phase")
LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "2"}
MAX_CALLS = 160
RULE = "All five head F1 nondecreasing, at least one strictly improves; valid graph items and Phase responses not fewer; every paired provider input-token count present and nonincreasing; all 160 requests dispatched. Development screen only, no default promotion."


def same(a, b):
    if a != b:
        raise ValueError("frozen data mismatch")


def hydrate(body, image_bytes):
    out = deepcopy(body)
    blocks = out["messages"][0]["content"][1:]
    if len(blocks) != len(image_bytes):
        raise ValueError("image count mismatch")
    for block, data in zip(blocks, image_bytes, strict=True):
        info = block["image_url"]
        info.pop("data_url_sha256")
        info["url"] = "data:image/png;base64," + base64.b64encode(data).decode()
    same(redact_images(out), body)
    return out


def bodies(output, selected, adapter):
    base = build_gemini_base(InferenceOnlyAdapter(adapter), selected)
    data = {"graph": [im.content for im in base.images],
            "phase": [Path(im["path"]).read_bytes() for im in selected["images"]]}
    return {arm: {branch: {seat: hydrate(read(output / "requests" / selected["key"] /
                f"{arm}_{branch}_{seat}.json"), data[branch]) for seat in SEATS}
            for branch in BRANCHES} for arm in ARMS}


def verify(output):
    plan = read(output / "plan.json")
    same((plan["profile"], plan["limits"], plan["max_calls"], plan["success_rule"]),
         (PROFILE, LIMITS, MAX_CALLS, RULE))
    for name, digest in plan["sources"].items():
        same(sha(ROOT / name), digest)
    for name, digest in plan["inputs"].items():
        same(sha(output / name), digest)
    for s in plan["selection"]:
        for im in s["images"]:
            same(sha(im["path"]), im["sha256"])
    return plan


def prepare(output, adapter):
    if output.exists():
        raise ValueError("fresh directory required")
    old, done = read(SOURCE / "plan.json"), read(SOURCE / "completion.json")
    if done["fatal_error"] or not read(SOURCE / "budget.json")["stopped"]:
        raise ValueError("source must be closed without fatal error")
    for name in ("plan", "initial_state", "predictions", "budget"):
        same(sha(SOURCE / f"{name}.json"), done[f"{name}_sha256"])
    for name, digest in done["inference_artifact_sha256"].items():
        same(sha(SOURCE / name), digest)
    selection = old["selection"]
    same(len(selection), 8)
    same(Counter(s["video_id"] for s in selection), Counter({v: 2 for v in ("VID103", "VID23", "VID31", "VID96")}))
    audit = {r["source"].replace("\\", "/"): r for r in read(TOKEN_AUDIT)["rows"]}
    initials = {}
    for s, initial in zip(selection, read(SOURCE / "initial_state.json")["targets"], strict=True):
        same(s["key"], initial["key"])
        if adapter.entries[s["video_id"]].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        same([(im["path"], im["sha256"]) for im in s["images"]],
             [(im["path"], im["sha256"]) for im in old["phase_inputs"][s["key"]]["images"]])
        record = read(SOURCE / "targets" / s["key"] / "pipeline.json")
        initials[s["key"]] = {"h0": initial["h0"], "pool": record["graph"]["pool"]}
        if not record["graph"]["pool"]["propositions"]:
            raise ValueError("this fixed cohort requires nonempty pools")
        for branch in BRANCHES:
            for seat in SEATS:
                matches = list((SOURCE / "branches" / branch / "calls").glob(f"*_{s['key']}_{branch}_review_{seat}/request.json"))
                same(len(matches), 1)
                path = matches[0]
                body = read(path)
                same(body["model"], MODELS[seat])
                packet = json.loads(body["messages"][0]["content"][0]["text"])
                if branch == "graph":
                    same(packet["propositions"], record["graph"]["pool"]["propositions"])
                revised = compact_review_wire(body, branch, count_tokens=len)
                import hashlib
                row = audit[path.relative_to(SOURCE).as_posix()]
                for arm, request, field in (("control", body, "before"), ("compact", revised, "after")):
                    text = request["messages"][0]["content"][0]["text"]
                    same(hashlib.sha256(text.encode()).hexdigest(), row[field + "_text_sha256"])
                    save(output / "requests" / s["key"] / f"{arm}_{branch}_{seat}.json", request)
                if any(c["after"] > c["before"] for c in row["text_tokens_proxy"].values()):
                    raise ValueError("proxy token budget exceeded")
    save(output / "initial_state.json", initials)
    for s in selection:
        bodies(output, s, adapter)  # Hash-bound real image reconstruction, no network.
    save(output / "metadata.json", metadata_preflight())
    deps = {*frozen_sources(), Path(__file__).resolve(), ROOT / "scripts/run_graph_review_trial.py",
            ROOT / "scripts/run_new_training_verb_guard_trial.py", ROOT / "scripts/run_prior_feedback_continuation.py",
            ROOT / "scripts/score_five_head_repair_trial.py", ROOT / "scripts/score_prior_feedback_continuation.py",
            ROOT / "scripts/score_graph_review_trial.py"}
    deps.update((ROOT / "src/surgical_agent/research/verification/prompts").glob("compact_*.txt"))
    inputs = [p for p in output.rglob("*.json")]
    save(output / "plan.json", {"profile": PROFILE, "created_utc": now(), "selection": selection,
         "arms": ARMS, "models": MODELS, "rates": RATES_V2, "providers": PROVIDERS,
         "limits": LIMITS, "max_calls": MAX_CALLS, "success_rule": RULE,
         "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(deps)},
         "inputs": {p.relative_to(output).as_posix(): sha(p) for p in inputs},
         "source_completion_sha256": sha(SOURCE / "completion.json"), "token_audit_sha256": sha(TOKEN_AUDIT),
         "previous_budget": read(PREVIOUS / "budget.json")["occupied"],
         "previous_budget_sha256": sha(PREVIOUS / "budget.json"),
         "scope": "Eight previously inspected Training targets; fixed H0/pools; fresh 80 requests per arm; alternate arm order; graph and Phase parallel, ten seats max; no retries, H0, proposal, GT during inference, Tracker or Gate."})
    verify(output)
    print(json.dumps({"prepared": True, "requests": MAX_CALLS, "limits": LIMITS}), flush=True)


class BoundCalls(TimedCalls):
    def __init__(self, output):
        previous = read(PREVIOUS / "budget.json")
        if not previous["stopped"]:
            raise ValueError("previous trial must be stopped")
        same(sha(PREVIOUS / "budget.json"), read(output / "plan.json")["previous_budget_sha256"])
        super().__init__(output, previous_budget=previous["occupied"],
                         limits={k: Decimal(v) for k, v in LIMITS.items()}, rates=RATES_V2,
                         providers=PROVIDERS, max_calls=MAX_CALLS, reasoning_seats=("grok", "gemini"))
        self.attempted = set()

    def call(self, target, stage, seat, body):
        same(redact_images(body), read(self.output / "requests" / target / f"{stage}_{seat}.json"))
        with self.lock:
            ident = (target, stage, seat)
            if ident in self.attempted:
                raise ValueError("retry forbidden")
            self.attempted.add(ident)
        return super().call(target, stage, seat, body)


def run_arm(calls, key, initial, wire, arm):
    start = perf_counter()
    jobs = [(b, s) for b in BRANCHES for s in SEATS]
    with ThreadPoolExecutor(max_workers=10) as workers:
        values = list(workers.map(lambda x: calls.call(key, arm + "_" + x[0], x[1], wire[x[0]][x[1]]), jobs))
    raw = {b: {s: v for (branch, s), v in zip(jobs, values, strict=True) if branch == b} for b in BRANCHES}
    count = len(wire["graph"][SEATS[0]]["messages"][0]["content"]) - 1
    pool = initial["pool"]
    normalized, formatting = normalize_five(raw["graph"], pool, count)
    means, diagnostics = panel.aggregate(normalized, pool, image_count=count)
    graph = panel.select(initial["h0"], pool, means, threshold=4)
    final, decision = apply_phase_choices(graph, raw["phase"], count)
    return {"graph": graph, "final": final, "raw": raw, "formatting": formatting,
            "means": means, "diagnostics": diagnostics, "phase_decision": decision,
            "phase_errors": {s: phase_choice_error(raw["phase"][s], count) for s in SEATS},
            "seconds": perf_counter() - start}


def execute(output, adapter):
    plan = verify(output)
    with (output / "execution.lock").open("x") as handle:
        handle.write(sha(output / "plan.json"))
    initials = read(output / "initial_state.json")
    calls = BoundCalls(output)
    calls.persist()
    rows = [{"key": s["key"], "video_id": s["video_id"], "frame_id": s["frame_id"],
             "h0": initials[s["key"]]["h0"], **{a: deepcopy(initials[s["key"]]["h0"]) for a in ARMS}} for s in plan["selection"]]
    start, fatal = perf_counter(), None
    try:
        for index, (s, row) in enumerate(zip(plan["selection"], rows, strict=True)):
            wire = bodies(output, s, adapter)
            for arm in (ARMS if index % 2 == 0 else ARMS[::-1]):
                if calls.stopped:
                    continue
                result = run_arm(calls, s["key"], initials[s["key"]], wire[arm], arm)
                save(output / "targets" / s["key"] / f"{arm}.json", result)
                row[arm] = result["final"]
                save(output / "predictions.json", rows)
                print(json.dumps({"target": s["key"], "arm": arm, "calls": len(calls.rows),
                                  "seconds": round(result["seconds"], 2), "occupied": {k: str(v) for k, v in calls.occupied.items()}}), flush=True)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", rows)
        try:
            verify(output)
        except Exception:
            fatal = fatal or "FROZEN_SOURCE_CHANGED"
            raise
        finally:
            files = [p for p in output.rglob("*.json") if "completion" not in p.name]
            save(output / "completion.json", {"fatal_error": fatal, "closed_utc": now(),
                 "seconds": perf_counter() - start, "post_calls": len(calls.rows),
                 "hashes": {p.relative_to(output).as_posix(): sha(p) for p in files}})


def score(output, adapter):
    plan = verify(output)
    done, ledger = read(output / "completion.json"), read(output / "budget.json")
    if done["fatal_error"] or not ledger["stopped"]:
        raise ValueError("closed nonfatal run required")
    for name, digest in done["hashes"].items():
        same(sha(output / name), digest)
    calls = ledger["calls"]
    same(len(calls), done["post_calls"])
    if len(calls) > MAX_CALLS or len({(c['target'], c['stage'], c['seat']) for c in calls}) != len(calls):
        raise ValueError("duplicate or excess requests")
    for account, limit in LIMITS.items():
        total = sum(Decimal(c["charge"]) for c in calls if c["account"] == account)
        same(total + Decimal(ledger["carried_occupied"][account]), Decimal(ledger["occupied"][account]))
        if Decimal(ledger["occupied"][account]) > Decimal(limit):
            raise ValueError("budget exceeded")
    replay = ReplayCalls(output, calls)
    initials, rows = read(output / "initial_state.json"), read(output / "predictions.json")
    records = {}
    for s, row in zip(plan["selection"], rows, strict=True):
        same((row["key"], row["video_id"], row["frame_id"], row["h0"]),
             (s["key"], s["video_id"], s["frame_id"], initials[s["key"]]["h0"]))
        wire = bodies(output, s, adapter)
        for arm in ARMS:
            path = output / "targets" / s["key"] / f"{arm}.json"
            if not path.exists():
                same(row[arm], row["h0"])
                continue
            result = run_arm(replay, s["key"], initials[s["key"]], wire[arm], arm)
            stored = read(path)
            same({k: v for k, v in result.items() if k != "seconds"}, {k: v for k, v in stored.items() if k != "seconds"})
            same(row[arm], result["final"])
            records[s["key"], arm] = stored
    same(len(replay.rows), len(calls))
    # First access to GT label values occurs only after closure and raw replay.
    _, truth = score_saved(adapter, [{**r, "h1": None, "final": r["compact"]} for r in rows])
    truths = {(t["video_id"], t["frame_id"]): t for t in truth}
    metrics = {a: compute_repair_comparison([{**truths[r["video_id"], r["frame_id"]],
               "h0": r["h0"], "h1": None, "final": r[a]} for r in rows])["arms"]["final"] for a in ("h0", *ARMS)}
    deltas = [{"key": r["key"], **frame_delta(r["control"], r["compact"], truths[r["video_id"], r["frame_id"]]["gt"],
               truths[r["video_id"], r["frame_id"]]["mask"])} for r in rows]
    usage_pairs = []
    for s in plan["selection"]:
        for branch in BRANCHES:
            for seat in SEATS:
                pair = {a: next((c for c in calls if (c["target"], c["stage"], c["seat"]) ==
                                  (s["key"], a + "_" + branch, seat)), {}) for a in ARMS}
                tokens = {a: pair[a].get("usage", {}).get("prompt_tokens") for a in ARMS}
                usage_pairs.append({"target": s["key"], "branch": branch, "seat": seat, "prompt_tokens": tokens,
                     "nonincreasing": tokens["compact"] <= tokens["control"] if all(type(v) is int for v in tokens.values()) else None})
    totals = {}
    for arm in ARMS:
        own = [c for c in calls if c["stage"].startswith(arm + "_")]
        totals[arm] = {"calls": len(own), "statuses": dict(Counter(c["status"] for c in own)),
            "costs": {acc: {kind: str(sum(Decimal(c["charge"]) for c in own if c["account"] == acc and c["charge_kind"] == kind))
                            for kind in ("native", "conservative_estimate", "unknown_reserved")} for acc in LIMITS},
            "panel_seconds": sum(v["seconds"] for (key, a), v in records.items() if a == arm),
            "reported_prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) or 0 for c in own),
            "reported_completion_tokens": sum(c.get("usage", {}).get("completion_tokens", 0) or 0 for c in own),
            "usage_missing_calls": sum("prompt_tokens" not in c.get("usage", {}) for c in own),
            "valid_graph_items": sum(sum(x is not None for x in v["means"].values()) for (key, a), v in records.items() if a == arm),
            "valid_phase_responses": sum(sum(e is None for e in v["phase_errors"].values()) for (key, a), v in records.items() if a == arm)}
    before, after = (metrics[a]["tasks"] for a in ARMS)
    checks = {"all_calls": len(calls) == MAX_CALLS,
              "all_head_f1_nondecreasing": all(after[t]["micro_f1"] >= before[t]["micro_f1"] for t in before),
              "some_f1_improves": any(after[t]["micro_f1"] > before[t]["micro_f1"] for t in before),
              "all_input_pairs_nonincreasing": all(p["nonincreasing"] is True for p in usage_pairs),
              **{k: totals["compact"][k] >= totals["control"][k] for k in ("valid_graph_items", "valid_phase_responses")}}
    report = {"success": all(checks.values()), "success_checks": checks, "metrics": metrics, "totals": totals,
              "input_token_pairs": usage_pairs, "comparisons": summarize_deltas(deltas), "raw_replay_passed": True,
              "elapsed_seconds": done["seconds"], "limitations": plan["scope"] + " Previously inspected Training development sample, not test-set validation."}
    save(output / "metrics.json", report)
    save(output / "scored_truth.json", truth)
    save(output / "frame_deltas.json", deltas)
    print(json.dumps({"success": report["success"], "checks": checks, "totals": totals}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    globals()[args.command](args.output, adapter)
