"""Frozen-H0 development replay with five visual reviewers, at most 3 rounds.

GT is read for scoring only after all paid inference stops. The old baseline,
old protocols and old experiments are not edited. No automatic retries.
"""
import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from threading import RLock

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import (
    ACADEMIC,
    MODELS,
    body_for,
    key_for,
    redact_images,
)
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import build_base, now, read, save
from surgical_agent.api.credentials import assert_secret_absent
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.perception.main_h0 import _LABEL_BOUNDARY
from surgical_agent.perception.ontology_prompt import load_prompt_ontology_text
from surgical_agent.research.verification.candidate_coordinator import (
    SEATS,
    proposal_schema,
    run,
)

SOURCE = ROOT / "artifacts/preflight/h0_prior_panel_openrouter_20260907_v7"
BASE_MODEL = "qwen/qwen3.8-max-0902"
PROVIDERS = {"base": "Alibaba", "deepseek": "Fireworks", "gpt": "OpenAI", "gemini": "Google AI Studio"}
# Conservative token-rate envelopes, USD except qwen (CNY). Cached input is
# bounded by the highest input rate. Native billing replaces the envelope.
RATES = {"base": ("0.0000025", "0.000006"), "grok": ("0.00000125", "0.0000025"),
         "gpt": ("0.0000004", "0.0000016"), "gemini": ("0.0000001", "0.0000004"),
         "deepseek": ("0.00000044", "0.00000132"), "qwen": ("0.000001", "0.00001")}
LIMITS = {"openrouter_usd": Decimal("1.50"), "xai_usd": Decimal("0.50"), "aliyun_cny": Decimal(1)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposal_body(base, selected, state, pool, issues):
    body = body_for("qwen", base, selected, pool)
    packet = {
        "academic_context": ACADEMIC,
        "task": "Propose only additional plausible CURRENT-frame candidate label IDs for independent visual review.",
        "instructions": ("Look across all visible tools and contact regions; multiple IVTs may coexist. "
                         "Use only these causal images. You cannot accept, delete or change the final answer. "
                         "Do not repeat candidate IDs already in the pool. Do not add alternatives merely to fill a quota. "
                         "Return empty lists if no new candidate has visual support. Issues are uncertain model judgments, not ground truth. "
                         "Return exactly the four candidate lists in the JSON schema, no commentary."),
        "target_frame_id": selected["frame_id"], "causal_frame_ids": selected["causal_frame_ids"],
        "seconds_relative_to_target": [(f-selected["frame_id"])/25 for f in selected["causal_frame_ids"]],
        "current_prediction": state, "candidate_pool": pool, "issues": issues,
        "full_ontology": load_prompt_ontology_text(), "label_boundaries": _LABEL_BOUNDARY,
        "response_schema": proposal_schema(),
    }
    body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    body.update(model=BASE_MODEL, max_tokens=4096, reasoning={"effort": "low"},
                provider={"only": ["alibaba"], "order": ["alibaba"], "allow_fallbacks": False,
                          "require_parameters": True})
    body.pop("enable_thinking", None)
    wire_schema = proposal_schema()
    for field in wire_schema["properties"].values():
        field.pop("uniqueItems", None)  # Alibaba rejects this array keyword; local validation remains strict.
    body["response_format"] = {"type": "json_schema", "json_schema": {
        "name": "candidate_proposal_v1", "strict": True, "schema": wire_schema}}
    return body


def bucket(seat):
    return "aliyun_cny" if seat == "qwen" else "xai_usd" if seat == "grok" else "openrouter_usd"


def envelope(seat, body, rates=None):
    # UTF-8 bytes over-bound text tokens; a deliberately loose image allowance
    # is applied to every supplied <1MP image, including explicit reference
    # galleries. No tools/audio/search. Never reserve for just the query images.
    image_count = sum(block.get("type") == "image_url" for message in body["messages"]
                      if isinstance(message["content"], list) for block in message["content"])
    input_bound = len(json.dumps(redact_images(body), ensure_ascii=False).encode()) + image_count*16384
    inp, out = map(Decimal, (rates or RATES)[seat])
    return Decimal(input_bound)*inp + Decimal(body["max_tokens"])*out


def usage_charge(seat, usage, reserve, rates=None):
    if seat == "grok" and usage.get("cost_in_usd_ticks") is not None:
        return Decimal(str(usage["cost_in_usd_ticks"]))/Decimal(10**10), "native"
    if bucket(seat) == "openrouter_usd" and usage.get("cost") is not None:
        return Decimal(str(usage["cost"])), "native"
    if seat == "qwen" and "prompt_tokens" in usage and "completion_tokens" in usage:
        inp, out = map(Decimal, (rates or RATES)[seat])
        return Decimal(usage["prompt_tokens"])*inp + Decimal(usage["completion_tokens"])*out, "conservative_estimate"
    return reserve, "unknown_reserved"


class Calls:
    def __init__(self, output, previous_budget=None, *, limits=None, rates=None,
                 providers=None, max_calls=72, reasoning_seats=()):
        self.output, self.lock, self.rows = output, RLock(), []
        self.limits = dict(limits or LIMITS)
        self.rates = dict(rates or RATES)
        self.providers = dict(PROVIDERS if providers is None else providers)
        self.max_calls = max_calls
        self.reasoning_seats = frozenset(reasoning_seats)
        self.carried = previous_budget or {k: "0" for k in self.limits}
        self.occupied = {k: Decimal(self.carried[k]) for k in self.limits}
        self.stopped = False

    def call(self, target, stage, seat, body):
        secret = key_for("gpt" if seat == "base" else seat)
        endpoint = "https://openrouter.ai/api/v1" if seat == "base" else MODELS[seat][0]
        reserve = envelope(seat, body, self.rates)
        account = bucket(seat)
        with self.lock:
            if self.stopped or len(self.rows) >= self.max_calls or self.occupied[account] + reserve > self.limits[account]:
                self.stopped = True
                return None
            self.occupied[account] += reserve
            row = {"index": len(self.rows), "target": target, "stage": stage, "seat": seat,
                   "account": account, "model": body["model"], "status": "DISPATCHED", "started_utc": now(),
                   "reserve": str(reserve), "charge": str(reserve), "charge_kind": "unknown_reserved"}
            self.rows.append(row)
            folder = self.output / "calls" / f"{row['index']:03d}_{target}_{stage}_{seat}"
            save(folder / "request.json", redact_images(body))
            save(folder / "record.json", row)
            self.persist()
        parsed = None
        try:
            r = requests.post(endpoint + "/chat/completions", json=body,
                              headers={"Authorization": "Bearer " + secret.reveal()}, timeout=(15, 120))
            try:
                raw = r.json()
            except ValueError:
                raw = {"non_json_response": r.text}
            raw = json.loads(json.dumps(raw).replace(secret.reveal(), "[REDACTED]"))
            save(folder / "response.json", {"http_status": r.status_code, "body": raw})
            usage = raw.get("usage", {}) or {}
            charge, kind = usage_charge(seat, usage, reserve, self.rates)
            row.update(http_status=r.status_code, usage=usage, charge=str(charge), charge_kind=kind)
            row["status"] = "API_FAILED"
            if r.status_code in (400, 401, 402, 404):
                with self.lock:
                    self.stopped = True  # Do not repeat a deterministic config failure across targets.
            if r.ok and not raw.get("error"):
                if raw.get("model") != body["model"]:
                    raise ValueError("model identity mismatch")
                expected_provider = self.providers.get(body["model"], self.providers.get(seat))
                if expected_provider is not None and raw.get("provider") != expected_provider:
                    raise ValueError("provider identity mismatch")
                choice = raw["choices"][0]
                if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
                    raise ValueError("incomplete or refused response")
                if seat not in {"base", *self.reasoning_seats} and (usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                                        or choice["message"].get("reasoning_content")):
                    raise ValueError("reviewer emitted reasoning despite disabled thinking")
                parsed = json.loads(choice["message"]["content"])
                row["status"] = "JSON_PARSED"  # Semantic contract checked by coordinator.
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            row.update(status="FAILED", exception_type=type(exc).__name__)
        finally:
            with self.lock:
                self.occupied[account] += Decimal(row["charge"]) - reserve
                row["finished_utc"] = now()
                save(folder / "record.json", row)
                self.persist()
            assert_secret_absent(secret, folder.glob("*.json"))
        return parsed

    def persist(self):
        save(self.output / "budget.json", {"limits": {k: str(v) for k, v in self.limits.items()},
             "carried_occupied": self.carried,
             "occupied": {k: str(v) for k, v in self.occupied.items()}, "calls": self.rows, "stopped": self.stopped})


def score(output, adapter, rows, results):
    report, truth = score_saved(adapter, rows)
    save(output / "scored_predictions.json", truth)
    save(output / "comparison.json", report)
    rounds = {}
    for index in range(3):
        values = [{**row, "final": result["snapshots"][index]} for row, result in zip(rows, results, strict=True)]
        comparison, _ = score_saved(adapter, values)
        rounds[str(index+1)] = comparison
    save(output / "round_comparisons.json", rounds)
    details = []
    for gt, result in zip(truth, results, strict=True):
        # score_saved groups by video; this frozen sample order is grouped too.
        assert (gt["video_id"], gt["frame_id"]) == (result["video_id"], result["frame_id"])
        coverage = {}
        for task in ("instrument", "verb", "target", "ivt"):
            if not gt["mask"][task]:
                continue
            expected = set(gt["gt"][task])
            pools = [h["pool"] for h in result["history"]]
            available = {p["label_id"] for pool in pools for p in pool["propositions"] if p["task"] == task}
            before, after = set(gt["h0"][task]), set(gt["final"][task])
            added, removed = after-before, before-after
            coverage[task] = {"gt_labels": len(expected), "h0_hits": len(expected & before),
                              "reviewed_pool_hits": len(expected & available), "pool_size": len(available),
                              "beneficial_label_edits": len(added & expected) + len(removed-expected),
                              "harmful_label_edits": len(added-expected) + len(removed & expected)}
        details.append({"video_id": gt["video_id"], "frame_id": gt["frame_id"], "tasks": coverage,
                        "stop_reason": result["stop_reason"],
                        "valid_rounds": sum(h["status"] == "VALID" for h in result["history"])})
    save(output / "candidate_coverage_and_edits.json", details)
    return report, details


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--previous-run", type=Path, help="carry prior native charges and unresolved reserves")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("new directory required; no automatic paid replay")
    selected = read(SOURCE / "plan.json")["selection"][:4]
    cached = {(r["video_id"], r["frame_id"]): r["h0"] for r in read(SOURCE / "predictions.json")}
    previous_budget = None
    if args.previous_run:
        if read(args.previous_run / "plan.json")["selection"] != selected:
            raise ValueError("previous budget must belong to the same frozen target cohort")
        previous_budget = read(args.previous_run / "budget.json")["occupied"]
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    bases = [build_base(adapter, s) for s in selected]
    dependencies = [Path(__file__), ROOT / "scripts/check_candidate_panel_providers.py",
                    ROOT / "scripts/run_prior_panel_trial.py", ROOT / "scripts/run_grounded_api_pipeline.py",
                    *sorted((ROOT / "src/surgical_agent/research/verification").glob("*.py")),
                    ROOT / "src/surgical_agent/evaluation/repair_comparison.py",
                    ROOT / "src/surgical_agent/perception/main_h0.py",
                    ROOT / "src/surgical_agent/perception/ontology_prompt.py",
                    ROOT / "src/surgical_agent/perception/final_only.py"]
    hashes = {str(f.relative_to(ROOT)): sha(f) for f in dependencies}
    plan = {"created_utc": now(), "selection": selected, "h0_source": str(SOURCE),
            "h0_source_sha256": sha(SOURCE / "predictions.json"), "models": MODELS,
            "base_proposer": BASE_MODEL, "scope": "four_old_Training_targets_development_replay",
            "h0_reused": True, "gt_not_in_inference": True, "max_calls": 72, "round_cap": 3,
            "review_max_output_tokens": 1536, "proposal_max_output_tokens": 4096,
            "limits": {k: str(v) for k, v in LIMITS.items()}, "rate_envelopes": RATES,
            "previous_run": str(args.previous_run) if args.previous_run else None,
            "carried_occupied": previous_budget,
            "source_sha256": hashes,
            "caveats": ["Not independent confirmation; these development targets were analyzed previously.",
                        "H0 is cached original output, not a new H0 inference.",
                        "Reviewer scores are not calibrated confidence or a correctness oracle."]}
    save(args.output / "plan.json", plan)
    for f in dependencies:
        dest = args.output / "frozen_source" / f.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dest)
    save(args.output / "frozen_ontology.json", {"ontology": load_prompt_ontology_text(), "label_boundary": _LABEL_BOUNDARY})
    rows = [{"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": cached[s["video_id"], s["frame_id"]],
             "h1": None, "final": cached[s["video_id"], s["frame_id"]], "status": "NOT_ATTEMPTED"} for s in selected]
    save(args.output / "predictions.json", rows)
    if not args.execute:
        print("Prepared only; no API requests.")
        return
    calls, results = Calls(args.output, previous_budget), []
    for index, (s, base) in enumerate(zip(selected, bases, strict=True)):
        key = s["key"]
        h0 = rows[index]["h0"]
        def propose(n, state, pool, issues, base=base, s=s, key=key):
            return calls.call(key, f"proposal_{n}", "base", proposal_body(base, s, state, pool, issues))
        def review(n, state, pool, base=base, s=s, key=key):
            del state  # Judges are not told which candidate was H0 or previously accepted.
            def one(seat):
                body = body_for(seat, base, s, pool, gemini_json_object=True)
                body["max_tokens"] = 1536  # Enough for the bounded 64-ID pool.
                return calls.call(key, f"review_{n}", seat, body)
            with ThreadPoolExecutor(max_workers=5) as workers:
                return dict(zip(SEATS, workers.map(one, SEATS), strict=True))
        result = run(h0, propose, review, lambda stage, value, key=key: save(args.output / "targets" / key / f"{stage}.json", value))
        result.update(video_id=s["video_id"], frame_id=s["frame_id"])
        results.append(result)
        rows[index].update(final=result["final"], status=result["stop_reason"])
        save(args.output / "predictions.json", rows)
        print(json.dumps({"target": key, "stop": result["stop_reason"], "rounds": len(result["history"]),
                          "changed": result["final"] != h0, "calls_so_far": len(calls.rows)}), flush=True)
    calls.stopped = True
    calls.persist()
    frozen_predictions = sha(args.output / "predictions.json")
    if any(sha(ROOT / f) != h for f, h in hashes.items()):
        raise ValueError("inference source changed during run")
    report, details = score(args.output, adapter, rows, results)
    if sha(args.output / "predictions.json") != frozen_predictions:
        raise ValueError("predictions changed while scoring")
    native = {k: sum(Decimal(r["charge"]) for r in calls.rows if r["account"] == k and r["charge_kind"] == "native") for k in LIMITS}
    summary = {"targets": len(rows), "prediction_sha256": frozen_predictions, "post_calls": len(calls.rows),
               "stops": dict(Counter(r["status"] for r in rows)), "native_costs": {k: str(v) for k, v in native.items()},
               "unpriced_calls": sum(r["charge_kind"] != "native" for r in calls.rows),
               "final_metrics": report["arms"]["final"], "h0_metrics": report["arms"]["h0"],
               "changes": report["paired"]["h0_to_final"], "details": details}
    save(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
