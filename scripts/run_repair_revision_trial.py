"""Isolated, paired Training trial for tolerant review and distributed proposals.

No GT is consumed by inference; score is a separate offline command. Historical
code and outputs stay intact. New reviewer proposals require a later review.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import requests

from scripts.check_candidate_panel_providers import redact_images
from scripts.panel_proposal_wire import (
    review_and_propose_wire,
    split_review_and_proposals,
)
from scripts.run_candidate_panel_trial import Calls, sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import (
    MODELS,
    PROVIDERS,
    RATES_V2,
    ROUTES,
    review_wire,
)
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.review_json_compat import (
    parse_review_json_compatible as parse_review_json,
)
from surgical_agent.research.verification.review_normalization import (
    normalize_review,
)

PROFILE = "repair_revision_tolerant_distributed_v1"
ARMS = ("single_proposer_tolerant", "distributed_ivt_tolerant")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
ALLOWANCE = {"openrouter_usd": Decimal(4), "xai_usd": Decimal(4), "aliyun_cny": Decimal(4)}


class RevisionCalls(Calls):
    """Recover complete JSON boundary fences and identical review-core duplicates."""

    def __init__(self, *args, reuse_source=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reuse_source = reuse_source
        self.cache_hits = []
        self.cache = {}
        if reuse_source:
            previous = read(reuse_source / "budget.json")
            if not previous["stopped"]:
                raise ValueError("cache source must be closed")
            if any(Decimal(self.carried[k]) != Decimal(previous["occupied"][k])
                   or self.limits[k] > Decimal(previous["limits"][k]) for k in self.limits):
                raise ValueError("cached ledger balance or limit changed")
            for row in previous["calls"]:
                if row["status"] not in ("JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"):
                    continue
                key = row["target"], row["stage"], row["seat"]
                folder = reuse_source / "calls" / f"{row['index']:03d}_{key[0]}_{key[1]}_{key[2]}"
                self.cache[key] = folder

    def call(self, target, stage, seat, body):
        folder = self.cache.get((target, stage, seat))
        if folder is not None:
            if read(folder / "request.json") != redact_images(body):
                raise ValueError("cached request differs; do not reuse another protocol")
            response = read(folder / "response.json")
            raw = response["body"]
            expected = self.providers.get(body["model"], self.providers.get(seat))
            if (response["http_status"] != 200 or raw.get("error") or raw.get("model") != body["model"]
                    or (expected is not None and raw.get("provider") != expected)
                    or raw["choices"][0]["finish_reason"] != "stop"
                    or raw["choices"][0]["message"].get("refusal")):
                raise ValueError("cached transport identity or completion invalid")
            text = response["body"]["choices"][0]["message"]["content"]
            parsed = json.loads(text) if seat == "base" else parse_review_json(text)[0]
            if parsed is None:
                raise ValueError("cached response no longer valid")
            with self.lock:
                self.cache_hits.append({"target": target, "stage": stage, "seat": seat, "source": str(folder),
                                        "request_sha256": sha(folder / "request.json"),
                                        "response_sha256": sha(folder / "response.json")})
                save(self.output / "cache_hits.json", self.cache_hits)
            return parsed
        parsed = super().call(target, stage, seat, body)
        if seat == "base":
            return parsed
        rows = [r for r in self.rows if (r["target"], r["stage"], r["seat"]) == (target, stage, seat)]
        if len(rows) != 1:
            return None
        row = rows[0]
        eligible = row.get("status") == "JSON_PARSED" or row.get("exception_type") == "JSONDecodeError"
        if row.get("http_status") != 200 or not eligible:
            return None  # Never bypass refusal, truncation, model/provider or reasoning checks.
        folder = self.output / "calls" / f"{row['index']:03d}_{target}_{stage}_{seat}"
        raw = read(folder / "response.json")
        try:
            parsed, diagnostic = parse_review_json(raw["body"]["choices"][0]["message"]["content"])
        except (ValueError, TypeError, KeyError, IndexError):
            return None
        save(folder / "parse_recovery.json", {"response_sha256": sha(folder / "response.json"),
                                              "diagnostic": diagnostic, "parsed": parsed})
        with self.lock:
            row["original_exception_type"] = row.pop("exception_type", None)
            row["status"] = ("SAFE_JSON_REJECTED" if parsed is None else
                             "JSON_PARSED_FENCE_NORMALIZED" if diagnostic["removed_fences"] else "JSON_PARSED")
            save(folder / "record.json", row)
            self.persist()
        return parsed


def merge_suggestions(current, pool, suggestions):
    """Capacity ranking only, never acceptance: at most four novel IVTs/round."""
    existing = {p["label_id"] for p in pool["propositions"] if p["task"] == "ivt"}
    support = Counter()
    for seat in SEATS:
        values = suggestions.get(seat, {}).get("ivt", [])
        if (not isinstance(values, list) or len(values) > 2
                or any(type(v) is not int or not 0 <= v < 100 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("invalid normalized reviewer suggestions")
        support.update(set(values) - existing)
    ordered = sorted(support, key=lambda v: (-support[v], v))
    kept, dropped = [], []
    result = deepcopy(pool)
    for value in ordered:
        if len(kept) >= 4:
            dropped.append({"ivt": value, "reason": "ROUND_CAP"})
            continue
        try:
            result = make_pool(current, {"instrument": [], "verb": [], "target": [], "ivt": [value]}, result)
        except (ApiSchemaError, ValueError):
            dropped.append({"ivt": value, "reason": "POOL_CAP"})
        else:
            kept.append(value)
    return result, {"proposer_count": dict(support), "queued_ivt": kept, "dropped": dropped,
                    "counts_are_not_visual_confidence": True}


def normalize_five(raw, pool, image_count):
    reviews, diagnostics = {}, {}
    for seat in SEATS:
        reviews[seat], diagnostics[seat] = normalize_review(raw[seat], pool, seat=seat, image_count=image_count)
    return reviews, diagnostics


def frozen_sources():
    names = ["scripts/run_repair_revision_trial.py", "scripts/panel_proposal_wire.py",
             "scripts/check_candidate_panel_providers.py", "scripts/run_candidate_panel_trial.py",
             "scripts/run_complete_gt_semantic_trial.py", "scripts/run_grounded_api_pipeline.py",
             "scripts/run_openrouter_gemini_h0_trial.py", "scripts/run_recent_mean_panel_trial.py",
             "scripts/run_prior_panel_trial.py", "scripts/run_presence_review_trial.py",
             "scripts/run_semantic_candidate_trial.py", "scripts/replay_semantic_review.py",
             "configs/perception/joint_openrouter_h0.yaml"]
    paths = {ROOT / p for p in names}
    for package in ("api", "perception", "data", "config", "evaluation", "artifacts", "research/verification"):
        paths.update((ROOT / "src/surgical_agent" / package).rglob("*.py"))
    paths.update(p for p in (ROOT / "src/surgical_agent/perception/prompts").glob("*") if p.is_file())
    return sorted(paths)


def prepare(output, previous, adapter, stage, reuse_source=None):
    if output.exists():
        raise ValueError("new output directory required")
    prior = read(previous / "budget.json")
    if not prior["stopped"]:
        raise ValueError("previous ledger must be closed")
    selection_path = ROOT / "artifacts/training/gate/final_only_preparation_20260908_v2/pilot_selection.json"
    manifest = read(selection_path)
    if not manifest["no_gt_label_values_used_for_selection"]:
        raise ValueError("selection must be mask-only")
    indices = (1, 6) if stage == "development" else (3, 8)
    selected = []
    for video in ("VID103", "VID23", "VID31", "VID96"):
        if adapter.entries[video].split is not DatasetSplit.TRAINING:
            raise ValueError("Training targets only")
        available = [r for r in manifest["selection"] if r["video_id"] == video]
        for index in indices:
            row = deepcopy(available[index])
            sample = next(s for s in adapter.iter_inference_video(video) if s.target_frame_id == row["frame_id"])
            if not all(row["gt_availability_only"].values()) or list(sample.causal_frame_ids) != row["causal_frame_ids"]:
                raise ValueError("invalid source selection")
            for im, path in zip(row["images"], sample.media_refs, strict=True):
                if sha(path) != im["sha256"]:
                    raise ValueError("image changed")
            row["key"] = f"{video}_{row['frame_id']}"
            row["request_metadata"] = canonical_request_metadata(build_gemini_base(adapter, row)).to_mapping()
            selected.append(row)
    metadata = {}
    for seat, route in ROUTES.items():
        model = PROPOSER if seat == "base" else MODELS[seat]
        response = requests.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=30)
        response.raise_for_status()
        data = response.json()
        endpoint = next(e for e in data["data"]["endpoints"] if e["tag"] == route)
        if endpoint["status"] != 0 or not {"response_format", "reasoning"} <= set(endpoint["supported_parameters"]):
            raise ValueError("unavailable fixed model route")
        if any(Decimal(endpoint["pricing"][f]) > Decimal(rate)
               for f, rate in zip(("prompt", "completion"), RATES_V2[seat], strict=True)):
            raise ValueError("price exceeds reserved rate")
        metadata[seat] = data
    plan = {"profile": PROFILE, "stage": stage, "created_utc": now(), "selection": selected,
            "selection_source": str(selection_path), "selection_sha256": sha(selection_path),
            "arms": list(ARMS), "models": MODELS, "h0": PROPOSER, "threshold": 4, "round_cap": 3,
            "review_json_parser": "complete_document_identical_review_core_duplicates_v1",
            "review_envelope_policy": "preserve_all_actual_envelopes_for_normalizer_v2",
            "max_calls": len(selected) * (1 + 18 + 15),
            "previous_budget": str(previous), "previous_budget_sha256": sha(previous / "budget.json"),
            "carried_occupied": prior["occupied"], "incremental_allowances": {k: str(v) for k, v in ALLOWANCE.items()},
            "limits": {k: str(Decimal(prior["occupied"][k]) + v) for k, v in ALLOWANCE.items()},
            "rates": RATES_V2, "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in frozen_sources()},
            "gt_policy": "no GT in inference; offline score after immutable round snapshots",
            "distributed_arm_changes": ["five reviewers also propose", "IVT-only new suggestions", "no standalone proposer", "new proposals reviewed next round"],
            "primary_success_rule": "Both Target and IVT micro-F1 and exact-set accuracy nondecreasing vs shared H0 and comparator; fewer total label errors; no extra worsened frames; record all heads.",
            "confirmation_boundary": "Different time targets from the same Training videos, not independent surgery or Test generalization."}
    if reuse_source is not None:
        if reuse_source.resolve() != previous.resolve():
            raise ValueError("reuse the most recent closed ledger")
        old_plan = read(reuse_source / "plan.json")
        if old_plan["selection"] != selected or old_plan["arms"] != list(ARMS):
            raise ValueError("cache selection or arms changed")
        plan["reuse_source"] = str(reuse_source)
        plan["reuse_sha256"] = {str(p.resolve().relative_to(ROOT)): sha(p) for p in reuse_source.rglob("*.json")
                                if "frozen_source" not in p.parts}
        plan["max_calls"] = old_plan["max_calls"] + 1 - len(prior["calls"])
        plan["logical_call_cap"] = old_plan["max_calls"]
        plan["total_batch_post_cap_including_schema_probe"] = old_plan["max_calls"] + 1
        # Reuse does not reset the original batch's unspent budget.
        plan["limits"] = prior["limits"]
        plan["incremental_allowances"] = {k: str(Decimal(prior["limits"][k]) - Decimal(prior["occupied"][k])) for k in ALLOWANCE}
        plan["allowance_note"] = "remaining original batch allowance, not an additional grant"
    save(output / "plan.json", plan)
    save(output / "model_endpoints.json", metadata)
    for row in selected:
        save(output / "h0_preflight" / f"{row['key']}.json", redact_images(gemini_h0_wire(build_gemini_base(adapter, row))))
    for path in frozen_sources():
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    print(json.dumps({"prepared": str(output), "targets": [r["key"] for r in selected],
                      "max_calls": plan["max_calls"], "allowance": plan["incremental_allowances"]}), flush=True)


def verify_plan(plan):
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")
    for row in plan["selection"]:
        for image in row["images"]:
            if sha(image["path"]) != image["sha256"]:
                raise ValueError("prepared image changed")
    for name, digest in plan.get("reuse_sha256", {}).items():
        if sha(ROOT / name) != digest:
            raise ValueError("cached source changed")


def execute(output, adapter):
    plan = read(output / "plan.json")
    if plan["profile"] != PROFILE or (output / "execution.lock").exists():
        raise ValueError("invalid profile or paid replay prohibited")
    verify_plan(plan)
    if sha(Path(plan["previous_budget"]) / "budget.json") != plan["previous_budget_sha256"]:
        raise ValueError("previous ledger changed")
    bases = {s["key"]: build_gemini_base(adapter, s) for s in plan["selection"]}
    for s in plan["selection"]:
        if canonical_request_metadata(bases[s["key"]]).to_mapping() != s["request_metadata"]:
            raise ValueError("H0 input changed")
    with (output / "execution.lock").open("x") as marker:
        marker.write(sha(output / "plan.json"))
    calls = RevisionCalls(output, plan["carried_occupied"], limits={k: Decimal(v) for k, v in plan["limits"].items()},
                  rates=plan["rates"], providers=PROVIDERS, max_calls=plan["max_calls"], reasoning_seats=("grok", "gemini"),
                  reuse_source=Path(plan["reuse_source"]) if plan.get("reuse_source") else None)
    states = {s["key"]: {"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": None,
                        "arms": {a: {"current": None, "pool": None, "issues": [], "active": True,
                                     "status": "NOT_ATTEMPTED", "history": []} for a in ARMS}}
              for s in plan["selection"]}
    for number in range(1, plan["round_cap"] + 1):
        for sample_index, selected in enumerate(plan["selection"]):
            key, base = selected["key"], bases[selected["key"]]
            state = states[key]
            if calls.stopped:
                break
            if number == 1:
                raw = calls.call(key, "h0", "base", gemini_h0_wire(base))
                try:
                    validate_final_only(raw)
                except (ApiSchemaError, TypeError, ValueError):
                    for arm in state["arms"].values():
                        arm.update(active=False, status="H0_FAILED")
                    calls.stopped = True
                    break
                state["h0"] = {t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"] for t in TASKS}
                for arm in state["arms"].values():
                    arm.update(current=deepcopy(state["h0"]), pool=make_pool(state["h0"]))
            # AB/BA time ordering; each arm owns its states and pending proposals.
            for arm_name in ARMS if sample_index % 2 == 0 else tuple(reversed(ARMS)):
                arm = state["arms"][arm_name]
                if not arm["active"] or calls.stopped:
                    continue
                record = {"round": number, "before": deepcopy(arm["current"])}
                pool = arm["pool"]
                if arm_name == ARMS[0]:
                    proposal = calls.call(key, f"{arm_name}_proposal_{number}", "base",
                                          gemini_proposal(base, selected, arm["current"], pool, arm["issues"]))
                    record["proposal"] = proposal
                    try:
                        if proposal is None:
                            raise ValueError("no proposal")
                        pool = make_pool(arm["current"], proposal, pool)
                    except (ApiSchemaError, ValueError, TypeError, KeyError):
                        arm.update(active=False, status="PROPOSAL_FAILED")
                        save(output / "targets" / key / f"{arm_name}_round_{number}_failed.json", record)
                        continue
                    if not pool["propositions"]:
                        arm.update(active=False, status="EMPTY_POOL_UNVERIFIED")
                        continue
                def one(seat, base=base, selected=selected, pool=pool,
                        arm_name=arm_name, key=key, number=number):
                    body = (review_wire(seat, base, selected, pool) if arm_name == ARMS[0]
                            else review_and_propose_wire(seat, base, selected, pool))
                    return calls.call(key, f"{arm_name}_review_{number}", seat, body)
                with ThreadPoolExecutor(max_workers=5) as workers:
                    raw_reviews = dict(zip(SEATS, workers.map(one, SEATS), strict=True))
                review_parts, suggestions, proposal_diagnostics = {}, {}, {}
                for seat in SEATS:
                    if arm_name == ARMS[1]:
                        review_parts[seat], suggestions[seat], proposal_diagnostics[seat] = split_review_and_proposals(raw_reviews[seat], seat, pool)
                    else:
                        review_parts[seat] = raw_reviews[seat]
                reviews, format_diagnostics = normalize_five(review_parts, pool, len(base.images))
                if pool["propositions"]:
                    means, diagnostics = panel.aggregate(reviews, pool, image_count=len(base.images))
                    try:
                        after = panel.select(arm["current"], pool, means)
                        issues = panel.unresolved(after, pool, means, diagnostics)
                    except (ApiSchemaError, TypeError, ValueError, KeyError):
                        arm.update(active=False, status="SELECTION_FAILED")
                        after, issues = deepcopy(arm["current"]), arm["issues"]
                else:
                    means, diagnostics, after, issues = {}, {}, deepcopy(arm["current"]), []
                next_pool, pending = (merge_suggestions(after, pool, suggestions) if arm_name == ARMS[1]
                                      else (pool, {"queued_ivt": []}))
                failed_proposal = arm_name == ARMS[1] and any(
                    d.get("proposal_status") != "VALID" for d in proposal_diagnostics.values())
                unresolved = bool(issues or pending["queued_ivt"] or pending.get("dropped")
                                  or not pool["propositions"] or failed_proposal)
                if arm["status"] != "SELECTION_FAILED":
                    arm.update(active=unresolved and number < plan["round_cap"],
                               status="UNRESOLVED" if unresolved else "MODEL_PASS")
                arm.update(current=after, pool=next_pool, issues=issues)
                record.update(pool=pool, next_pool=next_pool, raw_reviews=raw_reviews, reviews=reviews,
                              format_diagnostics=format_diagnostics, means=means, diagnostics=diagnostics,
                              suggestions=suggestions, proposal_diagnostics=proposal_diagnostics, pending=pending,
                              after=after, issues=issues, status=arm["status"])
                arm["history"].append(record)
                save(output / "targets" / key / f"{arm_name}_round_{number}.json", record)
                save(output / "inference_states.json", states)
                print(json.dumps({"target": key, "arm": arm_name, "round": number, "status": arm["status"],
                                  "reviewed_pool": len(pool["propositions"]), "new_ivt_next_round": pending["queued_ivt"],
                                  "calls": len(calls.rows)}), flush=True)
                if number == 1 and sample_index == 0 and pool["propositions"]:
                    broken = [seat for seat in SEATS if all(seat in d["invalid"] for d in diagnostics.values())]
                    if broken:
                        calls.stopped = True
                        save(output / "contract_failure.json", {"arm": arm_name, "seats": broken})
        rows = [{"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": s["h0"],
                 "arms": {a: {"final": v["current"], "status": v["status"], "actual_rounds": len(v["history"])}
                          for a, v in s["arms"].items()}} for s in states.values()]
        save(output / f"round_{number}_predictions.json", rows)
        save(output / "inference_states.json", states)
        subprocess.run([sys.executable, str(Path(__file__)), "score", "--output", str(output), "--round", str(number)],
                       cwd=ROOT, check=True)
        verify_plan(plan)
        if calls.stopped or not any(a["active"] for s in states.values() for a in s["arms"].values()):
            break
    calls.stopped = True
    calls.persist()
    save(output / "completion.json", {"targets": len(states), "post_calls": len(calls.rows), "last_snapshot": number,
         "new_charges": {a: str(sum(Decimal(r["charge"]) for r in calls.rows if r["account"] == a)) for a in ALLOWANCE},
         "unknown_reserved_calls": sum(r["charge_kind"] == "unknown_reserved" for r in calls.rows), "finished_utc": now()})


def score(output, adapter, number):
    path = output / f"round_{number}_predictions.json"
    digest, rows = sha(path), read(path)
    reports, paired = {}, {}
    paired_rows = [r for r in rows if r["h0"] is not None
                   and all(r["arms"][a]["actual_rounds"] > 0 for a in ARMS)]
    for arm in ARMS:
        data = [{"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": r["h0"], "h1": None,
                 "final": r["arms"][arm]["final"]} for r in rows]
        reports[arm], truth = score_saved(adapter, data)
        save(output / "scores" / f"round_{number}_{arm}_truth.json", truth)
        if paired_rows:
            subset = [{"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": r["h0"], "h1": None,
                       "final": r["arms"][arm]["final"]} for r in paired_rows]
            paired[arm], _ = score_saved(adapter, subset)
    coverage = {"selected": len(rows), "h0_available": sum(r["h0"] is not None for r in rows),
                "paired_with_both_reviewed": len(paired_rows),
                "statuses": {a: dict(Counter(r["arms"][a]["status"] for r in rows)) for a in ARMS},
                "unfinished_observations_are_not_gate_negative_labels": True}
    save(output / "scores" / f"round_{number}.json", {"prediction_sha256": digest, "coverage": coverage,
                                                     "reports": reports, "paired_reviewed_subset": paired})
    if sha(path) != digest:
        raise ValueError("prediction changed while scoring")
    print(json.dumps({"round": number, "metrics": {a: {t: m["micro_f1"] for t, m in r["arms"]["final"]["tasks"].items()}
                                                   for a, r in reports.items()}}), flush=True)


def replay(source, output, adapter):
    if output.exists():
        raise ValueError("new replay output required")
    states = read(source / "inference_states.json")
    reports = {}
    for number in (1, 2, 3):
        rows, changes = [], []
        for key, state in states.items():
            previous = [r for r in state["history"] if r["round"] <= number]
            record = previous[-1] if previous else None
            old_final = record["primary"] if record else state["h0"]
            final = old_final
            if record and record["round"] == number and record["pool"]["propositions"]:
                raw_reviews = deepcopy(record["reviews"])
                recoveries = {}
                for seat in SEATS:
                    if raw_reviews[seat] is not None:
                        continue
                    paths = list((source / "calls").glob(f"*_{key}_review_{number}_{seat}/response.json"))
                    if len(paths) != 1:
                        continue
                    original_record = read(paths[0].parent / "record.json")
                    if original_record.get("exception_type") != "JSONDecodeError":
                        continue
                    raw = read(paths[0])
                    try:
                        body = raw["body"]
                        if raw["http_status"] != 200 or body["choices"][0]["finish_reason"] != "stop":
                            continue
                        parsed, info = parse_review_json(body["choices"][0]["message"]["content"])
                    except (ValueError, KeyError, TypeError, IndexError):
                        continue
                    if parsed is None:
                        continue
                    raw_reviews[seat] = parsed
                    recoveries[seat] = {"source": str(paths[0]), "sha256": sha(paths[0]), "parse": info}
                reviews, diag = normalize_five(raw_reviews, record["pool"], 3)
                means, _ = panel.aggregate(reviews, record["pool"])
                final = panel.select(record["before"], record["pool"], means)
                strict_means, _ = panel.aggregate(record["reviews"], record["pool"])
                assert panel.select(record["before"], record["pool"], strict_means) == record["primary"]
                changes.append({"key": key, "normalization": diag, "parse_recoveries": recoveries,
                                "restored_means": [p for p in means if means[p] is not None and strict_means[p] is None]})
            rows.append({"video_id": state["video_id"], "frame_id": state["frame_id"], "h0": state["h0"],
                         "h1": old_final, "final": final})
        report, truth = score_saved(adapter, rows)
        save(output / f"round_{number}_predictions.json", rows)
        save(output / f"round_{number}_truth.json", truth)
        save(output / f"round_{number}_normalization.json", changes)
        reports[str(number)] = report
        print(json.dumps({"round": number, "f1": {a: {t: m["micro_f1"] for t, m in v["tasks"].items()}
                                                    for a, v in report["arms"].items()}}))
    save(output / "report.json", {"source": str(source), "source_sha256": sha(source / "inference_states.json"),
          "new_api_calls": 0, "kind": "each round reuses its original before state; not an independent closed loop",
          "rounds": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score", "replay"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--reuse-source", type=Path)
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, args.previous, adapter, args.stage, args.reuse_source)
    elif args.command == "execute":
        execute(args.output, adapter)
    elif args.command == "score":
        score(args.output, adapter, args.round)
    else:
        replay(args.source, args.output, adapter)
