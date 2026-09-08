"""Frozen Training trial: recent reviewers, means 4/3.5, <=3 rounds.

The threshold-4 trajectory alone drives repair. Threshold 3.5 is a shadow replay
of identical candidate pools and judgments, not a separate closed-loop trial.
Round scoring runs in a separate subprocess; no GT result is read by inference.
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import RATES, Calls, sha
from scripts.run_complete_gt_semantic_trial import normalize_review_wire, review_body_v3
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_presence_review_trial import _mask_only
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_semantic_candidate_trial import PROPOSER, gemini_proposal
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.verification import recent_mean_panel as mean_panel
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool

PROFILE = "recent_five_family_mean_20260908_v2_empty_pool_guard"
MODELS = {"grok": "grok-4.6", "qwen": "qwen3.8-flash", "gpt": "openai/gpt-5.6-luna",
          "gemini": "google/gemini-3.5-flash-lite", "deepseek": "deepseek/deepseek-v4-flash-vision-exp"}
RATES_V2 = {**RATES, "grok": ("0.000002", "0.000006"), "gpt": ("0.0000002", "0.0000012"),
            "gemini": ("0.0000003", "0.0000025")}
ALLOWANCE = {"openrouter_usd": Decimal(3), "xai_usd": Decimal("1.50"), "aliyun_cny": Decimal(2)}
ROUTES = {"base": "google-ai-studio", "gpt": "openai", "gemini": "google-ai-studio", "deepseek": "fireworks"}
PROVIDERS = {"base": "Google AI Studio", "gpt": "OpenAI", "gemini": "Google AI Studio", "deepseek": "Fireworks"}


def review_wire(seat, base, selected, pool):
    body = review_body_v3(seat, base, selected, pool)
    body["model"] = MODELS[seat]
    if seat == "gpt":
        body.pop("temperature", None)  # Fixed OpenAI route does not advertise this parameter.
        body["reasoning"] = {"effort": "none"}
    elif seat == "gemini":
        body["reasoning"] = {"effort": "minimal"}
    elif seat == "grok":
        body["reasoning_effort"] = "low"  # 4.6 cannot disable thinking.
    elif seat == "qwen":
        body["enable_thinking"] = False
        packet = json.loads(body["messages"][0]["content"][0]["text"])
        packet["output_format"] = "Return exactly one JSON object matching response_schema."
        body["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return body


class CachedCalls(Calls):
    """Reuse only successful exact-wire calls from an explicitly frozen source."""
    def __init__(self, *args, reuse_source, **kwargs):
        super().__init__(*args, **kwargs)
        self.reuse_source = Path(reuse_source)
        self.cache_hits = []
        self.cache = {}
        for record in read(self.reuse_source / "budget.json")["calls"]:
            if record["status"] == "JSON_PARSED":
                key = (record["target"], record["stage"], record["seat"])
                folder = self.reuse_source / "calls" / f"{record['index']:03d}_{'_'.join(key)}"
                self.cache[key] = folder

    def call(self, target, stage, seat, body):
        folder = self.cache.get((target, stage, seat))
        if folder is None:
            return super().call(target, stage, seat, body)
        if self.stopped:
            return None
        if read(folder / "request.json") != redact_images(body):
            raise ValueError("cached wire differs; no silent paid replacement")
        response = read(folder / "response.json")["body"]
        with self.lock:
            self.cache_hits.append({"target": target, "stage": stage, "seat": seat, "source": str(folder),
                                    "request_sha256": sha(folder / "request.json"),
                                    "response_sha256": sha(folder / "response.json")})
            save(self.output / "cache_hits.json", self.cache_hits)
        return json.loads(response["choices"][0]["message"]["content"])


def choose_samples(adapter):
    chosen = []
    # Three Training videos with frame-level interaction GT. The video list,
    # quantiles and mask-only selection are fixed before seeing model outputs.
    for video in ("VID103", "VID23", "VID96"):
        samples = list(adapter.iter_inference_video(video))
        masks = {r.inference.target_frame_id: _mask_only(r) for r in adapter.iter_video(video)}
        eligible = [s for s in samples if masks.get(s.target_frame_id) and all(masks[s.target_frame_id].values())]
        used = set()
        for numerator in (1, 2, 3, 4):
            anchor = samples[len(samples) * numerator // 5].target_frame_id
            s = min((s for s in eligible if s.target_frame_id not in used),
                    key=lambda s: (abs(s.target_frame_id - anchor), s.target_frame_id))
            if s.source_split is not DatasetSplit.TRAINING:
                raise ValueError("Training only")
            used.add(s.target_frame_id)
            item = {"key": f"{video}_{s.target_frame_id}", "video_id": video, "frame_id": s.target_frame_id,
                    "anchor_frame_id": anchor, "sampling_rule": f"nearest all-head-mask target to {numerator}/5; earlier tie",
                    "causal_frame_ids": list(s.causal_frame_ids), "gt_availability_only": masks[s.target_frame_id],
                    "images": [{"frame_id": f, "path": str(p), "sha256": sha(p)}
                               for f, p in zip(s.causal_frame_ids, s.media_refs, strict=True)]}
            base = build_gemini_base(adapter, item)
            item["request_metadata"] = canonical_request_metadata(base).to_mapping()
            chosen.append(item)
    return chosen


def load_gate_selection(path, adapter):
    """Bind a mask-only Gate collection plan to the actual Training images."""
    manifest = read(path)
    if (manifest.get("schema_version") != "final_only_gate_collection_selection_v1"
            or manifest.get("no_gt_label_values_used_for_selection") is not True):
        raise ValueError("unsupported Gate selection manifest")
    selected = []
    seen = set()
    for row in manifest["selection"]:
        video, frame = row["video_id"], row["frame_id"]
        if (type(frame) is not int or (video, frame) in seen
                or adapter.entries[video].split is not DatasetSplit.TRAINING):
            raise ValueError("Gate selection must contain unique Training targets")
        seen.add((video, frame))
        sample = next(s for s in adapter.iter_inference_video(video) if s.target_frame_id == frame)
        resolved = next(adapter.iter_video(video, frame_ids=[frame]))
        mask = _mask_only(resolved)
        if not all(mask.values()) or mask != row["gt_availability_only"]:
            raise ValueError("Gate selection requires complete interaction GT availability")
        if list(sample.causal_frame_ids) != row["causal_frame_ids"]:
            raise ValueError("Gate selection causal frames changed")
        images = [{"frame_id": f, "path": str(p), "sha256": sha(p)}
                  for f, p in zip(sample.causal_frame_ids, sample.media_refs, strict=True)]
        if [im["sha256"] for im in images] != [im["sha256"] for im in row["images"]]:
            raise ValueError("Gate selection image hash changed")
        item = {"key": f"{video}_{frame}", "video_id": video, "frame_id": frame,
                "anchor_frame_id": row["anchor_frame_id"], "causal_frame_ids": list(sample.causal_frame_ids),
                "gt_availability_only": mask, "images": images,
                "sampling_rule": manifest["selection_rule"]}
        item["request_metadata"] = canonical_request_metadata(build_gemini_base(adapter, item)).to_mapping()
        selected.append(item)
    if not selected:
        raise ValueError("Gate selection is empty")
    return selected


def prepare(output, previous, adapter, reuse_source=None, selection_manifest=None, allowances=None):
    if output.exists():
        raise ValueError("new directory required")
    reuse_source = reuse_source.resolve() if reuse_source else None
    prior = read(previous / "budget.json")
    if not prior["stopped"]:
        raise ValueError("previous inference still active")
    if reuse_source and (selection_manifest is not None or allowances is not None):
        raise ValueError("exact-wire reuse cannot change targets or allowances")
    if selection_manifest is not None:
        selected = load_gate_selection(selection_manifest, adapter)
    else:
        selected = read(reuse_source / "plan.json")["selection"] if reuse_source else choose_samples(adapter)
    old = read(ROOT / "artifacts/preflight/openrouter_gemini_h0_panel_20260908_v1/plan.json")["selection"]
    if {s["key"] for s in selected} & {s["key"] for s in old}:
        raise ValueError("must not reuse the previous four targets")
    metadata = {}
    for seat, route in ROUTES.items():
        model = PROPOSER if seat == "base" else MODELS[seat]
        response = requests.get(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=30)
        response.raise_for_status()
        data = response.json()
        endpoint = next(e for e in data["data"]["endpoints"] if e["tag"] == route)
        needed = {"response_format", "reasoning"} | ({"temperature"} if seat != "gpt" else set())
        if endpoint["status"] != 0 or not needed <= set(endpoint["supported_parameters"]):
            raise ValueError(f"unavailable/incompatible model route: {seat}")
        if any(Decimal(endpoint["pricing"][field]) > Decimal(rate)
               for field, rate in zip(("prompt", "completion"), RATES_V2[seat], strict=True)):
            raise ValueError(f"price exceeds reserve rate: {seat}")
        metadata[seat] = data
    sources = [Path(__file__), ROOT / "scripts/check_candidate_panel_providers.py",
               ROOT / "scripts/run_candidate_panel_trial.py", ROOT / "scripts/run_complete_gt_semantic_trial.py",
               ROOT / "scripts/run_grounded_api_pipeline.py", ROOT / "scripts/run_openrouter_gemini_h0_trial.py",
               ROOT / "scripts/run_presence_review_trial.py", ROOT / "scripts/run_prior_panel_trial.py",
               ROOT / "scripts/run_semantic_candidate_trial.py", ROOT / "scripts/replay_semantic_review.py",
               ROOT / "configs/perception/joint_openrouter_h0.yaml"]
    for package in ("api", "perception", "data", "config", "evaluation", "artifacts", "research/verification"):
        sources.extend((ROOT / "src/surgical_agent" / package).rglob("*.py"))
    sources.extend((ROOT / "src/surgical_agent/perception/prompts").glob("*"))
    if selection_manifest is not None:
        sources.append(selection_manifest.resolve())
    sources = sorted({p for p in sources if p.is_file()})
    if reuse_source:
        if previous.resolve() != reuse_source.resolve():
            raise ValueError("reuse source must be most recent closed ledger")
        allowance = {k: Decimal(prior["limits"][k]) - Decimal(prior["occupied"][k]) for k in ALLOWANCE}
    else:
        allowance = ALLOWANCE if allowances is None else allowances
        if (set(allowance) != set(ALLOWANCE)
                or any(not v.is_finite() or v <= 0 for v in allowance.values())):
            raise ValueError("allowances require three positive finite account amounts")
    plan = {"profile": PROFILE, "created_utc": now(), "selection": selected, "models": MODELS,
            "h0": PROPOSER, "proposer": PROPOSER, "h0_cache_reused": bool(reuse_source), "round_cap": 3,
            "threshold_primary": 4.0, "threshold_shadow": 3.5, "max_calls": len(selected) * (1 + 3 * 6),
            "gt_policy": "availability masks only for selection; independent scoring after each saved round; no scores fed back",
            "stopping": "nonempty pool decisions pass, empty pool abstention, 3 real reviews, failure or budget; empty refill alone does not stop",
            "admission": "five valid evidence ratings required; arithmetic mean >= threshold adds, <= 6-threshold removes",
            "shadow_caveat": "3.5 uses primary trajectory pools/votes; not an independent closed loop",
            "grok_caveat": "4.6 is a flagship, not a small model; no Fast available in account model listing",
            "reasoning": {"grok": "low mandatory", "qwen": "disabled", "gpt": "none", "gemini": "minimal mandatory", "deepseek": "disabled"},
            "previous_budget": str(previous), "previous_budget_sha256": sha(previous / "budget.json"),
            "carried_occupied": prior["occupied"], "incremental_allowances": {k: str(v) for k, v in allowance.items()},
            "limits": {k: str(Decimal(prior["occupied"][k]) + v) for k, v in allowance.items()}, "rates": RATES_V2,
            "reuse_source": str(reuse_source) if reuse_source else None,
            "reuse_sha256": {str(p.relative_to(ROOT)): sha(p) for p in reuse_source.rglob("*.json")
                             if "frozen_source" not in p.parts} if reuse_source else {},
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources}}
    save(output / "plan.json", plan)
    save(output / "model_endpoints.json", metadata)
    for s in selected:
        save(output / "h0_preflight" / f"{s['key']}.json", redact_images(gemini_h0_wire(build_gemini_base(adapter, s))))
    for p in sources:
        dest = output / "frozen_source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    print(json.dumps({"selection": [s["key"] for s in selected], "max_calls": plan["max_calls"],
                      "allowances": plan["incremental_allowances"]}), flush=True)


def score_round(output, adapter, number):
    path = output / f"round_{number}_predictions.json"
    digest = sha(path)
    rows = read(path)
    reports = {}
    for arm in ("primary", "shadow"):
        values = [{"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": r["h0"],
                   "h1": None, "final": r[arm]} for r in rows]
        report, truth = score_saved(adapter, values)
        reports[arm] = report
        save(output / "scores" / f"round_{number}_{arm}_truth.json", truth)
    if sha(path) != digest:
        raise ValueError("predictions changed during scoring")
    save(output / "scores" / f"round_{number}.json", {"prediction_sha256": digest, "reports": reports})
    print(json.dumps({"round": number, "actual_reviews": sum(r["reviewed_this_round"] for r in rows),
          "primary_f1": {t: m["micro_f1"] for t, m in reports["primary"]["arms"]["final"]["tasks"].items()},
          "changes": reports["primary"]["paired"]["h0_to_final"]["frames"]}), flush=True)


def verify_sources(plan):
    if any(sha(ROOT / p) != value for p, value in plan["source_sha256"].items()):
        raise ValueError("frozen inference source changed")
    if any(sha(ROOT / p) != value for p, value in plan.get("reuse_sha256", {}).items()):
        raise ValueError("cached evidence changed")


def execute(output, adapter):
    plan = read(output / "plan.json")
    if plan["profile"] != PROFILE or (output / "execution.lock").exists():
        raise ValueError("wrong profile or paid replay prohibited")
    verify_sources(plan)
    if sha(Path(plan["previous_budget"]) / "budget.json") != plan["previous_budget_sha256"]:
        raise ValueError("previous ledger changed")
    with (output / "execution.lock").open("x") as marker:
        marker.write(sha(output / "plan.json"))
    call_class = CachedCalls if plan.get("reuse_source") else Calls
    extra = {"reuse_source": plan["reuse_source"]} if plan.get("reuse_source") else {}
    calls = call_class(output, plan["carried_occupied"], limits={k: Decimal(v) for k, v in plan["limits"].items()},
                      rates=plan["rates"], providers=PROVIDERS, max_calls=plan["max_calls"],
                      reasoning_seats=("grok", "gemini"), **extra)
    states = {}
    selected_by_key = {s["key"]: s for s in plan["selection"]}
    bases = {k: build_gemini_base(adapter, s) for k, s in selected_by_key.items()}
    for k, s in selected_by_key.items():
        if canonical_request_metadata(bases[k]).to_mapping() != s["request_metadata"]:
            raise ValueError("prepared input differs")
        states[k] = {"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": None,
                     "primary": None, "shadow": None, "pool": None, "issues": [], "history": [],
                     "active": True, "status": "NOT_ATTEMPTED", "reviewed_this_round": False}
    for number in range(1, 4):
        for state in states.values():
            state["reviewed_this_round"] = False
        for key, state in states.items():
            if not state["active"] or calls.stopped:
                continue
            s, base = selected_by_key[key], bases[key]
            if number == 1:
                raw = calls.call(key, "h0", "base", gemini_h0_wire(base))
                try:
                    validate_final_only(raw)
                except (ApiSchemaError, TypeError, ValueError):
                    state.update(active=False, status="H0_FAILED")
                    calls.stopped = True
                    break
                h0 = {t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"]
                      for t in ("instrument", "verb", "target", "ivt", "phase")}
                state.update(h0=h0, primary=deepcopy(h0), shadow=deepcopy(h0), pool=make_pool(h0))
            proposal = calls.call(key, f"proposal_{number}", "base",
                                  gemini_proposal(base, s, state["primary"], state["pool"], state["issues"]))
            try:
                if proposal is None:
                    raise ValueError("proposal unavailable")
                pool = make_pool(state["primary"], proposal, state["pool"])
            except (ApiSchemaError, TypeError, ValueError, KeyError):
                state.update(active=False, status="PROPOSAL_FAILED")
                continue
            if not pool["propositions"]:
                state.update(active=False, status="EMPTY_POOL_UNVERIFIED", pool=pool)
                save(output / "targets" / key / f"empty_pool_{number}.json",
                     {"proposal": proposal, "pool": pool, "status": state["status"],
                      "reason": "No proposition was audited; empty judgments cannot prove whole-frame absence."})
                save(output / "inference_states.json", states)
                continue  # No five paid reviewers for a vacuous zero-item contract.
            def one(seat, key=key, number=number, base=base, s=s, pool=pool):
                raw = calls.call(key, f"review_{number}", seat, review_wire(seat, base, s, pool))
                return normalize_review_wire(seat, raw, pool)
            with ThreadPoolExecutor(max_workers=5) as workers:
                reviews = dict(zip(SEATS, workers.map(one, SEATS), strict=True))
            means, diagnostics = mean_panel.aggregate(reviews, pool, image_count=len(base.images))
            record = {"round": number, "proposal": proposal, "pool": pool, "reviews": reviews,
                      "means": means, "diagnostics": diagnostics, "before": deepcopy(state["primary"])}
            state["reviewed_this_round"] = True
            try:
                primary = mean_panel.select(state["primary"], pool, means, threshold=4.0)
                shadow = mean_panel.select(state["shadow"], pool, means, threshold=3.5)
                state.update(primary=primary, shadow=shadow)
                state["issues"] = mean_panel.unresolved(state["primary"], pool, means, diagnostics)
                state["status"] = "MODEL_PASS" if not state["issues"] else "UNRESOLVED"
                state["active"] = bool(state["issues"])
            except (ApiSchemaError, ValueError, TypeError, KeyError):
                # Selection is atomic across arms; no partial patch on overflow.
                state["primary"] = record["before"]
                state.update(active=False, status="SELECTION_FAILED")
            state["pool"] = pool
            record.update(primary=deepcopy(state["primary"]), shadow=deepcopy(state["shadow"]),
                          issues=deepcopy(state["issues"]), status=state["status"])
            state["history"].append(record)
            save(output / "targets" / key / f"round_{number}.json", record)
            save(output / "inference_states.json", states)
            print(json.dumps({"target": key, "round": number, "status": state["status"], "pool": len(pool["propositions"]),
                              "invalid_candidates": sum(v is None for v in means.values()), "calls": len(calls.rows)}), flush=True)
            # First production target doubles as the five-provider interface
            # smoke. A wholly malformed/unavailable seat stops paid dispatch.
            if number == 1 and key == next(iter(states)):
                broken = [seat for seat in SEATS if all(
                    d["invalid"].get(seat) == "SCHEMA_INVALID" for d in diagnostics.values())]
                if broken:
                    calls.stopped = True
                    save(output / "contract_failure.json", {"seats": broken})
        verify_sources(plan)
        rows = [{k: deepcopy(v) for k, v in state.items() if k not in ("pool", "issues", "history", "active")}
                for state in states.values()]
        save(output / f"round_{number}_predictions.json", rows)
        # This process gets only completed immutable predictions. Its stdout is
        # informational; no metrics, labels or exit output enter the API bodies.
        subprocess.run([sys.executable, str(Path(__file__)), "score", "--output", str(output),
                        "--round", str(number)], cwd=ROOT, check=True)
        save(output / "inference_states.json", states)
        if calls.stopped or not any(s["active"] for s in states.values()):
            break
    calls.stopped = True
    calls.persist()
    verify_sources(plan)
    save(output / "completion.json", {"finished_utc": now(), "post_calls": len(calls.rows),
          "targets": len(states), "actual_rounds": {k: len(s["history"]) for k, s in states.items()},
          "statuses": {k: s["status"] for k, s in states.items()}, "round_snapshots_written": number,
          "new_charges": {account: str(sum(Decimal(r["charge"]) for r in calls.rows if r["account"] == account))
                          for account in ALLOWANCE},
          "unknown_reserved_calls": sum(r["charge_kind"] == "unknown_reserved" for r in calls.rows)})
    print(json.dumps(read(output / "completion.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--reuse", type=Path, help="closed exact-wire source for successful request reuse")
    parser.add_argument("--selection-manifest", type=Path, help="frozen mask-only Gate Training selection")
    parser.add_argument("--openrouter-allowance", type=Decimal)
    parser.add_argument("--xai-allowance", type=Decimal)
    parser.add_argument("--aliyun-allowance", type=Decimal)
    parser.add_argument("--round", type=int)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        amounts = (args.openrouter_allowance, args.xai_allowance, args.aliyun_allowance)
        if any(a is not None for a in amounts) and not all(a is not None for a in amounts):
            raise ValueError("specify all three account allowances together")
        allowances = dict(zip(ALLOWANCE, amounts, strict=True)) if all(a is not None for a in amounts) else None
        prepare(args.output, args.previous, adapter, args.reuse, args.selection_manifest, allowances)
    elif args.command == "execute":
        execute(args.output, adapter)
    else:
        score_round(args.output, adapter, args.round)
