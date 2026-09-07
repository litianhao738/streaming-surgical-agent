"""Isolated Training-only H0 -> three-judge panel -> local patch experiment.

Commands prepare/smoke/run/score are explicit; no automatic request retries.
The original baseline and older research drivers are never modified.
"""
import argparse
import ast
import base64
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from threading import RLock

import requests
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_grounded_api_pipeline import (
    TASK_ATTRS,
    _evaluation_target,
    _task_masks,
    score_saved,
)
from scripts.run_presence_review_trial import _mask_only, safe_wire, wire_body
from surgical_agent.api.credentials import assert_secret_absent, resolve_api_key
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.config.loader import load_api_config
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.final_only import final_only_schema
from surgical_agent.perception.joint_api_vlm import JointPerceptionRequestBuilder
from surgical_agent.perception.main_h0 import _LABEL_BOUNDARY, load_main_h0_prompt
from surgical_agent.perception.ontology_prompt import (
    _TASK_NAMES,
    load_prompt_ontology_text,
)
from surgical_agent.research.verification.prior_panel import (
    COMPONENTS,
    TASKS,
    build_universe,
    digest,
    fit_prior,
    labels,
    normalize_review,
    run_arm,
    video_counts,
)

CONTRACTS = ROOT / "docs/protocols/h0_prior_panel_v1"
PREPARED = ROOT / "artifacts/research/h0_prior_panel_protocol_20260907"
HISTORICAL = ROOT / "artifacts/preflight/verifier_goal_training_collect_20260907"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL_TAGS = [("google/gemini-2.5-flash-lite", "google-ai-studio"),
              ("openai/gpt-4.1-mini", "openai"),
              ("anthropic/claude-haiku-4.5", "anthropic"),
              ("qwen/qwen3.8-max-0902", "alibaba")]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    atomic_write_json(path, value)


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime_sources(source_hashes):
    """Exclude unrelated training engines, retain actual imported dependencies."""
    loaded = set()
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            try:
                loaded.add(str(Path(path).resolve().relative_to(ROOT)))
            except ValueError:
                pass
    prefixes = ("src/surgical_agent/api/", "src/surgical_agent/perception/", "src/surgical_agent/config/",
                "src/surgical_agent/data/", "src/surgical_agent/evaluation/", "src/surgical_agent/artifacts/",
                "src/surgical_agent/research/verification/", "src/surgical_agent/research/signals/",
                "docs/protocols/", "configs/perception/")
    return {p: h for p, h in source_hashes.items()
            if p in loaded or p.replace("\\", "/").startswith(prefixes) or not p.endswith(".py")}


def truth_row(resolved):
    target = _evaluation_target(resolved)
    mask = _task_masks(target)
    gt = {}
    for q, attr in TASK_ATTRS.items():
        value = getattr(target, attr) if mask[q] else None
        gt[q] = ([value] if q == "phase" else list(value)) if mask[q] else None
    return {"video_id": resolved.inference.video_id, "frame_id": resolved.inference.target_frame_id,
            "gt": gt, "mask": mask}


def source_paths(resolved, root):
    result = []
    for name in ("annotation_source", "phase_source", "frame_action_source", "manifest_source"):
        value = getattr(resolved.provenance, name)
        if value:
            p = Path(value)
            p = p if p.is_absolute() else root / p
            if p.is_file():
                result.append(p.resolve())
    return result


def build_base(adapter, selected):
    sample = next((s for s in adapter.iter_inference_video(selected["video_id"])
                   if s.target_frame_id == selected["frame_id"]), None)
    if sample is None or sample.source_split is not DatasetSplit.TRAINING:
        raise ValueError("only exact canonical Training target accepted")
    if list(sample.causal_frame_ids) != selected["causal_frame_ids"]:
        raise ValueError("causal frame sequence changed")
    for item in selected["images"]:
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("source image changed")
    config = load_api_config(ROOT / "configs/perception/joint_openrouter_h0.yaml")
    loaded = CausalApiMediaLoader().load(sample)
    context = CausalPerceptionContextBuilder(max_frames=3, max_images=3, selection_strategy="fixed_all",
                                           history_image_detail="low", target_image_detail="high")
    base = JointPerceptionRequestBuilder(config=config).build(context.build(
        loaded.runtime_sample, loaded.frames, workflow_snapshot={}, memory_snapshot={},
        prior_finalized_prediction=None))
    if base.payload["system_text"] != load_main_h0_prompt():
        raise ValueError("H0 prompt drift")
    wire_body(base)  # validates original model/routing/generation envelope
    return base


def pricing_snapshot():
    results = []
    for model, tag in MODEL_TAGS:
        url = f"https://openrouter.ai/api/v1/models/{model}/endpoints"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        body = r.json()
        endpoint = next(e for e in body["data"]["endpoints"] if e["tag"] == tag)
        if endpoint["status"] != 0:
            raise ValueError("fixed provider unavailable")
        cap = 4096 if tag == "alibaba" else 16384
        required = {"temperature", "max_tokens", "response_format", "structured_outputs"}
        if not required <= set(endpoint["supported_parameters"]) or endpoint["max_completion_tokens"] < cap:
            raise ValueError("fixed endpoint incompatible")
        rates = {k: Decimal(str(v)) for k, v in endpoint["pricing"].items()}
        if any(not v.is_finite() or v < 0 for v in rates.values()):
            raise ValueError("invalid price")
        known = {"prompt", "completion", "image", "internal_reasoning", "discount", "web_search",
                 "audio", "input_audio_cache", "input_cache_read", "input_cache_write", "input_cache_write_1h"}
        if any(v for k, v in rates.items() if k not in known):
            raise ValueError("unknown billable service")
        # Conservative full-context envelope. Images are charged again at their
        # listed rate over the entire context; native reasoning is also reserved
        # over the entire context, even when included in completion usage.
        # Audio/search/tools/cache_control are never sent by this driver.
        input_rate = max(rates.get(k, Decimal(0)) for k in
                         ("prompt", "input_cache_read", "input_cache_write", "input_cache_write_1h"))
        context = Decimal(endpoint["context_length"])
        reserve = (context * (input_rate + rates.get("image", Decimal(0))
                              + rates.get("internal_reasoning", Decimal(0)))
                   + cap * rates["completion"] + Decimal("0.001"))
        results.append({"model": model, "tag": tag, "provider_name": endpoint["provider_name"],
                        "max_output": cap, "reserve_usd": str(reserve), "endpoint": endpoint,
                        "retrieved_utc": now(), "url": url, "response_sha256": digest(body)})
    return results


def prepare(args):
    import torch
    torch.set_num_threads(2)
    if args.output.exists():
        raise ValueError("prepare requires a fresh output directory")
    args.output.mkdir(parents=True)
    proposal = read(PREPARED / "proposed_training_images.json")
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    selected = deepcopy(proposal["selection"])
    hashes, counts = {}, {}
    for entry in adapter.entries.values():
        if entry.split is not DatasetSplit.TRAINING:
            continue
        rows = []
        for resolved in adapter.iter_video(entry.video_id):
            rows.append(truth_row(resolved))
            for p in source_paths(resolved, args.dataset_root):
                if str(p) not in hashes:
                    hashes[str(p)] = sha256_file(p)
        counts[entry.video_id] = video_counts(rows)
        print(json.dumps({"prepared_training_statistics": entry.video_id, "frames": len(rows)}), flush=True)
    save(args.output / "training_sufficient_statistics.json", counts)
    for video in sorted({s["video_id"] for s in selected}):
        save(args.output / "priors" / f"{video}.json", fit_prior(counts, video))
    historical_rows = read(HISTORICAL / "predictions.json")
    old_selection = read(HISTORICAL / "plan.json")["selection"]
    for row in selected:
        if any(r["video_id"] == row["video_id"] and abs(r["frame_id"] - row["frame_id"]) <= 50
               for r in historical_rows):
            raise ValueError("new target overlaps historic experiment")
        base = build_base(adapter, row)
        row["request_metadata"] = canonical_request_metadata(base).to_mapping()
        resolved = list(adapter.iter_video(row["video_id"], frame_ids=[row["frame_id"]]))
        if len(resolved) != 1:
            raise ValueError("canonical GT availability does not cover selected image")
        row["task_masks"] = _mask_only(resolved[0])
        for image in row["images"]:
            hashes[image["path"]] = image["sha256"]
    smoke = deepcopy(old_selection[0])
    smoke_base = build_base(adapter, smoke)
    if canonical_request_metadata(smoke_base).to_mapping() != smoke["request_metadata"]:
        raise ValueError("cached smoke H0 request binding differs")
    smoke_h0 = next(r["h0"] for r in historical_rows
                    if r["video_id"] == smoke["video_id"] and r["frame_id"] == smoke["frame_id"])
    save(args.output / "smoke_input.json", {"selection": smoke, "h0": smoke_h0})
    for image in smoke["images"]:
        hashes[image["path"]] = image["sha256"]
    # Freeze all runtime Python/config dependencies, excluding credentials/artifacts.
    sources = list((ROOT / "src/surgical_agent").rglob("*.py"))
    sources += [Path(__file__), ROOT / "scripts/run_presence_review_trial.py",
                ROOT / "scripts/run_grounded_api_pipeline.py", ROOT / "scripts/run_diff_review_trial.py",
                ROOT / "configs/perception/joint_openrouter_h0.yaml", ROOT / "requirements-prior-panel.txt"]
    sources += list(CONTRACTS.glob("*"))
    sources += list((ROOT / "src/surgical_agent/perception/prompts").glob("*.txt"))
    sources += [ROOT / "src/surgical_agent/research/signals/resources/ivt_components_v1.csv"]
    source_hashes = {}
    for p in sources:
        source_hashes[str(p.relative_to(ROOT))] = sha256_file(p)
        destination = args.output / "frozen_source" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    carry_cost, carry_calls = Decimal(0), 0
    predecessors = []
    for previous in args.previous_run:
        previous_plan, ledger = read(previous / "plan.json"), read(previous / "budget.json")
        for record in ledger["calls"].values():
            if record["state"] not in {"NOT_SENT", "RESERVED"}:
                carry_calls += 1
                carry_cost += Decimal(record["cost_usd"] if record["cost_usd"] is not None else record["reserve_usd"])
        predecessors.append({"path": str(previous.resolve()), "plan_sha256": digest(previous_plan),
                             "ledger_sha256": sha256_file(previous / "budget.json")})
    plan = {"version": "h0_prior_panel_runtime_v7", "created_utc": now(), "split": "Training",
            "budget_usd": str(Decimal("15.00") - carry_cost), "max_calls": 188 - carry_calls, "selection": selected,
            "total_authorized_usd": "15.00", "predecessors": predecessors,
            "carryover_liability_usd": str(carry_cost), "carryover_calls": carry_calls,
            "models": pricing_snapshot(), "source_sha256": source_hashes, "data_sha256": hashes,
            "runtime_source_sha256": runtime_sources(source_hashes),
            "design_sha256": sha256_file(CONTRACTS / "protocol.json"),
            "protocol_changes": ["User authorized execution with $15 instead of proposed $20",
                                 "Stable feedback fingerprint excludes coordinate jitter and free-text changes",
                                 "Google wire schema omits numeric/length/pattern constraints; full original local validation and prompt retained",
                                 "Google AI Studio fixed route replaces rate-limited Vertex EU; same model and prices",
                                 "Repair receives deterministic allowed edit roles; judges still never receive prior origin metadata",
                                 "Per-request judge schema fixes assessment count to |U|; full coverage remains mandatory locally",
                                 "Main panel sends its three pre-reserved independent requests concurrently; no vote sharing",
                                 "Explicit continuation only for never-dispatched H0/arms; an attempted failed panel is retained without retry",
                                 "Explicit revised compatibility run: predecessor attempts counted, unsettled rejected-schema cost reserved in full"],
            "historical_predictions_sha256": sha256_file(HISTORICAL / "predictions.json"),
            "runtime_tests_required": True}
    resume_artifacts = []
    if args.resume_source is not None:
        previous = args.resume_source.resolve()
        if previous not in {p.resolve() for p in args.previous_run}:
            raise ValueError("continuation must account for source-run costs")
        old_plan = read(previous / "plan.json")
        completion = read(previous / "prediction_completion.json")
        if sha256_file(previous / "predictions.json") != completion["prediction_sha256"]:
            raise ValueError("continuation predictions changed")
        if old_plan["selection"] != selected:
            raise ValueError("continuation selection/request bindings differ")
        if read(previous / "smoke_result.json")["passed"] is not True:
            raise ValueError("continuation requires fully successful source compatibility smoke")
        for relative, expected in old_plan["source_sha256"].items():
            if (relative in plan["runtime_source_sha256"] and
                    relative.replace("\\", "/").startswith(("docs/protocols/", "src/")) and source_hashes[relative] != expected):
                raise ValueError("continuation inference contract changed")
        # Reuse smoke only if the request construction functions are identical,
        # including wire adaptation, prompts/ontology and dynamic item counts.
        def function_nodes(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef)}
        old_functions = function_nodes(previous / "frozen_source/scripts/run_prior_panel_trial.py")
        new_functions = function_nodes(Path(__file__))
        for name in ("build_base", "request_body", "provider_schema", "packet_for", "repair_roles", "fixture_universe"):
            if old_functions[name] != new_functions[name]:
                raise ValueError("continuation request builder changed; new smoke required")
        if [(m["model"], m["tag"]) for m in old_plan["models"]] != [(m["model"], m["tag"]) for m in plan["models"]]:
            raise ValueError("continuation models/routes differ")
        history = merge_dispatch_history([read(p / "budget.json")["calls"] for p in args.previous_run])
        resume = {"source": str(previous), "rows": read(previous / "predictions.json"),
                  "source_calls": history,
                  "source_prediction_sha256": completion["prediction_sha256"]}
        save(args.output / "resume_input.json", resume)
        smoke_result = read(previous / "smoke_result.json")
        smoke_result["reused_from"] = str(previous)
        save(args.output / "smoke_result.json", smoke_result)
        for child in (previous / "targets").iterdir():
            shutil.copytree(child, args.output / "targets" / child.name)
        plan["resume_source"] = str(previous)
        resume_artifacts = [args.output / "resume_input.json", args.output / "smoke_result.json"]
    plan["artifact_sha256"] = {str(p.relative_to(args.output)): sha256_file(p) for p in
                              [args.output / "smoke_input.json", args.output / "training_sufficient_statistics.json",
                               *sorted((args.output / "priors").glob("*.json")), *resume_artifacts]}
    save(args.output / "plan.json", plan)
    # Zero-API coverage on prior frozen H0, distinct from the eight new targets.
    pools = []
    for row in historical_rows:
        if row.get("h0") is None:
            continue
        prior = fit_prior(counts, row["video_id"])
        pools.append({"video_id": row["video_id"], "frame_id": row["frame_id"], "h0": row["h0"],
                      "pools": {arm: build_universe(row["h0"], video_id=row["video_id"], prior=prior if arm != "none" else None,
                                                     global_only=arm == "global")
                                for arm in ("none", "global", "phase_global")}})
    save(args.output / "offline_candidate_pools.json", pools)
    truth = {}
    for video in sorted({r["video_id"] for r in pools}):
        frames = [r["frame_id"] for r in pools if r["video_id"] == video]
        for resolved in adapter.iter_video(video, frame_ids=frames):
            truth[video, resolved.inference.target_frame_id] = truth_row(resolved)
    coverage = {}
    for arm in ("none", "global", "phase_global"):
        valid = hits = extra = missed = 0
        by_video = {}
        for row in pools:
            gt = truth[row["video_id"], row["frame_id"]]
            if not gt["mask"]["ivt"]:
                continue
            u = {p["label_id"] for p in row["pools"][arm]["propositions"] if p["task"] == "ivt"}
            new = u - set(row["h0"]["ivt"])
            correct = new & set(gt["gt"]["ivt"])
            valid += 1
            hits += len(correct)
            extra += len(new)
            missed += len(set(gt["gt"]["ivt"]) - set(row["h0"]["ivt"]))
            by_video[row["video_id"]] = by_video.get(row["video_id"], 0) + len(correct)
        coverage[arm] = {"valid_targets": valid, "new_ivt_hits": hits, "new_ivt_candidates": extra,
                         "h0_missed_ivt": missed, "new_precision": hits / extra if extra else None,
                         "miss_coverage": hits / missed if missed else None, "hits_by_video": by_video}
    coverage["note"] = "Frozen prior parameters. Coverage is not panel repair accuracy; no GT-selected changes."
    save(args.output / "offline_candidate_coverage.json", coverage)
    print(json.dumps({"status": "PREPARED", "targets": len(selected), "coverage": coverage,
                      "paid_calls": 0}, ensure_ascii=False), flush=True)


def assert_frozen(output, plan):
    for relative, expected in plan.get("runtime_source_sha256", plan["source_sha256"]).items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError("runtime source changed since preflight: " + relative)
    for path, expected in plan["data_sha256"].items():
        if sha256_file(path) != expected:
            raise ValueError("dataset source changed")
    for relative, expected in plan["artifact_sha256"].items():
        if sha256_file(output / relative) != expected:
            raise ValueError("fitted prior or smoke input changed")


class DirectBudget:
    """Single-process durable batch reservations. Crash liabilities remain charged."""
    def __init__(self, output, plan, secret, sender=None):
        self.output, self.plan, self.secret = output, plan, secret
        self.sender = sender or requests.post
        self.lock = RLock()
        self.path = output / "budget.json"
        self.state = read(self.path) if self.path.exists() else {
            "budget_usd": plan["budget_usd"], "max_calls": plan["max_calls"], "calls": {}, "stop": None}
        self.models = {m["model"]: m for m in plan["models"]}

    def reserve(self, names_models):
        if self.state["stop"]:
            return False
        if any(name in self.state["calls"] for name, _ in names_models):
            raise ValueError("duplicate dispatch name")
        liability = sum(Decimal(r["cost_usd"] if r["cost_usd"] is not None else r["reserve_usd"])
                        for r in self.state["calls"].values())
        extra = sum(Decimal(self.models[model]["reserve_usd"]) for _, model in names_models)
        if liability + extra > Decimal(self.state["budget_usd"]) or len(self.state["calls"]) + len(names_models) > self.state["max_calls"]:
            self.state["stop"] = "BUDGET_RESERVATION_STOP"
            save(self.path, self.state)
            return False
        for name, model in names_models:
            self.state["calls"][name] = {"model": model, "state": "RESERVED", "cost_usd": None,
                                          "reserve_usd": self.models[model]["reserve_usd"]}
        save(self.path, self.state)
        return True

    def call(self, name, body, schema):
        with self.lock:
            if self.state["stop"]:
                return None
            row = self.state["calls"][name]
            if row["state"] != "RESERVED" or row["model"] != body["model"]:
                raise ValueError("request absent from reservation or already sent")
            model = self.models[row["model"]]
            if (body["provider"]["only"] != [model["tag"]] or body["provider"]["allow_fallbacks"]
                    or not body["provider"]["require_parameters"] or body["max_tokens"] != model["max_output"]
                    or body.get("tools") or body.get("plugins") or body.get("models")):
                raise ValueError("request changed reserved endpoint or service")
            directory = self.output / "calls" / name
            directory.mkdir(parents=True, exist_ok=False)
            save(directory / "request.json", {"wire": safe_wire(body), "wire_sha256": digest(body)})
            row.update(state="DISPATCHED", started_utc=now())
            save(self.path, self.state)
            row = deepcopy(row)
        payload, started = None, time.perf_counter()
        stop_reason = None
        try:
            r = self.sender(ENDPOINT, json=body, headers={"Authorization": "Bearer " + self.secret.reveal()},
                            timeout=(15, 240), allow_redirects=False)
            save(directory / "http_response.json", {"status_code": r.status_code,
                  "body": r.text.replace(self.secret.reveal(), "[REDACTED]")})
            native = r.json()
            usage = native.get("usage") or {}
            cost = usage.get("cost")
            if type(cost) in (int, float) and Decimal(str(cost)).is_finite() and cost >= 0:
                row["cost_usd"] = str(cost)
            row.update(http_status=r.status_code, usage=usage, provider=native.get("provider"),
                       returned_model=native.get("model"), generation_id=native.get("id"))
            if r.status_code != 200:
                raise ValueError("HTTP failure")
            if native.get("model") != row["model"] or native.get("provider") != model["provider_name"]:
                stop_reason = "MODEL_OR_PROVIDER_DRIFT"
                raise ValueError("identity mismatch")
            if native["choices"][0].get("finish_reason") != "stop":
                raise ValueError("incomplete model output")
            payload = json.loads(native["choices"][0]["message"]["content"])
            Draft202012Validator(schema).validate(payload)
            save(directory / "parsed.json", payload)
            row["state"] = "OK"
        except (requests.RequestException, OSError, ValueError, TypeError, KeyError, IndexError, ValidationError) as exc:
            # Never persist exception text: it may contain a key or full headers.
            payload = None
            row.update(state="FAILED", error=type(exc).__name__)
        row["latency_seconds"] = round(time.perf_counter() - started, 3)
        if row["cost_usd"] is None:
            stop_reason = "UNKNOWN_COST_STOP"
        elif Decimal(row["cost_usd"]) > Decimal(row["reserve_usd"]):
            stop_reason = "COST_EXCEEDED_RESERVATION"
        with self.lock:
            self.state["calls"][name] = row
            if stop_reason:
                self.state["stop"] = stop_reason
            save(directory / "result.json", row)
            save(self.path, self.state)
        print(json.dumps({"call": name, "status": row["state"], "cost_usd": row["cost_usd"],
                          "latency_seconds": row["latency_seconds"], "stop": self.state["stop"]}), flush=True)
        return payload

    def release_unsent(self):
        for row in self.state["calls"].values():
            if row["state"] == "RESERVED":
                row.update(state="NOT_SENT", cost_usd="0")
        save(self.path, self.state)


def provider_schema(schema, tag):
    """Google rejects the large bounded grammar; preserve all local constraints."""
    if not tag.startswith("google-"):
        return deepcopy(schema)
    removed = {"minLength", "maxLength", "minItems", "maxItems", "minimum", "maximum", "pattern"}

    def simplify(value):
        if isinstance(value, dict):
            return {k: simplify(v) for k, v in value.items() if k not in removed}
        if isinstance(value, list):
            return [simplify(v) for v in value]
        return value
    return simplify(schema)


def request_body(base, model, schema, template, packet):
    schema = deepcopy(schema)
    if "assessments" in schema["properties"]:
        count = len(packet["propositions"])
        schema["properties"]["assessments"].update(minItems=count, maxItems=count)
    content = [{"type": "text", "text": json.dumps(packet, ensure_ascii=False, sort_keys=True)}]
    for im, detail in zip(base.images, base.payload["image_details"], strict=True):
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{im.mime_type};base64," + base64.b64encode(im.content).decode(), "detail": detail}})
    system = template + "\n" + load_prompt_ontology_text() + _LABEL_BOUNDARY + "\nResponse schema:\n" + json.dumps(schema)
    body = {"model": model["model"], "messages": [{"role": "system", "content": system},
             {"role": "user", "content": content}], "temperature": 0, "max_tokens": model["max_output"],
            "stream": False, "provider": {"only": [model["tag"]], "order": [model["tag"]],
                                         "allow_fallbacks": False, "require_parameters": True},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema["properties"]["schema_version"]["const"], "strict": True,
                "schema": provider_schema(schema, model["tag"])}}}
    if model["tag"] == "alibaba":
        body["reasoning"] = {"effort": "low"}
    return body


def packet_for(base, selection, universe, current):
    return {"video_id": selection["video_id"], "target_frame_id": selection["frame_id"],
            "images": [{"ref": im.identifier, "sha256": hashlib.sha256(im.content).hexdigest(),
                        "frame_id": f, "relative_seconds": (f - selection["frame_id"]) / 25,
                        "detail": d} for im, f, d in zip(base.images, selection["causal_frame_ids"],
                                                          base.payload["image_details"], strict=True)],
            "current_prediction": current, "propositions": universe["propositions"]}


def repair_roles(universe):
    return {p["proposition_id"]: "IVT_EDIT" if p["task"] == "ivt" else
            "INDEPENDENT_COMPONENT" if universe["eligibility"][p["proposition_id"]]["independent"] else
            "IVT_COMPONENT_ONLY" for p in universe["propositions"]}


def fixture_universe(h0):
    u = build_universe(h0, video_id="ENGINEERING_FIXTURE")
    existing = {(p["task"], p["label_id"]) for p in u["propositions"]}
    for c in range(100):
        if len(u["propositions"]) == 48:
            break
        if ("ivt", c) not in existing:
            comp = COMPONENTS[c]
            u["propositions"].append({"proposition_id": f"p{len(u['propositions']) + 1:03}",
                "task": "ivt", "label_id": c, "components": comp,
                "name": "/".join(_TASK_NAMES[k][v] for k, v in comp.items())})
    return u


def was_dispatched(calls, prefix):
    return any((name == prefix or name.startswith(prefix + "/")) and
               row["state"] not in {"NOT_SENT", "RESERVED"} for name, row in calls.items())


def merge_dispatch_history(histories):
    """Retain transitive executions when a continuation itself is interrupted."""
    combined = {}
    for history in histories:
        for name, row in history.items():
            if name.startswith("smoke/") or row["state"] in {"NOT_SENT", "RESERVED"}:
                continue
            if name in combined:
                raise ValueError("the same semantic stage was dispatched in multiple source runs")
            combined[name] = row
    return combined


def execute(args, smoke=False):
    import torch
    torch.set_num_threads(2)
    plan = read(args.output / "plan.json")
    assert_frozen(args.output, plan)
    # No paid calls yet: recheck live prices against the frozen envelope.
    live = pricing_snapshot()
    if any(a["endpoint"]["pricing"] != b["endpoint"]["pricing"] or a["reserve_usd"] != b["reserve_usd"]
           for a, b in zip(live, plan["models"], strict=True)):
        raise ValueError("pricing changed; fresh preflight required")
    secret = resolve_api_key(api_key=None, api_key_file=args.api_key_file)
    name = "smoke" if smoke else "run"
    with (args.output / (name + ".lock")).open("x", encoding="utf-8") as handle:
        handle.write("Single invocation; no implicit redispatch.\n")
    budget = DirectBudget(args.output, plan, secret)
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    js, ps = read(CONTRACTS / "judge_response.schema.json"), read(CONTRACTS / "repair_response.schema.json")
    jp, pp = [(CONTRACTS / f"{role}_prompt.txt").read_text(encoding="utf-8") for role in ("judge", "repair")]
    models = plan["models"]
    if smoke:
        source = read(args.output / "smoke_input.json")
        s, h0 = source["selection"], source["h0"]
        base = build_base(adapter, s)
        u = fixture_universe(h0)
        packet = packet_for(base, s, u, h0)
        packet["engineering_fixture"] = "48 legal ontology propositions; not semantic evaluation or a prior-generated pool"
        save(args.output / "smoke_packet.json", packet)
        requested = [(f"smoke/judge_{i}", m["model"]) for i, m in enumerate(models[:3])]
        requested.append(("smoke/repair", models[3]["model"]))
        if not budget.reserve(requested):
            raise ValueError("insufficient smoke reservation")
        valid = []
        for i, m in enumerate(models[:3]):
            response = budget.call(f"smoke/judge_{i}", request_body(base, m, js, jp, packet), js)
            try:
                normalize_review(response, u, [im.identifier for im in base.images], js)
                valid.append(True)
            except (ValueError, TypeError, KeyError, ValidationError):
                valid.append(False)
        repair_packet = {**packet, "h0": h0, "last_accepted": h0, "issues": [],
                         "instruction": "No actionable issue provided; return an empty edit list."}
        patch = budget.call("smoke/repair", request_body(base, models[3], ps, pp, repair_packet), ps)
        valid.append(patch is not None and patch["edits"] == [])
        budget.release_unsent()
        save(args.output / "smoke_result.json", {"passed": all(valid), "checks": valid,
                                                  "engineering_only": True, "capacity": 48})
    else:
        if read(args.output / "smoke_result.json")["passed"] is not True:
            raise ValueError("all four provider smoke checks must pass before main experiment")
        resume = read(args.output / "resume_input.json") if plan.get("resume_source") else {"rows": [], "source_calls": {}}
        cached = {(r["video_id"], r["frame_id"]): r for r in resume["rows"]}
        results = []
        for index, s in enumerate(plan["selection"]):
            key = s["key"]
            previous_row = cached.get((s["video_id"], s["frame_id"]))
            row = {"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": None, "arms": {}}
            base = build_base(adapter, s)
            if canonical_request_metadata(base).to_mapping() != s["request_metadata"]:
                raise ValueError("H0 request hash drift")
            already_h0 = was_dispatched(resume["source_calls"], f"{key}/h0")
            if already_h0:
                if previous_row is None:
                    raise ValueError("attempted H0 has no preserved result")
                h0_payload = previous_row["h0"]
                row["h0_reused_from"] = resume["source"]
            elif budget.reserve([(f"{key}/h0", models[3]["model"])]):
                h0_payload = budget.call(f"{key}/h0", wire_body(base), final_only_schema())
            else:
                h0_payload = None
            if h0_payload is not None:
                row["h0"] = labels(h0_payload)
                prior = read(args.output / "priors" / f"{s['video_id']}.json")
                order = ("no_prior", "prior") if index % 2 == 0 else ("prior", "no_prior")
                for arm in order:
                    if was_dispatched(resume["source_calls"], f"{key}/{arm}"):
                        row["arms"][arm] = deepcopy(previous_row["arms"][arm])
                        row.setdefault("arms_reused_from", {})[arm] = resume["source"]
                        continue
                    u = build_universe(row["h0"], video_id=s["video_id"], prior=prior if arm == "prior" else None)
                    destination = args.output / "targets" / key / arm
                    save(destination / "universe.json", u)

                    def panel_call(round_no, current, key=key, arm=arm, base=base, s=s, u=u):
                        names = [(f"{key}/{arm}/r{round_no}_judge_{i}", m["model"]) for i, m in enumerate(models[:3])]
                        if round_no == 1 and not budget.reserve(names):
                            return None
                        packet = packet_for(base, s, u, current)
                        bodies = [request_body(base, m, js, jp, packet) for m in models[:3]]
                        with ThreadPoolExecutor(max_workers=3) as pool:
                            futures = [pool.submit(budget.call, n, body, js)
                                       for (n, _), body in zip(names, bodies, strict=True)]
                            return [future.result() for future in futures]

                    def patch_call(round_no, current, issues, key=key, arm=arm, base=base, s=s, u=u, row=row):
                        n = f"{key}/{arm}/p{round_no}"
                        reserve = [(n, models[3]["model"])] + [
                            (f"{key}/{arm}/r{round_no + 1}_judge_{i}", m["model"]) for i, m in enumerate(models[:3])]
                        if not budget.reserve(reserve):
                            return None
                        packet = packet_for(base, s, u, current)
                        packet.update(h0=row["h0"], last_accepted=current, issues=issues)
                        packet["allowed_edit_roles"] = repair_roles(u)
                        return budget.call(n, request_body(base, models[3], ps, pp, packet), ps)

                    row["arms"][arm] = run_arm(row["h0"], u, panel_call, patch_call, js, ps,
                        [im.identifier for im in base.images], lambda stage, value, destination=destination: save(destination / f"{stage}.json", value))
                    budget.release_unsent()
            results.append(row)
            save(args.output / "predictions.json", results)
        save(args.output / "prediction_completion.json", {"targets": len(results),
             "prediction_sha256": sha256_file(args.output / "predictions.json"), "completed_utc": now()})
    assert_secret_absent(secret, args.output.rglob("*.json"))


def score(args):
    completed = read(args.output / "prediction_completion.json")
    if sha256_file(args.output / "predictions.json") != completed["prediction_sha256"]:
        raise ValueError("predictions changed after completion")
    rows = read(args.output / "predictions.json")
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3)
    reports = {}
    for arm in ("no_prior", "prior"):
        for round_index in range(3):
            values = [{"video_id": r["video_id"], "frame_id": r["frame_id"], "h0": r["h0"],
                       "h1": None, "final": r["arms"].get(arm, {}).get("snapshots", [r["h0"]] * 3)[round_index]}
                      for r in rows]
            report, scored = score_saved(adapter, values)
            edits = {q: {"beneficial_add": 0, "harmful_add": 0, "beneficial_remove": 0, "harmful_remove": 0}
                     for q in TASKS}
            for r in scored:
                if r["h0"] is None or r["final"] is None:
                    continue
                for q in TASKS:
                    if not r["mask"][q]:
                        continue
                    gt, before, after = set(r["gt"][q]), set(r["h0"][q]), set(r["final"][q])
                    for c in after - before:
                        edits[q]["beneficial_add" if c in gt else "harmful_add"] += 1
                    for c in before - after:
                        edits[q]["harmful_remove" if c in gt else "beneficial_remove"] += 1
            reports[f"{arm}_D{round_index + 1}"] = {"comparison": report, "edits": edits}
            save(args.output / "scored" / f"{arm}_D{round_index + 1}.json", scored)
    save(args.output / "metrics.json", reports)
    print(json.dumps({"status": "SCORED", "targets": len(rows), "reports": list(reports)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "smoke", "run", "score"))
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=ROOT / "docs/API.txt")
    parser.add_argument("--previous-run", type=Path, action="append", default=[],
                        help="Explicit prior compatibility attempts charged against the same $15/188 cap")
    parser.add_argument("--resume-source", type=Path,
                        help="Only finish never-dispatched stages; preserve attempted failed arms and cached H0")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "score":
        score(args)
    else:
        execute(args, smoke=args.mode == "smoke")
