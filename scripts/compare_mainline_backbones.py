"""Six paired Training targets, three base models, unchanged 13-call mainline.

Model selection development experiment only. No Testing labels or Gate dataset.
The base model fills H0, proposal and Phase recommendation; all five seats fixed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import prior_gated_mainline_release as release
from scripts import run_prior_gated_joint_mainline as main
from scripts.run_repair_revision_trial import parse_review_json
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.schemas import DatasetSplit

old = main.old
PROFILE = "paired_mainline_backbones_training6_v1"
COHORT = {"VID103": 2, "VID23": 2, "VID31": 1, "VID96": 1}
MODELS = {
    "sol": {"model": "openai/gpt-5.6-sol", "tag": "openai", "provider": "OpenAI"},
    "gemini": {"model": "google/gemini-3.8-flash", "tag": "google-ai-studio", "provider": "Google AI Studio"},
    "qwen": {"model": "qwen/qwen3.8-max", "tag": "alibaba", "provider": "Alibaba"},
}
LIMITS = {"openrouter_usd": "4", "aliyun_cny": "0.7", "xai_usd": "0"}  # per model
VALID = {"JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"}
STANDARD = {
    "selection": "Training only; VID103/23 two time quantiles each, VID31/96 one each; all five masks valid, complete causal three images, >175 raw-frame gap. No selection by prediction or GT label value.",
    "rank": "First require six predicted targets. Rank eligible models by final five-head mean micro-F1, then IVT F1, then Phase F1; exact ties use lower measured USD occupancy, then elapsed time. If none complete, no model is qualified. Report all call failures, H0 quality and repair harm separately.",
    "interpretation": "Six-target development pilot only, not independent confirmation or significance. Same semantic prompts/images/schema and role-specific reasoning effort; Sol omits unsupported temperature. No thresholds tuned. Do not replace frozen base model automatically.",
    "failures": "No retries, no replacement or removal of failed targets; missing predictions score as empty sets. Deterministic configuration failure stops only that model.",
    "prior": "For each target video fit the identical prior from the other Training videos, explicitly excluding the whole query video; no Validation or Testing annotations.",
}


def base_body(body, name):
    out = deepcopy(body)
    spec = MODELS[name]
    out["model"] = spec["model"]
    out["provider"] = {"only": [spec["tag"]], "order": [spec["tag"]],
                       "allow_fallbacks": False, "require_parameters": True}
    if name == "sol":
        out.pop("temperature", None)  # OpenAI reasoning route does not support it.
    return out


class ModelCalls(main.MainlineCalls):
    def __init__(self, output, name, plan):
        self.name = name
        rates = {**old.joint.roster.RATES, "base": plan["base_rates"][name]}
        providers = {**old.joint.roster.PROVIDERS, "base": MODELS[name]["provider"],
                     MODELS[name]["model"]: MODELS[name]["provider"]}
        super().__init__(output, limits={k: Decimal(v) for k, v in LIMITS.items()}, rates=rates,
                         providers=providers, max_calls=78, reasoning_seats=("gemini",))

    def call(self, target, stage, seat, body):
        return super().call(target, stage, seat, base_body(body, self.name) if seat == "base" else body)


def annotation(video):
    return release.DATASET / "Training" / video / f"{video.lower()}.json"


def bases_for(plan):
    adapter = old.common.CholecTrack20DatasetAdapter(release.DATASET, causal_window_size=3)
    selected = {s["key"]: s for s in plan["selection"]}
    bases = {}
    for video in COHORT:
        for sample in adapter.iter_inference_video(video):
            key = f"{video}_{sample.target_frame_id}"
            if key not in selected:
                continue
            if sample.source_split is not DatasetSplit.TRAINING:
                raise ValueError("Training inference sample required")
            base = release.make_base(sample)
            for image, frozen in zip(base.images, selected[key]["images"], strict=True):
                if image.content != Path(frozen["path"]).read_bytes():
                    raise ValueError("image bytes changed")
            bases[key] = base
    if set(bases) != set(selected):
        raise ValueError("missing targets")
    return bases


def prepare(output):
    if output.exists():
        raise ValueError("fresh output required")
    old.save(output / "protocol.json", {"profile": PROFILE, "created_utc": old.now(), "cohort": COHORT,
              "models": MODELS, "standard": STANDARD, "max_calls": 234, "limits_per_model": LIMITS,
              "concurrency": "three models parallel per target; each runs fixed two panel branches; no cross-target overlap",
              "gate_data_collected": False})
    frozen = old.read(ROOT / "BEST_PIPELINE_VERSION.json")
    if frozen["version"] != main.VERSION:
        raise ValueError("wrong frozen mainline")
    changes = [rel for rel, h in frozen["runtime_source_sha256"].items() if old.sha(ROOT / rel) != h]
    if set(changes) - {"scripts/run_pipeline.py"}:
        raise ValueError("unexpected frozen source drift: " + str(changes))
    metadata, rates = {}, {}
    for name, spec in MODELS.items():
        response = requests.get("https://openrouter.ai/api/v1/models/" + spec["model"] + "/endpoints", timeout=30)
        response.raise_for_status()
        data = response.json()
        old.save(output / "provider_metadata" / f"{name}.json", data)
        endpoint = next(e for e in data["data"]["endpoints"] if e["tag"] == spec["tag"])
        required = {"reasoning", "max_tokens", "response_format", "structured_outputs"}
        if name != "sol":
            required.add("temperature")
        if endpoint["status"] != 0 or not required <= set(endpoint["supported_parameters"]):
            raise ValueError("incompatible exact model route")
        metadata[name] = endpoint
        rates[name] = [str(max([Decimal(endpoint["pricing"][field]),
                               *(Decimal(o.get(field, "0")) for o in endpoint["pricing"].get("overrides", []))]))
                       for field in ("prompt", "completion")]
    adapter = old.common.CholecTrack20DatasetAdapter(release.DATASET, causal_window_size=3)
    selection, annotation_hashes, evaluation_sources = [], {}, {}
    for video, count in COHORT.items():
        if adapter.entries[video].split is not DatasetSplit.TRAINING:
            raise ValueError("wrong split")
        resolved = list(adapter.iter_video(video))
        masks = {r.inference.target_frame_id: old.truth_row(r)["mask"] for r in resolved}
        # VID31 uses the project's explicit frame-level IVT/Phase sidecars.
        for path in old.collector.original.source_paths(resolved[0], release.DATASET):
            evaluation_sources[str(path)] = old.sha(path)
        annotation_hashes[video] = old.sha(annotation(video))
        samples = list(adapter.iter_inference_video(video))
        used = set()
        for n in range(1, count + 1):
            anchor = samples[len(samples) * n // (count + 1)].target_frame_id
            eligible = [s for s in samples if len(s.causal_frame_ids) == 3
                        and s.target_frame_id in masks and all(masks[s.target_frame_id].values())
                        and all(abs(f - other) > 175 for f in s.causal_frame_ids for other in used)]
            sample = min(eligible, key=lambda s: (abs(s.target_frame_id - anchor), s.target_frame_id))
            used.update(sample.causal_frame_ids)
            base = release.make_base(sample)
            images = []
            for fid, image in zip(sample.causal_frame_ids, base.images, strict=True):
                path = output / "images" / video / f"{fid}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(image.content)
                images.append({"frame_id": fid, "path": str(path.resolve()), "sha256": old.sha(path)})
            selection.append({"key": f"{video}_{sample.target_frame_id}", "video_id": video,
                              "frame_id": sample.target_frame_id, "anchor_frame_id": anchor,
                              "causal_frame_ids": list(sample.causal_frame_ids), "images": images})
    # Fit without ever loading the query video's rows into that video's prior builder.
    priors = {}
    for excluded in COHORT:
        counts = {}
        for v, entry in sorted(adapter.entries.items()):
            if entry.split is not DatasetSplit.TRAINING or v == excluded:
                continue
            resolved = list(adapter.iter_video(v))
            counts[v] = old.collector.video_counts([old.truth_row(r) for r in resolved])
            for path in old.collector.original.source_paths(resolved[0], release.DATASET):
                evaluation_sources[str(path)] = old.sha(path)
        prior = old.collector.fit_prior(counts, excluded)
        if excluded in prior["fit_videos"] or len(prior["fit_videos"]) != 9:
            raise ValueError("LOVO prior contaminated")
        old.save(output / "priors" / f"{excluded}.json", prior)
        priors[excluded] = old.sha(output / "priors" / f"{excluded}.json")
    source = {p.relative_to(ROOT).as_posix(): old.sha(p) for p in release.runtime_paths()}
    for rel in source:
        target = output / "frozen_source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    plan = {"profile": PROFILE, "mechanism": main.VERSION, "models": MODELS, "selection": selection,
            "reviewers": old.joint.roster.MODELS, "gate": old.GATE, "phase_threshold": 4,
            "base_rates": rates, "provider_metadata": metadata, "limits_per_model": LIMITS,
            "max_calls": 234, "source_sha256": source, "prior_sha256": priors,
            "annotation_sha256": annotation_hashes, "protocol_sha256": old.sha(output / "protocol.json"),
            "evaluation_source_sha256": evaluation_sources,
            "default_sha256": old.sha(ROOT / "DEFAULT_PIPELINE_VERSION.json"),
            "best_sha256": old.sha(ROOT / "BEST_PIPELINE_VERSION.json"),
            "credential_root": str(ROOT), "predeclared_standard": STANDARD,
            "metadata_hashes": {name: old.sha(output / "provider_metadata" / f"{name}.json") for name in MODELS}}
    old.save(output / "plan.json", plan)
    print(json.dumps({"prepared": str(output), "targets": [s["key"] for s in selection], "max_calls": 234}), flush=True)


def verify(output):
    p = old.read(output / "plan.json")
    if p["profile"] != PROFILE or p["models"] != MODELS or p["gate"] != old.GATE:
        raise ValueError("protocol changed")
    for rel, h in p["source_sha256"].items():
        if old.sha(ROOT / rel) != h or old.sha(output / "frozen_source" / rel) != h:
            raise ValueError("runtime source changed: " + rel)
    for name, h in p["prior_sha256"].items():
        path = output / "priors" / f"{name}.json"
        prior = old.read(path)
        if old.sha(path) != h or prior["excluded_video"] != name or name in prior["fit_videos"]:
            raise ValueError("prior changed")
    for s in p["selection"]:
        for image in s["images"]:
            if old.sha(image["path"]) != image["sha256"]:
                raise ValueError("image changed")
    for name, h in p["metadata_hashes"].items():
        if old.sha(output / "provider_metadata" / f"{name}.json") != h:
            raise ValueError("provider metadata changed")
    if old.sha(output / "protocol.json") != p["protocol_sha256"]:
        raise ValueError("protocol changed")
    for path, digest in p["evaluation_source_sha256"].items():
        if old.sha(path) != digest:
            raise ValueError("evaluation or prior source changed")
    return p


class Mock(old.MockCalls):
    def __init__(self, name):
        super().__init__()
        self.name, self.wires = name, {}

    def call(self, target, stage, seat, body):
        wire = base_body(body, self.name) if seat == "base" else body
        self.wires[stage, seat] = old.joint.roster.transport.redact_images(wire)
        return super().call(target, stage, seat, wire)


def preflight(output):
    plan, reports = verify(output), []
    bases = bases_for(plan)
    with old.joint.roster.lightweight_protocol():
        for selected in plan["selection"]:
            wires = {}
            for name in MODELS:
                calls = Mock(name)
                record = main.run_target(calls, bases[selected["key"]], selected,
                                         old.read(output / "priors" / f"{selected['video_id']}.json"), plan["gate"])
                if Counter(r["stage"] for r in calls.rows) != Counter(main.STAGES):
                    raise ValueError("incorrect stage count")
                if any(record["predictions"][main.PRIMARY][t] != record["predictions"]["gated_control"][t] for t in old.TASKS[:4]):
                    raise ValueError("four heads changed")
                wires[name] = calls.wires
            for key in wires["gemini"]:
                for name in MODELS:
                    ref, got = deepcopy(wires["gemini"][key]), deepcopy(wires[name][key])
                    if key[1] == "base":
                        for d in (ref, got):
                            for field in ("model", "provider", "temperature"):
                                d.pop(field, None)
                    if ref != got:
                        raise ValueError("unplanned paired request difference")
            reports.append({"key": selected["key"], "mock_calls": 39, "paired_wire_identity": True})
    old.save(output / "preflight.json", {"plan_sha256": old.sha(output / "plan.json"), "api_calls": 0, "targets": reports})
    print(json.dumps({"preflight_targets": len(reports), "api_calls": 0}), flush=True)


def execute(output):
    plan = verify(output)
    check = old.read(output / "preflight.json")
    if check["plan_sha256"] != old.sha(output / "plan.json") or len(check["targets"]) != 6:
        raise ValueError("preflight incomplete")
    bases = bases_for(plan)
    with (output / "execution.lock").open("x", encoding="utf-8") as lock:
        lock.write(old.sha(output / "plan.json"))
    rows, started, fatal = {name: [] for name in MODELS}, perf_counter(), None
    with old.joint.credential_context(plan), old.joint.roster.lightweight_protocol():
        calls = {name: ModelCalls(output / name, name, plan) for name in MODELS}
        try:
            for selected in plan["selection"]:
                def one(name, selected=selected):
                    row = {k: selected[k] for k in ("key", "video_id", "frame_id")}
                    stamp = perf_counter()
                    if calls[name].stopped:
                        row.update(status="NOT_DISPATCHED", predictions={a: None for a in main.ARMS})
                    else:
                        try:
                            record = main.run_target(calls[name], bases[selected["key"]], selected,
                                old.read(output / "priors" / f"{selected['video_id']}.json"), plan["gate"])
                        except (ValueError, TypeError, KeyError, ApiSchemaError) as exc:
                            row.update(status="TARGET_FAILED", error_type=type(exc).__name__, predictions={a: None for a in main.ARMS})
                        else:
                            old.save(output / name / "targets" / selected["key"] / "result.json", record)
                            row.update(status="PREDICTED", predictions=record["predictions"], timing=record["timing_seconds"])
                    row["seconds"] = perf_counter() - stamp
                    return name, row
                with ThreadPoolExecutor(max_workers=3) as workers:
                    for name, row in workers.map(one, MODELS):
                        rows[name].append(row)
                        old.save(output / name / "predictions.json", {"targets": rows[name]})
                        print(json.dumps({"model": name, "target": row["key"], "status": row["status"],
                              "completed": len(rows[name]), "calls": len(calls[name].rows), "seconds": row["seconds"]}), flush=True)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            for name, c in calls.items():
                c.stopped = True
                try:
                    c.persist()
                finally:
                    c.close_ledger()
                old.save(output / name / "predictions.json", {"targets": rows[name]})
            sealed = {p.relative_to(output).as_posix(): old.sha(p) for name in MODELS
                      for p in (output / name).rglob("*.json")}
            sealed["plan.json"] = old.sha(output / "plan.json")
            old.save(output / "completion.json", {"closed_utc": old.now(), "fatal_error": fatal,
                     "seconds": perf_counter() - started, "calls": sum(len(c.rows) for c in calls.values()),
                     "evidence_sha256": sealed, "gate_data_collected": False})


class Replay:
    def __init__(self, folder, name):
        self.folder, self.name = folder, name
        rows = old.read(folder / "budget.json")["calls"]
        self.rows = {(r["target"], r["stage"], r["seat"]): r for r in rows}
        if len(self.rows) != len(rows):
            raise ValueError("duplicate dispatch")
        self.used = set()

    def call(self, target, stage, seat, body):
        key = target, stage, seat
        row = self.rows.get(key)
        if row is None:
            return None  # Frozen budget/config stop, not a provider request.
        self.used.add(key)
        folder = self.folder / "calls" / f"{row['index']:03d}_{target}_{stage}_{seat}"
        wire = base_body(body, self.name) if seat == "base" else body
        if old.read(folder / "request.json") != old.joint.roster.transport.redact_images(wire):
            raise ValueError("raw request replay mismatch")
        if row["status"] not in VALID:
            return None
        raw = old.read(folder / "response.json")["body"]
        if raw["model"] != wire["model"]:
            raise ValueError("raw model identity mismatch")
        content = raw["choices"][0]["message"]["content"]
        return json.loads(content) if seat == "base" else parse_review_json(content)[0]


def score(output):
    plan, done = verify(output), old.read(output / "completion.json")
    if done["fatal_error"]:
        raise ValueError("fatal execution failure")
    for rel, digest in done["evidence_sha256"].items():
        if old.sha(output / rel) != digest:
            raise ValueError("sealed evidence changed")
    bases, results = bases_for(plan), {}
    with old.joint.roster.lightweight_protocol():
        for name in MODELS:
            rows = old.read(output / name / "predictions.json")["targets"]
            if [r["key"] for r in rows] != [s["key"] for s in plan["selection"]]:
                raise ValueError("complete six-target denominator required")
            replay = Replay(output / name, name)
            for row, selected in zip(rows, plan["selection"], strict=True):
                if row["status"] == "NOT_DISPATCHED":
                    continue
                try:
                    record = main.run_target(replay, bases[row["key"]], selected,
                        old.read(output / "priors" / f"{selected['video_id']}.json"), plan["gate"])
                except (ValueError, TypeError, KeyError, ApiSchemaError):
                    if row["status"] != "TARGET_FAILED":
                        raise
                else:
                    if record["predictions"] != row["predictions"]:
                        raise ValueError("raw prediction replay mismatch")
            if replay.used != set(replay.rows):
                raise ValueError("unreplayed calls")
            results[name] = {"rows": rows, "replayed_requests": len(replay.used)}
    # Scoring labels are attached only after all three models' predictions are sealed and replayed.
    truth = {}
    adapter = old.common.CholecTrack20DatasetAdapter(release.DATASET, causal_window_size=3)
    for video in COHORT:
        if old.sha(annotation(video)) != plan["annotation_sha256"][video]:
            raise ValueError("annotation changed")
        wanted = {s["frame_id"] for s in plan["selection"] if s["video_id"] == video}
        for resolved in adapter.iter_video(video):
            frame_id = resolved.inference.target_frame_id
            if frame_id not in wanted:
                continue
            truth[f"{video}_{frame_id}"] = old.truth_row(resolved)
    old.save(output / "scored_truth.json", truth)
    for name, result in results.items():
        rows = result.pop("rows")
        result["metrics"] = release.metrics(rows, truth)
        result["complete"] = all(r["status"] == "PREDICTED" for r in rows)
        result["target_statuses"] = dict(Counter(r["status"] for r in rows))
        result["seconds_sum"] = sum(r["seconds"] for r in rows)
        result["h0_seconds_sum"] = sum(r.get("timing", {}).get("h0", 0) for r in rows)
        budget = old.read(output / name / "budget.json")
        charges, stage_charges = defaultdict(Decimal), defaultdict(Decimal)
        for c in budget["calls"]:
            charges[c["account"] + "|" + c["charge_kind"]] += Decimal(c["charge"])
            stage_charges[c["stage"] + "|" + c["account"] + "|" + c["charge_kind"]] += Decimal(c["charge"])
        result["charges"] = {k: str(v) for k, v in charges.items()}
        result["stage_charges"] = {k: str(v) for k, v in stage_charges.items()}
        result["usd_occupied"] = budget["occupied"]["openrouter_usd"]
        result["call_statuses"] = dict(Counter(c["status"] for c in budget["calls"]))
        result["failures"] = [{k: c.get(k) for k in ("target", "stage", "seat", "status", "exception_type", "http_status")}
                              for c in budget["calls"] if c["status"] not in VALID]
        edits = Counter()
        for row in rows:
            h, f = row["predictions"]["h0"], row["predictions"][main.PRIMARY]
            if h is None or f is None:
                continue
            for t in old.TASKS:
                before, after, gt = set(h[t]), set(f[t]), set(truth[row["key"]]["gt"][t])
                edits["beneficial_labels"] += len((after - before) & gt) + len((before - after) - gt)
                edits["harmful_labels"] += len((before - after) & gt) + len((after - before) - gt)
            if h["phase"] != f["phase"]:
                gt = truth[row["key"]]["gt"]["phase"]
                edits["phase_fixed" if f["phase"] == gt else "phase_broken" if h["phase"] == gt else "phase_wrong_to_wrong"] += 1
        result["edits"] = dict(edits)
    def rank(name):
        result = results[name]
        m = result["metrics"][main.PRIMARY]
        return (-m["mean_f1"], -m["ivt"]["f1"], -m["phase"]["f1"], Decimal(result["usd_occupied"]), result["seconds_sum"])
    eligible = sorted((n for n in MODELS if results[n]["complete"]), key=rank)
    report = {"profile": PROFILE, "scored_utc": old.now(), "models": results,
              "recommendation": eligible[0] if eligible else None, "ranking": eligible,
              "predeclared_standard": STANDARD, "wall_seconds": done["seconds"],
              "calls": done["calls"], "testing_used": False, "gate_data_collected": False}
    old.save(output / "comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args.output.resolve())
