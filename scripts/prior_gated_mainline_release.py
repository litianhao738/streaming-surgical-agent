"""Preparation, bounded prospective Testing confirmation, and immutable release freeze.

Public CLI is run_prior_gated_joint_mainline.py. No Gate examples or weights.
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from time import perf_counter

from scripts import run_prior_gated_joint_mainline as main
from scripts.run_repair_revision_trial import parse_review_json
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.parser import parse_annotation_file
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.frame_ground_truth import aggregate_frame_target

ROOT, old = main.ROOT, main.old
DATASET = Path("D:/cholec_dataset")
VIDEOS = ("VID25", "VID92", "VID111")
PER_VIDEO = 8
LIMITS = {"openrouter_usd": "5", "aliyun_cny": "3", "xai_usd": "0"}
T2B = ROOT / "artifacts/research/prior_gated_add_guard_20260911_v1.json"
T1 = ROOT / "artifacts/preflight/prior_gated_joint_mainline_replay_20260911_v1/replay.json"
STANDARD = {
    "mainline_vs_h0": "Every planned target produces a prediction; pooled mean F1 strictly higher, total FP+FN strictly lower, IVT F1 no lower, zero H0-correct Phase broken.",
    "mainline_vs_gated_control": "Four interaction heads identical, pooled mean F1 no lower, total FP+FN no higher.",
    "incremental_phase_benefit": "Separately report whether at least one H0-wrong Phase is fixed with zero H0-correct Phase broken; a tie is not improvement.",
    "failures": "Keep every planned target. Unavailable predictions are empty sets for scoring and reported separately; no API retries or target substitution.",
    "use": "Frozen prospective Testing subset evaluation only. No model, threshold, prompt or rule selection from these outcomes; no Gate data collection.",
}


def runtime_paths():
    paths = set(ROOT.glob("scripts/*.py"))
    for folder in ("src", "configs"):
        paths.update(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".txt", ".json", ".csv", ".yaml", ".yml"})
    paths.update((ROOT / "pyproject.toml", ROOT / "requirements.txt"))
    return sorted(paths)


def annotation(video):
    return DATASET / "Testing" / video / f"{video.lower()}.json"


def masks_only(video):
    native = parse_annotation_file(annotation(video), expected_split=DatasetSplit.TESTING)
    return {f.frame_id: {task: bool(f.instances) and all(getattr(i.mask, task) for i in f.instances)
                         for task in old.TASKS} for f in native.frames}


def make_base(sample):
    original = old.collector.original
    config = original.load_api_config(ROOT / "configs/perception/joint_openrouter_h0.yaml")
    loaded = original.CausalApiMediaLoader().load(sample)
    context = original.CausalPerceptionContextBuilder(max_frames=3, max_images=3, selection_strategy="fixed_all",
                                                     history_image_detail="low", target_image_detail="high")
    base = original.JointPerceptionRequestBuilder(config=config).build(context.build(
        loaded.runtime_sample, loaded.frames, workflow_snapshot={}, memory_snapshot={}, prior_finalized_prediction=None))
    if base.payload["system_text"] != original.load_main_h0_prompt():
        raise ValueError("H0 prompt drift")
    return old.collector.gemini_request(base)


def load_bases(plan):
    adapter = old.common.CholecTrack20DatasetAdapter(DATASET, causal_window_size=3)
    selected = {s["key"]: s for s in plan["selection"]}
    bases = {}
    for video in plan["videos"]:
        ids = [s["frame_id"] for s in selected.values() if s["video_id"] == video]
        for sample in adapter.iter_inference_video(video):
            if sample.target_frame_id not in ids:
                continue
            key = f"{video}_{sample.target_frame_id}"
            if sample.source_split is not DatasetSplit.TESTING:
                raise ValueError("expected Testing inference-only sample")
            base = make_base(sample)
            if list(sample.causal_frame_ids) != selected[key]["causal_frame_ids"]:
                raise ValueError("causal window changed")
            for image, archived in zip(base.images, selected[key]["images"], strict=True):
                if image.content != Path(archived["path"]).read_bytes():
                    raise ValueError("decoded image changed")
            bases[key] = base
    if set(bases) != set(selected):
        raise ValueError("missing exact targets")
    return bases


def prepare(output):
    if output.exists():
        raise ValueError("new single-use output directory required")
    if old.read(T2B)["adopt"]:
        raise ValueError("this release freezes the original add rule; reconcile before Testing")
    if len(old.read(T1)["targets"]) != 16:
        raise ValueError("T1 replay incomplete")
    # Freeze the protocol before accessing even Testing masks/selection.
    old.save(output / "protocol.json", {"registered_utc": old.now(), "version": main.VERSION,
              "videos": VIDEOS, "targets_per_video": PER_VIDEO, "standard": STANDARD,
              "selection": "Eight temporal quantiles per video; nearest canonical target with a complete causal window and all-five valid masks; minimum 175 source-frame gap between every selected image.",
              "excluded_video": "VID06: previously used for window-density development",
              "scope": "24-target prospective Testing subset, not full Testing. Grok/Qwen baseline archives exist but their predictions and metrics are not read or used here.",
              "limits": LIMITS, "max_calls": len(VIDEOS) * PER_VIDEO * main.CALLS_PER_TARGET,
              "gate_data_collection": False, "selection_after_results": False})
    old.save(output / "previous_best_manifest.json", old.read(ROOT / "BEST_PIPELINE_VERSION.json"))
    adapter = old.common.CholecTrack20DatasetAdapter(DATASET, causal_window_size=3)
    selection, priors, annotations = [], {}, {}
    for video in VIDEOS:
        if adapter.entries[video].split is not DatasetSplit.TESTING:
            raise ValueError("wrong split")
        prior = old.collector.training_prior(adapter, video)
        old.save(output / "priors" / f"{video}.json", prior)
        priors[video] = old.sha(output / "priors" / f"{video}.json")
        masks = masks_only(video)
        annotations[video] = old.sha(annotation(video))
        samples = list(adapter.iter_inference_video(video))
        used = set()
        for numerator in range(1, PER_VIDEO + 1):
            anchor = samples[len(samples) * numerator // (PER_VIDEO + 1)].target_frame_id
            eligible = [s for s in samples if len(s.causal_frame_ids) == 3
                        and all(masks.get(s.target_frame_id, {}).values()) and s.target_frame_id in masks
                        and all(abs(f - known) > 175 for f in s.causal_frame_ids for known in used)]
            if not eligible:
                raise ValueError("no eligible target")
            sample = min(eligible, key=lambda s: (abs(s.target_frame_id - anchor), s.target_frame_id))
            used.update(sample.causal_frame_ids)
            base = make_base(sample)
            images = []
            for fid, image in zip(sample.causal_frame_ids, base.images, strict=True):
                if image.mime_type != "image/png":
                    raise ValueError("frozen PNG encoding required")
                path = output / "images" / video / f"{fid:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(image.content)
                images.append({"path": str(path.resolve()), "sha256": old.sha(path), "frame_id": fid})
            selection.append({"key": f"{video}_{sample.target_frame_id}", "video_id": video,
                              "frame_id": sample.target_frame_id, "anchor_frame_id": anchor,
                              "causal_frame_ids": list(sample.causal_frame_ids), "images": images,
                              "gt_availability_only": masks[sample.target_frame_id]})
    sources = {p.relative_to(ROOT).as_posix(): old.sha(p) for p in runtime_paths()}
    for rel in sources:
        dest = output / "frozen_source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    with old.joint.roster.lightweight_protocol():
        # Transport registry retains historical model names; the wire roster is authoritative.
        endpoint_models = {s: [old.joint.roster.transport.MODELS[s][0], old.joint.roster.MODELS[s]] for s in old.SEATS}
    plan = {"version": main.VERSION, "created_utc": old.now(), "videos": VIDEOS, "selection": selection,
            "arms": main.ARMS, "candidate": main.PRIMARY, "gate": old.GATE, "phase_threshold": 4,
            "h0_model": "google/gemini-3.8-flash", "h0_route": "google-ai-studio",
            "models": old.joint.roster.MODELS, "transports": old.joint.roster.TRANSPORTS,
            "providers": old.joint.roster.PROVIDERS, "rates": old.joint.roster.RATES,
            "reviewer_endpoints_and_model_ids": endpoint_models,
            "limits": LIMITS, "credential_root": str(ROOT), "max_calls": len(selection) * 13,
            "stages": main.STAGES, "automatic_retries": 0, "gate_training": False,
            "protocol_sha256": old.sha(output / "protocol.json"), "predeclared_standard": STANDARD,
            "source_sha256": sources, "prior_sha256": priors, "annotation_sha256": annotations,
            "default_sha256_before": old.sha(ROOT / "DEFAULT_PIPELINE_VERSION.json"),
            "t1_sha256": old.sha(T1), "t2b_sha256": old.sha(T2B),
            "environment": {"python": sys.version, "platform": platform.platform()}}
    old.save(output / "plan.json", plan)
    print(json.dumps({"prepared": str(output), "targets": len(selection), "max_calls": plan["max_calls"], "api_calls": 0}), flush=True)


def verify(output):
    plan = old.read(output / "plan.json")
    if (plan["version"] != main.VERSION or plan["gate"] != old.GATE or plan["models"] != old.joint.roster.MODELS
            or plan["stages"] != main.STAGES or plan["max_calls"] != len(plan["selection"]) * 13):
        raise ValueError("release contract changed")
    if old.sha(output / "protocol.json") != plan["protocol_sha256"]:
        raise ValueError("protocol changed")
    for rel, digest in plan["source_sha256"].items():
        if old.sha(ROOT / rel) != digest or old.sha(output / "frozen_source" / rel) != digest:
            raise ValueError("runtime source changed: " + rel)
    for video, digest in plan["prior_sha256"].items():
        path = output / "priors" / f"{video}.json"
        if old.sha(path) != digest:
            raise ValueError("prior changed")
        prior = old.read(path)
        if prior["excluded_video"] != video or video in prior["fit_videos"] or set(prior["fit_videos"]) & set(VIDEOS):
            raise ValueError("Testing contaminated prior")
    for s in plan["selection"]:
        for image in s["images"]:
            if old.sha(image["path"]) != image["sha256"]:
                raise ValueError("image changed")
    return plan


class RecordingMock(old.MockCalls):
    def call(self, target, stage, seat, body):
        if stage not in main.STAGES:
            raise ValueError("unexpected stage")
        return super().call(target, stage, seat, body)


def preflight(output):
    plan = verify(output)
    bases = load_bases(plan)
    checks = []
    with old.joint.roster.lightweight_protocol():
        for s in plan["selection"]:
            calls = RecordingMock()
            prior = old.read(output / "priors" / f"{s['video_id']}.json")
            result = main.run_target(calls, bases[s["key"]], s, prior, plan["gate"])
            if dict(Counter(c["stage"] for c in calls.rows)) != main.STAGES:
                raise ValueError("wrong call count")
            if any(result["predictions"][main.PRIMARY][t] != result["predictions"]["gated_control"][t] for t in old.TASKS[:4]):
                raise ValueError("Phase changed four heads")
            checks.append({"key": s["key"], "mock_calls": len(calls.rows), "passed": True})
    old.save(output / "preflight.json", {"plan_sha256": old.sha(output / "plan.json"), "api_calls": 0, "targets": checks})
    print(json.dumps({"preflight_targets": len(checks), "api_calls": 0}), flush=True)


def execute(output):
    plan = verify(output)
    check = old.read(output / "preflight.json")
    if check["plan_sha256"] != old.sha(output / "plan.json") or len(check["targets"]) != len(plan["selection"]):
        raise ValueError("preflight incomplete")
    bases = load_bases(plan)
    with (output / "execution.lock").open("x", encoding="utf-8") as lock:
        lock.write(old.sha(output / "plan.json"))
    rows, start, fatal = [], perf_counter(), None
    with old.joint.credential_context(plan), old.joint.roster.lightweight_protocol():
        calls = main.MainlineCalls(output, limits={a: Decimal(v) for a, v in plan["limits"].items()},
                                  rates=old.joint.roster.RATES, providers=old.joint.roster.PROVIDERS,
                                  max_calls=plan["max_calls"], reasoning_seats=("gemini",))
        try:
            for s in plan["selection"]:
                row = {k: s[k] for k in ("key", "video_id", "frame_id")}
                if calls.stopped:
                    row.update(status="NOT_DISPATCHED_BUDGET_OR_CONFIG_STOP", predictions={a: None for a in main.ARMS})
                else:
                    try:
                        prior = old.read(output / "priors" / f"{s['video_id']}.json")
                        record = main.run_target(calls, bases[s["key"]], s, prior, plan["gate"])
                    except (ValueError, TypeError, KeyError, ApiSchemaError) as exc:
                        row.update(status="TARGET_FAILED", error_type=type(exc).__name__, predictions={a: None for a in main.ARMS})
                    else:
                        old.save(output / "targets" / s["key"] / "result.json", record)
                        row.update(status="PREDICTED", predictions=record["predictions"], timing_seconds=record["timing_seconds"])
                rows.append(row)
                old.save(output / "predictions.json", {"targets": rows})
                print(json.dumps({"target": row["key"], "status": row["status"], "completed": len(rows),
                                  "calls": len(calls.rows), "occupied": {k: str(v) for k, v in calls.occupied.items()}}), flush=True)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            calls.stopped = True
            try:
                calls.persist()
            finally:
                calls.close_ledger()
            old.save(output / "predictions.json", {"targets": rows})
            evidence = {p.relative_to(output).as_posix(): old.sha(p) for folder in ("calls", "targets")
                        for p in (output / folder).rglob("*.json")}
            evidence.update({name: old.sha(output / name) for name in ("plan.json", "predictions.json", "budget.json")})
            old.save(output / "completion.json", {"closed_utc": old.now(), "fatal_error": fatal,
                     "targets": len(rows), "calls": len(calls.rows), "seconds": perf_counter() - start,
                     "statuses": dict(Counter(c["status"] for c in calls.rows)), "evidence_sha256": evidence,
                     "test_gt_used_during_inference": False, "gate_training_data_collected": False})


def metrics(rows, truth):
    result = {}
    for arm in main.ARMS:
        tasks = {}
        for task in old.TASKS:
            tp = fp = fn = 0
            for row in rows:
                item = truth[row["key"]]
                if not item["mask"][task]:
                    continue
                pred = row["predictions"][arm]
                got, gt = set(pred[task] if pred is not None else []), set(item["gt"][task] or [])
                tp += len(got & gt)
                fp += len(got - gt)
                fn += len(gt - got)
            tasks[task] = {"tp": tp, "fp": fp, "fn": fn, "f1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                           "precision": 100 * tp / (tp + fp) if tp + fp else 0}
        result[arm] = {**tasks, "mean_f1": sum(t["f1"] for t in tasks.values()) / 5,
                       "mean_precision": sum(t["precision"] for t in tasks.values()) / 5,
                       "errors": sum(t["fp"] + t["fn"] for t in tasks.values())}
    return result


def score(output):
    plan = verify(output)
    done = old.read(output / "completion.json")
    rows = old.read(output / "predictions.json")["targets"]
    if done["fatal_error"] or [r["key"] for r in rows] != [s["key"] for s in plan["selection"]]:
        raise ValueError("complete planned cohort required")
    for rel, digest in done["evidence_sha256"].items():
        if old.sha(output / rel) != digest:
            raise ValueError("sealed evidence changed")
    # Reconstruct from actual HTTP responses, not merely the cached parsed payloads.
    budget = old.read(output / "budget.json")
    for row in rows:
        if row["status"] != "PREDICTED":
            continue
        record = old.read(output / "targets" / row["key"] / "result.json")
        raw_by_stage = {}
        for c in budget["calls"]:
            if c["target"] != row["key"]:
                continue
            parsed = None
            if c["status"] in {"JSON_PARSED", "JSON_PARSED_FENCE_NORMALIZED"}:
                folder = output / "calls" / f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
                text = old.read(folder / "response.json")["body"]["choices"][0]["message"]["content"]
                parsed = json.loads(text) if c["seat"] == "base" else parse_review_json(text)[0]
            raw_by_stage.setdefault(c["stage"], {})[c["seat"]] = parsed
        h0 = old.gated.h0_from_raw(raw_by_stage["h0"]["base"])
        pool = old.make_pool(h0)
        proposal = raw_by_stage["proposal"]["base"]
        if proposal is not None:
            try:
                pool = old.make_pool(h0, proposal, pool)
            except ValueError:
                pass
        if pool != record["pool"]:
            raise ValueError("raw pool replay mismatch")
        prior = old.read(output / "priors" / f"{row['video_id']}.json")
        preds, _ = old.decide_record(h0, pool, raw_by_stage.get("control_graph", {}), None,
                                    raw_by_stage.get("joint_r1", {}), prior, plan["gate"])
        if {a: preds[a] for a in main.ARMS} != row["predictions"]:
            raise ValueError("raw prediction replay mismatch")
    # Only now access Testing labels for evaluation.
    truth = {}
    for video in plan["videos"]:
        if old.sha(annotation(video)) != plan["annotation_sha256"][video]:
            raise ValueError("annotation source changed")
        native = parse_annotation_file(annotation(video), expected_split=DatasetSplit.TESTING)
        wanted = {r["frame_id"] for r in rows if r["video_id"] == video}
        for frame in native.frames:
            if frame.frame_id not in wanted:
                continue
            target = aggregate_frame_target(frame, allowed_tasks=frozenset(old.TASKS), source=str(annotation(video)))
            attrs = dict(zip(old.TASKS, ("instrument_ids", "verb_ids", "target_ids", "triplet_ids", "phase_id"), strict=True))
            item = {"video_id": video, "frame_id": frame.frame_id,
                    "mask": {t: getattr(target.mask, t) for t in old.TASKS},
                    "gt": {t: [target.phase_id] if t == "phase" else list(getattr(target, attr)) for t, attr in attrs.items()}}
            truth[f"{video}_{frame.frame_id}"] = item
    m = metrics(rows, truth)
    edits = Counter()
    for row in rows:
        h0, final = row["predictions"]["h0"], row["predictions"][main.PRIMARY]
        if h0 is None or final is None:
            continue
        if final["phase"] != h0["phase"]:
            gt = truth[row["key"]]["gt"]["phase"]
            edits["fixed" if final["phase"] == gt else "broken" if h0["phase"] == gt else "wrong_to_wrong"] += 1
    c, h, g = m[main.PRIMARY], m["h0"], m["gated_control"]
    passed = (c["mean_f1"] > h["mean_f1"] and c["errors"] < h["errors"] and c["ivt"]["f1"] >= h["ivt"]["f1"]
              and edits["broken"] == 0 and c["mean_f1"] >= g["mean_f1"] and c["errors"] <= g["errors"]
              and all(r["status"] == "PREDICTED" for r in rows))
    report = {"scored_utc": old.now(), "targets": len(rows), "metrics": m, "passed": passed,
              "incremental_phase_benefit": edits["fixed"] > 0 and edits["broken"] == 0,
              "phase_edits": edits, "target_statuses": dict(Counter(r["status"] for r in rows)),
              "by_video": {v: metrics([r for r in rows if r["video_id"] == v], truth) for v in plan["videos"]},
              "predeclared_standard": STANDARD, "plan_sha256": old.sha(output / "plan.json"),
              "scope": "Frozen 24-target Testing subset; not full Testing; no tuning after these results.",
              "raw_replay_verified": True}
    old.save(output / "scored_truth.json", list(truth.values()))
    old.save(output / "metrics.json", report)
    print(json.dumps({"passed": passed, "phase_edits": edits, "metrics": {a: {k: x[k] for k in ("mean_f1", "errors")} for a, x in m.items()}}, indent=2))


def freeze(output):
    plan = verify(output)
    result = old.read(output / "metrics.json")
    if old.sha(ROOT / "DEFAULT_PIPELINE_VERSION.json") != plan["default_sha256_before"]:
        raise ValueError("default changed")
    paths = [p for p in output.rglob("*") if p.is_file() and p.name != "release_manifest.json"]
    evidence = {p.relative_to(output).as_posix(): old.sha(p) for p in paths if "frozen_source" not in p.parts}
    historical = {}
    for name in ("prior_gated_joint_vid110_confirm_20260911_v1_resume1",
                 "prior_gated_vid110_confirm_20260911_v1", "validation_vid110_confirm_20260911_v2_resume1"):
        folder = ROOT / "artifacts/preflight" / name
        historical[name] = {"path": str(folder), "sha256": {
            p.relative_to(folder).as_posix(): old.sha(p) for p in folder.rglob("*.json")
            if "frozen_source" not in p.parts}}
    replay_path = ROOT / "artifacts/research/prior_gated_joint_replay_20260911.json"
    manifest = {"schema_version": "frozen_prior_gated_joint_mainline_v1", "version": main.VERSION,
                "frozen_utc": old.now(), "status": "frozen_candidate_not_default", "entrypoint": "scripts/run_prior_gated_joint_mainline.py",
                "calls_per_target": 13, "stages": main.STAGES, "recommended_form": "compact four-head review + original prior gate (H0 Phase bucket) + joint-panel Phase admission; no blind Phase vote",
                "h0_model": plan["h0_model"], "h0_route": plan["h0_route"], "models": plan["models"],
                "transports": plan["transports"], "providers": plan["providers"], "gate": plan["gate"],
                "reviewer_endpoints_and_model_ids": plan["reviewer_endpoints_and_model_ids"],
                "phase_threshold": 4, "phase_feedback_to_prior": False, "joint_prompt_variant": "v1",
                "h0_prompt_version": "joint_perception_main_h0_v1", "review_prompt_profile": "verifier_compact_wording_v2_json_mode_candidate",
                "runtime_source_sha256": plan["source_sha256"], "frozen_source": str(output / "frozen_source"),
                "evidence_archive": str(output), "evidence_sha256": evidence,
                "historical_evidence": historical,
                "historical_phase_replay": {"path": str(replay_path), "sha256": old.sha(replay_path)},
                "prior_sha256": plan["prior_sha256"], "default_unchanged_sha256": plan["default_sha256_before"],
                "t1_replay": {"path": str(T1), "sha256": old.sha(T1), "targets": 16, "request_checks": 208},
                "t2b": {"path": str(T2B), "sha256": old.sha(T2B), "adopted": False},
                "historical_joint_phase_edits": {"Training40": {"fixed": 1, "broken": 0}, "VID11032": {"fixed": 1, "broken": 0}, "VID110_new16": {"fixed": 0, "broken": 0}, "total": {"fixed": 2, "broken": 0}, "note": "First two are historical joint-protocol replays; their recommendation included prior hints. Exact new no-hint policy confirmed only on last batch."},
                "independent_confirmation": {"passed": result["passed"], "incremental_phase_benefit": result["incremental_phase_benefit"], "targets": result["targets"], "videos": plan["videos"], "phase_edits": result["phase_edits"]},
                "gate_training_data_collected": False, "default_replaced": False,
                "limitations": ["Prior benefit aligns dataset Phase-3 annotation conventions; no new visual evidence.",
                    "Joint Phase rarely switches and shares H0 localization errors; expensive and not reliably better than retaining H0 Phase.",
                    "Historical evidence came from one Validation video, mostly Phases 1/2/3; Phase 4/6 prior buckets may fall back globally.",
                    "Archived true-Phase plus pool-expansion diagnostic changed new16 four-head errors only 74 to 73; oracle labels never used in inference.",
                    "Prospective Testing evidence covers a small frozen subset, not the full split or a significance test; no subsequent tuning is allowed on it.",
                    "Single writer removes in-process concurrent budget writers; external filesystem locks may still fail and are surfaced without retries."]}
    if (output / "release_manifest.json").exists():
        raise ValueError("release already frozen")
    old.save(output / "release_manifest.json", manifest)
    old.save(ROOT / "BEST_PIPELINE_VERSION.json", manifest)
    print(json.dumps({"frozen": main.VERSION, "default_unchanged": True, "gate_training_data_collected": False, "confirmation_passed": result["passed"]}))
