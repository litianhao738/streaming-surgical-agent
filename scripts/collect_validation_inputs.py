"""Freeze metadata-selected Validation targets and their shared upstream.

The Validation split has never carried a verifier or repair experiment. This
collects, once, the inputs every later arm shares: the original Gemini H0, the
original graph candidate generation, and a Training-only prior.

Targets are chosen from time quantiles using media availability and task masks
only; no label value is read for selection. The prior is fitted on all ten
Training videos and records the Validation video as excluded, so the retrieval
contract still proves the query video never contributed to its own hints.

Commands: prepare (zero API), execute (single-use paid, two calls per target).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_joint_phase_feedback_trial as trial
from scripts import run_prior_panel_trial as original
from scripts.run_openrouter_gemini_h0_trial import gemini_h0_wire, gemini_request
from scripts.run_prior_candidate_trial import proposal_wire
from scripts.run_prior_panel_trial import truth_row
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.verification.candidate_coordinator import make_pool
from surgical_agent.research.verification.prior_panel import fit_prior, video_counts

PROFILE = "validation_fresh_upstream_v1"
LIMITS = {"openrouter_usd": "1", "aliyun_cny": "0", "xai_usd": "0"}


def build_validation_base(adapter, selected):
    """Use the original media/prompt recipe with an explicit Validation guard."""
    sample = next((s for s in adapter.iter_inference_video(selected["video_id"])
                   if s.target_frame_id == selected["frame_id"]), None)
    if sample is None or sample.source_split is not DatasetSplit.VALIDATION:
        raise ValueError("only exact canonical Validation target accepted")
    if list(sample.causal_frame_ids) != selected["causal_frame_ids"]:
        raise ValueError("causal frame sequence changed")
    for item in selected["images"]:
        if trial.sha(item["path"]) != item["sha256"]:
            raise ValueError("source image changed")
    config = original.load_api_config(ROOT / "configs/perception/joint_openrouter_h0.yaml")
    loaded = original.CausalApiMediaLoader().load(sample)
    context = original.CausalPerceptionContextBuilder(
        max_frames=3, max_images=3, selection_strategy="fixed_all",
        history_image_detail="low", target_image_detail="high")
    base = original.JointPerceptionRequestBuilder(config=config).build(context.build(
        loaded.runtime_sample, loaded.frames, workflow_snapshot={}, memory_snapshot={},
        prior_finalized_prediction=None))
    if base.payload["system_text"] != original.load_main_h0_prompt():
        raise ValueError("H0 prompt drift")
    return gemini_request(base)


def validation_bases(adapter, plan):
    inference = trial.common.InferenceOnlyAdapter(adapter)
    return {s["key"]: build_validation_base(inference, s) for s in plan["selection"]}


def training_prior(adapter, video):
    """Statistics from every Training video; the query video is Validation."""
    counts = {}
    for name, entry in sorted(adapter.entries.items()):
        if entry.split is DatasetSplit.TRAINING:
            counts[name] = video_counts([truth_row(r) for r in adapter.iter_video(name)])
    prior = fit_prior(counts, video)
    if prior["excluded_video"] != video or video in prior["fit_videos"] or len(prior["fit_videos"]) != len(counts):
        raise ValueError("prior must be fitted on every Training video and exclude the query video")
    return prior


def choose(adapter, video, count, gap):
    samples = list(adapter.iter_inference_video(video))
    masks = {r.inference.target_frame_id: truth_row(r)["mask"] for r in adapter.iter_video(video)}
    selection, used = [], set()
    for numerator in range(1, count + 1):
        anchor = samples[len(samples) * numerator // (count + 1)].target_frame_id
        eligible = [s for s in samples if len(s.causal_frame_ids) == 3
                    and all(masks.get(s.target_frame_id, {}).get(t, False) for t in trial.TASKS)
                    and all(abs(f - known) > gap for f in s.causal_frame_ids for known in used)]
        if not eligible:
            raise ValueError(f"no metadata-eligible sample left for {video} at quantile {numerator}")
        s = min(eligible, key=lambda x: (abs(x.target_frame_id - anchor), x.target_frame_id))
        selection.append({"key": f"{video}_{s.target_frame_id}", "video_id": video,
                          "frame_id": s.target_frame_id, "anchor_frame_id": anchor,
                          "causal_frame_ids": list(s.causal_frame_ids),
                          "gt_availability_only": masks[s.target_frame_id],
                          "images": [{"frame_id": f, "path": str(p), "sha256": trial.sha(p)}
                                     for f, p in zip(s.causal_frame_ids, s.media_refs, strict=True)]})
        used.update(s.causal_frame_ids)
    return selection


def prepare(output, adapter, video, count, gap):
    if output.exists():
        raise ValueError("new output required")
    if adapter.entries[video].split is not DatasetSplit.VALIDATION:
        raise ValueError("this collector is for the Validation split")
    selection = choose(adapter, video, count, gap)
    trial.save(output / "priors" / f"{video}.json", training_prior(adapter, video))
    deps = sorted({*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py"), *ROOT.glob("src/**/*.txt"),
                   *ROOT.glob("src/**/*.json"), *ROOT.glob("src/**/*.csv"),
                   ROOT / "configs/perception/joint_openrouter_h0.yaml"})
    deps = [p for p in deps if p.is_file()]
    for p in deps:
        destination = output / "frozen_source" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    trial.save(output / "plan.json", {
        "profile": PROFILE, "created_utc": trial.now(), "video": video, "selection": selection,
        "selection_rule": (f"{count} time quantiles of {video}; nearest sample with three real causal "
                           f"frames, all five task masks valid, and every image id more than {gap} raw "
                           "frames from an already selected image. Masks only, no label values."),
        "upstream_protocol": "Original Gemini H0 and original graph candidate generation, unchanged.",
        "prior_scope": "Fitted on all ten Training videos; the Validation query video is recorded as excluded.",
        "gap": gap, "credential_root": str(ROOT.resolve()), "limits": LIMITS,
        "max_calls": 2 * len(selection),
        "prior_sha256": {video: trial.sha(output / "priors" / f"{video}.json")},
        "sources": {p.relative_to(ROOT).as_posix(): trial.sha(p) for p in deps}})
    print(json.dumps({"prepared": True, "video": video, "targets": len(selection),
                      "max_calls": 2 * len(selection), "api_calls": 0}), flush=True)


def verify(output):
    plan = trial.read(output / "plan.json")
    for rel, digest in plan["sources"].items():
        if trial.sha(ROOT / rel) != digest:
            raise ValueError("source changed: " + rel)
    for video, digest in plan["prior_sha256"].items():
        if trial.sha(output / "priors" / f"{video}.json") != digest:
            raise ValueError("prior changed")
    for s in plan["selection"]:
        for image in s["images"]:
            if trial.sha(image["path"]) != image["sha256"]:
                raise ValueError("image changed")
    return plan


def execute(output, adapter):
    plan = verify(output)
    if (output / "execution.lock").exists():
        raise ValueError("single-use collection; no paid overwrite")
    bases = validation_bases(adapter, plan)
    with (output / "execution.lock").open("x", encoding="utf-8") as handle:
        handle.write(trial.sha(output / "plan.json"))
    start, fatal, rows = perf_counter(), None, []
    with trial.credential_context(plan), trial.roster.lightweight_protocol():
        calls = trial.roster.GLMCalls(output, limits={a: Decimal(v) for a, v in LIMITS.items()},
                                      rates=trial.roster.RATES, providers=trial.roster.PROVIDERS,
                                      max_calls=plan["max_calls"])
        try:
            for selected in plan["selection"]:
                if calls.stopped:
                    break
                key, base = selected["key"], bases[selected["key"]]
                raw = calls.call(key, "h0", "base", gemini_h0_wire(base))
                validate_final_only(raw)
                h0 = {t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"]
                      for t in trial.TASKS}
                pool = make_pool(h0)
                prior = trial.read(output / "priors" / f"{selected['video_id']}.json")
                hints = retrieve_candidate_hints(h0, prior, video_id=selected["video_id"])
                proposal = calls.call(key, "proposal", "base",
                                      proposal_wire(base, selected, h0, pool, hints["packet"]))
                if proposal is None:
                    raise ValueError("upstream candidate call failed; preserve the failure")
                record = {"h0_raw": raw, "h0": h0, "pool": make_pool(h0, proposal, pool),
                          "hints": hints, "proposal_raw": proposal}
                trial.save(output / "targets" / key / "result.json", record)
                rows.append({**{k: selected[k] for k in ("key", "video_id", "frame_id")}, "h0": h0})
                print(json.dumps({"collected": len(rows), "target": key,
                                  "calls": len(calls.rows)}), flush=True)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            calls.stopped = True
            calls.persist()
            trial.save(output / "predictions.json", rows)
            paths = [p for p in output.rglob("*.json")
                     if "frozen_source" not in p.parts and p.name != "completion.json"]
            trial.save(output / "completion.json", {
                "closed_utc": trial.now(), "fatal_error": fatal, "calls": len(calls.rows),
                "elapsed_seconds": perf_counter() - start,
                "hashes": {p.relative_to(output).as_posix(): trial.sha(p) for p in paths}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video", default="VID110")
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--gap", type=int, default=175)
    args = parser.parse_args()
    dataset = trial.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, dataset, args.video, args.targets, args.gap)
    else:
        execute(args.output, dataset)
