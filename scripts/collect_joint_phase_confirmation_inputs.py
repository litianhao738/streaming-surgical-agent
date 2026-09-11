"""Freeze fresh metadata-selected Training targets and shared original upstream.

Separate collection, never a new H0 prompt or candidate-generator variant.
Only prepare is needed before selecting a verifier/repair revision.
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

from scripts import run_expanded_split_review as upstream
from scripts import run_joint_phase_feedback_trial as trial
from scripts.run_openrouter_gemini_h0_trial import gemini_h0_wire
from scripts.run_prior_candidate_trial import proposal_wire
from surgical_agent.perception.final_only import validate_final_only
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.verification.candidate_coordinator import make_pool

LIMITS = {"openrouter_usd": "1", "aliyun_cny": "0", "xai_usd": "0"}


def prepare(output, adapter, gap):
    if output.exists():
        raise ValueError("new output required")
    excluded, source_hashes = upstream.history()
    selection = []
    for video in upstream.VIDEOS:
        if adapter.entries[video].split is not upstream.old.DatasetSplit.TRAINING:
            raise ValueError("Training only")
        samples = list(adapter.iter_inference_video(video))
        masks = {r.inference.target_frame_id: upstream.masks_only(r) for r in adapter.iter_video(video)}
        used = set(excluded[video])
        for numerator in range(1, 5):
            anchor = samples[len(samples) * numerator // 5].target_frame_id
            eligible = [s for s in samples if len(s.causal_frame_ids) == 3
                and all(masks.get(s.target_frame_id, {}).get(t, False) for t in trial.TASKS)
                and all(abs(f - known) > gap for f in s.causal_frame_ids for known in used)]
            if not eligible:
                raise ValueError(f"No metadata-eligible sample for {video}, gap {gap}, quantile {numerator}/5; zero API")
            s = min(eligible, key=lambda x: (abs(x.target_frame_id - anchor), x.target_frame_id))
            selection.append({"key": f"{video}_{s.target_frame_id}", "video_id": video,
                "frame_id": s.target_frame_id, "anchor_frame_id": anchor, "causal_frame_ids": list(s.causal_frame_ids),
                "gt_availability_only": masks[s.target_frame_id],
                "images": [{"frame_id": f, "path": str(p), "sha256": trial.sha(p)}
                    for f, p in zip(s.causal_frame_ids, s.media_refs, strict=True)]})
            used.update(s.causal_frame_ids)
    trial.save(output / "history.json", {"excluded_targets_by_video": {v: sorted(fs) for v, fs in excluded.items()},
        "source_sha256": source_hashes})
    for video in upstream.VIDEOS:
        prior = trial.read(upstream.SOURCE / "priors" / f"{video}.json")
        upstream.validate_prior(prior, video, upstream.old.InferenceOnlyAdapter(adapter))
        trial.save(output / "priors" / f"{video}.json", prior)
    deps = sorted({*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py"), *ROOT.glob("src/**/*.txt"),
        *ROOT.glob("src/**/*.json"), *ROOT.glob("src/**/*.csv"), ROOT / "configs/perception/joint_openrouter_h0.yaml"})
    for p in deps:
        if p.is_file():
            target = output / "frozen_source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
    trial.save(output / "plan.json", {"profile": "joint_phase_fresh_upstream_v1", "created_utc": trial.now(),
        "selection": selection, "selection_rule": f"Nearest eligible 1/5..4/5 per original four Training videos; all image IDs >{gap} raw frames from historical/planned IDs; masks only, no label values.",
        "upstream_protocol": "Original Gemini H0 and original graph candidate generation, shared by all future verifier arms; does not change current main H0 config.",
        "gap": gap, "sources": {p.relative_to(ROOT).as_posix(): trial.sha(p) for p in deps if p.is_file()},
        "prior_sha256": {v: trial.sha(output / "priors" / f"{v}.json") for v in upstream.VIDEOS},
        "history_sha256": trial.sha(output / "history.json"), "credential_root": str(ROOT.resolve()),
        "limits": LIMITS, "max_calls": 32})
    print(json.dumps({"prepared": True, "targets": [s["key"] for s in selection], "gap": gap, "api_calls": 0}), flush=True)


def verify(output):
    plan = trial.read(output / "plan.json")
    for rel, digest in plan["sources"].items():
        if trial.sha(ROOT / rel) != digest:
            raise ValueError("source changed: " + rel)
    for video, digest in plan["prior_sha256"].items():
        if trial.sha(output / "priors" / f"{video}.json") != digest:
            raise ValueError("prior changed")
    for s in plan["selection"]:
        for im in s["images"]:
            if trial.sha(im["path"]) != im["sha256"]:
                raise ValueError("image changed")
    return plan


def execute(output, adapter):
    plan = verify(output)
    bases = trial.bases_for(adapter, plan)
    with (output / "execution.lock").open("x") as f:
        f.write(trial.sha(output / "plan.json"))
    start, fatal, rows = perf_counter(), None, []
    with trial.credential_context(plan), trial.roster.lightweight_protocol():
        calls = trial.roster.GLMCalls(output, limits={a: Decimal(v) for a, v in LIMITS.items()},
            rates=trial.roster.RATES, providers=trial.roster.PROVIDERS, max_calls=32)
        try:
            for selected in plan["selection"]:
                key, base = selected["key"], bases[selected["key"]]
                raw = calls.call(key, "h0", "base", gemini_h0_wire(base))
                validate_final_only(raw)
                h0 = {t: [raw[t]["selected_id"]] if t == "phase" else raw[t]["selected_ids"] for t in trial.TASKS}
                pool = make_pool(h0)
                prior = trial.read(output / "priors" / f"{selected['video_id']}.json")
                hints = retrieve_candidate_hints(h0, prior, video_id=selected["video_id"])
                proposal = calls.call(key, "proposal", "base", proposal_wire(base, selected, h0, pool, hints["packet"]))
                if proposal is None:
                    raise ValueError("upstream candidate call failed; preserve failure")
                pool = make_pool(h0, proposal, pool)
                record = {"h0_raw": raw, "h0": h0, "pool": pool, "hints": hints, "proposal_raw": proposal}
                trial.save(output / "targets" / key / "result.json", record)
                rows.append({**{k: selected[k] for k in ("key", "video_id", "frame_id")}, "h0": h0})
                print(json.dumps({"collected": len(rows), "target": key, "calls": len(calls.rows)}), flush=True)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            calls.stopped = True
            calls.persist()
            trial.save(output / "predictions.json", rows)
            paths = [p for p in output.rglob("*.json") if "frozen_source" not in p.parts and p.name != "completion.json"]
            trial.save(output / "completion.json", {"closed_utc": trial.now(), "fatal_error": fatal,
                "calls": len(calls.rows), "elapsed_seconds": perf_counter() - start,
                "hashes": {p.relative_to(output).as_posix(): trial.sha(p) for p in paths}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gap", type=int, default=175)
    args = parser.parse_args()
    adapter = trial.common.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, adapter, args.gap)
    else:
        execute(args.output, adapter)
