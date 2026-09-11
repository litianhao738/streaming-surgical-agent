"""Freeze new Training targets using time, history identities and masks only."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.prepare_final_only_gate_training import masks_only
from scripts.run_candidate_panel_trial import sha
from scripts.run_prior_panel_trial import now, read, save
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit

VIDEOS = ("VID103", "VID23", "VID31", "VID96")
SOURCE = ROOT / "artifacts/preflight/prior_candidate_eight_20260908_v1"
GAP = 250


def prepare(inventory_path, output, dataset_root):
    if output.exists():
        raise ValueError("single-use new selection directory required")
    inventory = read(inventory_path)
    # The independent inventory exposes identities only, never GT label values.
    excluded = inventory["excluded_targets_by_video"]
    ad = CholecTrack20DatasetAdapter(dataset_root, causal_window_size=3)
    selection, summaries = [], {}
    for video in VIDEOS:
        if ad.entries[video].split is not DatasetSplit.TRAINING:
            raise ValueError("Training only")
        samples = list(ad.iter_inference_video(video))
        masks = {r.inference.target_frame_id: masks_only(r) for r in ad.iter_video(video)}
        old = set(excluded.get(video, []))
        eligible = [s for s in samples if len(s.causal_frame_ids) == 3
                    and masks.get(s.target_frame_id, {}).get("verb", False)
                    and masks.get(s.target_frame_id, {}).get("ivt", False)
                    and all(abs(s.target_frame_id - f) > GAP for f in old)]
        selected = set()
        for numerator in range(1, 7):
            anchor = samples[len(samples) * numerator // 7].target_frame_id
            choices = [s for s in eligible
                       if all(abs(s.target_frame_id - f) > GAP for f in selected)]
            if not choices:
                raise ValueError("not enough history-separated targets; no silent resampling")
            sample = min(choices, key=lambda s: (abs(s.target_frame_id - anchor), s.target_frame_id))
            frame = sample.target_frame_id
            selected.add(frame)
            selection.append({
                "key": f"{video}_{frame}", "video_id": video, "frame_id": frame,
                "source_split": "Training", "anchor_frame_id": anchor,
                "quantile": f"{numerator}/7", "causal_frame_ids": list(sample.causal_frame_ids),
                "gt_availability_only": masks[frame],
                "nearest_historical_distance_frames": min(abs(frame - f) for f in old) if old else None,
                "images": [{"frame_id": f, "path": str(p), "sha256": sha(p)}
                           for f, p in zip(sample.causal_frame_ids, sample.media_refs, strict=True)],
            })
        summaries[video] = {
            "inference_targets": len(samples), "historical_targets": len(old),
            "eligible_targets": len(eligible),
            "selected_frames": sorted(selected),
            "valid_by_task": {t: sum(bool(m.get(t)) for m in masks.values())
                              for t in ("instrument", "verb", "target", "ivt", "phase")},
        }
    # Reuse precisely the prior tables of the original four-video graph trial.
    source_plan = read(SOURCE / "plan.json")
    for path, digest in source_plan["knowledge_source_sha256"].items():
        if sha(path) != digest:
            raise ValueError("original knowledge source changed")
    for video in VIDEOS:
        p = SOURCE / "priors" / f"{video}.json"
        if sha(p) != source_plan["prior_sha256"][video]:
            raise ValueError("original prior changed")
        prior = read(p)
        if video in prior["fit_videos"] or prior["excluded_video"] != video:
            raise ValueError("query-video leakage")
        if any(ad.entries[v].split is not DatasetSplit.TRAINING for v in prior["fit_videos"]):
            raise ValueError("non-Training prior source")
    output.mkdir(parents=True)
    (output / "priors").mkdir()
    for video in VIDEOS:
        shutil.copyfile(SOURCE / "priors" / f"{video}.json", output / "priors" / f"{video}.json")
    save(output / "selection.json", {
        "schema_version": "new_training_verb_confirmation_selection_v1",
        "created_utc": now(), "no_gt_label_values_used_for_selection": True,
        "selection": selection, "video_inventory": summaries,
        "history_inventory": str(inventory_path.resolve()), "history_inventory_sha256": sha(inventory_path),
        "selection_source_sha256": sha(__file__),
        "selection_rule": "Six temporal seventiles per video; nearest earlier-tie eligible frame. Verb and IVT mask true, three real causal frames, strictly more than 250 raw frames from all recorded/planned targets and within-cohort targets. No label values, images or model outputs used to choose targets.",
        "scope": "New target confirmation in previously used Training videos; not unseen-video or independent full-system validation.",
        "prior_source": str(SOURCE), "prior_source_plan_sha256": sha(SOURCE / "plan.json"),
        "prior_sha256": {v: sha(output / "priors" / f"{v}.json") for v in VIDEOS},
        "gt_policy": "Only masks are retained from selection. Reused per-query-video LOVO priors contain other Training videos, including other evaluation-cohort videos. No query-video GT enters its own prior or inference.",
    })
    print(json.dumps({"selection_path": str(output / "selection.json"), "targets": len(selection),
                      "frames": {v: summaries[v]["selected_frames"] for v in VIDEOS}}, ensure_ascii=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = p.parse_args()
    prepare(args.inventory, args.output, args.dataset_root)
