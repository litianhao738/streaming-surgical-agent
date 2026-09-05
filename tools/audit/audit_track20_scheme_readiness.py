"""Read-only Track20 media/supervision audit; no test class statistics or API calls."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import cv2
from surgical_agent.data.constants import TASK_CLASS_COUNTS
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.label_policy import is_valid_task_id
from surgical_agent.data.parser import parse_annotation_file
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.evaluation.frame_ground_truth import aggregate_evaluation_target, aggregate_frame_target
from surgical_agent.perception.ontology_prompt import _INSTRUMENTS, _VERBS, _TARGETS, _PHASES, _ivt_rows

TASKS = ("instrument", "verb", "target", "ivt", "phase")
ATTRS = dict(zip(TASKS, ("instrument_ids", "verb_ids", "target_ids", "triplet_ids", "phase_id")))
WINDOWS = (3, 6, 8)
LABEL_NAMES = {"instrument": _INSTRUMENTS, "verb": _VERBS, "target": _TARGETS, "phase": _PHASES,
               "ivt": tuple(f"{_INSTRUMENTS[i]}-{_VERBS[v]}-{_TARGETS[t]}" for _, i, v, t in _ivt_rows())}


def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def index_summary(ids):
    ids = sorted(ids)
    runs, gaps, length = [], [], 0
    previous = None
    streak = {}
    for value in ids:
        if previous is not None and value - previous != 25:
            gaps.append({"after": previous, "before": value, "delta": value - previous})
            runs.append(length)
            length = 0
        length += 1
        streak[value] = length
        previous = value
    if length:
        runs.append(length)
    summary = {
        "count": len(ids), "first": ids[0] if ids else None, "last": ids[-1] if ids else None,
        "index_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "expected_raw_frame_step": 25, "deltas": dict(sorted(Counter(b-a for a,b in zip(ids, ids[1:])).items())),
        "contiguous_runs": len(runs), "run_lengths": runs, "gaps": gaps,
        "full_causal_media_windows": {str(w): sum(max(0, n-w+1) for n in runs) for w in WINDOWS},
    }
    return summary, streak


def empty_supervision():
    return {"frames": 0, "valid_frames": dict.fromkeys(TASKS, 0), "all_five_valid_frames": 0,
            "empty_valid_action_frames": 0,
            "full_causal_windows_by_valid_target_task": {str(w): dict.fromkeys(TASKS, 0) for w in WINDOWS}}


def add_target(stats, target, history_length, histograms=None):
    stats["frames"] += 1
    flags = {task: bool(getattr(target.mask, task)) for task in TASKS}
    stats["all_five_valid_frames"] += int(all(flags.values()))
    stats["empty_valid_action_frames"] += int(flags["ivt"] and not target.triplet_ids)
    for task, valid in flags.items():
        stats["valid_frames"][task] += int(valid)
        for window in WINDOWS:
            stats["full_causal_windows_by_valid_target_task"][str(window)][task] += int(valid and history_length >= window)
        if valid and histograms is not None:
            ids = [target.phase_id] if task == "phase" else getattr(target, ATTRS[task])
            histograms[task].update(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/preflight/track20_scheme_audit_20260905.json")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    adapter = CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=8)
    report = {
        "schema_version": "track20_scheme_readiness_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root.resolve()), "script_sha256": sha(__file__),
        "policy": {"dataset_modified": False, "paid_calls": 0, "test_class_distributions_collected": False,
                   "frame_mask": "nonempty instance list AND all instances valid for each task; VID31 explicit frame sidecar",
                   "window_policy": "Training/Validation: 3/6/8 consecutive PNG images at raw-index step25, reset at PNG gaps. Testing: full MP4 history t-25*k subject to frame bounds, no historical GT required. Target-only supervision required.",
                   "full_windows_are_audit_only": "Do not drop warmup/gap targets in a final benchmark; these counts describe full context availability."},
        "repair_manifest": {"path": str(args.dataset_root / "repair_manifest.json"), "sha256": sha(args.dataset_root / "repair_manifest.json"), "adapter_integrity_check_passed": True},
        "code_hashes": {str(path.relative_to(ROOT)): sha(path) for path in [ROOT/"src/surgical_agent/data/dataset.py", ROOT/"src/surgical_agent/data/parser.py", ROOT/"src/surgical_agent/data/masks.py", ROOT/"src/surgical_agent/data/label_policy.py", ROOT/"src/surgical_agent/evaluation/frame_ground_truth.py", ROOT/"src/surgical_agent/perception/ontology_prompt.py", ROOT/"src/surgical_agent/research/signals/resources/ivt_components_v1.csv"]},
        "videos": [], "splits": {},
    }
    train_hist = {task: Counter() for task in TASKS}
    train_video_coverage = {task: Counter() for task in TASKS}
    train_instance_hist = {task: Counter() for task in TASKS}
    for entry in adapter.entries.values():
        print(f"Auditing {entry.split.value}/{entry.video_id}", flush=True)
        annotation_path = Path(entry.annotation_file)
        raw = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        raw_ids = sorted(map(int, raw["annotations"]))
        raw_index, raw_streak = index_summary(raw_ids)
        instances = [instance for row in raw["annotations"].values() for instance in row]
        record = {"video_id": entry.video_id, "split": entry.split.value,
                  "raw_annotation": {"path": str(annotation_path), "sha256": sha(annotation_path), "metadata": raw["video"],
                                     "frame_index": raw_index, "instances": len(instances),
                                     "empty_instance_frames": sum(not row for row in raw["annotations"].values())}}
        record["raw_annotation"]["valid_instance_counts"] = {
            task: sum(is_valid_task_id(task, instance["triplet" if task == "ivt" else task]) for instance in instances)
            for task in TASKS
        }
        media = Path(entry.media_source)
        if media.is_dir():
            pngs = sorted(media.glob("*.png"))
            ids = sorted(int(path.stem) for path in pngs)
            media_index, streak = index_summary(ids)
            first = cv2.imread(str(pngs[0])) if pngs else None
            record["media"] = {"type": "png_frames", "path": str(media), "frame_index": media_index,
                               "first_image_shape": list(first.shape) if first is not None else None,
                               "raw_annotation_frames_without_exact_png": len(set(raw_ids)-set(ids)),
                               "png_without_raw_annotation_frame": len(set(ids)-set(raw_ids)),
                               "raw_mp4_count": len(list(media.parent.glob("*.mp4")))}
        else:
            cap = cv2.VideoCapture(str(media))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {media}")
            count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            decode_checks = []
            for fid in (raw_ids[0], raw_ids[len(raw_ids)//2], raw_ids[-1]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid-1)
                ok, frame = cap.read()
                decode_checks.append({"annotation_frame_id": fid, "decoder_index": fid-1, "decoded": bool(ok), "shape": list(frame.shape) if ok else None})
            cap.release()
            record["media"] = {"type": "mp4", "path": str(media), "size_bytes": media.stat().st_size,
                               "frame_count_header": count, "fps_header": fps, "width": width, "height": height,
                               "duration_seconds_header": count/fps, "annotation_frames_out_of_video_bounds": sum(not 1 <= f <= count for f in raw_ids),
                               "decode_checks": decode_checks, "note": "Header frame count and three sampled exact-index decodes; not exhaustive decode/content alignment proof."}
            streak = {fid: (fid - 1) // 25 + 1 for fid in raw_ids if 1 <= fid <= count}
            record["media"]["causal_history_policy"] = "MP4 t-25*k; annotation-key gaps are not media gaps and do not reset visual history"
        stats = empty_supervision()
        stats["valid_instance_counts"] = dict.fromkeys(TASKS, 0)
        if entry.split is DatasetSplit.TESTING:
            stats["valid_instance_counts"] = record["raw_annotation"]["valid_instance_counts"].copy()
            canonical = parse_annotation_file(annotation_path, expected_split=entry.split)
            for frame in canonical.frames:
                target = aggregate_frame_target(frame, allowed_tasks=frozenset(TASKS), source=str(annotation_path))
                add_target(stats, target, streak.get(frame.frame_id, 0))
            record["effective_source"] = {"route": "official_native_test_annotation_for_availability_audit_only", "sha256": sha(annotation_path), "runtime_adapter_hides_gt": True}
        else:
            per_video_hist = {task: Counter() for task in TASKS} if entry.split is DatasetSplit.TRAINING else None
            supervision_sources = set()
            for sample in adapter.iter_video(entry.video_id):
                if sample.evaluation is not None and sample.evaluation.instance_supervision_available:
                    target = aggregate_evaluation_target(sample.evaluation, source=sample.provenance.annotation_source or "")
                    for instance in sample.evaluation.instances:
                        for task in TASKS:
                            stats["valid_instance_counts"][task] += int(getattr(instance.mask, task))
                    if entry.split is DatasetSplit.TRAINING:
                        for instance in sample.evaluation.instances:
                            for task in TASKS:
                                if getattr(instance.mask, task):
                                    train_instance_hist[task].update([getattr(instance, "triplet_id" if task == "ivt" else f"{task}_id")])
                else:
                    target = sample.frame_supervision
                if target is None:
                    raise RuntimeError(f"Unexpected missing effective supervision: {entry.video_id}")
                assert len(sample.inference.causal_frame_ids) == min(8, streak[target.frame_id])
                add_target(stats, target, streak[target.frame_id], per_video_hist)
                supervision_sources.update(s for s in (sample.provenance.annotation_source, sample.provenance.phase_source, sample.provenance.frame_action_source) if s)
            record["effective_source"] = {"route": "derived_vid31_frame_level_only" if entry.video_id == "VID31" else "candidate_repaired_vid30" if entry.video_id == "VID30" else "official_raw", "files": [{"path": path, "sha256": sha(path)} for path in sorted(supervision_sources)], "bbox_and_instance_association_allowed": entry.video_id != "VID31"}
            if per_video_hist is not None:
                for task in TASKS:
                    train_hist[task].update(per_video_hist[task])
                    train_video_coverage[task].update(per_video_hist[task].keys())
        record["effective_frame_supervision"] = stats
        report["videos"].append(record)
    for split in DatasetSplit:
        rows = [v for v in report["videos"] if v["split"] == split.value]
        report["splits"][split.value] = {
            "videos": len(rows), "raw_annotation_frames": sum(v["raw_annotation"]["frame_index"]["count"] for v in rows),
            "raw_instances": sum(v["raw_annotation"]["instances"] for v in rows),
            "png_frames": sum(v["media"].get("frame_index", {}).get("count", 0) for v in rows),
            "mp4_frames_header": sum(v["media"].get("frame_count_header", 0) for v in rows),
            "effective_supervision_frames": sum(v["effective_frame_supervision"]["frames"] for v in rows),
            "valid_frames": {t: sum(v["effective_frame_supervision"]["valid_frames"][t] for v in rows) for t in TASKS},
            "all_five_valid_frames": sum(v["effective_frame_supervision"]["all_five_valid_frames"] for v in rows),
            "full_causal_windows_by_valid_target_task": {str(w): {t: sum(v["effective_frame_supervision"]["full_causal_windows_by_valid_target_task"][str(w)][t] for v in rows) for t in TASKS} for w in WINDOWS},
        }
    report["train_only_class_statistics"] = {}
    for task in TASKS:
        hist = train_hist[task]
        report["train_only_class_statistics"][task] = {
            "granularity": "positive frame counts on strictly task-valid effective frame labels, including VID31 frame-only sidecar",
            "ontology_classes": TASK_CLASS_COUNTS[task], "observed_classes": len(hist),
            "absent_class_ids": [i for i in range(TASK_CLASS_COUNTS[task]) if i not in hist],
            "positive_frame_counts": {str(i): hist[i] for i in range(TASK_CLASS_COUNTS[task])},
            "video_coverage_counts": {str(i): train_video_coverage[task][i] for i in range(TASK_CLASS_COUNTS[task])},
            "observed_classes_below_10_positive_frames": sum(0 < n < 10 for n in hist.values()),
            "observed_classes_below_100_positive_frames": sum(0 < n < 100 for n in hist.values()),
            "observed_classes_in_one_training_video": sum(n == 1 for n in train_video_coverage[task].values()),
            "top_five_classes_by_positive_frame_count": hist.most_common(5),
        }
    report["train_only_instance_class_statistics"] = {
        task: {"granularity": "valid individual instances from current effective instance route, including partial frames; excludes VID31 frame-only sidecar",
               "valid_instances": sum(hist.values()), "observed_classes": len(hist),
               "positive_instance_counts": {str(i): hist[i] for i in range(TASK_CLASS_COUNTS[task])}}
        for task, hist in train_instance_hist.items()
    }
    report["train_only_any_effective_supervision_class_coverage"] = {
        task: {"granularity": "union of task-valid frame labels and individually valid instance labels; neither equivalent to fully labeled frames nor independent examples",
               "observed_classes": len(set(train_hist[task]) | set(train_instance_hist[task])),
               "absent_class_ids": sorted(set(range(TASK_CLASS_COUNTS[task])) - set(train_hist[task]) - set(train_instance_hist[task]))}
        for task in TASKS
    }
    for task, values in report["train_only_any_effective_supervision_class_coverage"].items():
        values["absent_classes"] = [{"id": i, "name": LABEL_NAMES[task][i]} for i in values["absent_class_ids"]]
    report["totals"] = {"videos": len(report["videos"]), "raw_annotation_frames": sum(s["raw_annotation_frames"] for s in report["splits"].values()), "raw_instances": sum(s["raw_instances"] for s in report["splits"].values())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "totals": report["totals"], "splits": report["splits"], "train_coverage": {t: {k: v for k,v in x.items() if k not in ("positive_frame_counts", "video_coverage_counts")} for t,x in report["train_only_class_statistics"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
