"""Offline, fixed-rule instrument fusion on existing Training predictions only."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from surgical_agent.research.gate.pgp_ambiguity import repair
from surgical_agent.research.verification.prior_panel import COMPONENTS
from surgical_agent.research.gate.final_only_training import canonical_labels
from surgical_agent.perception.final_only import final_only_schema

TASKS = ("instrument", "verb", "target", "ivt", "phase")
VARIANTS = ("pgp", "tracker_replace", "tracker_union", "tracker_ivt_closure")
MAX_INSTRUMENTS = final_only_schema()["properties"]["instrument"]["properties"]["selected_ids"]["maxItems"]


def fused(prediction, tracker_classes, variant):
    result = deepcopy(prediction)
    if variant == "tracker_replace":
        result["instrument"] = sorted(tracker_classes)
    elif variant == "tracker_union":
        result["instrument"] = sorted(set(prediction["instrument"]) | tracker_classes)
    elif variant == "tracker_ivt_closure":
        result["instrument"] = sorted(tracker_classes | {
            COMPONENTS[c]["instrument"] for c in prediction["ivt"]
        })
    elif variant != "pgp":
        raise ValueError(variant)
    overflow = len(result["instrument"]) > MAX_INSTRUMENTS
    if overflow:
        # Preserve the existing contract; do not rank/truncate classes using GT.
        result["instrument"] = list(prediction["instrument"])
    assert all(result[t] == prediction[t] for t in TASKS[1:])
    return canonical_labels(result), overflow


def counts(prediction, truth):
    out = []
    for task in TASKS:
        p, g = set(prediction[task]), set(truth[task])
        out.append((len(p & g), len(p - g), len(g - p)))
    return np.asarray(out, dtype=np.int64)


def quality(c):
    pooled = c.sum(axis=0)
    heads = {}
    for task, (tp, fp, fn) in zip(TASKS, pooled):
        heads[task] = dict(tp=int(tp), fp=int(fp), fn=int(fn),
                           f1=100 * 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 100.,
                           precision=100 * tp / (tp + fp) if tp + fp else 100.,
                           recall=100 * tp / (tp + fn) if tp + fn else 100.)
    return {"five_head_mean_f1": float(np.mean([v["f1"] for v in heads.values()])),
            "total_errors": int(pooled[:, 1:].sum()), "by_head": heads}


def main():
    output = ROOT / "artifacts/preflight/tracker_zero_call_fusion_20260914_r1"
    if output.exists():
        raise ValueError("preserve existing results; choose a new audit output")
    gate = ROOT / "artifacts/training/gate"
    rows_path = gate / "official_behavior_net_v2_20260913/primary_rows.json"
    array_path = gate / "pgp_ambiguity_assessment_20260913_r2/predictions.npz"
    rows = json.loads(rows_path.read_text("utf8"))
    with np.load(array_path) as z:
        data = {k: z[k] for k in ("ids", "videos", "cheap", "ambiguity", "qwen_hgb_route")}
    assert list(data["ids"]) == [r["sample_id"] for r in rows]
    assert set(np.unique(data["qwen_hgb_route"])).issubset({0, 1, False, True})
    assert all(r["source_split"] == "Training" and all(r["mask"].values()) for r in rows)
    folder = ROOT / "artifacts/training/tracker_clip_v2_oof5_20260906/oof"
    index_path = folder / "index.json"
    index = json.loads(index_path.read_text("utf8"))
    frames, bindings = {}, {}
    videos = sorted(set(data["videos"]))
    for rel in sorted({index["video_to_artifact"][v] for v in videos}):
        raw = (folder / rel).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == index["artifacts"][rel]
        bindings[rel] = digest
        for v, doc in json.loads(raw)["videos"].items():
            if v in videos:
                assert doc["source_split"] == "training"
                frames[v] = {f["frame_id"]: f["tracks"] for f in doc["frames"]}
    summaries = {v: [] for v in VARIANTS}
    edits = {v: Counter() for v in VARIANTS}
    unsupported = {v: Counter() for v in VARIANTS}
    prediction_rows = []
    for i, row in enumerate(rows):
        baseline = canonical_labels(repair(row["cheap_labels"], row["final_labels"])
                                    if data["qwen_hgb_route"][i] else row["cheap_labels"])
        expected = data["ambiguity"][i] if data["qwen_hgb_route"][i] else data["cheap"][i]
        assert np.array_equal(counts(baseline, row["gt"]), expected)
        # Frozen predictions already apply the original detector score threshold.
        current = frames[row["video_id"]][row["frame_id"]]
        tracker_classes = {t["instrument_id"] for t in current}
        gt = set(row["gt"]["instrument"])
        old = set(baseline["instrument"])
        predictions = {}
        for variant in VARIANTS:
            prediction, overflow = fused(baseline, tracker_classes, variant)
            predictions[variant] = prediction
            summaries[variant].append(counts(prediction, row["gt"]))
            new = set(prediction["instrument"])
            edits[variant].update(added_correct=len((new - old) & gt),
                                  added_incorrect=len((new - old) - gt),
                                  removed_incorrect=len((old - new) - gt),
                                  removed_correct=len((old - new) & gt),
                                  changed_frames=int(new != old),
                                  improved_frames=int(len(new ^ gt) < len(old ^ gt)),
                                  harmed_frames=int(len(new ^ gt) > len(old ^ gt)),
                                  schema_capacity_fallback_frames=int(overflow))
            missing = {COMPONENTS[c]["instrument"] for c in prediction["ivt"]} - new
            unsupported[variant].update(frames=int(bool(missing)), missing_instrument_components=len(missing))
        prediction_rows.append({"key": row["sample_id"], "video_id": row["video_id"],
                                "tracker_classes": sorted(tracker_classes), "predictions": predictions})
    report = {"rows": len(rows), "api_calls": 0, "new_training": False, "default_changed": False,
              "scope": "Post-hoc fixed-rule comparison on existing four Training videos; PGP uses saved outer-video-held-out routes, Tracker uses frozen indexed predictions. Upstream is not fully nested. Not independent confirmation.",
              "primary_design": "tracker_ivt_closure: Tracker supplies instrument classes; retain instrument components of unchanged PGP IVTs.",
              "rules": "Three explicit formulas, no threshold sweep or fitting; current-frame frozen detections only. If a formula exceeds the existing instrument maxItems=3, retain the original PGP output rather than truncate classes.",
              "pgp_logical_calls_all_variants": 37157,
              "additional_model_calls_all_variants": 0,
              "local_tracker_compute_included_in_api_cost": False,
              "variants": {}}
    for variant in VARIANTS:
        c = np.asarray(summaries[variant])
        report["variants"][variant] = {**quality(c), "instrument_edits": dict(edits[variant]),
                                       "ivt_instrument_consistency": dict(unsupported[variant]),
                                       "by_video": {str(v): quality(c[data["videos"] == v]) for v in videos}}
    assert abs(report["variants"]["pgp"]["five_head_mean_f1"] - 61.698902333008025) < 1e-5
    assert report["variants"]["pgp"]["total_errors"] == 36150
    report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (rows_path, array_path, index_path, Path(__file__))}
    report["tracker_sha256"] = bindings
    output.mkdir()
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf8")
    with (output / "predictions.jsonl").open("x", encoding="utf8") as stream:
        for row in prediction_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: {"f1": v["five_head_mean_f1"], "errors": v["total_errors"],
                           "instrument_f1": v["by_head"]["instrument"]["f1"],
                           "ivt_f1": v["by_head"]["ivt"]["f1"],
                           "edits": v["instrument_edits"], "consistency": v["ivt_instrument_consistency"],
                           "videos": {a: {"f1": b["five_head_mean_f1"], "errors": b["total_errors"]}
                                      for a, b in v["by_video"].items()}}
                      for k, v in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
