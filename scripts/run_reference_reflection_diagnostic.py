"""Two-case cross-video labeled-reference diagnostic; not baseline inference.

Reference classes are all legal targets for the common H0 I/V pair. References
come exclusively from other Training videos, selected by a fixed median rule.
Current target GT never enters the request. References are NOT causal history.
"""
import argparse
import base64
import json
import shutil
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import PROVIDERS, Calls, sha
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_prior_panel_trial import build_base, now, read, save
from scripts.run_reflection_diagnostic import reflection_body
from scripts.run_semantic_candidate_trial import PROPOSER
from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.candidate_coordinator import COMPONENTS
from surgical_agent.research.verification.reflective_repair import (
    accept_reflection,
    reflection_schema,
)


def references(excluded_video, pair):
    wanted = {c for c, p in COMPONENTS.items() if (p["instrument"], p["verb"]) == pair}
    bins = defaultdict(list)
    root = Path("D:/cholec_dataset/Training")
    for path in sorted(root.glob("*/*.json")):
        if path.parent.name == excluded_video or path.stem.lower() != path.parent.name.lower():
            continue
        raw = read(path)
        if raw["video"]["split"].lower() != "training":
            raise ValueError("references must be Training only")
        for frame, annotations in raw["annotations"].items():
            for a in annotations:
                c = a.get("triplet")
                if c not in wanted or a.get("visible") != 1 or a.get("tool_bbox", [0, 0, 0, 0])[2] <= 0:
                    continue
                image = path.parent / "Frames" / f"{int(frame):06d}.png"
                if image.exists():
                    bins[c].append({"video_id": path.parent.name, "frame_id": int(frame), "triplet_id": c,
                                    "components": COMPONENTS[c], "tool_bbox": a["tool_bbox"],
                                    "annotation_source": str(path), "image_path": str(image)})
    chosen = []
    for c in sorted(wanted):
        items = sorted(bins[c], key=lambda r: (r["video_id"], r["frame_id"]))
        if items:
            item = items[len(items) // 2]
            item.update(image_sha256=sha(item["image_path"]), annotation_sha256=sha(item["annotation_source"]))
            chosen.append(item)
    return chosen, sorted(c for c in wanted if not bins[c])


def reference_body(base, selected, h0, pool, gallery):
    body = reflection_body(base, selected, h0, pool)
    packet = json.loads(body["messages"][0]["content"][0]["text"])
    count = len(gallery)
    packet["reference_scope"] = (
        "The first images are labeled examples from OTHER Training videos, not earlier frames of this query. "
        "Their annotations belong only to their reference image and tool box. Use them to interpret dataset "
        "category boundaries, never copy their labels into the query. Re-evaluate actual query visual evidence."
    )
    packet["reference_images"] = [{"image_index": i, "source_video": r["video_id"], "source_frame": r["frame_id"],
        "annotated_tool_bbox_xywh_normalized": r["tool_bbox"], "annotated_triplet_id": r["triplet_id"],
        "annotated_relation": "/".join(_TASK_NAMES[q][v] for q, v in r["components"].items())}
        for i, r in enumerate(gallery)]
    packet["query_images"] = [{"image_index": count + i, "frame_id": f,
                               "seconds_relative_to_target": (f-selected["frame_id"])/25}
                              for i, f in enumerate(selected["causal_frame_ids"])]
    current = count + len(base.images) - 1
    packet["instructions"] = packet["instructions"].replace(
        f"Cite current image index {len(base.images)-1}", f"Cite current image index {current}")
    packet["response_schema"] = reflection_schema(pool, current + 1)
    refs = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(Path(r["image_path"]).read_bytes()).decode(),
                                                 "detail": "high"}} for r in gallery]
    body["messages"][0]["content"] = [{"type": "text", "text": json.dumps(packet, ensure_ascii=False)},
                                     *refs, *body["messages"][0]["content"][1:]]
    return body


def main(output):
    if output.exists():
        raise ValueError("new output directory required")
    source = ROOT / "artifacts/preflight/repair_v2_diagnostic_20260908_v1"
    previous = ROOT / "artifacts/preflight/repair_joint_reflection_20260908_v1"
    plan, cached, budget = read(source / "plan.json"), read(source / "predictions.json"), read(previous / "budget.json")
    selection, cached = plan["selection"][:2], cached[:2]
    if {s["video_id"] for s in selection} != {"VID103"} or not budget["stopped"]:
        raise ValueError("fixed diagnostic cohort changed or prior inference active")
    pairs = {(COMPONENTS[c]["instrument"], COMPONENTS[c]["verb"]) for r in cached for c in r["h0"]["ivt"]}
    if len(pairs) != 1:
        raise ValueError("diagnostic expects one common H0 I/V pair, not GT-selected classes")
    gallery, unavailable = references("VID103", next(iter(pairs)))
    if not gallery or any(r["video_id"] == "VID103" for r in gallery):
        raise ValueError("missing or contaminated references")
    limits = {k: Decimal(v) for k, v in budget["occupied"].items()}
    limits["openrouter_usd"] += Decimal("0.60")
    paths = [Path(__file__), ROOT / "scripts/run_reflection_diagnostic.py", ROOT / "scripts/run_candidate_panel_trial.py",
             ROOT / "src/surgical_agent/research/verification/reflective_repair.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    save(output / "plan.json", {"created_utc": now(), "selection": selection, "gallery": gallery,
        "missing_reference_classes": unavailable, "reference_rule": "one median sample per legal H0 instrument/verb target class, other Training videos",
        "max_calls": 2, "limits": {k: str(v) for k, v in limits.items()}, "source_sha256": hashes,
        "source_prediction_sha256": sha(source / "predictions.json"), "previous_ledger": str(previous),
        "reference_gt_in_requests": True, "query_gt_in_requests": False,
        "scope": "two known Training difficulty cases; supervised-reference diagnostic, not independent confirmation"})
    for p in paths:
        dest = output / "frozen_source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    print(json.dumps({"references": [(r["video_id"], r["frame_id"], r["triplet_id"]) for r in gallery],
                      "query_targets": [s["key"] for s in selection], "max_calls": 2}), flush=True)
    calls = Calls(output, budget["occupied"], limits=limits, providers={**PROVIDERS, PROPOSER: "Google AI Studio"}, max_calls=2)
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    rows = []
    for s, r in zip(selection, cached, strict=True):
        base = build_base(adapter, s)
        body = reference_body(base, s, r["h0"], r["pool"], gallery)
        raw = calls.call(s["key"], "labeled_reference_reflection", "base", body)
        final, status = r["h0"], "FAILED_KEEP"
        try:
            final = accept_reflection(r["h0"], r["pool"], raw, image_count=len(gallery)+len(base.images))
            status = "VALID_PATCH" if final != r["h0"] else "VALID_KEEP"
        except (ApiSchemaError, TypeError, ValueError, KeyError) as exc:
            status += ":" + type(exc).__name__
        save(output / "targets" / f"{s['key']}.json", {"raw": raw, "h0": r["h0"], "pool": r["pool"], "final": final, "status": status})
        rows.append({"video_id": s["video_id"], "frame_id": s["frame_id"], "h0": r["h0"], "h1": None, "final": final, "status": status})
        save(output / "predictions.json", rows)
        print(json.dumps({"target": s["key"], "status": status}), flush=True)
    calls.stopped = True
    calls.persist()
    frozen = sha(output / "predictions.json")
    if any(sha(ROOT / p) != v for p, v in hashes.items()):
        raise ValueError("inference source changed")
    report, truth = score_saved(adapter, rows)
    assert sha(output / "predictions.json") == frozen
    save(output / "scored_predictions.json", truth)
    save(output / "comparison.json", report)
    save(output / "summary.json", {"post_calls": len(calls.rows), "prediction_sha256": frozen, "comparison": report,
        "native_usd": str(sum(Decimal(r["charge"]) for r in calls.rows if r["charge_kind"] == "native"))})
    print(json.dumps(report["arms"]["final"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
