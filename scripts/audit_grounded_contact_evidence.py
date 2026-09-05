"""Offline audit of a completed grounded-contact run; never makes API calls.

GT geometry is diagnostic only, not contact-point ground truth or inference input.
Reads release-native video annotations; use the normal evaluator for derived labels.
"""

import argparse
import base64
import hashlib
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgical_agent.api.contracts import ApiImageInput
from surgical_agent.data.label_policy import is_valid_task_id
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.perception.context_builder import encode_rgb_png
from surgical_agent.perception.ontology_prompt import _ivt_rows
from surgical_agent.research.verification.grounded_repair import make_contact_crops


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@lru_cache(maxsize=32)
def request_image_bytes(path):
    # Requests contain a canonical lossless RGB re-encoding, not the disk PNG
    # container bytes. Reuse the actual normalization and encoder for verification.
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    frame = CausalApiMediaLoader._normalize_rgb_window((pixels,))[0]
    encoded = encode_rgb_png(frame)
    with Image.open(io.BytesIO(encoded)) as decoded:
        if not np.array_equal(pixels, np.asarray(decoded)):
            raise ValueError("Request encoding changed original RGB pixels")
    return encoded


def snapshot(directory):
    return {str(p.relative_to(directory)): digest(p.read_bytes())
            for p in sorted(directory.rglob("*")) if p.is_file()}


def geometry(box, gt):
    x, y, w, h = gt
    other = [x, y, x + w, y + h]
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    intersection = (max(0, min(box[2], other[2]) - max(box[0], other[0])) *
                    max(0, min(box[3], other[3]) - max(box[1], other[1])))
    union = area + w * h - intersection
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return {"iou": intersection / union if union else 0,
            "box_fraction_inside_gt": intersection / area if area else 0,
            "center_inside_gt": x <= cx <= x + w and y <= cy <= y + h}


def run(args):
    if args.output.exists():
        raise ValueError("Fresh output required; completed evidence must be preserved")
    if args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("Audit output must be outside the source run")
    before = snapshot(args.run)
    annotation = args.video / (args.video.name.lower() + ".json")
    raw_bytes = annotation.read_bytes()
    raw = json.loads(raw_bytes)
    rows = read(args.run / "predictions.json")
    summary = read(args.run / "summary.json")
    components = {r[0]: r[1:] for r in _ivt_rows()}
    tasks = ("instrument", "verb", "target", "ivt", "phase")
    scored = {r["frame_id"]: r for r in summary["evaluation"]["h0"]["frames"]}
    totals = {arm: {t: dict(tp=0, fp=0, fn=0, exact=0, valid=0) for t in tasks}
              for arm in ("h0", "final")}
    args.output.mkdir(parents=True)
    findings, verified_images, usage = [], 0, []
    for row in rows:
        fid = row["frame_id"]
        instances = raw["annotations"][str(fid)]
        gt = {t: sorted({i["triplet" if t == "ivt" else t] for i in instances
                         if is_valid_task_id(t, i["triplet" if t == "ivt" else t])}) for t in tasks}
        # This native-data audit intentionally refuses partial labels. The original
        # scoring masks, rather than empty sets, must govern any broader audit.
        if any(not is_valid_task_id(t, i["triplet" if t == "ivt" else t])
               for i in instances for t in tasks):
            raise ValueError("Partial labels require the adapter's task-wise mask evaluator")
        if gt != scored[fid]["gt"] or not all(scored[fid]["mask"].values()):
            raise ValueError("Release-native GT differs from saved scoring evidence")
        if any(tuple(i[t] for t in tasks[:3]) != components[i["triplet"]] for i in instances):
            raise ValueError("Native IVT component conflict requires separate scope audit")
        for arm in totals:
            for t in tasks:
                pred, truth = set(row[arm][t]), set(gt[t])
                values = dict(tp=len(pred & truth), fp=len(pred - truth), fn=len(truth - pred),
                              exact=int(pred == truth), valid=1)
                for k, v in values.items():
                    totals[arm][t][k] += v
        h0_request = read(args.run / "calls" / f"{fid}_h0" / "request.json")
        meta = h0_request["metadata"]["images"][-1]
        target = ApiImageInput(meta["identifier"], meta["mime_type"],
                               request_image_bytes(args.video / "Frames" / f"{fid:06d}.png"))
        crops, manifest = make_contact_crops(target, row["locator"], target_frame_id=fid)
        if manifest != row["crop_manifest"]:
            raise ValueError("Reconstructed crop manifest mismatch")
        crop_bytes = {c.identifier: c.content for c in crops}
        with Image.open(io.BytesIO(target.content)) as im:
            width, height = im.size
        pairs = []
        for loc, crop, entry in zip(row["locator"]["instances"], crops, manifest):
            (args.output / f"{fid}_instance{loc['instance_id']}.png").write_bytes(crop.content)
            crop_box = [p / (width if n % 2 == 0 else height)
                        for n, p in enumerate(entry["pixel_box_xyxy"])]
            pairs.append({"instance_id": loc["instance_id"], "tip_box": loc["tip_box"],
                          "all_gt_pairs": [{"gt_index": n, "instrument": item["instrument"],
                                            "tip": geometry(loc["tip_box"], item["tool_bbox"]),
                                            "padded_crop": geometry(crop_box, item["tool_bbox"])}
                                           for n, item in enumerate(instances)]})
        calls = []
        for directory in sorted((args.run / "calls").glob(f"{fid}_*")):
            req, wire = read(directory / "request.json"), read(directory / "wire_request.json")
            content = wire["messages"][1]["content"]
            texts = [c["text"] for c in content if c["type"] == "text"]
            if texts != [req["payload"]["input_text"]]:
                raise ValueError("Wire input differs from saved request")
            if wire["messages"][0]["content"] != req["payload"]["system_text"]:
                raise ValueError("Wire system instruction differs from saved request")
            images = [c["image_url"] for c in content if c["type"] == "image_url"]
            metadata = req["metadata"]["images"]
            if len(images) != len(metadata):
                raise ValueError("Wire image count mismatch")
            for n, (wire_image, image_meta) in enumerate(zip(images, metadata)):
                ident = image_meta["identifier"]
                data = crop_bytes.get(ident)
                if data is None:
                    frame_id = int(ident.rsplit(":", 1)[1])
                    data = request_image_bytes(args.video / "Frames" / f"{frame_id:06d}.png")
                uri = f"data:{image_meta['mime_type']};base64," + base64.b64encode(data).decode("ascii")
                if digest(data) != image_meta["sha256"] or len(data) != image_meta["size_bytes"]:
                    raise ValueError("Source image hash/size mismatch")
                if digest(uri.encode()) != wire_image["data_url_sha256"]:
                    raise ValueError("Actual wire image digest mismatch")
                if wire_image["detail"] != req["payload"]["image_details"][n]:
                    raise ValueError("Actual wire image detail mismatch")
                verified_images += 1
            data = json.loads(texts[0])
            stage = directory.name.split("_", 1)[1]
            calls.append({"stage": stage, "input_keys": sorted(data),
                          "image_details": [i["detail"] for i in images],
                          "request_hash": req["metadata"]["request_hash"]})
            usage.extend(read_jsonl(directory / "api_usage.jsonl"))
        findings.append({"frame_id": fid, "raw_gt_instances": instances, "gt": gt,
                         "h0": row["h0"], "final": row["final"], "geometry": pairs,
                         "calls": calls, "proposal": row["proposal"], "review": row["review"],
                         "admission": row["admission"]})
    for arm in totals.values():
        for v in arm.values():
            v["micro_precision"] = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else None
            denom = 2 * v["tp"] + v["fp"] + v["fn"]
            v["micro_f1"] = 2 * v["tp"] / denom if denom else None
    if before != snapshot(args.run) or digest(raw_bytes) != digest(annotation.read_bytes()):
        raise ValueError("Source evidence changed during audit")
    output = {"scope": "offline native-label evidence audit; geometry is not contact-point GT",
              "source_run": str(args.run.resolve()), "source_sha256": before,
              "annotation_sha256": digest(raw_bytes), "source_unchanged": True,
              "verified_wire_image_occurrences": verified_images,
              "metrics": totals, "frames": findings,
              "provider_calls": sum(u["provider_call_count"] for u in usage),
              "ledger_cost": round(sum(u.get("provider_cost") or 0 for u in usage), 6)}
    (args.output / "audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k not in {"frames", "source_sha256"}}, indent=2))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "video", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
