"""Deterministic auxiliary views from causal predicted boxes, never annotation boxes."""
from __future__ import annotations

import base64
import io
import json
import math
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

VERSION = "oof_auxiliary_candidate_views_v1"
ARMS = ("control", "roi_current", "roi_temporal")
MIN_SCORE = 0.5
MAX_REGIONS = 2
PADDING = 0.35
TILE_SIZE = (384, 288)


def regions(tracks):
    """Use at most two high-score tool boxes; scores/classes never enter the LLM."""
    chosen = []
    for track in sorted(tracks, key=lambda t: (-t["score"], t["track_id"])):
        box = track["bbox_tlwh"]
        if (len(box) != 4 or any(not math.isfinite(x) for x in box)
                or not math.isfinite(track["score"]) or track["score"] < MIN_SCORE):
            continue
        x, y, w, h = box
        if w <= 0 or h <= 0:
            continue
        px, py = max(w * PADDING, 0.06), max(h * PADDING, 0.06)
        rect = [max(0, x - px), max(0, y - py), min(1, x + w + px), min(1, y + h + py)]
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            continue
        chosen.append({"track_id": track["track_id"], "rect_xyxy_normalized": rect})
        if len(chosen) == MAX_REGIONS:
            break
    return chosen


def create_views(selected, tracks, output):
    """Current ROI or three-column fixed-coordinate causal ROI strip.

    Past views use the SAME normalized region, not a claimed tracked contact point.
    No synthetic motion, object identity, or tool-tip location is inferred.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    images = [Image.open(im["path"]).convert("RGB") for im in selected["images"]]
    ids = selected["causal_frame_ids"]
    if len(images) != 3 or ids != [selected["frame_id"] - 50, selected["frame_id"] - 25, selected["frame_id"]]:
        raise ValueError("exact three real causal frames required")
    result = {arm: [] for arm in ARMS}
    for index, region in enumerate(regions(tracks)):
        rect = region["rect_xyxy_normalized"]
        crops = []
        for image in images:
            box = (math.floor(rect[0] * image.width), math.floor(rect[1] * image.height),
                   math.ceil(rect[2] * image.width), math.ceil(rect[3] * image.height))
            crops.append(image.crop(box))
        current = crops[-1].copy()
        current.thumbnail((768, 768))
        current_path = output / f"region_{index}_current.jpg"
        current.save(current_path, quality=95)
        result["roi_current"].append({"path": str(current_path.resolve()), "region": region,
                                      "source_frame_ids": [ids[-1]], "layout": "single current-frame crop"})
        strip = Image.new("RGB", (TILE_SIZE[0] * 3, TILE_SIZE[1] + 24), "black")
        draw = ImageDraw.Draw(strip)
        for column, (crop, frame) in enumerate(zip(crops, ids, strict=True)):
            tile = ImageOps.contain(crop, TILE_SIZE)
            left = column * TILE_SIZE[0] + (TILE_SIZE[0] - tile.width) // 2
            strip.paste(tile, (left, 24 + (TILE_SIZE[1] - tile.height) // 2))
            draw.text((column * TILE_SIZE[0] + 8, 5), f"{(frame - ids[-1]) / 25:g}s", fill="white")
        temporal_path = output / f"region_{index}_temporal.jpg"
        strip.save(temporal_path, quality=95)
        result["roi_temporal"].append({"path": str(temporal_path.resolve()), "region": region,
            "source_frame_ids": ids, "layout": "left -2s, middle -1s, right current; same coordinates, not tracked identity"})
    return result


def augment_proposal(body, views):
    if not views:
        return deepcopy(body)
    out = deepcopy(body)
    content = out["messages"][0]["content"]
    packet = json.loads(content[0]["text"])
    original_count = sum(b.get("type") == "image_url" for b in content)
    packet["auxiliary_visual_evidence"] = {
        "instructions": "The first three images remain the original causal full frames; image 2 is the full target. "
        "Additional images are deterministic crops from those same frames, not later events or new tools. "
        "Use them only to inspect visible tool-tissue boundaries and motion; return current-frame candidates only. "
        "The predicted regions are imperfect localization hints, not tool-tip/contact annotations or evidence of a class. "
        "A region may contain no contact. Check full frames for all other tools. Do not count repeated views as additional instances. "
        "No extra candidates are required. Keep the original ontology, output schema and candidate limits.",
        "views": [{"image_index": original_count + i, "source_frame_ids": v["source_frame_ids"],
                   "region_xyxy_normalized": v["region"]["rect_xyxy_normalized"], "layout": v["layout"]}
                  for i, v in enumerate(views)]}
    content[0]["text"] = json.dumps(packet, ensure_ascii=False)
    for view in views:
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
            base64.b64encode(Path(view["path"]).read_bytes()).decode(), "detail": "high"}})
    return out


def augment_review(body, views):
    """Keep three image identities; target panel contains full scene plus ROI zooms.

    Full target pixels are pasted without downscaling; original history is unchanged.
    Review image index 2 still denotes the target, preserving existing evidence rules.
    """
    if not views:
        return deepcopy(body)
    out = deepcopy(body)
    content = out["messages"][0]["content"]
    blocks = [b for b in content if b.get("type") == "image_url"]
    if len(blocks) != 3:
        raise ValueError("three original image identities required")
    url = blocks[-1]["image_url"]["url"]
    full = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    gallery = Image.new("RGB", (full.width + 256, full.height), "black")
    gallery.paste(full, (0, 0))
    draw = ImageDraw.Draw(gallery)
    row_height = full.height // len(views)
    for i, view in enumerate(views):
        crop = Image.open(view["path"]).convert("RGB")
        tile = ImageOps.contain(crop, (256, max(1, row_height - 22)))
        gallery.paste(tile, (full.width + (256 - tile.width) // 2, i * row_height + 22))
        draw.text((full.width + 4, i * row_height + 4), f"Target ROI {i + 1}", fill="white")
    if gallery.width * gallery.height > 1_000_000:
        raise ValueError("gallery exceeds the fixed image accounting envelope")
    buffer = io.BytesIO()
    gallery.save(buffer, format="PNG")
    blocks[-1]["image_url"]["url"] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    packet = json.loads(content[0]["text"])
    packet["target_view_layout"] = {
        "image_index": 2, "full_scene_width_pixels": full.width,
        "instructions": "Target image 2 has the unchanged full scene on the LEFT and enlarged crops of the SAME target "
        "on the RIGHT. These are repeated views, not additional tools or later events. Predicted boxes only localize "
        "possible tool regions and do not prove any class or contact. Judge propositions using actual visible evidence. "
        "WHOLE_FRAME absence still requires inspecting the complete left scene, not just the crops. "
        "Images 0 and 1 are unchanged causal history. All output and acceptance rules remain the same.",
        "regions_xyxy_normalized": [v["region"]["rect_xyxy_normalized"] for v in views]}
    content[0]["text"] = json.dumps(packet, ensure_ascii=False)
    return out
