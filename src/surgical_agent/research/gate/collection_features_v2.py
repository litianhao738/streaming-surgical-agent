"""Frozen pre-review H0, leave-query-video-out prior and normalized track features.

No repair proposal, reviewer response or target GT is accepted. The five prior
features are explicitly defined here; they are not an unarchived exploratory fit.
"""
import math
from itertools import combinations

from surgical_agent.research.gate import mainline_training as v1
from surgical_agent.research.verification.prior_gated_repair import phase_rate

FEATURE_VERSION = "mainline_gate_h0_prior_tracker_v2"
PRIOR_FEATURES = ("prior_nonnull_ivt_count", "prior_ivt_rate_mean", "prior_ivt_rate_min",
                  "prior_ivt_veto_count", "prior_ivt_high_support_count")
SPATIAL_FEATURES = tuple("tracker_" + key for key in (
    "area_mean", "area_max", "center_x_mean", "center_y_mean", "edge_fraction",
    "pair_distance_min", "pair_available", "age_mean", "age_max", "matched_count",
    "displacement_mean", "displacement_max", "area_change_mean", "score_change_mean",
    "class_switch_fraction", "matched_iou_mean", "motion_available", "aspect_ratio_mean"))
FEATURE_ORDER = (*v1.FEATURE_ORDER, *PRIOR_FEATURES, *SPATIAL_FEATURES)
VIEWS = ("h0", "h0_prior", "h0_tracker_basic", "h0_tracker_full", "h0_prior_tracker_full", "h0_prior_tracker_no_scores")


def avg(values):
    return sum(values)/len(values) if values else 0.0


def box(track):
    x, y, w, h = track["bbox_tlwh"]
    if (not all(type(v) in (float, int) and math.isfinite(v) for v in (x, y, w, h))
            or min(x, y) < 0 or min(w, h) <= 0 or x+w > 1+1e-6 or y+h > 1+1e-6):
        raise ValueError("expected normalized clipped Tracker box")
    if type(track["age"]) is not int or track["age"] < 0:
        raise ValueError("invalid track age")
    return float(x), float(y), float(w), float(h)


def iou(a, b):
    x, y, w, h = a
    u, v, s, t = b
    intersect = max(0, min(x+w, u+s)-max(x, u))*max(0, min(y+h, v+t)-max(y, v))
    return intersect/(w*h+s*t-intersect)


def extract(h0, *, video_id, target_frame_id, causal_frame_ids, tracker_snapshot, prior, gate):
    if prior["excluded_video"] != video_id or video_id in prior["fit_videos"]:
        raise ValueError("query video must be excluded from prior")
    values = v1.extract_features(h0, target_frame_id=target_frame_id,
                                 causal_frame_ids=causal_frame_ids, tracker_snapshot=tracker_snapshot)
    labels = v1.canonical_labels(h0)
    rates = [float(phase_rate(prior, "ivt", c, labels["phase"][0])) for c in labels["ivt"] if c < 94]
    if any(not math.isfinite(r) or not 0 <= r <= 1 for r in rates):
        raise ValueError("invalid prior rates")
    values.update(dict(zip(PRIOR_FEATURES, (float(len(rates)), avg(rates), min(rates, default=0),
        float(sum(r < gate["veto_rate"] for r in rates)), float(sum(r >= gate["add_rate"] for r in rates))))))
    values.update(dict.fromkeys(SPATIAL_FEATURES, 0.0))
    if tracker_snapshot is not None:
        if tracker_snapshot["video_id"] != video_id:
            raise ValueError("Tracker video mismatch")
        frames = tracker_snapshot["frames"]
        current = {t["track_id"]: t for t in frames[-1]["tracks"]}
        previous = {t["track_id"]: t for t in frames[-2]["tracks"]} if len(frames) > 1 else {}
        boxes = {k: box(t) for k, t in current.items()}
        old_boxes = {k: box(t) for k, t in previous.items()}
        centers = {k: (b[0]+b[2]/2, b[1]+b[3]/2) for k, b in boxes.items()}
        areas = [b[2]*b[3] for b in boxes.values()]
        distances = [math.dist(a,b)/math.sqrt(2) for a,b in combinations(centers.values(),2)]
        matched = sorted(set(current) & set(previous))
        dt = (frames[-1]["frame_id"]-frames[-2]["frame_id"])/25 if len(frames)>1 else 1
        motion = [math.dist(centers[k], (old_boxes[k][0]+old_boxes[k][2]/2, old_boxes[k][1]+old_boxes[k][3]/2))/math.sqrt(2)/dt for k in matched]
        features = (avg(areas), max(areas,default=0), avg([c[0] for c in centers.values()]),
                    avg([c[1] for c in centers.values()]),
                    avg([float(min(b[0],b[1],1-b[0]-b[2],1-b[1]-b[3]) <= .05) for b in boxes.values()]),
                    min(distances,default=0), float(bool(distances)),
                    avg([t["age"] for t in current.values()]), max([t["age"] for t in current.values()],default=0),
                    float(len(matched)), avg(motion), max(motion,default=0),
                    avg([abs(boxes[k][2]*boxes[k][3]-old_boxes[k][2]*old_boxes[k][3]) for k in matched]),
                    avg([abs(current[k]["score"]-previous[k]["score"]) for k in matched]),
                    avg([float(current[k]["instrument_id"] != previous[k]["instrument_id"]) for k in matched]),
                    avg([iou(boxes[k],old_boxes[k]) for k in matched]), float(bool(matched)),
                    avg([b[2]/b[3] for b in boxes.values()]))
        values.update({k:float(v) for k,v in zip(SPATIAL_FEATURES,features,strict=True)})
    validate(values)
    return values


def validate(values):
    if set(values) != set(FEATURE_ORDER) or any(type(v) not in (int,float) or not math.isfinite(v) for v in values.values()):
        raise ValueError("v2 feature names or finite numeric contract violated")


def view(values, name):
    validate(values)
    if name not in VIEWS:
        raise ValueError("unknown feature view")
    keep = set(v1.BASE_FEATURES)
    if "prior" in name:
        keep.update(PRIOR_FEATURES)
    if "tracker" in name:
        keep.update(v1.TRACKER_FEATURES)
        if name != "h0_tracker_basic":
            keep.update(SPATIAL_FEATURES)
    if name.endswith("no_scores"):
        keep.difference_update(("tracker_detector_score_mean", "tracker_detector_score_min", "tracker_score_change_mean"))
    return {k:float(values[k]) if k in keep else 0.0 for k in FEATURE_ORDER}
