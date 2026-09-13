"""Offline value-of-verification models. Only frozen pre-review inputs are used.

No transport, dataset loader, deployment hook, or paid model is imported here.
"""
from __future__ import annotations

from collections import defaultdict
import time

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from surgical_agent.data.constants import TASK_ID_BOUNDS
from surgical_agent.research.gate import official_net_training as old

VERSION = "accuracy-constrained-value-gate-v4-20260913"
TASKS = old.TASKS
SIZES = np.array([TASK_ID_BOUNDS[t][1] + 1 for t in TASKS], float)
SEMANTIC = tuple(f"{kind}_{t}_{i}" for kind in ("h0", "cheap", "added", "removed")
                 for t in TASKS for i in range(int(SIZES[TASKS.index(t)])))
HISTORY = ("previous_available", "log_frame_gap", "previous_cheap_equal",
           *(f"previous_{t}_symmetric_difference" for t in TASKS),
           *(f"previous_{t}_equal" for t in TASKS))
VARIANTS = {
    "value_semantic_history_robust": {"view": "temporal", "target": "bh", "kappa": 1.0},
    "value_counts": {"view": "counts", "target": "bh", "kappa": 1.0},
    "value_semantic": {"view": "semantic", "target": "bh", "kappa": 1.0},
    "value_semantic_history_no_penalty": {"view": "temporal", "target": "bh", "kappa": 0.0},
    "direct_error_gain": {"view": "temporal", "target": "gain", "kappa": 0.0},
    "weighted_gain_classifier": {"view": "temporal", "target": "logistic", "kappa": 0.0},
}
PRIMARY = "value_semantic_history_robust"


def recipe():
    return {
        "version": VERSION, "datasets": {"primary": 6059, "sensitivity": 7053, "strict_complete": 7008},
        "videos": list(old.VIDEOS), "primary_variant": PRIMARY, "variants": VARIANTS,
        "ridge_alphas": [1.0, 10.0, 100.0],
        "cost_lambdas": [0.0, .0025, .005, .01, .02, .04, .08, .16, .32],
        "classifier_thresholds": [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95],
        "classifier": "L2-regularized logistic C=1; weight=video_weight*abs(error_gain); neutral rows excluded",
        "video_weights": "mean-one n/(number_of_videos*n_video); scaler and estimator both weighted",
        "uncertainty": "population std of gain predictions from leave-one-fitting-video-out Ridge models; not a confidence bound",
        "support_guard": "skip when any active h0/cheap semantic label bit was never active in this model's fitting videos; counts variant has no identity guard",
        "label": "per-head corrected and corrupted label bits, B-H equals cheap_errors-full_errors",
        "feature_order": {v: feature_order(v) for v in ("counts", "semantic", "temporal")},
        "history": "only immediately preceding original-cohort target's cheap labels; missing observation resets history; sorted frame_id then sample_id",
        "proposal_identity": "not used: common sealed row schema lacks original-attempt pre-Gate proposal identities",
        "history_uses_review_results": False, "feature_view": "no_tracker",
        "outer_split": "four-video LOVO", "inner_split": "LOVO of the other three videos",
        "selection": "minimum logical calls among pooled F1>=max(cheap,full), errors<=min(cheap,full), and each-video no harm vs cheap; tie max worst-video F1 gain, max pooled F1, larger regularization and threshold",
        "f1_tolerance_percentage_points": 1e-10, "allowed_error_increase": 0,
        "no_feasible_policy": "skip all, mark INFEASIBLE; never relax accuracy",
        "random_controls": {"draws": 1000, "seed": 3407, "block_sizes": [15, 30, 60], "primary_block_size": 30, "quantile": "linear"},
        "acceptance": "quality constraints, strictly cheaper than full, positive review count, F1>30-block p97.5 AND errors<30-block p2.5; all outer selections feasible",
        "cost": "primary logical calls; frozen known-USD scenario and unpriced GLM/DS request counts; no live price lookup",
        "old_v2_budget_cap": "diagnostic cap on saved outer routes only; not a refit of the proposed v3 selector",
        "unknown": "never imputed; full 7372 cohort retained; coverage and historical investment reported separately",
        "curve": "vary fixed cost lambda on outer-fold selected models; descriptive only, no workpoint selection from curves",
        "upstream_fully_nested": False, "statistical_precision_guarantee": False,
        "Testing_access": False, "VID110_access": False, "api_calls": 0, "deployable": False,
        "registration": "frozen before this fit after prior exploration of the same four Training videos",
    }


def feature_order(view):
    result = list(old.FEATURE_ORDER)
    if view in ("semantic", "temporal"):
        result += list(SEMANTIC)
    if view == "temporal":
        result += list(HISTORY)
    if view not in ("counts", "semantic", "temporal"):
        raise ValueError("unknown view")
    return result


def checked_labels(value):
    if set(value) != set(TASKS):
        raise ValueError("five label heads required")
    result = {}
    for t, size in zip(TASKS, SIZES):
        seq = value[t]
        if len(seq) != len(set(seq)) or any(type(i) is not int or not 0 <= i < size for i in seq):
            raise ValueError("invalid labels")
        if t == "phase" and len(seq) != 1:
            raise ValueError("one phase required")
        result[t] = set(seq)
    return result


def features(rows, cohort):
    """Project to pre-review fields before any history or feature construction."""
    projected = {r["sample_id"]: {"h0": checked_labels(r["h0_labels"]),
                 "cheap": checked_labels(r["cheap_labels"]), "counts": r["features_postcheap"]} for r in rows}
    positions = {r["sample_id"]: i for i, r in enumerate(rows)}
    arrays = {v: np.zeros((len(rows), len(feature_order(v)))) for v in ("counts", "semantic", "temporal")}
    grouped = defaultdict(list)
    for member in cohort:
        grouped[member["video_id"]].append(member)
    seen = set()
    for video in sorted(grouped):
        previous = None
        for member in sorted(grouped[video], key=lambda m: (m["frame_id"], m["key"])):
            key = member["key"]
            if key not in projected:
                previous = None
                continue
            seen.add(key)
            row = projected[key]
            base = [float(row["counts"][k]) for k in old.FEATURE_ORDER]
            maps = {"h0": row["h0"], "cheap": row["cheap"],
                    "added": {t: row["cheap"][t] - row["h0"][t] for t in TASKS},
                    "removed": {t: row["h0"][t] - row["cheap"][t] for t in TASKS}}
            bits = [float(i in maps[kind][t]) for kind in maps for t, size in zip(TASKS, SIZES) for i in range(int(size))]
            history = [0.] * len(HISTORY)
            if previous is not None:
                prev, frame = previous
                gap = member["frame_id"] - frame
                history = [1., float(np.log1p(min(gap, 100000))), float(row["cheap"] == prev),
                           *[float(len(row["cheap"][t] ^ prev[t])) for t in TASKS],
                           *[float(row["cheap"][t] == prev[t]) for t in TASKS]]
            i = positions[key]
            arrays["counts"][i] = base
            arrays["semantic"][i] = base + bits
            arrays["temporal"][i] = base + bits + history
            previous = row["cheap"], member["frame_id"]
    if seen != set(positions) or any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("feature membership or finite-value mismatch")
    return arrays


def targets(rows):
    out = np.zeros((len(rows), 10), float)
    for i, r in enumerate(rows):
        if any(r["mask"].get(t) is not True for t in TASKS):
            raise ValueError("unknown supervision forbidden in regression")
        a, b, g = [checked_labels(r[k]) for k in ("cheap_labels", "final_labels", "gt")]
        for j, t in enumerate(TASKS):
            out[i, j] = len((a[t] - b[t]) - g[t]) + len((b[t] - a[t]) & g[t])
            out[i, j + 5] = len((b[t] - a[t]) - g[t]) + len((a[t] - b[t]) & g[t])
    return out


def prepare_arrays(rows, cohort):
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate rows")
    for r in rows:
        if r["source_split"] != "Training" or r["video_id"] not in old.VIDEOS:
            raise ValueError("only frozen Training videos allowed")
        if set(r["features_postcheap"]) != set(old.FEATURE_ORDER):
            raise ValueError("feature schema drift")
        if any(r["features_postcheap"][k] != 0 for k in old.FEATURE_ORDER if k.startswith("tracker_")):
            raise ValueError("Tracker not masked")
        if old.derive_labels(r["cheap_labels"], r["final_labels"], r["gt"], r["mask"]) != r["labels"]:
            raise ValueError("sealed utility mismatch")
    data = {"X_" + k: v for k, v in features(rows, cohort).items()}
    data.update(y=targets(rows), cheap=old.counts(rows, "cheap_labels"), full=old.counts(rows, "final_labels"),
                merged_cheap=old.counts(rows, "cheap_labels", True), merged_full=old.counts(rows, "final_labels", True),
                ids=np.array([r["sample_id"] for r in rows]), videos=np.array([r["video_id"] for r in rows]),
                frames=np.array([r["frame_id"] for r in rows]), rules=np.array([bool(r["features_postcheap"]["proposal_rule"]) for r in rows]))
    gain = data["y"][:, :5].sum(axis=1) - data["y"][:, 5:].sum(axis=1)
    assert np.array_equal(gain, (data["cheap"] - data["full"])[:, :, 1:].sum(axis=(1, 2)))
    return data


def video_weights(videos):
    unique = np.unique(videos)
    return np.array([len(videos) / (len(unique) * np.sum(videos == v)) for v in videos])


def fit_model(X, y, videos, alpha, target, names):
    weights = video_weights(videos)
    support_columns = [i for i, name in enumerate(names) if name in SEMANTIC and (name.startswith("h0_") or name.startswith("cheap_"))]
    unseen = [i for i in support_columns if not np.any(X[:, i] > 0)]
    if target == "logistic":
        gain = y[:, :5].sum(axis=1) - y[:, 5:].sum(axis=1)
        selected = gain != 0
        if not selected.any():
            return {"constant": 0., "target": target, "unseen_columns": unseen}
        X, weights, gain = X[selected], weights[selected] * np.abs(gain[selected]), gain[selected]
        labels = (gain > 0).astype(int)
        if len(np.unique(labels)) == 1:
            return {"constant": float(labels[0]), "target": target, "unseen_columns": unseen}
        scaler = StandardScaler().fit(X, sample_weight=weights)
        model = LogisticRegression(C=1., solver="liblinear", max_iter=2000, random_state=3407).fit(scaler.transform(X), labels, sample_weight=weights)
    else:
        labels = y if target == "bh" else (y[:, :5].sum(axis=1) - y[:, 5:].sum(axis=1))[:, None]
        scaler = StandardScaler().fit(X, sample_weight=weights)
        model = Ridge(alpha=alpha, solver="cholesky").fit(scaler.transform(X), labels, sample_weight=weights)
    coefficients = np.atleast_2d(model.coef_) / scaler.scale_[None, :]
    intercept = np.atleast_1d(model.intercept_) - coefficients @ scaler.mean_
    return {"target": target, "weights": coefficients.tolist(), "bias": intercept.tolist(), "unseen_columns": unseen}


def model_values(model, X):
    if "constant" in model:
        return np.full(len(X), model["constant"])
    raw = X @ np.array(model["weights"]).T + np.array(model["bias"])
    if model["target"] == "bh":
        values = np.clip(raw, 0, np.tile(SIZES, 2))
        return values[:, :5].sum(axis=1) - values[:, 5:].sum(axis=1)
    if model["target"] == "logistic":
        return 1 / (1 + np.exp(-np.clip(raw[:, 0], -700, 700)))
    return np.clip(raw[:, 0], -SIZES.sum(), SIZES.sum())


def predict_bundle(bundle, X):
    center = model_values(bundle["center"], X)
    uncertainty = np.std([model_values(m, X) for m in bundle["submodels"]], axis=0) if bundle["submodels"] else np.zeros(len(X))
    columns = bundle["center"]["unseen_columns"]
    supported = ~np.any(X[:, columns] > 0, axis=1) if columns else np.ones(len(X), bool)
    return center, uncertainty, supported


class ModelCache:
    def __init__(self, data, variant):
        self.data, self.variant, self.cache = data, variant, {}
        self.X = data["X_" + variant["view"]]
        self.names = feature_order(variant["view"])

    def single(self, videos, alpha):
        key = tuple(sorted(videos)), alpha
        if key not in self.cache:
            fit = np.isin(self.data["videos"], videos)
            self.cache[key] = fit_model(self.X[fit], self.data["y"][fit], self.data["videos"][fit], alpha, self.variant["target"], self.names)
        return self.cache[key]

    def bundle(self, videos, alpha):
        videos = sorted(videos)
        return {"center": self.single(videos, alpha),
                "submodels": [self.single([w for w in videos if w != v], alpha) for v in videos] if self.variant["kappa"] else [],
                "fit_videos": videos, "alpha": alpha, "feature_order": self.names,
                "kappa": self.variant["kappa"], "deployable": False}


def route_from_scores(mean, uncertainty, supported, rules, parameter, variant):
    if parameter is None:
        return np.zeros(len(mean), bool)
    if variant["target"] == "logistic":
        return supported & (mean >= parameter)
    score = mean - variant["kappa"] * uncertainty
    return supported & (score > parameter * (12 - rules.astype(int)))


def cheap_cap(route, videos, frames, ids, q=.2):
    capped = np.zeros(len(route), bool)
    for v in sorted(set(videos)):
        order = sorted(np.where(videos == v)[0], key=lambda i: (frames[i], ids[i]))
        count = 0
        for t, i in enumerate(order, 1):
            if route[i] and count + 1 <= int(np.floor(q * t + 1e-12)):
                capped[i] = True
                count += 1
    return capped


def quality(data, indices, route):
    c, f, vids = data["cheap"][indices], data["full"][indices], data["videos"][indices]
    chosen = np.where(route[:, None, None], f, c)
    before, after, all_review = old.pooled(c), old.pooled(chosen), old.pooled(f)
    gains, by_video = [], {}
    for v in sorted(set(vids)):
        mask = vids == v
        a, b = old.pooled(c[mask]), old.pooled(chosen[mask])
        df = b["five_head_mean_f1"] - a["five_head_mean_f1"]
        de = b["total_errors"] - a["total_errors"]
        by_video[v] = {"delta_f1": df, "delta_errors": de, "reviewed": int(route[mask].sum())}
        gains.append(df)
    checks = {"f1_at_least_cheap_and_full": after["five_head_mean_f1"] + 1e-10 >= max(before["five_head_mean_f1"], all_review["five_head_mean_f1"]),
              "errors_at_most_cheap_and_full": after["total_errors"] <= min(before["total_errors"], all_review["total_errors"]),
              "each_video_no_harm": all(r["delta_f1"] >= -1e-10 and r["delta_errors"] <= 0 for r in by_video.values())}
    calls = int((1 + data["rules"][indices].astype(int) + route * (12 - data["rules"][indices].astype(int))).sum())
    return {**after, "reviewed": int(route.sum()), "logical_calls": calls, "checks": checks,
            "feasible": all(checks.values()), "worst_video_delta_f1": min(gains), "by_video_delta": by_video}


def outer_fold(data, held_video, variant, config, cache=None):
    """Held-out labels are never indexed during model/parameter selection."""
    cache = cache or ModelCache(data, variant)
    fit_videos = sorted(set(data["videos"]) - {held_video})
    fit = np.where(data["videos"] != held_video)[0]
    held = np.where(data["videos"] == held_video)[0]
    alphas = [1.] if variant["target"] == "logistic" else config["ridge_alphas"]
    parameters = config["classifier_thresholds"] if variant["target"] == "logistic" else config["cost_lambdas"]
    candidates, inner_archive = [], []
    # All-skip competes honestly on cost if it already meets accuracy.
    candidates.append({"alpha": alphas[0], "parameter": None, **quality(data, fit, np.zeros(len(fit), bool))})
    for alpha in alphas:
        means, spreads = np.zeros(len(fit)), np.zeros(len(fit))
        supports = np.zeros(len(fit), bool)
        for inner_video in fit_videos:
            local = np.where(data["videos"][fit] == inner_video)[0]
            bundle = cache.bundle([v for v in fit_videos if v != inner_video], alpha)
            means[local], spreads[local], supports[local] = predict_bundle(bundle, cache.X[fit[local]])
        inner_archive.append({"alpha": alpha, "sample_ids": data["ids"][fit].tolist(), "mean": means.tolist(), "uncertainty": spreads.tolist(), "supported": supports.tolist()})
        for parameter in parameters:
            route = route_from_scores(means, spreads, supports, data["rules"][fit], parameter, variant)
            candidates.append({"alpha": alpha, "parameter": parameter, **quality(data, fit, route)})
    feasible = [r for r in candidates if r["feasible"]]
    selected = min(feasible, key=lambda r: (r["logical_calls"], -r["worst_video_delta_f1"], -r["five_head_mean_f1"], -r["alpha"], -float(r["parameter"] if r["parameter"] is not None else 1e9))) if feasible else candidates[0]
    selected = {**selected, "selection_state": "FEASIBLE" if feasible else "INFEASIBLE_SKIP_ALL"}
    bundle = cache.bundle(fit_videos, selected["alpha"])
    mean, spread, support = predict_bundle(bundle, cache.X[held])
    route = route_from_scores(mean, spread, support, data["rules"][held], selected["parameter"], variant)
    fold = {"held_video": held_video, "fit_videos": fit_videos, "selected": selected, "candidates": candidates,
            "inner_scores": inner_archive, "held_indices": held.tolist(), "mean": mean.tolist(), "uncertainty": spread.tolist(),
            "supported": support.tolist(), "route": route.tolist(), "model": bundle}
    return fold


def route_report(data, route, costs):
    out = quality(data, np.arange(len(route)), route)
    out["review_fraction"] = float(np.mean(route))
    out["cost"] = old.policy_cost(route, data["rules"], costs)
    out["merged_grasp_retract"] = old.pooled(np.where(route[:, None, None], data["merged_full"], data["merged_cheap"]))
    out["by_video"] = {}
    gain = data["y"][:, :5].sum(axis=1) - data["y"][:, 5:].sum(axis=1)
    out["selected_error_gain"] = float(gain[route].sum())
    out["selected_with_any_harm"] = int((route & (data["y"][:, 5:].sum(axis=1) > 0)).sum())
    for v in sorted(set(data["videos"])):
        ix = data["videos"] == v
        chosen = np.where(route[ix, None, None], data["full"][ix], data["cheap"][ix])
        out["by_video"][v] = {**old.pooled(chosen), "reviewed": int(route[ix].sum()), "targets": int(ix.sum()),
            "merged_grasp_retract": old.pooled(np.where(route[ix, None, None], data["merged_full"][ix], data["merged_cheap"][ix])),
            "cost": old.policy_cost(route[ix], data["rules"][ix], costs)}
    out["video_equal_mean_delta_f1"] = float(np.mean([r["delta_f1"] for r in out["by_video_delta"].values()]))
    return out


def random_reference(data, route, costs, config):
    draws = config["random_controls"]["draws"]
    seed = config["random_controls"]["seed"]
    samples = defaultdict(list)
    for block in config["random_controls"]["block_sizes"]:
        for masks in old.control_masks(route, data["videos"], data["frames"].tolist(), draws=draws, block=block, seed=seed):
            for name, mask in masks.items():
                if block != 30 and name != "block_within_video":
                    continue
                key = name if name != "block_within_video" else f"block_{block}"
                a = old.pooled(np.where(mask[:, None, None], data["full"], data["cheap"]))
                b = old.pooled(np.where(mask[:, None, None], data["merged_full"], data["merged_cheap"]))
                c = old.policy_cost(mask, data["rules"], costs)
                samples[key].append([a["five_head_mean_f1"], a["total_errors"], b["five_head_mean_f1"], b["total_errors"], c["logical_calls"], c["known_usd_estimate"]])
    return {name: {"draws": draws, "seed": seed, "f1_mean": float(np.mean(a, axis=0)[0]),
                  "f1_p97_5": float(np.quantile(a, .975, axis=0, method="linear")[0]),
                  "errors_mean": float(np.mean(a, axis=0)[1]), "errors_p2_5": float(np.quantile(a, .025, axis=0, method="linear")[1]),
                  "merged_f1_mean": float(np.mean(a, axis=0)[2]), "merged_errors_mean": float(np.mean(a, axis=0)[3]),
                  "logical_calls_mean": float(np.mean(a, axis=0)[4]), "known_usd_mean": float(np.mean(a, axis=0)[5]),
                  "raw_draw_metrics": a, "columns": ["f1", "errors", "merged_f1", "merged_errors", "logical_calls", "known_usd"],
                  "reference_is_independent_video_confidence_interval": False} for name, a in samples.items()}


def evaluate_variant(data, costs, name, config, on_fold=None, progress=None):
    variant = config["variants"][name]
    cache, folds = ModelCache(data, variant), []
    n = len(data["ids"])
    route = np.zeros(n, bool)
    mean, spread = np.zeros(n), np.zeros(n)
    supported = np.zeros(n, bool)
    started = time.perf_counter()
    for v in sorted(set(data["videos"])):
        fold = outer_fold(data, v, variant, config, cache)
        ix = np.array(fold["held_indices"])
        route[ix], mean[ix], spread[ix], supported[ix] = fold["route"], fold["mean"], fold["uncertainty"], fold["supported"]
        if on_fold:
            on_fold(fold)
        folds.append({k: val for k, val in fold.items() if k not in ("model", "inner_scores", "mean", "uncertainty", "supported", "route", "held_indices")})
        if progress:
            progress(v)
    fit_seconds = time.perf_counter() - started
    out = route_report(data, route, costs)
    out["controls"] = random_reference(data, route, costs, config)
    ref = out["controls"]["block_30"]
    checks = {**out["checks"], "cost_below_full": out["cost"]["logical_calls"] < 13 * n,
              "nonzero_reviews": bool(route.any()), "all_outer_choices_feasible": all(f["selected"]["selection_state"] == "FEASIBLE" for f in folds),
              "f1_above_block_p97_5": out["five_head_mean_f1"] > ref["f1_p97_5"],
              "errors_below_block_p2_5": out["total_errors"] < ref["errors_p2_5"]}
    out.update(development_checks=checks, conditional_development_pass=all(checks.values()), folds=folds,
               fit_and_selection_seconds=fit_seconds, name=name, feature_order=cache.names, deployable=False,
               oof=[{"sample_id": str(data["ids"][i]), "video_id": str(data["videos"][i]), "mean": float(mean[i]),
                     "uncertainty": float(spread[i]), "supported": bool(supported[i]), "routed": bool(route[i])} for i in range(n)])
    if name == PRIMARY:
        out["descriptive_cost_curve"] = [{"lambda": parameter, **route_report(data, route_from_scores(mean, spread, supported, data["rules"], parameter, variant), costs)} for parameter in config["cost_lambdas"]]
    out["prediction_diagnostics_by_video"] = {}
    gain = data["y"][:, :5].sum(axis=1) - data["y"][:, 5:].sum(axis=1)
    for v in sorted(set(data["videos"])):
        ix = np.where(data["videos"] == v)[0]
        out["prediction_diagnostics_by_video"][v] = {
            "mean_predicted_gain": float(mean[ix].mean()) if variant["target"] != "logistic" else None,
            "mean_observed_gain": float(gain[ix].mean()), "unsupported_count": int((~supported[ix]).sum()),
            "mean_uncertainty": float(spread[ix].mean()),
            "gain_mae": float(np.mean(np.abs(mean[ix] - gain[ix]))) if variant["target"] != "logistic" else None,
            "score_quintiles": [{"n": len(part), "mean_score": float(mean[part].mean()), "observed_error_gain": float(gain[part].sum())}
                                for part in np.array_split(ix[np.argsort(-mean[ix], kind="stable")], min(5, len(ix)))]}
    return out
