"""Offline, versioned help-minus-hurt Gate. No transport or dataset access."""
from __future__ import annotations

from copy import deepcopy
import math
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from surgical_agent.research.gate import escalation_rule_training as features
from surgical_agent.research.verification.candidate_coordinator import _TASK_NAMES
from surgical_agent.research.verification.prior_panel import COMPONENTS

VERSION = "official-help-minus-hurt-gate-v2-20260913"
LABEL_VERSION = "official-observed-behavior-utility-v2-20260913"
FAILURE_VERSION = "official-single-logical-attempt-fallback-v1-20260913"
TASKS = ("instrument", "verb", "target", "ivt", "phase")
VIDEOS = ("VID103", "VID23", "VID31", "VID96")
EPS = 1e-12
FEATURE_ORDER = tuple(sorted(features.FEATURE_ORDER))
GRASP = _TASK_NAMES["verb"].index("grasp")
RETRACT = _TASK_NAMES["verb"].index("retract")
GRASPER = _TASK_NAMES["instrument"].index("grasper")
IVT_MERGE = {c: d for c, a in COMPONENTS.items() if a["verb"] == GRASP
             for d, b in COMPONENTS.items() if b == {**a, "verb": RETRACT}}


def recipe():
    return {
        "version": VERSION, "label_version": LABEL_VERSION,
        "failure_policy": {
            "version": FAILURE_VERSION, "additional_logical_retries": 0,
            "scope": "First stored logical request in the original collection, including its recorded account failover; no later collection pass or offline syntax restoration.",
            "on_any_call_failure": "Return missing review evidence to the unchanged frozen admission/fallback rules.",
            "on_no_final_output": "UNKNOWN; exclude from utility fitting; preserve cohort membership.",
            "transient_one_retry_variant": "NOT ESTIMATED: would require a separately frozen exact-request replay of a uniform one-retry trajectory, not arbitrary mixing of archived passes.",
        },
        "estimator": {"type": "two_binary_logistic_regressions", "scaler": "StandardScaler",
                      "solver": "liblinear", "C": 1.0, "class_weight": "balanced",
                      "max_iter": 2000, "random_state": 3407,
                      "score": "predict_proba(help) - predict_proba(hurt)",
                      "scores_are_calibrated_probabilities": False,
                      "single_class_training_fold": "constant observed class probability; explicitly report"},
        "feature_version": features.FEATURE_VERSION, "feature_order": list(FEATURE_ORDER),
        "primary_feature_view": "features_postcheap", "tracker_variant_is_diagnostic": True,
        "primary_dataset": "original_attempt_observed_behavior",
        "sensitivity_dataset": "7053_completed_retry_or_syntax_observations",
        "strict_success_sensitivity": "7008 observations excluding all 45 separately accepted syntax recoveries; no-Tracker diagnostic only",
        "epsilon": EPS, "outer_split": "leave_one_Training_video_out",
        "inner_split": "leave_one_of_the_remaining_three_videos_out",
        "threshold_selection": {
            "candidates": "Every distinct inner OOF score plus 1+1e-12 (skip all)",
            "constraint": "inner pooled five-head FP+FN <= inner cheap FP+FN",
            "objective": "maximum unrounded inner pooled five-head mean F1",
            "tie_break": "fewer reviews, then larger threshold",
            "outer_labels_used": False,
        },
        "curve_fractions": [i / 100 for i in range(5, 51, 5)],
        "curve_scope": "Descriptive pooled OOF fixed-budget ranking, not an online policy or a source of deployment thresholds",
        "curve_ties": "numpy.argsort default quicksort on frozen cohort ordering, matching the original audit",
        "controls": {"draws": 300, "seed": 3407, "block_targets": 30,
                     "types": ["global", "within_video", "block_within_video"],
                     "block_definition": "Sort each video by frame_id, shuffle contiguous 30-target blocks, truncate only the last selected block to match review count",
                     "quantiles": "Unrounded empirical order statistics: sorted[int(.975*N)-1], sorted[int(.025*N)]",
                     "interval_interpretation": "Conditional random-routing reference, not independent-video confidence intervals"},
        "development_acceptance": [
            "F1 > cheap F1 AND errors < cheap errors",
            "F1 > within-video block random F1 p97.5 AND errors < its errors p2.5",
            "Compare calls and current-ledger known USD costs for Gate, cheap and full; disclose unpriced GLM/DS",
        ],
        "metric_views": ["original", "merged_grasp_retract"],
        "merge_is_primary_selection_metric": False,
        "merge": {"verb": {str(GRASP): RETRACT}, "ivt": {str(k): v for k, v in IVT_MERGE.items()},
                  "clinical_equivalence_claimed": False},
        "cost": {"source": "Read-only official budget.sqlite, original-pass per-stage/per-seat mean",
                 "cny_per_usd_scenario": 7.1, "unpriced_accounts": ["glm_requests", "deepseek_requests"],
                 "retry_costs": "Separate historical collection investment; not included as extra online requests in the single-attempt policy",
                 "calls_definition": "Logical API requests including reused historical responses at inference; monetary figures are estimates, not refunds"},
        "fitting_data_videos": list(VIDEOS), "Testing_access": False, "VID110_access": False,
        "VID110_future_calibration": "Only after user approval; it participated in mechanism development and is not independent validation. Freeze before Testing.",
        "future_testing_protocol": {"report_per_video": True, "metric_views_locked": ["original", "merged_grasp_retract"],
                                    "choose_metric_from_Testing_results": False, "match_frozen_inference_and_behavior_versions": True},
        "registration_scope": "Frozen before this fit, after exploratory audits of these same Training data; not an untouched-data confirmatory preregistration",
        "upstream_prior_tracker_folds_nested": False,
        "evaluation_scope": "Conditional Gate-only nested LOVO diagnostic; upstream priors/Tracker are fixed and may contain outer-fold information",
        "F7_action_split": "Not implemented; optional diagnostic, not selected as primary",
        "deployable": False,
    }


class SingleLogicalAttempt:
    """Execution wrapper for this behavior version; no recovery or retry dispatch.

    The delegate owns the existing single-call/account-failover behavior. A
    repeated logical request is rejected before the delegate can spend money.
    This wrapper is not installed into any frozen collection entry point.
    """
    def __init__(self, delegate):
        self.delegate = delegate
        self.seen = set()
        import threading
        self.lock = threading.Lock()

    def call(self, target, stage, seat, body):
        identity = (target, stage, seat)
        with self.lock:
            if identity in self.seen:
                raise ValueError("additional logical retry forbidden by " + FAILURE_VERSION)
            self.seen.add(identity)
        return self.delegate.call(target, stage, seat, body)


def derive_labels(cheap, final, gt, mask):
    old = features.label_outcome(cheap, final, gt=gt, mask=mask)
    delta = old["utility_delta"]
    known = delta is not None
    return {**old, "label_version": LABEL_VERSION,
            "help": int(delta > EPS) if known else None,
            "hurt": int(delta < -EPS) if known else None,
            "neutral": int(abs(delta) <= EPS) if known else None,
            "utility_class": ("help" if delta > EPS else "hurt" if delta < -EPS else "neutral") if known else None}


def merge_labels(labels):
    out = {t: set(labels[t]) for t in TASKS}
    out["verb"] = {RETRACT if c == GRASP else c for c in out["verb"]}
    out["ivt"] = {IVT_MERGE.get(c, c) for c in out["ivt"]}
    return out


def counts(rows, field, merged=False):
    out = np.zeros((len(rows), 5, 3), dtype=np.int64)
    for i, row in enumerate(rows):
        if not all(row["mask"].get(t, False) for t in TASKS):
            raise ValueError("all five GT masks required for observed fitting rows")
        pred = merge_labels(row[field]) if merged else row[field]
        gt = merge_labels(row["gt"]) if merged else row["gt"]
        for j, task in enumerate(TASKS):
            p, g = set(pred[task]), set(gt[task])
            out[i, j] = len(p & g), len(p - g), len(g - p)
    return out


def pooled(c):
    total = c.sum(axis=0) if c.ndim == 3 else c
    tp, fp, fn = total.T
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.ones(5, dtype=float), where=den != 0)
    return {"five_head_mean_f1": float(f1.mean() * 100), "ivt_f1": float(f1[3] * 100),
            "phase_f1": float(f1[4] * 100), "total_errors": int((fp + fn).sum()),
            "f1_by_task": dict(zip(TASKS, (f1 * 100).tolist()))}


def frame_utility(cheap, full):
    def f1(c):
        tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
        den = 2 * tp + fp + fn
        return np.divide(2 * tp, den, out=np.ones_like(tp, dtype=float), where=den != 0).mean(axis=1)
    return f1(full) - f1(cheap)


def fit_binary(X, y):
    classes = np.unique(y)
    if len(classes) == 1:
        return {"constant": float(classes[0])}
    return make_pipeline(StandardScaler(), LogisticRegression(
        solver="liblinear", C=1., class_weight="balanced", max_iter=2000, random_state=3407)).fit(X, y)


def predict(model, X):
    return np.full(len(X), model["constant"]) if isinstance(model, dict) else model.predict_proba(X)[:, 1]


def serialize(model, X):
    if isinstance(model, dict):
        return dict(model)
    scaler, lr = model.steps[0][1], model.steps[1][1]
    weights = lr.coef_[0] / scaler.scale_
    bias = float(lr.intercept_[0] - np.dot(weights, scaler.mean_))
    if not np.allclose(X @ weights + bias, model.decision_function(X), atol=1e-9):
        raise ValueError("weight export changed fitted scores")
    return {"weights": weights.tolist(), "bias": bias}


def oof_pair(X, help_y, hurt_y, videos, indices):
    scores = np.full(len(X), np.nan)
    for v in sorted(set(videos[indices])):
        held = indices[videos[indices] == v]
        fit = indices[videos[indices] != v]
        if not len(fit):
            raise ValueError("a held video must have separate fitting videos")
        scores[held] = predict(fit_binary(X[fit], help_y[fit]), X[held]) - predict(fit_binary(X[fit], hurt_y[fit]), X[held])
    return scores


def choose_threshold(scores, cheap, full):
    """Exact inner candidate search; never accepts an outer data argument."""
    if not len(scores) or not np.isfinite(scores).all() or len(scores) != len(cheap) or len(cheap) != len(full):
        raise ValueError("invalid threshold inputs")
    order = np.argsort(-scores)
    cumulative = np.cumsum((full - cheap)[order], axis=0)
    ends = np.r_[np.where(np.diff(scores[order]) != 0)[0], len(order) - 1]
    totals = np.concatenate([cheap.sum(axis=0)[None], cheap.sum(axis=0)[None] + cumulative[ends]])
    thresholds = np.r_[1. + EPS, scores[order][ends]]
    reviewed = np.r_[0, ends + 1]
    tp, fp, fn = totals[..., 0], totals[..., 1], totals[..., 2]
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.ones_like(tp, dtype=float), where=den != 0).mean(axis=1) * 100
    errors = (fp + fn).sum(axis=1)
    eligible = np.where(errors <= errors[0])[0]
    best = max(eligible, key=lambda i: (float(f1[i]), -int(reviewed[i]), float(thresholds[i])))
    return {"threshold": float(thresholds[best]), "inner_five_head_mean_f1": float(f1[best]),
            "inner_total_errors": int(errors[best]), "inner_cheap_errors": int(errors[0]),
            "inner_reviewed": int(reviewed[best]), "candidate_count": len(thresholds)}


def nested_scores(X, help_y, hurt_y, videos, cheap, full, sample_ids):
    n = len(X); scores = np.zeros(n); help_scores = np.zeros(n); hurt_scores = np.zeros(n)
    route = np.zeros(n, bool); folds = []
    for v in sorted(set(videos)):
        fit = np.where(videos != v)[0]; held = np.where(videos == v)[0]
        inner = oof_pair(X, help_y, hurt_y, videos, fit)
        selected = choose_threshold(inner[fit], cheap[fit], full[fit])
        hm, dm = fit_binary(X[fit], help_y[fit]), fit_binary(X[fit], hurt_y[fit])
        help_scores[held], hurt_scores[held] = predict(hm, X[held]), predict(dm, X[held])
        scores[held] = help_scores[held] - hurt_scores[held]
        route[held] = scores[held] >= selected["threshold"]
        folds.append({"held_out_video": v, "fit_videos": sorted(set(videos[fit])),
                      "inner_fit_videos_per_holdout": {w: sorted(set(videos[fit]) - {w}) for w in sorted(set(videos[fit]))},
                      "fit_samples": len(fit), "held_samples": len(held), **selected,
                      "held_reviewed": int(route[held].sum()),
                      "single_class_fit": {"help": isinstance(hm, dict), "hurt": isinstance(dm, dict)}})
    return scores, help_scores, hurt_scores, route, folds


def policy_cost(route, rules, costs):
    route = np.asarray(route, bool); rules = np.asarray(rules, bool)
    stage_counts = {"h0|base": len(route), "proposal|base": int((route | rules).sum()),
                    "phase_recommendation|base": int(route.sum())}
    for stage in ("control_graph", "joint_r1"):
        for seat in ("gpt", "gemini", "grok", "deepseek", "qwen"):
            stage_counts[stage + "|" + seat] = int(route.sum())
    known, unpriced = 0., {}
    for stage, n in stage_counts.items():
        price = costs["per_call_main_attempt"][stage]
        if price.get("requests_only"):
            unpriced[price["account"]] = unpriced.get(price["account"], 0) + n
        else:
            known += n * price["usd_per_call"]
    return {"logical_calls": sum(stage_counts.values()), "known_usd_estimate": known,
            "usd_complete": not any(unpriced.values()), "unpriced_request_counts": unpriced,
            "stage_calls": stage_counts, "currency_conversion_scenario_cny_per_usd": 7.1}


def control_masks(route, videos, frame_ids, *, draws=300, block=30, seed=3407):
    rng = random.Random(seed); n = len(route); k = int(route.sum()); unique = sorted(set(videos))
    ids = {v: np.where(videos == v)[0].tolist() for v in unique}
    order = {v: sorted(ids[v], key=lambda i: frame_ids[i]) for v in unique}
    for _ in range(draws):
        global_mask = np.zeros(n, bool); global_mask[rng.sample(range(n), k)] = True
        within, blocked = np.zeros(n, bool), np.zeros(n, bool)
        for v in unique:
            need = int(route[videos == v].sum())
            within[rng.sample(ids[v], need)] = True
            blocks = [order[v][i:i + block] for i in range(0, len(order[v]), block)]
            rng.shuffle(blocks)
            picked = [i for group in blocks for i in group][:need]
            blocked[picked] = True
        yield {"global": global_mask, "within_video": within, "block_within_video": blocked}


def evaluate_routes(rows, route, cheap, full, merged_cheap, merged_full, costs, *, controls=True):
    videos = np.array([r["video_id"] for r in rows]); frames = [r["frame_id"] for r in rows]
    rules = [bool(r["features_postcheap"]["proposal_rule"]) for r in rows]
    chosen = np.where(route[:, None, None], full, cheap)
    merged = np.where(route[:, None, None], merged_full, merged_cheap)
    out = {**pooled(chosen), "merged_grasp_retract": pooled(merged), "reviewed": int(route.sum()),
           "review_fraction": float(route.mean()), "cost": policy_cost(route, rules, costs),
           "by_video": {v: {**pooled(chosen[videos == v]), "merged_grasp_retract": pooled(merged[videos == v]),
                            "reviewed": int(route[videos == v].sum()),
                            "cost": policy_cost(route[videos == v], np.asarray(rules)[videos == v], costs)} for v in sorted(set(videos))}}
    if controls:
        samples = {name: [] for name in ("global", "within_video", "block_within_video")}
        for masks in control_masks(route, videos, frames):
            for name, mask in masks.items():
                a = pooled(np.where(mask[:, None, None], full, cheap))
                b = pooled(np.where(mask[:, None, None], merged_full, merged_cheap))
                random_cost = policy_cost(mask, rules, costs)
                samples[name].append((a["five_head_mean_f1"], a["total_errors"], b["five_head_mean_f1"], b["total_errors"],
                                      random_cost["known_usd_estimate"], random_cost["logical_calls"]))
        out["controls"] = {}
        for name, values in samples.items():
            array = np.array(values); ordered = np.sort(array, axis=0); count = len(values)
            out["controls"][name] = {"draws": count, "seed": 3407,
                "f1_mean": float(array[:, 0].mean()), "f1_p97_5": float(ordered[int(.975 * count) - 1, 0]),
                "errors_mean": float(array[:, 1].mean()), "errors_p2_5": float(ordered[int(.025 * count), 1]),
                "merged_grasp_retract": {"f1_mean": float(array[:, 2].mean()),
                    "f1_p97_5": float(ordered[int(.975 * count) - 1, 2]), "errors_mean": float(array[:, 3].mean()),
                    "errors_p2_5": float(ordered[int(.025 * count), 3])},
                "logical_calls_mean": float(array[:, 5].mean()),
                "logical_calls_min": int(array[:, 5].min()), "logical_calls_max": int(array[:, 5].max()),
                "matched_quantity": "reviewed targets; proposal-rule mix can change total logical calls",
                "unpriced_request_counts": out["cost"]["unpriced_request_counts"],
                "known_usd_mean": float(array[:, 4].mean()), "usd_complete": False}
        cheap_metrics, reference = pooled(cheap), out["controls"]["block_within_video"]
        out["development_checks"] = {
            "f1_above_cheap": out["five_head_mean_f1"] > cheap_metrics["five_head_mean_f1"],
            "errors_below_cheap": out["total_errors"] < cheap_metrics["total_errors"],
            "f1_above_block_random_p97_5": out["five_head_mean_f1"] > reference["f1_p97_5"],
            "errors_below_block_random_p2_5": out["total_errors"] < reference["errors_p2_5"],
            "current_cost_and_calls_disclosed": True,
        }
        out["conditional_development_pass"] = all(out["development_checks"].values())
    return out


def evaluate(rows, costs, *, tracker=False):
    if len({r["sample_id"] for r in rows}) != len(rows) or {r["video_id"] for r in rows} != set(VIDEOS):
        raise ValueError("exactly four Training videos and unique IDs required")
    if len({r["policy_id"] for r in rows}) != 1:
        raise ValueError("mixed inference policies")
    for r in rows:
        if r["source_split"] != "Training" or r["label_version"] != LABEL_VERSION or r["feature_version"] != features.FEATURE_VERSION:
            raise ValueError("Training/feature/label contract mismatch")
        if derive_labels(r["cheap_labels"], r["final_labels"], r["gt"], r["mask"]) != r["labels"]:
            raise ValueError("utility labels changed")
    view = "features_postcheap_with_tracker" if tracker else "features_postcheap"
    if any(set(r[view]) != set(FEATURE_ORDER) for r in rows):
        raise ValueError("exact pre-review feature contract required")
    X = np.array([[r[view][k] for k in FEATURE_ORDER] for r in rows], dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("nonfinite features")
    y = {name: np.array([r["labels"][name] for r in rows], dtype=int) for name in ("help", "hurt", "safe_benefit")}
    cheap, full = counts(rows, "cheap_labels"), counts(rows, "final_labels")
    mc, mf = counts(rows, "cheap_labels", True), counts(rows, "final_labels", True)
    if not np.allclose(frame_utility(cheap, full), [r["labels"]["utility_delta"] for r in rows], atol=1e-14, rtol=0):
        raise ValueError("independent set utility mismatch")
    videos = np.array([r["video_id"] for r in rows]); n = len(rows)
    score, hs, ds, route, folds = nested_scores(X, y["help"], y["hurt"], videos, cheap, full, [r["sample_id"] for r in rows])
    def report(mask, controls=True):
        return evaluate_routes(rows, mask, cheap, full, mc, mf, costs, controls=controls)
    result = {"rows": n, "feature_view": view, "feature_order": list(FEATURE_ORDER),
              "deployable": False, "nested_gate_only": True, "upstream_folds_nested": False,
              "class_counts": {k: int(y[k].sum()) for k in y}, "neutral": int(n - y["help"].sum() - y["hurt"].sum()),
              "folds": folds, "nested_selected_gate": report(route),
              "cheap": report(np.zeros(n, bool), False), "full": report(np.ones(n, bool), False),
              "frame_utility_oracle_unavailable_at_inference": report(y["help"].astype(bool), False),
              "oracle_is_pooled_metric_optimum": False, "curves": [], "diagnostics": {}}
    hc, hm = counts(rows, "h0_labels"), counts(rows, "h0_labels", True)
    result["h0"] = {**pooled(hc), "merged_grasp_retract": pooled(hm), "reviewed": 0,
        "cost": policy_cost(np.zeros(n, bool), np.zeros(n, bool), costs),
        "by_video": {v: {**pooled(hc[videos == v]), "merged_grasp_retract": pooled(hm[videos == v]),
                         "cost": policy_cost(np.zeros(int((videos == v).sum()), bool), np.zeros(int((videos == v).sum()), bool), costs)} for v in sorted(set(videos))}}
    for fraction in recipe()["curve_fractions"]:
        mask = np.zeros(n, bool); mask[np.argsort(-score)[:round(fraction * n)]] = True
        result["curves"].append({"budget_fraction": fraction, **report(mask)})
    safe_scores = np.zeros(n)
    for v in sorted(set(videos)):
        held = videos == v
        safe_scores[held] = predict(fit_binary(X[~held], y["safe_benefit"][~held]), X[held])
    for name, s in (("benefit", hs), ("safe_benefit", safe_scores)):
        target = y["help"] if name == "benefit" else y[name]
        result["diagnostics"][name + "_fixed_0_5"] = {
            "auc": float(roc_auc_score(target, s)) if len(np.unique(target)) == 2 else None,
            "ap": float(average_precision_score(target, s)) if target.any() else None,
            **report(s >= .5)}
    result["hurt_auc"] = float(roc_auc_score(y["hurt"], ds)) if len(np.unique(y["hurt"])) == 2 else None
    result["oof"] = [{"sample_id": r["sample_id"], "video_id": r["video_id"],
                      "help_score": float(h), "hurt_score": float(d), "net_score": float(s),
                      "nested_routed": bool(b)} for r, h, d, s, b in zip(rows, hs, ds, score, route)]
    result["grasp_convention"] = {}
    for v in sorted(set(videos)):
        targets = [r for r in rows if r["video_id"] == v and GRASPER in r["gt"]["instrument"]]
        result["grasp_convention"][v] = {"grasper_frames": len(targets),
            "gt_grasp_rate": sum(GRASP in r["gt"]["verb"] for r in targets) / len(targets) if targets else None,
            "gt_retract_rate": sum(RETRACT in r["gt"]["verb"] for r in targets) / len(targets) if targets else None}
    weights = {"version": VERSION, "feature_view": view, "feature_order": list(FEATURE_ORDER),
               "models": {k: serialize(fit_binary(X, y[k]), X) for k in ("help", "hurt")},
               "score": "sigmoid(help) - sigmoid(hurt)", "threshold": None, "deployable": False,
               "calibration_required": True, "failure_policy_version": FAILURE_VERSION}
    return result, weights
