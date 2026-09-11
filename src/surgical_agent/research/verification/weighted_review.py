"""Training-fitted monotone score aggregation; no query GT or extra model calls."""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS

VERSION = "monotone_cross_video_review_v1"
POSITIVE = 0.65
NEGATIVE = 0.35
TASKS = ("instrument", "verb", "target", "ivt")


def fit(examples, excluded_video):
    retained = [e for e in examples if e["video_id"] != excluded_video]
    models = {}
    for task in TASKS:
        rows = [e for e in retained if e["task"] == task]
        count = len(rows)
        positives = sum(e["present"] for e in rows)
        if count < 20 or min(positives, count - positives) < 5:
            models[task] = {"fallback": "INSUFFICIENT_TRAINING", "count": count, "positives": positives}
            continue
        x = (np.asarray([e["scores"] for e in rows], dtype=float) - 3) / 2
        y = np.asarray([e["present"] for e in rows], dtype=float)

        def objective(theta, x=x, y=y):
            z = x @ theta[1:] + theta[0]
            loss = np.logaddexp(0, z).sum() - y @ z + .5 * theta[1:] @ theta[1:]
            residual = expit(z) - y
            gradient = np.concatenate(([residual.sum()], x.T @ residual + theta[1:]))
            return float(loss), gradient

        fitted = minimize(objective, np.zeros(6), jac=True, method="L-BFGS-B",
                          bounds=[(None, None)] + [(0, None)] * 5,
                          options={"maxiter": 1000, "ftol": 1e-12})
        if not fitted.success or not np.isfinite(fitted.x).all():
            raise ValueError("monotone reviewer fit did not converge")
        models[task] = {"intercept": float(fitted.x[0]), "weights": fitted.x[1:].tolist(),
                        "count": count, "positives": positives, "fallback": None}
    return {"version": VERSION, "excluded_video": excluded_video,
        "fit_videos": sorted({e["video_id"] for e in retained}), "models": models,
        "seats": list(SEATS), "positive_threshold": POSITIVE, "negative_threshold": NEGATIVE}


def select(h0, pool, reviews, fitted, video_id):
    if (fitted["excluded_video"] != video_id or video_id in fitted["fit_videos"]
            or fitted["version"] != VERSION or fitted["seats"] != list(SEATS)
            or fitted["positive_threshold"] != POSITIVE or fitted["negative_threshold"] != NEGATIVE):
        raise ValueError("frozen model must exclude query video")
    means, diagnostics = panel.aggregate(reviews, pool, image_count=3)
    adjusted, probabilities = deepcopy(means), {}
    for item in pool["propositions"]:
        pid, task = item["id"], item["task"]
        model = fitted["models"][task]
        scores = diagnostics[pid]["scores"]
        if means[pid] is None or model["fallback"]:
            continue
        z = model["intercept"] + sum(w * (s - 3) / 2 for w, s in zip(model["weights"], scores, strict=True))
        probability = 1 / (1 + math.exp(-max(-700, min(700, z))))
        probabilities[pid] = probability
        # A learned prior alone cannot establish visual presence or whole-frame absence.
        adjusted[pid] = (4.0 if probability >= POSITIVE and max(scores) >= 4 else
                         2.0 if probability <= NEGATIVE and min(scores) <= 2 else None)
    return {"prediction": panel.select(h0, pool, adjusted, threshold=4),
        "original_means": means, "adjusted_scores": adjusted, "fitted_probabilities": probabilities,
        "diagnostics": diagnostics, "probabilities_are_not_validated_confidence": True}
