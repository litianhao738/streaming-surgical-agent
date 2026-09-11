"""CPU pilot fitting for the final-only Gate; never installs an online policy.

Uses leave-one-video-out Gate fits with paired Tracker/no-Tracker views. These
are conditional diagnostics, not nested cross-validation of Tracker + Gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.research.gate.final_only_training import (
    FEATURE_ORDER,
    FEATURE_VERSION,
    TARGETS,
    readiness,
)


def fit_pilot(rows, *, target="benefit", seed=3407):
    check = readiness(rows, target=target)
    if not check["can_fit_pilot"]:
        raise ValueError("Gate training data not ready: " + "; ".join(check["blocking_reasons"]))
    usable = [row for row in rows if row["labels"][target] is not None]
    if any(row.get("feature_version") != FEATURE_VERSION for row in usable):
        raise ValueError("incompatible final-only feature version")
    labels = np.asarray([row["labels"][target] for row in usable])
    result = {"readiness": check, "target": target, "variants": {}, "artifacts": {},
              "feature_order": list(FEATURE_ORDER), "feature_version": FEATURE_VERSION,
              "policy_id": usable[0]["policy_id"], "deployable": False, "paper_final": False,
              "validation_calibrated": False, "seed": seed,
              "partition": "leave_one_video_out_gate_only_conditional_diagnostic",
              "training_recipe": {"estimator": "standardized_logistic_regression", "C": 1.0,
                                  "class_weight": "balanced", "solver": "liblinear", "max_iter": 2000},
              "uses_gt_for_feature_extraction": False}
    for variant in ("no_tracker", "with_tracker"):
        matrix = np.asarray([[row["features_" + variant][key] for key in FEATURE_ORDER] for row in usable], dtype=float)
        scores = np.zeros(len(usable))
        folds = []
        for video in sorted({row["video_id"] for row in usable}):
            held = np.asarray([row["video_id"] == video for row in usable])
            model = make_pipeline(StandardScaler(), LogisticRegression(
                solver="liblinear", C=1.0, class_weight="balanced", max_iter=2000, random_state=seed))
            model.fit(matrix[~held], labels[~held])
            scores[held] = model.predict_proba(matrix[held])[:, 1]
            folds.append({"held_out_video_id": video,
                          "fit_video_ids": sorted({row["video_id"] for i, row in enumerate(usable) if not held[i]}),
                          "fit_count": int((~held).sum()), "held_out_count": int(held.sum())})
        predicted = scores >= 0.5
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
        metrics = {"auroc": float(roc_auc_score(labels, scores)),
                   "average_precision": float(average_precision_score(labels, scores)),
                   "positive_prevalence": float(labels.mean()),
                   "fixed_threshold": 0.5, "precision": float(precision), "recall": float(recall), "f1": float(f1),
                   "routed_fraction": float(predicted.mean()), "folds": folds,
                   "scores_are_not_calibrated_probabilities": True,
                   "oof_scores": [{"sample_id": row["sample_id"], "label": int(labels[i]), "score": float(scores[i])}
                                  for i, row in enumerate(usable)]}
        if target in ("benefit", "safe_benefit"):
            delta = np.asarray([row["labels"]["utility_delta"] for row in usable], dtype=float)
            harms = np.asarray([row["labels"]["any_head_harm"] for row in usable], dtype=bool)
            metrics["observed_policy_comparison"] = {
                "never_repair_mean_utility_gain": 0.0,
                "always_repair_mean_utility_gain": float(delta.mean()),
                "gate_mean_utility_gain": float((delta * predicted).mean()),
                "beneficial_episodes_admitted": int(((labels == 1) & predicted).sum()),
                "nonbeneficial_episodes_admitted": int(((labels == 0) & predicted).sum()),
                "episodes_with_any_head_harm_admitted": int((harms & predicted).sum()),
                "cost_savings_measured": False,
            }
        result["variants"][variant] = metrics
        final = make_pipeline(StandardScaler(), LogisticRegression(
            solver="liblinear", C=1.0, class_weight="balanced", max_iter=2000, random_state=seed))
        final.fit(matrix, labels)
        scaler, linear = final.steps[0][1], final.steps[1][1]
        weights = linear.coef_[0] / scaler.scale_
        bias = float(linear.intercept_[0] - np.dot(weights, scaler.mean_))
        if not np.allclose(matrix @ weights + bias, final.decision_function(matrix), atol=1e-9):
            raise RuntimeError("serialized Gate differs from fitted model")
        result["artifacts"][variant] = {
            "schema_version": "final_only_gate_linear_pilot_v1", "feature_version": FEATURE_VERSION,
            "feature_order": list(FEATURE_ORDER), "weights": weights.tolist(), "bias": bias,
            "policy_id": usable[0]["policy_id"], "target": target,
            "training_video_ids": sorted({row["video_id"] for row in usable}),
            "deployable": False, "paper_final": False, "validation_calibrated": False,
        }
    return result


def run(args):
    manifest_path = args.prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = args.prepared / "training_examples.json"
    if (manifest["schema_version"] != "final_only_gate_preparation_v1"
            or sha256_file(examples) != manifest["training_examples_sha256"]
            or sha256_file(args.prepared / "features_before_gt.json") != manifest["features_before_gt_sha256"]):
        raise ValueError("prepared training data or pre-GT features changed")
    rows = json.loads(examples.read_text(encoding="utf-8"))
    if any(row["policy_id"] != manifest["policy_id"] for row in rows):
        raise ValueError("Gate data mixes a different frozen policy")
    frozen = json.loads((args.prepared / "features_before_gt.json").read_text(encoding="utf-8"))
    pre_gt = {row["sample_id"]: row for row in frozen}
    if len(pre_gt) != len(rows) or {row["sample_id"] for row in rows} != set(pre_gt):
        raise ValueError("pre-GT and labeled observation identities differ")
    for row in rows:
        if any(row.get(key) != value for key, value in pre_gt[row["sample_id"]].items()):
            raise ValueError("features or provenance changed after the GT join")
    check = readiness(rows, target=args.target)
    print(json.dumps(check), flush=True)
    if args.check_only:
        return check
    if args.output is None or args.output.exists():
        raise ValueError("training requires a new --output directory")
    # Validate every fit fold before creating any model output.
    result = fit_pilot(rows, target=args.target, seed=args.seed)
    result["preparation_manifest_sha256"] = sha256_file(manifest_path)
    result["training_examples_sha256"] = sha256_file(examples)
    args.output.mkdir(parents=True)
    artifacts = result.pop("artifacts")
    for variant, artifact in artifacts.items():
        atomic_write_json(args.output / f"{variant}_pilot.json", artifact)
    atomic_write_json(args.output / "training_report.json", result)
    print(json.dumps({"output": str(args.output), "deployable": False}), flush=True)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prepared", type=Path, required=True)
    result.add_argument("--target", choices=TARGETS, default="benefit")
    result.add_argument("--output", type=Path)
    result.add_argument("--seed", type=int, default=3407)
    result.add_argument("--check-only", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
