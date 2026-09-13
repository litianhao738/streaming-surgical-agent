"""Observed terminal utility of a fixed repair policy, including its fallbacks.

This is a separate supervision contract. It does not redefine the older
strict repair_observed field or modify its frozen observations.
"""
from collections import Counter
from copy import deepcopy

from surgical_agent.research.gate import mainline_training as strict

FEATURE_VERSION = strict.FEATURE_VERSION
FEATURE_ORDER = strict.FEATURE_ORDER
LABEL_VERSION = "mainline_actual_terminal_utility_v1"
TARGETS = ("benefit", "safe_benefit")


def label_outcome(h0, final, *, gt, mask):
    truth = strict._truth(gt, mask)
    before = strict.canonical_labels(h0) if h0 is not None else None
    after = strict.canonical_labels(final) if final is not None else None
    observed = before is not None and after is not None and all(mask.values())
    counts_before, counts_after, harms = {}, {}, {}
    for task in strict.TASKS:
        expected = truth[task]
        counts_before[task] = (strict._counts(set(before[task]), expected)
                               if before is not None and expected is not None else None)
        counts_after[task] = (strict._counts(set(after[task]), expected)
                              if after is not None and expected is not None else None)
        harms[task] = (len((set(after[task]) - set(before[task])) - expected)
                       + len((set(before[task]) - set(after[task])) & expected)
                       if observed else None)
    delta = (sum(counts_after[t]["f1"] - counts_before[t]["f1"] for t in strict.TASKS)
             / len(strict.TASKS) if observed else None)
    any_harm = any(harms.values()) if observed else None
    return {"label_version": LABEL_VERSION, "utility_definition": strict.UTILITY_DEFINITION,
            "utility_tasks": list(strict.TASKS), "terminal_observed": observed,
            "complete_five_head_gt": all(mask.values()),
            "benefit": int(delta > strict.EPSILON) if observed else None,
            "safe_benefit": int(delta > strict.EPSILON and not any_harm) if observed else None,
            "utility_delta": delta, "any_label_harm": any_harm,
            "harmful_edits_by_task": harms,
            "h0_counts_by_task": counts_before, "final_counts_by_task": counts_after}


def derive_rows(source_rows):
    result = []
    for original in source_rows:
        if (original["label_version"] != strict.LABEL_VERSION
                or original["labels"]["label_version"] != strict.LABEL_VERSION):
            raise ValueError("derive only from the explicit strict source contract")
        row = deepcopy(original)
        row["source_strict_labels"] = row.pop("labels")
        row["source_label_version"] = row["label_version"]
        row["label_version"] = LABEL_VERSION
        row["labels"] = label_outcome(row["h0_labels"], row["final_labels"],
                                      gt=row["gt"], mask=row["mask"])
        result.append(row)
    return result


def validate_rows(rows, target):
    if target not in TARGETS or not rows:
        raise ValueError("nonempty terminal-utility rows and a supported target required")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate identities")
    if len({(r["video_id"], r["frame_id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate frames")
    if len({r["policy_id"] for r in rows}) != 1 or not rows[0]["policy_id"]:
        raise ValueError("one frozen nonempty policy_id required")
    for row in rows:
        if (row["source_split"] != "Training" or row["feature_version"] != FEATURE_VERSION
                or row["label_version"] != LABEL_VERSION
                or row["source_label_version"] != strict.LABEL_VERSION):
            raise ValueError("Training feature/label provenance mismatch")
        strict.validate_features(row["features_with_tracker"])
        strict.validate_features(row["features_no_tracker"])
        if row["features_no_tracker"] != strict.mask_tracker(row["features_with_tracker"]):
            raise ValueError("paired feature views differ beyond Tracker masking")
        if row["labels"] != label_outcome(row["h0_labels"], row["final_labels"],
                                          gt=row["gt"], mask=row["mask"]):
            raise ValueError("terminal labels failed independent recomputation")
    usable = [r for r in rows if r["labels"][target] is not None]
    videos = sorted({r["video_id"] for r in usable})
    if len(videos) < 3:
        raise ValueError("at least three videos required")
    folds = []
    for video in videos:
        fit = [r for r in usable if r["video_id"] != video]
        counts = Counter(r["labels"][target] for r in fit)
        if min(counts[0], counts[1]) < 2:
            raise ValueError("insufficient fit classes for held-out " + video)
        folds.append({"held_out_video_id": video, "fit_count": len(fit),
                      "fit_negative": counts[0], "fit_positive": counts[1],
                      "held_out_count": len(usable) - len(fit)})
    return {"rows": len(rows), "known": len(usable), "unknown": len(rows) - len(usable),
            "positive": sum(r["labels"][target] for r in usable),
            "negative": sum(r["labels"][target] == 0 for r in usable), "folds": folds}
