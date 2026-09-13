"""Sealed, original-attempt Training replay for the branch PGP-Gate.

This module prepares evidence only. It never loads a Gate/Tracker checkpoint,
imports a live collection entry point, or sends requests. Action order is
cheap, interaction, phase, both. Costs include both already-observed Qwen probes.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.audit import gate_proposals_offline_20260913 as base
from surgical_agent.research.verification.prior_gated_joint import decide_phase

VERSION = "pgp-branch-original-replay-no-tracker-20260913-v1"
ACTIONS = ("cheap", "interaction", "phase", "both")
COST_COLUMNS = ("logical_calls", "known_usd", "glm_requests", "deepseek_requests", "reviewer_calls")
SOURCE = base.SOURCE
CACHE = base.DEFAULT
TASKS = base.TASKS
PREFIX_KEYS = ("h0|base", "proposal|base", "phase_recommendation|base",
               "control_graph|qwen", "joint_r1|qwen")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def partial_settled(observed, present):
    """No-change certificate using only the queried seats of ONE proposition.

Each value is a valid 1..5 rating or None for an observed invalid answer.
An unqueried seat is absent, never encoded as an invalid response.
"""
    require(len(observed) <= 5, "too many observed ratings")
    if any(score is None for score in observed):
        return True
    require(all(isinstance(s, (int, float)) and 1 <= s <= 5 for s in observed),
            "invalid observed rating")
    remaining = 5 - len(observed)
    total = sum(observed)
    return total + remaining > 10 if present else total + 5 * remaining < 20


def partial_phase_settled(observed, current):
    """No phase change is possible under any completion of unqueried ratings."""
    require(len(observed) == 7 and len({len(p) for p in observed}) == 1,
            "seven equally observed phase alternatives required")
    require(0 <= current < 7 and len(observed[0]) <= 5, "invalid phase prefix")
    if any(score is None for score in observed[current]):
        return True
    remaining = 5 - len(observed[current])
    lower = sum(observed[current]) + remaining
    for phase, scores in enumerate(observed):
        if phase == current or any(score is None for score in scores):
            continue
        require(all(isinstance(s, (int, float)) and 1 <= s <= 5 for s in scores),
                "invalid phase rating")
        upper = sum(scores) + 5 * remaining
        if upper >= 20 and upper > lower:
            return False
    return True


def _rating(diagnostic, seat):
    """Read exactly the rating/validity of the requested seat."""
    if seat in diagnostic["invalid"]:
        return None
    score = diagnostic["scores"][base.ORDER.index(seat)]
    require(isinstance(score, (int, float)) and 1 <= score <= 5,
            "a valid queried seat must have a 1..5 score")
    return score


def streaming_stop_plan(result, h0, order=None):
    """Conservative branch stop, without accessing unqueried seat ratings.

Qwen has already been observed in both branches at the decision point. The
mathematical stop may be zero, but billing applies max(1, stop) separately.
"""
    order = base.QORDER if order is None else list(order)
    require(len(order) == 5 and set(order) == set(base.ORDER), "reviewer order is not a permutation")
    props = result["pool"]["propositions"]
    control = [[] for _ in props]
    phases = [[] for _ in range(7)]
    kc = kj = None
    for k in range(6):
        if kc is None and all(partial_settled(scores, p["label_id"] in h0[p["task"]])
                              for p, scores in zip(props, control)):
            kc = k
        if kj is None and partial_phase_settled(phases, h0["phase"][0]):
            kj = k
        if k == 5:
            break
        if kc is None:
            for p, observed in zip(props, control):
                observed.append(_rating(result["diagnostics"][p["id"]], order[k]))
        if kj is None:
            for phase, observed in enumerate(phases):
                observed.append(_rating(result["joint_diagnostics"][f"phase_{phase}"], order[k]))
        if kc is not None and kj is not None:
            break
    return (5 if kc is None else kc), (5 if kj is None else kj)


def branch_outputs(cheap, full):
    """Four actions preserve interaction's four heads as an indivisible unit."""
    outputs = []
    for action in range(4):
        out = deepcopy(cheap)
        if action & 1:
            out.update({task: list(full[task]) for task in TASKS[:4]})
        if action & 2:
            out["phase"] = list(full["phase"])
        outputs.append(out)
    return outputs


def cost_vector(keys, ledger):
    require(len(keys) == len(set(keys)), "a logical request was charged twice")
    require(all(k in ledger for k in keys), "unknown request in cost ledger")
    return np.array([len(keys), sum(ledger[k].get("usd_per_call", 0) for k in keys),
                     sum(k.endswith("|grok") for k in keys),
                     sum(k.endswith("|deepseek") for k in keys),
                     sum(k.startswith(("control_graph|", "joint_r1|")) for k in keys)], dtype=float)


def branch_costs(stops, ledger):
    """All actions pay five-call prefix; only active branches can add calls."""
    kc, kj = map(int, stops)
    require(0 <= kc <= 5 and 0 <= kj <= 5, "invalid branch stop")
    costs = []
    for action in range(4):
        keys = list(PREFIX_KEYS)
        if action & 1:
            keys.extend("control_graph|" + s for s in base.QORDER[1:max(1, kc)])
        if action & 2:
            keys.extend("joint_r1|" + s for s in base.QORDER[1:max(1, kj)])
        costs.append(cost_vector(keys, ledger))
    return np.array(costs)


def feature_schema():
    names = [k for k in base.old.FEATURE_ORDER if not k.startswith("tracker_")]
    for group, suffixes in (("pool", ("present", "cheap_selected")),
                            ("qwen_control", ("score", "valid")),
                            ("qwen_joint", ("score", "valid"))):
        names.extend(f"{group}_{task}_{label}_{suffix}"
                     for task, label in base.LABELS for suffix in suffixes)
    require(len(names) == len(set(names)), "duplicate input feature names")
    return names


def validate_cached_row(row, result):
    require(result["h0"] == row["h0_labels"], "archived H0 mismatch")
    full = row["final_labels"]
    cheap = row["cheap_labels"]
    require(result["predictions"]["gated_control_jointphase"] == full,
            "archived original final mismatch; retries cannot be substituted")
    require(cheap["phase"] == row["h0_labels"]["phase"], "cheap phase must remain H0 phase")
    require(all(result["predictions"]["gated_control"][t] == full[t] for t in TASKS[:4]),
            "interaction prediction depends on phase branch")
    # Recompute phase admission from the sealed per-seat diagnostics, not stored
    # panel means. This is outcome validation, outside inference feature access.
    means = {}
    for phase in range(7):
        pid = f"phase_{phase}"
        d = result["joint_diagnostics"][pid]
        scores = [_rating(d, seat) for seat in base.ORDER]
        means[pid] = None if any(s is None for s in scores) else sum(scores) / 5
    phase, _ = decide_phase(row["h0_labels"], means)
    require(phase == full["phase"], "independently recomputed phase mismatch")
    plans = [streaming_stop_plan(result, row["h0_labels"], order)
             for order in (base.ORDER, base.QORDER)]
    for plan, order in zip(plans, (base.ORDER, base.QORDER)):
        require(plan == base.stop_plan(result, row["h0_labels"], order),
                "prefix-only stopping disagrees with frozen bounds")
        kc, kj = plan
        if kc < 5:
            require(all(full[t] == cheap[t] for t in TASKS[:4]),
                    "interaction stop does not preserve archived output")
        if kj < 5:
            require(full["phase"] == row["h0_labels"]["phase"],
                    "phase stop does not preserve archived output")
    outputs = branch_outputs(cheap, full)
    # Execute all four archived counterfactual actions, honoring partial stops.
    kc, kj = plans[1]
    available = {t: (cheap[t] if kc < 5 else full[t]) for t in TASKS[:4]}
    available["phase"] = row["h0_labels"]["phase"] if kj < 5 else phase
    replayed = branch_outputs(cheap, available)
    require(outputs == replayed, "four-action branch replay changed a prediction")
    return outputs, plans


def prepare(out: Path):
    """Validate 6,059 original Training observations and prepare new arrays.

`out` must be a new run directory (it may already contain the runner's recipe).
The replay-specific files must not exist. Nothing under old artifacts is written.
"""
    out = Path(out).resolve()
    require(out.is_relative_to(ROOT / "artifacts"), "output must stay under workspace artifacts")
    require(not out.is_relative_to(SOURCE) and not out.is_relative_to(CACHE), "old output directory forbidden")
    out.mkdir(parents=True, exist_ok=True)
    output_names = ("pgp_replay_arrays.npz", "pgp_reviewer_source_sha256.json", "pgp_replay_receipt.json")
    require(not any((out / name).exists() for name in output_names), "refusing to overwrite replay output")
    with base.offline():
        previous_receipt = base.read(CACHE / "receipt.json")
        require(previous_receipt["state"] == "COMPLETED", "prior cache not complete")
        bindings = {str((CACHE / "receipt.json").resolve()): base.sha(CACHE / "receipt.json")}
        for name in ("recipe.json", "prepared_arrays.npz", "reviewer_source_sha256.json"):
            path = (CACHE / name).resolve()
            digest = base.sha(path)
            require(digest == previous_receipt["output_sha256"][name], "changed prepared evidence: " + name)
            bindings[str(path)] = digest
        previous_recipe = base.read(CACHE / "recipe.json")
        for name in ("primary_rows.json", "costs.json", "preparation_receipt.json"):
            path = (SOURCE / name).resolve()
            digest = base.sha(path)
            require(digest == previous_recipe["old_artifacts_sha256"][str(path)], "changed original source: " + name)
            bindings[str(path)] = digest
        code_sources = [Path(__file__), Path(base.__file__),
                        ROOT / "src/surgical_agent/research/verification/prior_gated_joint.py",
                        ROOT / "src/surgical_agent/research/verification/phase_extension.py",
                        ROOT / "scripts/run_gate_escalation_mainline.py"]
        bindings.update({str(p.resolve()): base.sha(p) for p in code_sources})
        rows = base.read(SOURCE / "primary_rows.json")
        require(len(rows) == 6059, "only the frozen 6059-row primary cohort is accepted")
        require(all(r["source_split"] == "Training" and r["video_id"] in base.old.VIDEOS
                    and r["origin"] == "original_attempt_observed_behavior" for r in rows),
                "non-original or non-Training observation")
        ids = np.array([r["sample_id"] for r in rows])
        require(len(set(ids)) == len(ids), "duplicate Training sample")
        videos = np.array([r["video_id"] for r in rows])
        require(set(videos) == set(base.old.VIDEOS), "Training video set changed")
        source_bindings = base.read(CACHE / "reviewer_source_sha256.json")
        original_bindings = base.read(SOURCE / "preparation_receipt.json")["source_sha256"]
        cached = np.load(CACHE / "prepared_arrays.npz", allow_pickle=False)
        require(np.array_equal(videos, cached["videos"]), "cached row order changed")
        data = {"rows": rows, "ids": ids, "videos": videos, "feature_names": feature_schema(),
                "action_names": list(ACTIONS), "cost_columns": list(COST_COLUMNS)}
        for name, field in (("cheap", "cheap_labels"), ("full", "final_labels"), ("h0", "h0_labels")):
            for merged in (False, True):
                key = name + ("_merged" if merged else "")
                data[key] = base.old.counts(rows, field, merged)
                require(np.array_equal(data[key], cached[key]), "cached count mismatch: " + key)
        ledger = base.read(SOURCE / "costs.json")["per_call_main_attempt"]
        keep = [i for i, name in enumerate(base.old.FEATURE_ORDER) if not name.startswith("tracker_")]
        keep += list(range(len(base.old.FEATURE_ORDER), cached["X_qwen2"].shape[1]))
        data["X"] = cached["X_qwen2"][:, keep].copy()
        require(data["X"].shape == (len(rows), len(data["feature_names"])), "feature schema mismatch")
        output_rows = [[] for _ in ACTIONS]
        stops, costs = [], []
        for i, row in enumerate(rows):
            # Derived Tracker fields bundled in primary JSON are discarded. No
            # Tracker prediction file, model, or feature extractor is opened.
            row.pop("features_postcheap_with_tracker", None)
            path = (base.RAW / row["sample_id"] / "result.json").resolve()
            require(path.parent.parent == base.RAW.resolve(), "sample path escaped original cache")
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            require(digest == source_bindings[str(path)] == original_bindings[str(path)],
                    "unsealed original reviewer result: " + row["sample_id"])
            bindings[str(path)] = digest
            result = json.loads(raw)
            outputs, plans = validate_cached_row(row, result)
            for action, output in enumerate(outputs):
                output_rows[action].append({"gt": row["gt"], "mask": row["mask"], "out": output})
            pool, q1, q2 = base.qwen_features(result, row["cheap_labels"])
            x = [row["features_postcheap"][k] for k in base.old.FEATURE_ORDER if not k.startswith("tracker_")]
            require(np.array_equal(data["X"][i], np.array(x + pool + q1 + q2)),
                    "features cannot be reconstructed from cheap/Qwen-only evidence")
            stops.append(plans)
            costs.append(branch_costs(plans[1], ledger))
            if (i + 1) % 1000 == 0:
                base.log(f"PGP original evidence verified {i + 1}/{len(rows)}")
        data["stops"] = np.array(stops)
        require(np.array_equal(data["stops"], cached["stops"]), "frozen stopping cache differs")
        data["actions_counts"] = np.stack([base.old.counts(r, "out") for r in output_rows], axis=1)
        data["actions_merged_counts"] = np.stack([base.old.counts(r, "out", True) for r in output_rows], axis=1)
        data["action_costs"] = np.stack(costs)
        legacy_costs = base.make_costs(rows, data["stops"], ledger)
        data["baselines_costs"] = {"h0": legacy_costs["h0"], "cheap": legacy_costs["base"][0],
                                   "full": legacy_costs["full"], "always_qwen2_exact": legacy_costs["qwen2"][1]}
        require(np.allclose(data["action_costs"][:, 3], legacy_costs["qwen2"][1], atol=1e-12),
                "both-branch costs disagree with original Qwen2 exact replay")
        require(np.all(data["action_costs"][:, 0, 0] == 5), "Qwen2 prefix must cost five calls")
        require(np.isfinite(data["X"]).all() and np.isfinite(data["action_costs"]).all(), "non-finite data")
        baseline_metrics = {k: base.old.pooled(data[k]) for k in ("h0", "cheap", "full")}
        require(abs(baseline_metrics["cheap"]["five_head_mean_f1"] - 61.396292642) < 1e-7, "cheap anchor changed")
        require(abs(baseline_metrics["full"]["five_head_mean_f1"] - 61.666997916) < 1e-7, "full anchor changed")
        require(int(data["action_costs"][:, 3, 0].sum()) == 57541, "always-Qwen2 exact-call anchor changed")
        cached.close()
        for path, expected in bindings.items():
            require(base.sha(path) == expected, "source mutated during preparation: " + path)
        np.savez_compressed(out / output_names[0], **{k: v for k, v in data.items() if isinstance(v, np.ndarray)},
                            **{"cost_" + k: v for k, v in data["baselines_costs"].items()})
        base.write(out / output_names[1], {k: v for k, v in bindings.items() if k in source_bindings})
        receipt = {"version": VERSION, "state": "COMPLETED", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "rows": len(rows), "videos": sorted(set(videos)), "action_names": list(ACTIONS),
                   "feature_names": data["feature_names"], "cost_columns": list(COST_COLUMNS),
                   "source_sha256": bindings, "baseline_metrics": baseline_metrics,
                   "four_action_original": {name: base.old.pooled(data["actions_counts"][:, a])
                                            for a, name in enumerate(ACTIONS)},
                   "four_action_call_totals": dict(zip(ACTIONS, data["action_costs"][:, :, 0].sum(0).astype(int).tolist())),
                   "queried_prefix_only_stop_verified": True, "four_action_outputs_verified": True,
                   "tracker_used": False, "tracker_weights_loaded": False, "gate_weights_loaded": False,
                   "tracker_prediction_files_read": False, "api_calls": 0, "Testing_access": False, "VID110_access": False,
                   "limitations": ["original-attempt Training cache only; retries never substituted",
                                   "upstream frozen priors are not fully nested within Gate fitting",
                                   "known USD excludes unpriced GLM/DeepSeek; no latency inference",
                                   "exact stop preserves archived outcomes; new-video quality is unproven"],
                   "output_sha256": {name: base.sha(out / name) for name in output_names[:2]}}
        base.write(out / output_names[2], receipt)
        data["replay_receipt"] = receipt
        return data
