"""Prepare and train a new offline Gate version, never overwrite old artifacts.

check validates and constructs all arrays without fitting or writing output.
run performs prepare + train; the user explicitly starts this command.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
import sklearn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from surgical_agent.research.gate import accuracy_value_v4 as gate

SOURCE = ROOT / "artifacts/training/gate/official_behavior_net_v2_20260913"
TRAIN_RECEIPT_SHA = "b1007fd326e2aa0f7256fc4e8faa13639fd9333fe38c4b4bad5a43c7e0a75e51"
DEFAULT = ROOT / "artifacts/training/gate/accuracy_value_v4_20260913_r1"
PROTOCOL = ROOT / "docs/GATE_ACCURACY_VALUE_V4_TRAINING_PROTOCOL_2026-09-13.md"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return Path(path).read_text(encoding="utf-8")


def read_json(path):
    return json.loads(read(path))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


@contextmanager
def offline():
    def forbidden(*args, **kwargs):
        raise RuntimeError("Network/API is forbidden in offline Gate v4")
    with patch("socket.socket.connect", forbidden), patch("socket.socket.connect_ex", forbidden), \
         patch("socket.create_connection", forbidden), patch("requests.sessions.Session.request", forbidden), \
         patch("urllib.request.urlopen", forbidden):
        yield


def output_path(value):
    output = Path(value).resolve()
    parent = (ROOT / "artifacts/training/gate").resolve()
    if output.parent != parent or not output.name.startswith("accuracy_value_v4_"):
        raise ValueError("Output must be a direct child artifacts/training/gate/accuracy_value_v4_* outside all frozen datasets")
    return output


def log(output, message):
    line = f"{now()} {message}"
    print(line, flush=True)
    with (output / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def verify(bindings, description):
    for filename, digest in tqdm(bindings.items(), desc=description, mininterval=1):
        if sha(filename) != digest:
            raise ValueError("Hash changed: " + str(filename))


def source_bindings():
    if sha(SOURCE / "training_receipt.json") != TRAIN_RECEIPT_SHA:
        raise ValueError("The pinned v2 training receipt changed")
    receipt = read_json(SOURCE / "training_receipt.json")
    prepared = read_json(SOURCE / "preparation_receipt.json")
    bindings = {str(SOURCE / "training_receipt.json"): TRAIN_RECEIPT_SHA}
    for manifest in (receipt, prepared):
        for name, digest in manifest["output_sha256"].items():
            path = (SOURCE / name).resolve()
            if not path.is_relative_to(SOURCE.resolve()):
                raise ValueError("Receipt path escape")
            bindings[str(path)] = digest
    bindings.update({k: v for k, v in prepared["source_sha256"].items() if k.endswith(".py")})
    # This is a sealed-export check; it does not replay all original API traces.
    for path in (Path(__file__), ROOT / "src/surgical_agent/research/gate/accuracy_value_v4.py", PROTOCOL,
                 ROOT / "src/surgical_agent/data/constants.py"):
        bindings[str(path.resolve())] = sha(path)
    verify(bindings, "Verify frozen data/code")
    return bindings


def load_membership():
    frozen = read_json(SOURCE / "selection_frozen.json")
    cohort = frozen["original_cohort"]
    keys = {r["key"] for r in cohort}
    if len(cohort) != 7372 or len(keys) != 7372 or frozen["GT_used_for_membership"]:
        raise ValueError("Frozen membership invalid")
    if set(frozen["primary_ids"]) & set(frozen["unknown_ids"]) or set(frozen["primary_ids"]) | set(frozen["unknown_ids"]) != keys:
        raise ValueError("Primary/unknown partition invalid")
    if {r["video_id"] for r in cohort} != set(gate.old.VIDEOS):
        raise ValueError("Out-of-scope cohort")
    return frozen


def dataset_arrays(dataset, frozen):
    rows = read_json(SOURCE / (dataset + "_rows.json"))
    membership_key = {"primary": "primary_ids", "sensitivity": "completed_sensitivity_ids", "strict_complete": "strict_success_sensitivity_ids"}[dataset]
    if len(rows) != gate.recipe()["datasets"][dataset] or {r["sample_id"] for r in rows} != set(frozen[membership_key]):
        raise ValueError("Dataset membership changed")
    original = {r["key"]: (r["video_id"], r["frame_id"]) for r in frozen["original_cohort"]}
    if any(original[r["sample_id"]] != (r["video_id"], r["frame_id"]) for r in rows):
        raise ValueError("Target identity mismatch")
    data = gate.prepare_arrays(rows, frozen["original_cohort"])
    return data


def coverage(frozen):
    unknown = read_json(SOURCE / "unknown_targets.json")
    if any(r["supervision_observed"] or any(r[k] is not None for k in ("help", "hurt", "neutral", "utility_delta")) for r in unknown):
        raise ValueError("Unknown labels were imputed")
    primary, completed = set(frozen["primary_ids"]), set(frozen["completed_sensitivity_ids"])
    both = primary & completed
    by_video = {}
    for v in gate.old.VIDEOS:
        ids = {r["key"] for r in frozen["original_cohort"] if r["video_id"] == v}
        by_video[v] = {"total": len(ids), "primary_observed": len(ids & primary), "completed": len(ids & completed),
                       "primary_unknown": len(ids - primary), "unknown_in_both": len(ids - primary - completed)}
    return {"full_cohort": 7372, "observed_in_both": len(both), "primary_only_fallback": len(primary - completed),
            "completed_only": len(completed - primary), "unknown_in_both": 7372 - len(primary | completed),
            "unknown_first_attempt": len(unknown), "unknown_by_reason": dict(Counter(r["reason"] for r in unknown)),
            "by_video": by_video, "behavior_comparison": read_json(SOURCE / "behavior_comparison.json"),
            "claim_scope": "Metrics conditional on scored output. No imputed full-cohort accuracy or projected failure cost; historical ledger investment separate."}


def check():
    bindings = source_bindings()
    frozen = load_membership()
    inventory = {}
    for dataset in tqdm(gate.recipe()["datasets"], desc="Check arrays (no fitting)"):
        data = dataset_arrays(dataset, frozen)
        inventory[dataset] = {"rows": len(data["ids"]), "dimensions": {v: data["X_" + v].shape[1] for v in ("counts", "semantic", "temporal")}}
    result = {"state": "CHECK_PASSED_NO_FIT", "verified_bindings": len(bindings), "inventory": inventory,
              "coverage": coverage(frozen), "api_calls": 0, "Testing_access": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def prepare(output):
    output = output_path(output)
    if output.exists():
        raise FileExistsError("Output already exists; choose a NEW --output. Never overwrite or auto-resume a previous run.")
    bindings = source_bindings()
    frozen = load_membership()
    output.mkdir(parents=True, exist_ok=False)
    log(output, "PREPARING offline data; no fit, API, VID110, or Testing")
    config = gate.recipe()
    write_json(output / "recipe.json", config)
    write_json(output / "membership.json", {k: frozen[k] for k in ("original_cohort", "primary_ids", "unknown_ids", "completed_sensitivity_ids", "strict_success_sensitivity_ids")})
    write_json(output / "coverage.json", coverage(frozen))
    write_json(output / "unknown_targets.json", read_json(SOURCE / "unknown_targets.json"))
    write_json(output / "costs.json", read_json(SOURCE / "costs.json"))
    for dataset in tqdm(config["datasets"], desc="Prepare Training arrays", mininterval=1):
        data = dataset_arrays(dataset, frozen)
        np.savez_compressed(output / (dataset + ".npz"), **data)
        log(output, f"Prepared {dataset}: n={len(data['ids'])}; features counts/semantic/history=43/599/612")
    verify(bindings, "Recheck inputs before seal")
    products = {str(p.relative_to(output)): sha(p) for p in output.iterdir() if p.is_file() and p.name != "run.log"}
    write_json(output / "preparation_receipt.json", {"state": "PREPARED_NOT_FITTED", "version": gate.VERSION,
        "created_utc": now(), "source_bindings": bindings, "product_sha256": products, "api_calls": 0,
        "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__, "deployable": False})
    log(output, "PREPARED. Data and recipe sealed before fitting.")


def verify_report(data, report):
    oof = report["oof"]
    if [r["sample_id"] for r in oof] != data["ids"].tolist():
        raise ValueError("OOF ID mismatch")
    route = np.array([r["routed"] for r in oof], bool)
    totals = np.where(route[:, None, None], data["full"], data["cheap"]).sum(axis=0)
    den = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
    f1 = np.divide(2 * totals[:, 0], den, out=np.ones(5), where=den != 0).mean() * 100
    errors = int(totals[:, 1:].sum())
    if abs(f1 - report["five_head_mean_f1"]) > 1e-10 or errors != report["total_errors"]:
        raise ValueError("Independent count recomputation failed")


def markdown(report):
    lines = ["# Accuracy-constrained value Gate v4", "", f"State: **{report['state']}**", "",
             "Offline Training development only. No Testing, API calls, or deployment. No statistical accuracy guarantee.", "",
             "Only value_semantic_history_robust is the primary model. Other variants cannot replace it post hoc.", "",
             "| Dataset | Strategy | F1 | FP+FN | Reviewed | Logical calls | Known USD* | Dev pass |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for dataset, entry in report["datasets"].items():
        for name, e in {**entry["baselines"], **entry["evaluations"]}.items():
            lines.append(f"| {dataset} | {name} | {e['five_head_mean_f1']:.6f} | {e['total_errors']} | {e['reviewed']} | {e['cost']['logical_calls']} | {e['cost']['known_usd_estimate']:.4f} | {e.get('conditional_development_pass', 'reference')} |")
    for dataset, entry in report["datasets"].items():
        e = entry["evaluations"][gate.PRIMARY]
        lines += ["", f"## {dataset}: primary per-video", "", "| Video | n | Reviewed | F1 | Delta F1 vs cheap | Delta errors | Selection |", "|---|---:|---:|---:|---:|---:|---|"]
        for fold in e["folds"]:
            v = fold["held_video"]
            a, d = e["by_video"][v], e["by_video_delta"][v]
            lines.append(f"| {v} | {a['targets']} | {a['reviewed']} | {a['five_head_mean_f1']:.6f} | {d['delta_f1']:+.6f} | {d['delta_errors']:+d} | {fold['selected']['selection_state']} |")
        lines += ["", "Failed acceptance checks: " + ", ".join(k for k, v in e["development_checks"].items() if not v), ""]
    lines += ["* Known USD is a frozen ledger scenario, excluding unpriced GLM/DS amounts. It is not new spending.", "",
              "Counts are logical requests including reused responses. Full-cohort failure costs and historical retry investment are in costs.json/coverage.json; scored-cohort costs are not extrapolated to 7,372 targets.", "",
              "All saved models are outer-fold development models (trained on three videos); no final deployment model or calibrated deployment threshold is installed.", "",
              "Upstream priors/Tracker were not fully nested. Four repeatedly explored videos are not independent confirmatory validation."]
    return "\n".join(lines) + "\n"


def train(output):
    output = output_path(output)
    prep = read_json(output / "preparation_receipt.json")
    if (output / "training_started.json").exists() or (output / "training_receipt.json").exists():
        raise FileExistsError("Training already started in this output. Preserve it and use a new output with run.")
    verify(prep["source_bindings"], "Verify source seal")
    verify({str(output / k): v for k, v in prep["product_sha256"].items()}, "Verify prepared arrays")
    config = read_json(output / "recipe.json")
    if config != gate.recipe():
        raise ValueError("Recipe changed; prepare a fresh version")
    current_environment = (platform.python_version(), np.__version__, sklearn.__version__)
    if current_environment != (prep["python"], prep["numpy"], prep["sklearn"]):
        raise ValueError("Python/numpy/sklearn environment changed after prepare")
    write_json(output / "training_started.json", {"created_utc": now(), "recipe_sha256": sha(output / "recipe.json"), "api_calls": 0})
    log(output, "TRAINING CPU only; progress is fold-level; all models stay non-deployable")
    started = time.perf_counter()
    costs, old_report = read_json(output / "costs.json"), read_json(SOURCE / "training_report.json")
    report = {"version": gate.VERSION, "created_utc": now(), "datasets": {}, "api_calls": 0, "Testing_access": False, "VID110_access": False, "deployable": False}
    total_folds = len(config["datasets"]) * len(config["variants"]) * 4
    with tqdm(total=total_folds, desc="Offline nested LOVO", mininterval=1) as progress:
        for dataset in config["datasets"]:
            with np.load(output / (dataset + ".npz"), allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
            n = len(data["ids"])
            archived = old_report["evaluations"][dataset + "_no_tracker"]
            if [r["sample_id"] for r in archived["oof"]] != data["ids"].tolist():
                raise ValueError("Archived baseline cohort alignment failed")
            old_route = np.array([r["nested_routed"] for r in archived["oof"]], bool)
            masks = {"cheap": np.zeros(n, bool), "full": np.ones(n, bool), "archived_v2": old_route,
                     "archived_v2_cap20_diagnostic": gate.cheap_cap(old_route, data["videos"], data["frames"], data["ids"])}
            entry = {"baselines": {k: gate.route_report(data, r, costs) for k, r in masks.items()}, "evaluations": {}}
            for name in config["variants"]:
                log(output, f"Start {dataset}/{name}")
                def save_fold(fold):
                    write_json(output / "folds" / dataset / name / (fold["held_video"] + ".json"), fold)
                result = gate.evaluate_variant(data, costs, name, config, on_fold=save_fold, progress=lambda v: progress.update(1))
                verify_report(data, result)
                write_json(output / "evaluations" / f"{dataset}__{name}.json", result)
                entry["evaluations"][name] = result
                log(output, f"Done {dataset}/{name}: F1={result['five_head_mean_f1']:.6f}, errors={result['total_errors']}, reviewed={result['reviewed']}/{n}, dev_pass={result['conditional_development_pass']}")
            report["datasets"][dataset] = entry
    passes = {d: e["evaluations"][gate.PRIMARY]["conditional_development_pass"] for d, e in report["datasets"].items()}
    report.update(primary_passes_by_dataset=passes, state="CONDITIONAL_DEVELOPMENT_PASS" if all(passes.values()) else "DEVELOPMENT_NOT_PASSED",
                  elapsed_seconds=time.perf_counter() - started, completed_utc=now(),
                  statistical_accuracy_guarantee=False, final_deployment_threshold=None)
    write_json(output / "training_report.json", report)
    (output / "REPORT.md").write_text(markdown(report), encoding="utf-8")
    with (output / "summary.csv").open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "strategy", "f1", "errors", "reviewed", "logical_calls", "known_usd", "dev_pass"])
        for d, entry in report["datasets"].items():
            for name, r in {**entry["baselines"], **entry["evaluations"]}.items():
                writer.writerow([d, name, r["five_head_mean_f1"], r["total_errors"], r["reviewed"], r["cost"]["logical_calls"], r["cost"]["known_usd_estimate"], r.get("conditional_development_pass", "reference")])
    verify(prep["source_bindings"], "Recheck untouched sources")
    verify({str(output / k): v for k, v in prep["product_sha256"].items()}, "Recheck prepared products")
    products = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and p.name != "run.log"}
    write_json(output / "training_receipt.json", {"state": "TRAINING_FINISHED", "development_state": report["state"], "completed_utc": now(), "output_sha256": products,
               "api_calls": 0, "Testing_access": False, "VID110_access": False, "deployable": False})
    log(output, f"FINISHED: {report['state']}. See {output / 'REPORT.md'}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "train", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT, help="Fresh artifacts/training/gate/accuracy_value_v4_* directory")
    args = parser.parse_args(argv)
    with offline():
        if args.command == "check":
            check()
        elif args.command == "prepare":
            prepare(args.output)
        elif args.command == "train":
            train(args.output)
        else:
            prepare(args.output)
            train(args.output)


if __name__ == "__main__":
    main()
