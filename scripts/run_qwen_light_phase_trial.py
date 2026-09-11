"""Eight-target paired Qwen seat substitution; no default or prompt change."""
import argparse
import base64
import hashlib
import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import Calls
from surgical_agent.research.verification.phase_rating import (
    apply_ratings,
    rating_error,
)

SOURCE = ROOT / "artifacts/preflight/phase_temporal_ratings_same32_20260910_v2"
OUTPUT = ROOT / "artifacts/preflight/qwen_light_phase_eight_20260910_v1"
MODELS = {"original": "qwen3.8-flash", "light": "qwen3.5-35b-a3b"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request(row, arm):
    body = read(OUTPUT / "requests" / f"{row['key']}_{arm}.json")
    images = [x for x in body["messages"][0]["content"] if x["type"] == "image_url"]
    for block, im in zip(images, row["images"], strict=True):
        assert sha(Path(im["path"])) == im["sha256"]
        url = "data:image/png;base64,"+base64.b64encode(Path(im["path"]).read_bytes()).decode()
        assert hashlib.sha256(url.encode()).hexdigest() == block["image_url"].pop("data_url_sha256")
        block["image_url"]["url"] = url
    assert redact_images(body) == read(OUTPUT / "requests" / f"{row['key']}_{arm}.json")
    return body


def prepare():
    if OUTPUT.exists():
        raise ValueError("new output required")
    plan = read(SOURCE / "plan.json")
    selection = []
    for video in dict.fromkeys(r["video_id"] for r in plan["selection"]):
        group = [r for r in plan["selection"] if r["video_id"] == video]
        selection.extend([group[0], group[-1]])
    assert len(selection) == 8
    frozen_inputs = {}
    for row in selection:
        result_path = SOURCE / f"targets/{row['key']}/result.json"
        frozen_inputs[str(result_path)] = sha(result_path)
        save(OUTPUT / f"cached/{row['key']}.json", read(result_path))
        for arm, model in MODELS.items():
            path = SOURCE / f"preflight_requests/{row['key']}/compact_repeat_qwen.json"
            frozen_inputs[str(path)] = sha(path)
            body = read(path)
            body["model"] = model
            assert body["enable_thinking"] is False
            save(OUTPUT / f"requests/{row['key']}_{arm}.json", body)
            request(row, arm)
        a = read(OUTPUT / f"requests/{row['key']}_original.json")
        b = read(OUTPUT / f"requests/{row['key']}_light.json")
        a.pop("model")
        b.pop("model")
        assert a == b
    for module in list(sys.modules.values()):
        path = Path(getattr(module, "__file__", "") or ".").resolve()
        if path.is_file() and path.suffix == ".py" and path.is_relative_to(ROOT) and ".venv-p2" not in path.parts:
            target = OUTPUT / "frozen_source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    config = ROOT / "DEFAULT_PIPELINE_VERSION.json"
    save(OUTPUT / "plan.json", {"models": MODELS, "selection": selection,
        "rule": "First and last frozen targets per video; short Phase rating prompt, cached other FOUR reviewers; only Qwen model changes. Fresh old/new Qwen sequential, alternating order. All-five valid means>=4 unique higher-than-H0. No GT before closure. Already inspected Training development only; no default promotion.",
        "max_calls": 16, "budget_cny": "1", "rate_envelope_cny_per_token": ["0.000001", "0.00001"],
        "rates_note": "Conservative existing rate envelope, not model-specific actual billing; no actual CNY cost claim.",
        "default_sha256": sha(config), "archive_inputs": frozen_inputs,
        "frozen_hashes": {p.relative_to(OUTPUT).as_posix(): sha(p) for p in OUTPUT.rglob("*") if p.is_file()}})
    print(json.dumps({"prepared": [r["key"] for r in selection], "calls": 16, "mock_requests": 16, "api_calls": 0}))


def verify():
    plan = read(OUTPUT / "plan.json")
    for name, digest in plan["frozen_hashes"].items():
        assert sha(OUTPUT / name) == digest
    for name, digest in plan["archive_inputs"].items():
        assert sha(Path(name)) == digest
    assert sha(ROOT / "DEFAULT_PIPELINE_VERSION.json") == plan["default_sha256"]
    return plan


def execute():
    plan = verify()
    with (OUTPUT / "execution.lock").open("x", encoding="utf-8") as stream:
        stream.write(sha(OUTPUT / "plan.json"))
    calls = Calls(OUTPUT, limits={"aliyun_cny": Decimal(1), "xai_usd": Decimal(0), "openrouter_usd": Decimal(0)},
                  rates={"qwen": ("0.000001", "0.00001")}, providers={}, max_calls=16)
    results, error = [], None
    started = perf_counter()
    try:
        for i, row in enumerate(plan["selection"]):
            cached = read(OUTPUT / f"cached/{row['key']}.json")
            record = {"key": row["key"], "h0": cached["predictions"]["h0"], "branches": {}}
            for arm in (list(MODELS) if i % 2 == 0 else list(reversed(MODELS))):
                begin = perf_counter()
                raw = calls.call(row["key"], arm, "qwen", request(row, arm))
                elapsed = perf_counter()-begin
                panel = deepcopy(cached["branches"]["compact_repeat"]["raw"])
                panel["qwen"] = raw
                pred, decision = apply_ratings(cached["predictions"]["graph_before_phase"], panel, 3)
                record["branches"][arm] = {"raw_qwen": raw, "error": rating_error(raw, 3), "prediction": pred,
                    "decision": decision, "seconds": elapsed}
            results.append(record)
            save(OUTPUT / "predictions.json", results)
            print(json.dumps({"completed": i+1, "key": row["key"], "calls": len(calls.rows)}), flush=True)
            if calls.stopped:
                break
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        save(OUTPUT / "completion.json", {"error": error, "calls": len(calls.rows), "targets": len(results),
            "elapsed_seconds": perf_counter()-started, "plan_sha256": sha(OUTPUT / "plan.json"),
            "hashes": {p.relative_to(OUTPUT).as_posix(): sha(p) for p in OUTPUT.rglob("*") if p.is_file() and "frozen_source" not in p.parts}})


def score():
    verify()
    done = read(OUTPUT / "completion.json")
    assert done["error"] is None and done["calls"] == 16 and done["targets"] == 8
    for name, digest in done["hashes"].items():
        assert sha(OUTPUT / name) == digest
    rows = read(OUTPUT / "predictions.json")
    # Reconstruct all 16 results from archived raw responses before loading GT.
    for row in rows:
        cached = read(OUTPUT / f"cached/{row['key']}.json")
        for arm in MODELS:
            branch = row["branches"][arm]
            folders = list((OUTPUT / "calls").glob(f"*_{row['key']}_{arm}_qwen"))
            assert len(folders) == 1
            response = read(folders[0] / "response.json")
            if read(folders[0] / "record.json")["status"] == "JSON_PARSED":
                assert json.loads(response["body"]["choices"][0]["message"]["content"]) == branch["raw_qwen"]
            panel = deepcopy(cached["branches"]["compact_repeat"]["raw"])
            panel["qwen"] = branch["raw_qwen"]
            pred, decision = apply_ratings(cached["predictions"]["graph_before_phase"], panel, 3)
            assert pred == branch["prediction"] and decision == branch["decision"]
    truths = {f"{r['video_id']}_{r['frame_id']}": r for r in read(SOURCE / "scored_truth.json")}
    metrics = {}
    for arm in ("h0", *MODELS):
        per_task = {}
        for task in ("instrument", "verb", "target", "ivt", "phase"):
            tp = fp = fn = exact = n = 0
            for row in rows:
                truth = truths[row["key"]]
                if not truth["mask"][task]:
                    continue
                pred = row["h0"] if arm == "h0" else row["branches"][arm]["prediction"]
                expected, actual = set(truth["gt"][task]), set(pred[task])
                tp += len(expected & actual)
                fp += len(actual-expected)
                fn += len(expected-actual)
                exact += expected == actual
                n += 1
            per_task[task] = {"tp": tp, "fp": fp, "fn": fn, "valid": n,
                "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                "precision": tp/(tp+fp) if tp+fp else 0, "accuracy": exact/n if n else None}
        counts = Counter()
        for row in rows:
            old, gt = row["h0"]["phase"], truths[row["key"]]["gt"]["phase"]
            new = old if arm == "h0" else row["branches"][arm]["prediction"]["phase"]
            counts["unchanged" if new == old else "corrected" if new == gt else "harmed" if old == gt else "wrong_to_wrong"] += 1
        metrics[arm] = {"tasks": per_task, "phase_changes": counts}
    summary = {"metrics": metrics, "calls": done["calls"], "elapsed_seconds": done["elapsed_seconds"],
        "timing_note": "Only Qwen calls freshly timed; other four seats cached, not full panel latency.",
        "per_arm_seconds": {a: sum(r["branches"][a]["seconds"] for r in rows) for a in MODELS},
        "valid_qwen": {a: sum(r["branches"][a]["error"] is None for r in rows) for a in MODELS},
        "budget": read(OUTPUT / "budget.json")["occupied"], "plan_sha256": done["plan_sha256"],
        "truth_source_sha256": sha(SOURCE / "scored_truth.json"), "raw_replayed": True}
    save(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    args = parser.parse_args()
    globals()[args.command]()
