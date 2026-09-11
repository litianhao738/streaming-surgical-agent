"""Paired 8-target lightweight Phase seats, cached other three seats, no GT online."""
import argparse
import base64
import hashlib
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import Calls
from scripts.run_qwen_light_phase_trial import read, save, sha
from surgical_agent.research.verification.phase_rating import (
    apply_ratings,
    rating_error,
)

SOURCE = ROOT / "artifacts/preflight/phase_temporal_ratings_same32_20260910_v2"
OUTPUT = ROOT / "artifacts/preflight/light_phase_pair_eight_20260910_v1"
MODELS = {"original": {"grok": "grok-4.6", "qwen": "qwen3.8-flash"},
    "light": {"grok": "mistralai/ministral-8b-2512", "qwen": "qwen3.5-35b-a3b"}}
VARIANTS = ("original", "qwen_only", "mistral_only", "both_light")


def request(row, arm, seat):
    body = read(OUTPUT / f"requests/{row['key']}_{arm}_{seat}.json")
    images = [x for x in body["messages"][0]["content"] if x["type"] == "image_url"]
    for block, im in zip(images, row["images"], strict=True):
        assert sha(Path(im["path"])) == im["sha256"]
        url = "data:image/png;base64,"+base64.b64encode(Path(im["path"]).read_bytes()).decode()
        assert hashlib.sha256(url.encode()).hexdigest() == block["image_url"].pop("data_url_sha256")
        block["image_url"]["url"] = url
    assert redact_images(body) == read(OUTPUT / f"requests/{row['key']}_{arm}_{seat}.json")
    return body


def verify():
    plan = read(OUTPUT / "plan.json")
    for name, digest in plan["frozen_hashes"].items():
        assert sha(OUTPUT / name) == digest
        if name.startswith("frozen_source/"):
            assert sha(ROOT / name.removeprefix("frozen_source/")) == digest
    for name, digest in plan["archive_inputs"].items():
        assert sha(Path(name)) == digest
    assert sha(ROOT / "DEFAULT_PIPELINE_VERSION.json") == plan["default_sha256"]
    for row in plan["selection"]:
        for im in row["images"]:
            assert sha(Path(im["path"])) == im["sha256"]
    return plan


def prepare():
    if OUTPUT.exists():
        raise ValueError("new output required")
    source_plan = read(SOURCE / "plan.json")
    selection, frozen_inputs = [], {}
    completion = read(SOURCE / "completion.json")
    assert completion["fatal_error"] is None
    for video in dict.fromkeys(r["video_id"] for r in source_plan["selection"]):
        group = [r for r in source_plan["selection"] if r["video_id"] == video]
        selection.extend([group[0], group[-1]])
    assert len(selection) == 8
    for row in selection:
        source = SOURCE / f"targets/{row['key']}/result.json"
        assert sha(source) == completion["hashes"][source.relative_to(SOURCE).as_posix()]
        frozen_inputs[str(source)] = sha(source)
        save(OUTPUT / f"cached/{row['key']}.json", read(source))
        for arm, models in MODELS.items():
            for seat, model in models.items():
                source = SOURCE / f"preflight_requests/{row['key']}/compact_repeat_{seat}.json"
                assert sha(source) == source_plan["inputs"][source.relative_to(SOURCE).as_posix()]
                frozen_inputs[str(source)] = sha(source)
                body = read(source)
                body["model"] = model
                if arm == "light" and seat == "grok":
                    body.pop("reasoning_effort", None)
                    body["provider"] = {"only": ["mistral"], "order": ["mistral"],
                        "allow_fallbacks": False, "require_parameters": True}
                save(OUTPUT / f"requests/{row['key']}_{arm}_{seat}.json", body)
                request(row, arm, seat)
        for seat in ("grok", "qwen"):
            a, b = [read(OUTPUT / f"requests/{row['key']}_{arm}_{seat}.json") for arm in MODELS]
            assert a["messages"] == b["messages"] and a["response_format"] == b["response_format"]
    for module in list(sys.modules.values()):
        path = Path(getattr(module, "__file__", "") or ".").resolve()
        if path.is_file() and path.suffix == ".py" and path.is_relative_to(ROOT) and ".venv-p2" not in path.parts:
            target = OUTPUT / "frozen_source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    save(OUTPUT / "plan.json", {"models": MODELS, "selection": selection,
        "rule": "First/last target per video from same32 Training archive, no outcome-based subset selection. Cache GPT/Gemini/DeepSeek. Fresh original/light Qwen and Grok/Mistral pairs, two calls parallel per arm, alternating arm order. Identical short Phase scoring text/images/schema, same mean4 selector. Only model/provider/reasoning configuration changes. Counterfactual individual-seat substitutions offline. No GT before sealed execution; no retries or default promotion.",
        "max_calls": 32, "limits": {"aliyun_cny": "1", "xai_usd": "0.5", "openrouter_usd": "0.5"},
        "rates": {"qwen": ["0.000001", "0.00001"], "grok": ["0.000002", "0.000006"], "gpt": ["0.00000015", "0.00000015"]},
        "rates_note": "Qwen uses conservative CNY rate envelope, not actual billing. Mistral transported with gpt credential slot, never counted as an extra GPT reviewer.",
        "default_sha256": sha(ROOT / "DEFAULT_PIPELINE_VERSION.json"), "archive_inputs": frozen_inputs,
        "frozen_hashes": {p.relative_to(OUTPUT).as_posix(): sha(p) for p in OUTPUT.rglob("*") if p.is_file()}})
    print(json.dumps({"targets": [r["key"] for r in selection], "preflight_requests": 32, "api_calls": 0}))


def variants(cached, responses):
    results = {}
    for variant in VARIANTS:
        panel = deepcopy(cached["branches"]["compact_repeat"]["raw"])
        for seat in ("grok", "qwen"):
            light = variant == "both_light" or variant == ("qwen_only" if seat == "qwen" else "mistral_only")
            panel[seat] = responses["light" if light else "original"][seat]["raw"]
        pred, decision = apply_ratings(cached["predictions"]["graph_before_phase"], panel, 3)
        results[variant] = {"prediction": pred, "decision": decision}
    return results


def execute():
    plan = verify()
    with (OUTPUT / "execution.lock").open("x", encoding="utf-8") as stream:
        stream.write(sha(OUTPUT / "plan.json"))
    calls = Calls(OUTPUT, limits={k: Decimal(v) for k, v in plan["limits"].items()}, rates=plan["rates"],
        providers={"gpt": "Mistral"}, max_calls=32, reasoning_seats=("grok",))
    rows, error = [], None
    started = perf_counter()
    try:
        for index, row in enumerate(plan["selection"]):
            responses, times = {}, {}
            for arm in (list(MODELS) if index % 2 == 0 else list(reversed(MODELS))):
                def call(seat, arm=arm, row=row):
                    begin = perf_counter()
                    transport = "gpt" if seat == "grok" and arm == "light" else seat
                    raw = calls.call(row["key"], arm+"_"+seat, transport, request(row, arm, seat))
                    return {"raw": raw, "error": rating_error(raw, 3), "seconds": perf_counter()-begin, "transport": transport}
                begin = perf_counter()
                with ThreadPoolExecutor(2) as pool:
                    responses[arm] = dict(zip(("grok", "qwen"), pool.map(call, ("grok", "qwen")), strict=True))
                times[arm] = perf_counter()-begin
            cached = read(OUTPUT / f"cached/{row['key']}.json")
            rows.append({"key": row["key"], "h0": cached["predictions"]["h0"], "responses": responses,
                "pair_seconds": times, "variants": variants(cached, responses)})
            save(OUTPUT / "predictions.json", rows)
            print(json.dumps({"completed": index+1, "calls": len(calls.rows), "errors": {a: {s: v['error'] for s, v in group.items()} for a, group in responses.items()}}), flush=True)
            if calls.stopped:
                break
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        save(OUTPUT / "completion.json", {"error": error, "calls": len(calls.rows), "targets": len(rows),
            "elapsed_seconds": perf_counter()-started, "plan_sha256": sha(OUTPUT / "plan.json"),
            "hashes": {p.relative_to(OUTPUT).as_posix(): sha(p) for p in OUTPUT.rglob("*") if p.is_file() and "frozen_source" not in p.parts}})


def score():
    verify()
    done = read(OUTPUT / "completion.json")
    assert done["error"] is None and done["calls"] == 32 and done["targets"] == 8
    for name, digest in done["hashes"].items():
        assert sha(OUTPUT / name) == digest
    rows = read(OUTPUT / "predictions.json")
    for row in rows:
        for arm, group in row["responses"].items():
            for seat, record in group.items():
                folder, = (OUTPUT / "calls").glob(f"*_{row['key']}_{arm}_{seat}_{record['transport']}")
                if read(folder / "record.json")["status"] == "JSON_PARSED":
                    assert json.loads(read(folder / "response.json")["body"]["choices"][0]["message"]["content"]) == record["raw"]
                else:
                    assert record["raw"] is None
        cached = read(OUTPUT / f"cached/{row['key']}.json")
        assert variants(cached, row["responses"]) == row["variants"]
    truths = {f"{r['video_id']}_{r['frame_id']}": r for r in read(SOURCE / "scored_truth.json")}
    metrics = {}
    for arm in ("h0", *VARIANTS):
        per_task, changes = {}, Counter()
        for task in ("instrument", "verb", "target", "ivt", "phase"):
            tp = fp = fn = exact = n = 0
            for row in rows:
                truth = truths[row["key"]]
                if not truth["mask"][task]:
                    continue
                pred = row["h0"] if arm == "h0" else row["variants"][arm]["prediction"]
                gt, actual = set(truth["gt"][task]), set(pred[task])
                tp += len(gt & actual)
                fp += len(actual-gt)
                fn += len(gt-actual)
                exact += gt == actual
                n += 1
                if task == "phase":
                    old = set(row["h0"][task])
                    changes["unchanged" if actual == old else "corrected" if actual == gt else "harmed" if old == gt else "wrong_to_wrong"] += 1
            per_task[task] = {"tp": tp, "fp": fp, "fn": fn, "valid": n,
                "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                "precision": tp/(tp+fp) if tp+fp else 0, "accuracy": exact/n if n else None}
        metrics[arm] = {"tasks": per_task, "phase_changes": changes}
    summary = {"metrics": metrics, "calls": done["calls"], "elapsed_seconds": done["elapsed_seconds"],
        "timing_note": "Only substituted pair freshly timed; remaining three cached, not full panel latency.",
        "pair_seconds": {a: sum(r["pair_seconds"][a] for r in rows) for a in MODELS},
        "seat_seconds": {a: {s: sum(r["responses"][a][s]["seconds"] for r in rows) for s in ("grok", "qwen")} for a in MODELS},
        "valid": {a: {s: sum(r["responses"][a][s]["error"] is None for r in rows) for s in ("grok", "qwen")} for a in MODELS},
        "budget": read(OUTPUT / "budget.json")["occupied"], "plan_sha256": done["plan_sha256"],
        "truth_source_sha256": sha(SOURCE / "scored_truth.json"), "raw_replayed": True}
    save(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    args = parser.parse_args()
    globals()[args.command]()
