"""Screen candidate reviewer models on archived first-round joint review questions.

Isolated and single-use. Each candidate answers exactly the question the current
five seats already answered: the archived DeepSeek-seat request text, with the
images rebuilt by the pipeline's own base builder and checked against the
archived SHA-256 digests. Only the model, provider and reasoning bindings change.
Ground truth is read only by `score`, after the run is closed, and nothing here
feeds any pipeline or default.

Commands
  prepare   freeze the questions, image digests, candidate bindings and budget
  execute   one paid pass, every candidate on every question, no retries
  score     per-model discrimination (AUC) beside the archived five seats
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from statistics import median
from threading import RLock
from time import perf_counter

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_split_review_trial as common
from scripts.check_candidate_panel_providers import key_for, redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_prior_panel_trial import now, read, save
from surgical_agent.api.credentials import assert_secret_absent
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.review_normalization import (
    CORE_FIELDS,
    parse_review_json,
)
from surgical_agent.research.verification.semantic_coordinator import item_error

PROFILE = "reviewer_screening_joint_r1_v1"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
HEADS = ("instrument", "verb", "target", "ivt")
CURRENT = {"grok": "z-ai/glm-5.3-flash", "qwen": "qwen3.5-35b-a3b", "gpt": "openai/gpt-5.6-luna",
           "gemini": "google/gemini-3.5-flash-lite", "deepseek": "deepseek/deepseek-v4-flash-vision-exp"}
CURRENT_FAMILIES = {"grok": "GLM", "qwen": "Qwen", "gpt": "GPT", "gemini": "Gemini", "deepseek": "DeepSeek"}
CANDIDATES = {
    "deepseek-v4.1-flash": {"family": "DeepSeek", "model": "deepseek/deepseek-v4.1-flash",
                            "provider": "deepseek", "reasoning": {"enabled": False},
                            "temperature": 0, "rates": ("0.00000015", "0.0000006")},
    # Same model on an endpoint outside the paid-training data policy; FP8 serving.
    "deepseek-v4.1-flash-deepinfra": {"family": "DeepSeek", "model": "deepseek/deepseek-v4.1-flash",
                                      "provider": "deepinfra/fp8", "reasoning": {"enabled": False},
                                      "temperature": 0, "rates": ("0.0000003", "0.0000012")},
    "deepseek-v4.1-flash-novita": {"family": "DeepSeek", "model": "deepseek/deepseek-v4.1-flash",
                                   "provider": "novita", "reasoning": {"enabled": False},
                                   "temperature": 0, "rates": ("0.0000003", "0.0000012")},
    "claude-haiku-4.5": {"family": "Claude", "model": "anthropic/claude-haiku-4.5",
                         "provider": "anthropic", "reasoning": {"enabled": False},
                         "temperature": 0, "rates": ("0.000001", "0.000005")},
    # Gemini 3.8 Flash requires thinking; low effort is the least it accepts.
    "gemini-3.8-flash": {"family": "Gemini", "model": "google/gemini-3.8-flash",
                         "provider": "google-ai-studio", "reasoning": {"effort": "low"},
                         "temperature": 0, "rates": ("0.00000075", "0.00000375")},
    # The OpenAI route does not advertise temperature, as for the current GPT seat.
    "gpt-5.6-luna-pro": {"family": "GPT", "model": "openai/gpt-5.6-luna-pro",
                         "provider": "openai", "reasoning": {"effort": "none"},
                         "temperature": None, "rates": ("0.0000002", "0.0000012")},
    "qwen3.7-plus": {"family": "Qwen", "model": "qwen/qwen3.7-plus", "provider": "alibaba",
                     "reasoning": {"enabled": False}, "temperature": 0,
                     "rates": ("0.00000032", "0.00000128")},
    "muse-glimmer-30b": {"family": "Meta", "model": "meta/muse-glimmer-30b",
                         "provider": "fireworks", "reasoning": {"enabled": False},
                         "temperature": 0, "rates": ("0.00000035", "0.0000015")},
}
LIMIT_USD = Decimal("1.8")
RESERVE_TOKENS = (9000, 2500)
MAX_INVALID_PCT = 15.0
RULE = ("Rank by the mean of verb, target and IVT AUC on valid items; exclude any model whose "
        "invalid-item rate exceeds 15%; keep at most one model per family; break ties by cost.")


def current_roster_archives():
    """Closed archives whose five seats were exactly the current roster."""
    out = []
    for truth in sorted((ROOT / "artifacts/preflight").glob("*/scored_truth.json")):
        root = truth.parent
        budget = root / "budget.json"
        if not budget.is_file():
            continue
        rows = read(budget).get("calls") or []
        models = {s: Counter(r["model"] for r in rows if r.get("seat") == s).most_common(1)[0][0]
                  for s in SEATS if any(r.get("seat") == s for r in rows)}
        if models == CURRENT:
            out.append(root)
    return out


def data_urls(adapter, selected):
    base = common.build_gemini_base(adapter, selected)
    return ["data:" + im.mime_type + ";base64," + base64.b64encode(im.content).decode()
            for im in base.images]


def restore(request, urls):
    """Put the pipeline's own images back and prove they are the archived bytes."""
    body = deepcopy(request)
    blocks = [c for c in body["messages"][0]["content"] if c.get("type") == "image_url"]
    if len(blocks) != len(urls):
        raise ValueError("image count differs from the archived request")
    for block, url in zip(blocks, urls, strict=True):
        if hashlib.sha256(url.encode()).hexdigest() != block["image_url"].pop("data_url_sha256"):
            raise ValueError("rebuilt image differs from the archived request")
        block["image_url"]["url"] = url
    return body


def bind(template, candidate):
    """The archived question with only this candidate's transport binding."""
    spec = CANDIDATES[candidate]
    body = deepcopy(template)
    body["model"] = spec["model"]
    body["provider"] = {"only": [spec["provider"]], "order": [spec["provider"]],
                        "allow_fallbacks": False, "require_parameters": True}
    body["reasoning"] = deepcopy(spec["reasoning"])
    if spec["temperature"] is None:
        body.pop("temperature", None)
    else:
        body["temperature"] = spec["temperature"]
    body["response_format"] = {"type": "json_object"}
    return body


def collect_questions(adapter):
    questions, seen = [], set()
    for root in current_roster_archives():
        selection = {s["key"]: s for s in read(root / "plan.json")["selection"]}
        for path in sorted((root / "calls").glob("*_deepseek/request.json")):
            parts = path.parent.name.split("_", 1)[1].split("_")
            key, stage = "_".join(parts[:2]), "_".join(parts[2:-1])
            if stage != "joint_r1":
                continue
            request = read(path)
            digest = hashlib.sha256(request["messages"][0]["content"][0]["text"].encode()).hexdigest()
            if (key, digest) in seen:
                continue
            seen.add((key, digest))
            restore(request, data_urls(adapter, selection[key]))  # verification only
            questions.append({"archive": root.name, "key": key,
                              "request": str(path.relative_to(ROOT)), "text_sha256": digest})
    return questions


def prepare(output, adapter, only=None):
    if output.exists():
        raise ValueError("new single-use screening directory required")
    if only and not set(only) <= set(CANDIDATES):
        raise ValueError("unknown candidate")
    candidates = {k: CANDIDATES[k] for k in (only or CANDIDATES)}
    questions = collect_questions(adapter)
    plan = {"profile": PROFILE, "created_utc": now(), "questions": questions,
            "candidates": candidates, "current_roster": CURRENT,
            "current_families": CURRENT_FAMILIES, "limit_usd": str(LIMIT_USD),
            "max_calls": len(questions) * len(candidates), "automatic_retries": 0,
            "selection_rule": RULE, "max_invalid_pct": MAX_INVALID_PCT,
            "source_sha256": {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))},
            "limitations": [
                "Forty first-round questions from four already-analysed Training videos.",
                "Candidates answer the compact joint prompt written for the current roster.",
                "JSON-object mode for every candidate; Markdown fences are tolerated by the published parser.",
                "Phase items are answered but only the four interaction heads are ranked."]}
    save(output / "plan.json", plan)
    print(json.dumps({"prepared": str(output), "questions": len(questions),
                      "targets": len({q["key"] for q in questions}),
                      "max_calls": plan["max_calls"], "limit_usd": str(LIMIT_USD)}), flush=True)


class Ledger:
    def __init__(self, output):
        self.output, self.lock, self.rows = output, RLock(), []
        self.occupied, self.stopped, self.disabled = Decimal(0), False, set()

    def reserve(self, candidate):
        rin, rout = map(Decimal, CANDIDATES[candidate]["rates"])
        return rin * RESERVE_TOKENS[0] + rout * RESERVE_TOKENS[1]

    def persist(self):
        save(self.output / "budget.json", {"limit_usd": str(LIMIT_USD), "stopped": self.stopped,
                                           "disabled_candidates": sorted(self.disabled),
                                           "occupied_usd": str(self.occupied), "calls": self.rows})


def call(ledger, secret, question, candidate, body):
    reserve = ledger.reserve(candidate)
    with ledger.lock:
        if candidate in ledger.disabled:
            return None
        if ledger.stopped or ledger.occupied + reserve > LIMIT_USD:
            ledger.stopped = True
            return None
        ledger.occupied += reserve
        row = {"index": len(ledger.rows), "key": question["key"], "candidate": candidate,
               "model": body["model"], "status": "DISPATCHED", "reserve": str(reserve),
               "charge": str(reserve), "charge_kind": "unknown_reserved", "started_utc": now()}
        ledger.rows.append(row)
        folder = ledger.output / "calls" / f"{row['index']:03d}_{question['key']}_{candidate}"
        save(folder / "request.json", redact_images(body))
        ledger.persist()
    parsed, started = None, perf_counter()
    try:
        response = requests.post(ENDPOINT, json=body, timeout=(15, 150),
                                 headers={"Authorization": "Bearer " + secret.reveal()})
        try:
            raw = response.json()
        except ValueError:
            raw = {"non_json_response": response.text}
        raw = json.loads(json.dumps(raw).replace(secret.reveal(), "[REDACTED]"))
        save(folder / "response.json", {"http_status": response.status_code, "body": raw})
        usage = raw.get("usage") or {}
        row.update(http_status=response.status_code, usage=usage,
                   returned_model=raw.get("model"), provider=raw.get("provider"))
        if usage.get("cost") is not None:
            row.update(charge=str(Decimal(str(usage["cost"]))), charge_kind="native")
        row["status"] = "API_FAILED"
        if response.status_code in (400, 401, 402, 404) and usage.get("cost") is None:
            # A deterministic request rejection is not billed. Stop this one
            # candidate rather than repeating the same failure on every question.
            row.update(charge="0", charge_kind="rejected_unbilled")
            with ledger.lock:
                ledger.disabled.add(candidate)
        elif response.status_code == 429 and usage.get("cost") is None:
            # Rate limiting returns no usage and is not billed; do not hold a reserve.
            row.update(charge="0", charge_kind="rate_limited_unbilled")
        if response.ok and not raw.get("error"):
            choice = raw["choices"][0]
            row["finish_reason"] = choice.get("finish_reason")
            row["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            parsed, diagnostics = parse_review_json(choice["message"].get("content"))
            row["parse"] = diagnostics
            row["status"] = "JSON_PARSED" if parsed is not None else "UNPARSEABLE"
    except (requests.RequestException, KeyError, TypeError, IndexError) as exc:
        row.update(status="FAILED", exception_type=type(exc).__name__)
    finally:
        row["elapsed_seconds"] = perf_counter() - started
        with ledger.lock:
            ledger.occupied += Decimal(row["charge"]) - reserve
            row["finished_utc"] = now()
            save(folder / "record.json", row)
            ledger.persist()
        assert_secret_absent(secret, folder.glob("*.json"))
    return parsed


def execute(output, adapter):
    plan = read(output / "plan.json")
    if (output / "execution.lock").exists():
        raise ValueError("single-use screening; no paid overwrite")
    if plan["source_sha256"] != {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))}:
        raise ValueError("screening source changed after prepare")
    with (output / "execution.lock").open("x", encoding="utf-8") as handle:
        handle.write(sha(output / "plan.json"))
    secret, ledger, answers = key_for("gpt"), Ledger(output), {}
    selections = {}
    start = perf_counter()
    try:
        for question in plan["questions"]:
            if ledger.stopped:
                break
            archive = ROOT / "artifacts/preflight" / question["archive"]
            if question["archive"] not in selections:
                selections[question["archive"]] = {s["key"]: s for s in read(archive / "plan.json")["selection"]}
            template = restore(read(ROOT / question["request"]),
                               data_urls(adapter, selections[question["archive"]][question["key"]]))
            names = list(plan["candidates"])
            with ThreadPoolExecutor(max_workers=len(names)) as workers:
                results = dict(zip(names, workers.map(
                    lambda c, q=question, t=template: call(ledger, secret, q, c, bind(t, c)),
                    names), strict=True))
            answers[f"{question['archive']}|{question['key']}"] = results
            save(output / "answers.json", answers)
            print(json.dumps({"key": question["key"], "calls": len(ledger.rows),
                              "spent_usd": str(ledger.occupied),
                              "parsed": sum(v is not None for v in results.values())}), flush=True)
    finally:
        ledger.stopped = True
        ledger.persist()
        save(output / "completion.json", {"closed_utc": now(), "calls": len(ledger.rows),
                                          "statuses": dict(Counter(r["status"] for r in ledger.rows)),
                                          "spent_usd": str(ledger.occupied),
                                          "seconds": perf_counter() - start,
                                          "gt_not_loaded_during_inference": True})


def items_of(parsed, pool):
    """Valid interaction judgments only; anything malformed is an abstention."""
    ids = {p["id"]: p for p in pool["propositions"] if p["task"] in HEADS}
    if not isinstance(parsed, dict):
        return {}, len(ids)
    if isinstance(parsed.get("judgments"), dict):
        entries = list(parsed["judgments"].items())
    elif isinstance(parsed.get("rows"), list):
        entries = [(r.get("candidate_id"), r) for r in parsed["rows"] if isinstance(r, dict)]
    else:
        return {}, len(ids)
    out, counts = {}, Counter(pid for pid, _ in entries)
    for pid, item in entries:
        if pid not in ids or counts[pid] > 1 or not isinstance(item, dict):
            continue
        core = {k: item.get(k) for k in CORE_FIELDS}
        if CORE_FIELDS <= item.keys() and item_error(core, ids[pid]["task"], 3) is None:
            out[pid] = core["rating"]
    return out, len(ids) - len(out)


def auc(pairs):
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def score(output, merge=()):
    plan = read(output / "plan.json")
    if not (output / "completion.json").exists():
        raise ValueError("inference must be closed before ground truth is read")
    answers = read(output / "answers.json")
    ledger = read(output / "budget.json")["calls"]
    pairs = defaultdict(lambda: defaultdict(list))
    invalid = Counter()
    total = Counter()
    for question in plan["questions"]:
        archive = ROOT / "artifacts/preflight" / question["archive"]
        truth = {f"{t['video_id']}_{t['frame_id']}": t for t in read(archive / "scored_truth.json")}
        row = truth[question["key"]]
        node = read(archive / "targets" / question["key"] / "result.json")["joint_r1"]
        props = [p for p in node["pool"]["propositions"]
                 if p["task"] in HEADS and row["mask"].get(p["task"])]
        for p in props:
            correct = p["label_id"] in set(row["gt"][p["task"]] or [])
            scores = (node["diagnostics"].get(p["id"]) or {}).get("scores") or [None] * 5
            for seat, value in zip(SEATS, scores, strict=True):
                name = "current:" + CURRENT[seat]
                total[name] += 1
                if value is None:
                    invalid[name] += 1
                else:
                    pairs[name][p["task"]].append((value, correct, question["key"]))
        for candidate in plan["candidates"]:
            parsed = answers.get(f"{question['archive']}|{question['key']}", {}).get(candidate)
            valid, _ = items_of(parsed, node["pool"])
            name = "candidate:" + CANDIDATES[candidate]["model"]
            for p in props:
                total[name] += 1
                if p["id"] not in valid:
                    invalid[name] += 1
                else:
                    correct = p["label_id"] in set(row["gt"][p["task"]] or [])
                    pairs[name][p["task"]].append((valid[p["id"]], correct, question["key"]))
    families = {"current:" + CURRENT[s]: CURRENT_FAMILIES[s] for s in SEATS}
    families.update({"candidate:" + c["model"]: c["family"] for c in plan["candidates"].values()})
    # One model can be screened on several endpoints; the row must name the one used.
    candidate_keys = {"candidate:" + c["model"]: k for k, c in plan["candidates"].items()}
    cost = defaultdict(list)
    seconds = defaultdict(list)
    for r in ledger:
        if r["charge_kind"] == "native":
            cost["candidate:" + r["model"]].append(float(r["charge"]))
        if r.get("status") == "JSON_PARSED":
            seconds["candidate:" + r["model"]].append(r["elapsed_seconds"])
    table = []
    for name in pairs:
        a = {h: auc([(s, y) for s, y, _ in pairs[name][h]]) for h in HEADS}
        composite = [a[h] for h in ("verb", "target", "ivt") if a[h] is not None]
        table.append({"model": name, "family": families[name], "auc": a,
                      "candidate_key": candidate_keys.get(name),
                      "composite": sum(composite) / len(composite) if composite else None,
                      "invalid_pct": round(100 * invalid[name] / max(total[name], 1), 1),
                      "median_seconds": round(median(seconds[name]), 1) if seconds[name] else None,
                      "usd_per_call": round(sum(cost[name]) / len(cost[name]), 5) if cost[name] else None})
    # Rows scored by another screening run on the same questions join the ranking;
    # a model scored here is never replaced by an older row.
    for other in merge:
        for row in read(Path(other) / "selection.json")["table"]:
            if row["model"] not in {t["model"] for t in table}:
                table.append(row | {"merged_from": str(other)})
    eligible = [t for t in table if t["composite"] is not None and t["invalid_pct"] <= MAX_INVALID_PCT]
    eligible.sort(key=lambda t: (-t["composite"], t["usd_per_call"] or 0))
    chosen, used = [], set()
    for t in eligible:
        if t["family"] in used:
            continue
        chosen.append(t)
        used.add(t["family"])
        if len(chosen) == 5:
            break
    save(output / "selection.json", {"scored_utc": now(), "rule": plan["selection_rule"],
                                     "table": table, "chosen": chosen})
    print(f"{'model':<50}{'fam':<9}{'I':>6}{'V':>6}{'T':>6}{'IVT':>6}{'comp':>7}{'inv%':>6}{'sec':>6}{'$/call':>9}")
    for t in sorted(table, key=lambda t: -(t["composite"] or 0)):
        a = t["auc"]
        print(f"{t['model']:<50}{t['family']:<9}"
              + "".join(f"{a[h]:6.3f}" if a[h] is not None else f"{'-':>6}" for h in HEADS)
              + f"{t['composite'] or 0:7.3f}{t['invalid_pct']:6.1f}"
              + f"{t['median_seconds'] or 0:6.1f}{t['usd_per_call'] or 0:9.5f}")
    print("\nchosen roster:", [f"{t['family']}={t['model']}" for t in chosen])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="screen only these candidates")
    parser.add_argument("--merge", nargs="*", type=Path, default=(),
                        help="other screening outputs on the same questions to rank together")
    args = parser.parse_args()
    if args.command == "score":
        score(args.output, args.merge)
    else:
        adapter = common.InferenceOnlyAdapter(
            CholecTrack20DatasetAdapter("D:/cholec_dataset", causal_window_size=3))
        if args.command == "prepare":
            prepare(args.output, adapter, args.only)
        else:
            execute(args.output, adapter)
