"""A (development): prior-expanded candidate pool under a screened reviewer roster.

Isolated, single-use, Training development only; no default changes. It reuses
the confirm16 cached H0, initial graph pool, retrieval hints and Phase
recommendation, rebuilds images with the pipeline's own base builder, and sends
Codex's unchanged joint review wire. The two arms differ only in the pool:

  base      the published joint pool: the four-head graph pool plus seven Phases
  expanded  base plus leave-query-video-out prior candidates for the H0 phase:
            the top verb and target classes and the top same-instrument IVT
            relations, with the components those relations need

Both arms use the screened roster and v2.0.1 aggregation: an interaction
candidate is scored when at least three seats are valid, while Phase keeps the
published five-valid rule. Selection is Codex's `select_joint`, unchanged.
Archived arms (H0, v1.3.0 control, v2.0.0 joint_r1) and an offline v2.0.1
replay on the old roster are scored beside them at no extra cost.

Commands: prepare, preflight (zero API), execute (single-use), score.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from threading import RLock
from time import perf_counter

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import check_candidate_panel_providers as transport
from scripts import run_glm_parallel_repair as roster
from scripts import run_joint_phase_feedback_trial as joint
from scripts import run_reviewer_screening as screening
from scripts import run_split_review_trial as common
from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_prior_panel_trial import now, read, save
from surgical_agent.api.credentials import assert_secret_absent
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.perception.ontology_prompt import _TASK_NAMES
from surgical_agent.research.verification.candidate_coordinator import MAX_POOL, SEATS
from surgical_agent.research.verification.five_head_repair import normalize_five_heads
from surgical_agent.research.verification.prior_panel import COMPONENTS
from surgical_agent.research.verification.review_normalization import parse_review_json

PROFILE = "pool_expansion_dev16_v1"
SOURCE = ROOT / "artifacts/preflight/joint_phase_confirm16_independent_20260910_v1"
PAIRED = ROOT / "artifacts/preflight/joint_phase_confirm16_alternative_20260910_v1"
VARIANT = "v1"
MIN_VALID = 3
K_VERB_TARGET = 3
K_IVT = 6
NULL_VERB, NULL_TARGET, NULL_IVT_FROM = 9, 14, 94
HEADS = ("instrument", "verb", "target", "ivt")
ALL_TASKS = (*HEADS, "phase")
ARMS = ("base", "expanded")
COMPARATORS = ("h0", "control", "joint_r1")
LIMITS = {"openrouter_usd": Decimal("2.5"), "aliyun_cny": Decimal("1.5")}
FAMILY_SEAT = {"GLM": "grok", "Qwen": "qwen", "GPT": "gpt", "Gemini": "gemini", "DeepSeek": "deepseek"}
RUNTIME_FILES = [Path(__file__).resolve(), ROOT / "scripts/run_reviewer_screening.py"]


def initials_for():
    for root in (SOURCE, PAIRED):
        if (root / "initials.json").is_file():
            return read(root / "initials.json")
    raise ValueError("cached confirm16 initial state not found")


def find_prior(video, expected_sha):
    """The exact leave-video-out table the cached hints were retrieved from."""
    for path in sorted(glob.glob(str(ROOT / "artifacts/preflight/*/priors" / f"{video}.json"))):
        prior = read(path)
        if prior.get("table_sha256") == expected_sha:
            if prior["excluded_video"] != video or video in prior["fit_videos"]:
                raise ValueError("query video leaked into its own prior")
            return prior, path
    raise ValueError("no archived prior matches the hint digest for " + video)


def ranked(prior, head, phase):
    table = prior["tasks"][head]
    order = []
    for rows in (table["phase"].get(str(phase)) or [], table["global"]):
        for row in sorted((r for r in rows if r["eligible"]), key=lambda r: (-r["rate"], r["id"])):
            if row["id"] not in order:
                order.append(row["id"])
    return order


def proposition(task, label):
    components = COMPONENTS[label] if task == "ivt" else None
    name = ("/".join(_TASK_NAMES[k][v] for k, v in components.items()) if components
            else _TASK_NAMES[task][label])
    return {"id": f"{task}_{label}", "task": task, "label_id": label, "name": name,
            "components": components}


def expand(h0, pool, prior):
    """Deterministic prior candidates; membership is never a vote to accept them."""
    pairs = {(p["task"], p["label_id"]) for p in pool["propositions"] if p["task"] in HEADS}
    phase, added = h0["phase"][0], []
    for head, null in (("verb", NULL_VERB), ("target", NULL_TARGET)):
        for label in ranked(prior, head, phase):
            if sum(a[0] == head for a in added) >= K_VERB_TARGET:
                break
            if label != null and (head, label) not in pairs:
                pairs.add((head, label))
                added.append((head, label))
    instruments = set(h0["instrument"])
    for label in ranked(prior, "ivt", phase):
        if sum(a[0] == "ivt" for a in added) >= K_IVT:
            break
        if (label < NULL_IVT_FROM and ("ivt", label) not in pairs
                and COMPONENTS[label]["instrument"] in instruments):
            pairs.add(("ivt", label))
            added.append(("ivt", label))
    for task, label in list(pairs):
        if task == "ivt":
            pairs.update(COMPONENTS[label].items())
    if len(pairs) + 7 > MAX_POOL:
        raise ValueError("expanded pool exceeds the published cap; never truncate")
    four = {"propositions": [proposition(t, c) for t, c in sorted(pairs)]}
    return joint.joint_pool(four), added


def resolve_roster(selection):
    """Kept models keep their validated seat wire; new models take a free seat."""
    chosen = selection["chosen"]
    if len(chosen) != len(SEATS) or len({c["family"] for c in chosen}) != len(SEATS):
        raise ValueError("five models from five families required")
    seats, free = {}, list(SEATS)
    for c in chosen:
        kind, model = c["model"].split(":", 1)
        if kind == "current":
            seat = next(s for s in SEATS if screening.CURRENT[s] == model)
            seats[seat] = {"kind": "current", "model": model, "family": c["family"]}
            free.remove(seat)
    new = [c for c in chosen if c["model"].startswith("candidate:")]
    # A family keeps its own legacy seat label first; other families fill what is left.
    for c in sorted(new, key=lambda c: FAMILY_SEAT.get(c["family"]) not in free):
        model = c["model"].split(":", 1)[1]
        preferred = FAMILY_SEAT.get(c["family"])
        seat = preferred if preferred in free else free[0]
        name = c.get("candidate_key")
        if name is None:
            matches = [k for k, v in screening.CANDIDATES.items() if v["model"] == model]
            if len(matches) != 1:
                raise ValueError("ambiguous endpoint for " + model + "; the selection must name it")
            name = matches[0]
        if screening.CANDIDATES[name]["model"] != model:
            raise ValueError("selection names the wrong candidate for " + model)
        seats[seat] = {"kind": "candidate", "model": model, "family": c["family"], "candidate": name}
        free.remove(seat)
    return {s: seats[s] for s in SEATS}


def wire(seat, spec, base, selected, pool, current, recommendation):
    if spec["kind"] == "current":
        return joint.joint_wire(seat, base, selected, pool, current, recommendation, VARIANT)
    template = joint.joint_wire("deepseek", base, selected, pool, current, recommendation, VARIANT)
    return screening.bind(template, spec["candidate"])


def route(seat, spec):
    if spec["kind"] == "current" and seat == "qwen":
        return transport.MODELS["qwen"][0].rstrip("/") + "/chat/completions", "qwen", "aliyun_cny"
    return screening.ENDPOINT, "gpt", "openrouter_usd"


def rates(seat, spec):
    if spec["kind"] == "candidate":
        return tuple(map(Decimal, screening.CANDIDATES[spec["candidate"]]["rates"]))
    return tuple(map(Decimal, roster.RATES[seat]))


def aggregate(reviews, pool, min_valid=MIN_VALID):
    """v2.0.1: interaction candidates need `min_valid` seats; Phase keeps five."""
    means, diagnostics = {}, {}
    for p in pool["propositions"]:
        scores = [(reviews[s]["judgments"].get(p["id"]) or {}).get("rating") for s in SEATS]
        valid = [x for x in scores if x is not None]
        need = len(SEATS) if p["task"] == "phase" else min_valid
        means[p["id"]] = sum(valid) / len(valid) if len(valid) >= need else None
        diagnostics[p["id"]] = {"scores": scores, "valid_seats": len(valid),
                                "invalid": {s: "MISSING_OR_INVALID" for s, x in zip(SEATS, scores, strict=True)
                                            if x is None},
                                "explicit_conflict": bool(valid) and min(valid) <= 2 and max(valid) >= 4}
    return means, diagnostics


def decide(raw, pool, h0, min_valid=MIN_VALID):
    reviews, formats = normalize_five_heads(raw, pool, image_count=3)
    means, diagnostics = aggregate(reviews, pool, min_valid)
    prediction, decision = joint.select_joint(h0, pool, means)
    return {"reviews": reviews, "format_diagnostics": formats, "means": means,
            "diagnostics": diagnostics, "prediction": prediction, "phase_decision": decision}


class Ledger:
    def __init__(self, output):
        self.output, self.lock, self.rows = output, RLock(), []
        self.occupied = {k: Decimal(0) for k in LIMITS}
        self.stopped, self.disabled = False, set()

    def persist(self):
        save(self.output / "budget.json", {"limits": {k: str(v) for k, v in LIMITS.items()},
                                           "occupied": {k: str(v) for k, v in self.occupied.items()},
                                           "stopped": self.stopped,
                                           "disabled_seats": sorted(self.disabled), "calls": self.rows})


def call(ledger, key, stage, seat, spec, body):
    endpoint, slot, account = route(seat, spec)
    rin, rout = rates(seat, spec)
    reserve = rin * 9000 + rout * 2500
    with ledger.lock:
        if seat in ledger.disabled or ledger.stopped:
            return None
        if ledger.occupied[account] + reserve > LIMITS[account]:
            ledger.stopped = True
            return None
        ledger.occupied[account] += reserve
        row = {"index": len(ledger.rows), "key": key, "stage": stage, "seat": seat,
               "model": body["model"], "account": account, "status": "DISPATCHED",
               "reserve": str(reserve), "charge": str(reserve), "charge_kind": "unknown_reserved",
               "started_utc": now()}
        ledger.rows.append(row)
        folder = ledger.output / "calls" / f"{row['index']:03d}_{key}_{stage}_{seat}"
        save(folder / "request.json", redact_images(body))
        ledger.persist()
    secret, parsed, started = transport.key_for(slot), None, perf_counter()
    try:
        response = requests.post(endpoint, json=body, timeout=(15, 150),
                                 headers={"Authorization": "Bearer " + secret.reveal()})
        try:
            raw = response.json()
        except ValueError:
            raw = {"non_json_response": response.text}
        raw = json.loads(json.dumps(raw).replace(secret.reveal(), "[REDACTED]"))
        save(folder / "response.json", {"http_status": response.status_code, "body": raw})
        usage = raw.get("usage") or {}
        row.update(http_status=response.status_code, usage=usage, returned_model=raw.get("model"))
        if usage.get("cost") is not None:
            row.update(charge=str(Decimal(str(usage["cost"]))), charge_kind="native")
        elif account == "aliyun_cny" and usage.get("prompt_tokens") is not None:
            row.update(charge=str(rin * usage["prompt_tokens"] + rout * usage.get("completion_tokens", 0)),
                       charge_kind="conservative_estimate")
        row["status"] = "API_FAILED"
        if response.status_code in (400, 401, 402, 404) and usage.get("cost") is None:
            row.update(charge="0", charge_kind="rejected_unbilled")
            with ledger.lock:
                ledger.disabled.add(seat)
        elif response.status_code == 429 and usage.get("cost") is None:
            # Rate limiting returns no usage and is not billed; do not hold a reserve.
            row.update(charge="0", charge_kind="rate_limited_unbilled")
        if response.ok and not raw.get("error"):
            choice = raw["choices"][0]
            row["finish_reason"] = choice.get("finish_reason")
            parsed, diagnostics = parse_review_json(choice["message"].get("content"))
            row["parse"] = diagnostics
            row["status"] = "JSON_PARSED" if parsed is not None else "UNPARSEABLE"
    except (requests.RequestException, KeyError, TypeError, IndexError) as exc:
        row.update(status="FAILED", exception_type=type(exc).__name__)
    finally:
        row["elapsed_seconds"] = perf_counter() - started
        with ledger.lock:
            ledger.occupied[account] += Decimal(row["charge"]) - reserve
            row["finished_utc"] = now()
            save(folder / "record.json", row)
            ledger.persist()
        assert_secret_absent(secret, folder.glob("*.json"))
    return parsed


def target_inputs(adapter, plan_selection, initials, key):
    selected = next(s for s in plan_selection if s["key"] == key)
    ini = initials[key]
    result = read(SOURCE / "targets" / key / "result.json")
    proposal = result["phase_proposal"]
    recommendation = None if proposal["error"] else proposal["raw"]["phase_id"]
    base = common.build_gemini_base(adapter, selected)
    return selected, ini, result, recommendation, base


def pools_for(ini, video):
    prior, path = find_prior(video, ini["hints"]["audit"]["prior_sha256"])
    base_pool = joint.joint_pool(ini["pool"])
    expanded_pool, added = expand(ini["h0"], ini["pool"], prior)
    return base_pool, expanded_pool, added, path


def prepare(output, adapter, selection_path):
    if output.exists():
        raise ValueError("new single-use experiment directory required")
    seats = resolve_roster(read(selection_path))
    plan_source, initials = read(SOURCE / "plan.json"), initials_for()
    targets, fingerprints = [], {}
    for selected in plan_source["selection"]:
        key = selected["key"]
        _, ini, _, recommendation, base = target_inputs(adapter, plan_source["selection"], initials, key)
        base_pool, expanded_pool, added, prior_path = pools_for(ini, selected["video_id"])
        pools = {"base": base_pool, "expanded": expanded_pool}
        fingerprints[key] = {arm: {s: fingerprint(wire(s, seats[s], base, selected, pools[arm],
                                                        ini["h0"], recommendation)) for s in SEATS}
                             for arm in ARMS}
        targets.append({"key": key, "video_id": selected["video_id"],
                        "recommendation": recommendation, "prior": prior_path,
                        "added": added, "pool_sizes": {a: len(p["propositions"]) for a, p in pools.items()},
                        "same_pool": base_pool == expanded_pool})
    calls = sum(len(SEATS) * (1 if t["same_pool"] else 2) for t in targets)
    plan = {"profile": PROFILE, "created_utc": now(), "source_archive": str(SOURCE),
            "roster": seats, "roster_selection": str(selection_path),
            "roster_selection_sha256": sha(selection_path), "variant": VARIANT,
            "min_valid_interaction": MIN_VALID, "min_valid_phase": len(SEATS),
            "k_verb_target": K_VERB_TARGET, "k_ivt": K_IVT, "targets": targets,
            "limits": {k: str(v) for k, v in LIMITS.items()}, "max_calls": calls,
            "automatic_retries": 0, "wire_fingerprints": fingerprints,
            "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in RUNTIME_FILES},
            "policy": ("Cached confirm16 H0, graph pool, hints and Phase recommendation. Two arms "
                       "differ only by prior-expanded candidates. Screened roster, v2.0.1 aggregation "
                       "(interaction >=3 valid seats, Phase five), Codex select_joint unchanged. One "
                       "round, no retries, no GT in any request, no default change."),
            "limitations": [
                "Sixteen already-analysed Training targets: development, not confirmation.",
                "The roster was screened on overlapping Training videos.",
                "Expansion depth was fixed before this run; it is not tuned on this cohort.",
                "Phase recommendation is the archived one; no proposer call is repeated."]}
    save(output / "plan.json", plan)
    for path in RUNTIME_FILES:
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    print(json.dumps({"prepared": str(output), "targets": len(targets), "max_calls": calls,
                      "roster": {s: v["model"] for s, v in seats.items()}}), flush=True)


def mock_review(pool):
    item = {"rating": 3, "finding": "UNCLEAR", "scope": "UNCERTAIN", "image_indices": [],
            "observation": "Offline preflight answer."}
    return {"judgments": {p["id"]: dict(item) for p in pool["propositions"]}}


def preflight(output, adapter):
    """Zero API: archived faithfulness, pool audit and a mocked end-to-end pass."""
    plan, initials = read(output / "plan.json"), initials_for()
    plan_source = read(SOURCE / "plan.json")
    faithful = Counter()
    rows = []
    for t in plan["targets"]:
        key = t["key"]
        selected, ini, result, recommendation, base = target_inputs(
            adapter, plan_source["selection"], initials, key)
        archived = result["joint_r1"]
        # 1. Old roster, five-valid rule must reproduce the published round exactly.
        for seat in SEATS:
            folder = next((SOURCE / "calls").glob(f"*_{key}_joint_r1_{seat}"), None)
            if folder is not None:
                got = redact_images(joint.joint_wire(seat, base, selected, joint.joint_pool(ini["pool"]),
                                                     ini["h0"], recommendation, VARIANT))
                faithful["wire"] += got == read(folder / "request.json")
                faithful["wire_total"] += 1
        replay = decide(archived["raw"], archived["pool"], ini["h0"], min_valid=len(SEATS))
        faithful["means"] += replay["means"] == archived["means"]
        faithful["prediction"] += replay["prediction"] == archived["prediction"]
        faithful["targets"] += 1
        # 2. Frozen wires for the screened roster.
        base_pool, expanded_pool, added, _ = pools_for(ini, t["video_id"])
        pools = {"base": base_pool, "expanded": expanded_pool}
        for arm in ARMS:
            got = {s: fingerprint(wire(s, plan["roster"][s], base, selected, pools[arm],
                                       ini["h0"], recommendation)) for s in SEATS}
            if got != plan["wire_fingerprints"][key][arm]:
                raise ValueError("wire changed after prepare: " + key)
            # 3. Mocked pass through the real normalizer, aggregator and selector.
            decide({s: mock_review(pools[arm]) for s in SEATS}, pools[arm], ini["h0"])
        rows.append({"key": key, "added": added, "sizes": t["pool_sizes"]})
    save(output / "offline_preflight.json", {"checked_utc": now(), "api_calls": 0,
                                             "faithfulness": dict(faithful), "targets": rows})
    print(json.dumps({"preflight": str(output), "api_calls": 0, "faithfulness": dict(faithful),
                      "mean_added": round(sum(len(r["added"]) for r in rows) / len(rows), 2),
                      "pool_sizes": [r["sizes"] for r in rows[:3]]}), flush=True)


def execute(output, adapter):
    plan = read(output / "plan.json")
    if (output / "execution.lock").exists():
        raise ValueError("single-use experiment; no paid overwrite")
    for name, value in plan["source_sha256"].items():
        if sha(ROOT / name) != value:
            raise ValueError("runtime source changed after prepare: " + name)
    initials, plan_source = initials_for(), read(SOURCE / "plan.json")
    with (output / "execution.lock").open("x", encoding="utf-8") as handle:
        handle.write(sha(output / "plan.json"))
    ledger, start, fatal = Ledger(output), perf_counter(), None
    try:
        for t in plan["targets"]:
            if ledger.stopped:
                break
            key = t["key"]
            selected, ini, _, recommendation, base = target_inputs(
                adapter, plan_source["selection"], initials, key)
            base_pool, expanded_pool, added, _ = pools_for(ini, t["video_id"])
            pools = {"base": base_pool, "expanded": expanded_pool}
            record = {"key": key, "added": added, "arms": {}}
            for arm in ARMS:
                if arm == "expanded" and t["same_pool"]:
                    record["arms"][arm] = deepcopy(record["arms"]["base"]) | {"shared_from": "base"}
                    continue
                bodies = {s: wire(s, plan["roster"][s], base, selected, pools[arm],
                                  ini["h0"], recommendation) for s in SEATS}
                if {s: fingerprint(b) for s, b in bodies.items()} != plan["wire_fingerprints"][key][arm]:
                    raise ValueError("frozen wire differs at execution: " + key)
                with ThreadPoolExecutor(max_workers=len(SEATS)) as workers:
                    raw = dict(zip(SEATS, workers.map(
                        lambda s, k=key, a=arm, b=bodies: call(ledger, k, a, s, plan["roster"][s], b[s]),
                        SEATS),
                        strict=True))
                record["arms"][arm] = {"raw": raw, "pool": pools[arm], **decide(raw, pools[arm], ini["h0"])}
            save(output / "targets" / key / "result.json", record)
            print(json.dumps({"key": key, "calls": len(ledger.rows),
                              "spent": {k: str(v) for k, v in ledger.occupied.items()},
                              "added": len(added)}), flush=True)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        ledger.stopped = True
        ledger.persist()
        save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal,
                                          "calls": len(ledger.rows),
                                          "statuses": dict(Counter(r["status"] for r in ledger.rows)),
                                          "spent": {k: str(v) for k, v in ledger.occupied.items()},
                                          "seconds": perf_counter() - start,
                                          "gt_not_loaded_during_inference": True})


def score(output):
    plan = read(output / "plan.json")
    if not (output / "completion.json").exists():
        raise ValueError("inference must be closed before ground truth is read")
    truth = {f"{r['video_id']}_{r['frame_id']}": r for r in read(SOURCE / "scored_truth.json")}
    initials = initials_for()
    versions, coverage = {}, Counter()
    for t in plan["targets"]:
        key = t["key"]
        path = output / "targets" / key / "result.json"
        if not path.is_file():
            continue
        record, archived = read(path), read(SOURCE / "targets" / key / "result.json")
        row, h0 = truth[key], initials[key]["h0"]
        preds = {name: archived["predictions"][name] for name in COMPARATORS}
        preds["v2.0.1_old_roster"] = decide(archived["joint_r1"]["raw"], archived["joint_r1"]["pool"], h0)["prediction"]
        for arm in ARMS:
            preds[f"{arm}_new_roster"] = record["arms"][arm]["prediction"]
            pool = record["arms"][arm]["pool"]
            for head in HEADS:
                if row["mask"].get(head):
                    gt = set(row["gt"][head] or [])
                    have = {p["label_id"] for p in pool["propositions"] if p["task"] == head}
                    coverage[(arm, head, "gt")] += len(gt)
                    coverage[(arm, head, "in_pool")] += len(gt & have)
        for name, prediction in preds.items():
            versions.setdefault(name, []).append((prediction, row))
    metrics = {}
    for name, pairs in versions.items():
        entry = {}
        for task in ALL_TASKS:
            tp = fp = fn = 0
            for prediction, row in pairs:
                if row["mask"].get(task):
                    got, gt = set(prediction[task]), set(row["gt"][task] or [])
                    tp, fp, fn = tp + len(got & gt), fp + len(got - gt), fn + len(gt - got)
            entry[task] = {"tp": tp, "fp": fp, "fn": fn,
                           "f1": round(200 * tp / (2 * tp + fp + fn), 2) if 2 * tp + fp + fn else None,
                           "precision": round(100 * tp / (tp + fp), 2) if tp + fp else None}
        entry["mean_f1"] = round(sum(entry[t]["f1"] or 0 for t in ALL_TASKS) / len(ALL_TASKS), 2)
        entry["mean_precision"] = round(sum(entry[t]["precision"] or 0 for t in ALL_TASKS) / len(ALL_TASKS), 2)
        entry["errors"] = sum(entry[t]["fp"] + entry[t]["fn"] for t in ALL_TASKS)
        entry["targets"] = len(pairs)
        metrics[name] = entry
    cover = {f"{arm}/{head}": f"{coverage[(arm, head, 'in_pool')]}/{coverage[(arm, head, 'gt')]}"
             for arm in ARMS for head in HEADS}
    save(output / "metrics.json", {"scored_utc": now(), "metrics": metrics, "gt_in_pool": cover})
    print(f"{'arm':<22}" + "".join(f"{t:>11}" for t in ALL_TASKS) + f"{'meanF1':>8}{'meanP':>8}{'errors':>7}")
    for name, e in metrics.items():
        print(f"{name:<22}" + "".join(f"{e[t]['f1']:>11}" for t in ALL_TASKS)
              + f"{e['mean_f1']:>8}{e['mean_precision']:>8}{e['errors']:>7}")
    print("GT in pool:", cover)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roster-selection", type=Path)
    args = parser.parse_args()
    if args.command == "score":
        score(args.output)
    else:
        adapter = common.InferenceOnlyAdapter(
            CholecTrack20DatasetAdapter("D:/cholec_dataset", causal_window_size=3))
        if args.command == "prepare":
            prepare(args.output, adapter, args.roster_selection)
        elif args.command == "preflight":
            preflight(args.output, adapter)
        else:
            execute(args.output, adapter)
