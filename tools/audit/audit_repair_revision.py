"""Offline integrity and independent masked-score audit of paired repair trials.

No model calls, no round selection from GT, no changes to saved predictions.
--saved-truth-only supports synthetic fixtures and explicitly forgoes fresh GT
verification. A stopped or failed partial trial is never called completed.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import ACADEMIC, redact_images
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.research.verification.prior_panel import BOUNDS, COMPONENTS

TASKS = ("instrument", "verb", "target", "ivt", "phase")
ARMS = ("single_proposer_tolerant", "distributed_ivt_tolerant")
SEATS = ("grok", "qwen", "gpt", "gemini", "deepseek")
ROUTES = {"base": "google-ai-studio", "gpt": "openai", "gemini": "google-ai-studio", "deepseek": "fireworks"}
PROVIDERS = {"base": "Google AI Studio", "gpt": "OpenAI", "gemini": "Google AI Studio", "deepseek": "Fireworks"}
CATEGORIES = ("improved", "worsened", "equal_loss_changed", "unchanged", "mixed", "unscored")
FINAL_STATUSES = {"MODEL_PASS", "UNRESOLVED", "EMPTY_POOL_UNVERIFIED"}


class AuditFailure(ValueError):
    """Saved artifacts violate an explicitly checked invariant."""


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row):
    return f"{row['video_id']}_{row['frame_id']}"


def indexed(rows):
    result = {identity(row): row for row in rows}
    require(len(result) == len(rows), "duplicate target identities")
    return result


def labels(prediction, task):
    if prediction is None:
        return set()
    values = prediction[task]
    upper = 7 if task == "phase" else BOUNDS[task]
    require(isinstance(values, list) and all(type(v) is int and 0 <= v < upper for v in values),
            "invalid prediction IDs")
    require(len(values) == len(set(values)), "duplicate prediction IDs")
    return set(values)


def metrics(rows, arm):
    """Independent set arithmetic; failed predictions never get exact credit."""
    result = {}
    for task in TASKS:
        tp = fp = fn = exact = valid = failed = 0
        for row in rows:
            require(type(row["mask"][task]) is bool, "mask must be explicit boolean")
            if not row["mask"][task]:
                continue
            valid += 1
            gt, prediction = set(row["gt"][task]), row[arm]
            predicted = labels(prediction, task)
            tp += len(predicted & gt)
            fp += len(predicted - gt)
            fn += len(gt - predicted)
            exact += prediction is not None and predicted == gt
            failed += prediction is None
        result[task] = {
            "valid_targets": valid, "failed_predictions": failed,
            "tp": tp, "fp": fp, "fn": fn, "exact_matches": exact,
            "micro_precision": (tp / (tp + fp) if tp + fp else 0.0) if valid else None,
            "micro_recall": (tp / (tp + fn) if tp + fn else 0.0) if valid else None,
            "micro_f1": (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0) if valid else None,
            "exact_set_accuracy": exact / valid if valid else None,
        }
    return result


def change_counts(rows, before="h0", after="final"):
    """A mixed frame has improved and worsened heads, irrespective of net loss."""
    frames = Counter(dict.fromkeys(CATEGORIES, 0))
    tasks = {task: Counter() for task in TASKS}
    details = []
    for row in rows:
        categories, per_head = set(), {}
        all_left = all_right = True
        for task in TASKS:
            if not row["mask"][task]:
                continue
            gt, left, right = set(row["gt"][task]), labels(row[before], task), labels(row[after], task)
            left_loss, right_loss = len(left ^ gt), len(right ^ gt)
            changed = left != right or (row[before] is None) != (row[after] is None)
            category = ("improved" if right_loss < left_loss else "worsened" if right_loss > left_loss
                        else "equal_loss_changed" if changed else "unchanged")
            categories.add(category)
            left_exact = row[before] is not None and left == gt
            right_exact = row[after] is not None and right == gt
            added, removed = right - left, left - right
            beneficial = len(added & gt) + len(removed - gt)
            harmful = len(added - gt) + len(removed & gt)
            tasks[task].update({category: 1, "valid_targets": 1, "loss_before": left_loss, "loss_after": right_loss,
                               "wrong_to_exact": int(not left_exact and right_exact),
                               "exact_to_wrong": int(left_exact and not right_exact),
                               "added_correct": len(added & gt), "added_incorrect": len(added - gt),
                               "removed_correct": len(removed & gt), "removed_incorrect": len(removed - gt),
                               "beneficial_label_edits": beneficial, "harmful_label_edits": harmful})
            per_head[task] = {"category": category, "loss_before": left_loss, "loss_after": right_loss,
                              "added": sorted(added), "removed": sorted(removed),
                              "beneficial": beneficial, "harmful": harmful}
            all_left &= left_exact
            all_right &= right_exact
        category = ("unscored" if not categories else "mixed" if {"improved", "worsened"} <= categories
                    else "improved" if "improved" in categories else "worsened" if "worsened" in categories
                    else "equal_loss_changed" if "equal_loss_changed" in categories else "unchanged")
        frames.update({category: 1, "valid_targets": int(bool(categories)),
                       "wrong_to_exact": int(bool(categories) and not all_left and all_right),
                       "exact_to_wrong": int(bool(categories) and all_left and not all_right)})
        details.append({"key": identity(row), "category": category, "tasks": per_head})
    return {"frames": dict(frames), "tasks": {t: dict(v) for t, v in tasks.items()}, "details": details}


def same_number(a, b):
    return a is b if a is None or b is None else math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def check_metrics(actual, saved):
    for task, values in actual.items():
        for name, value in values.items():
            require(same_number(value, saved[task][name]), f"score mismatch: {task}.{name}")


def image_refs(body):
    return [(part["image_url"]["data_url_sha256"], part["image_url"]["detail"])
            for message in body["messages"] if isinstance(message["content"], list)
            for part in message["content"] if part.get("type") == "image_url"]


class _JSONPairs(list):
    """Retain object fields for independent duplicate-key verification."""


def _same_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _json_path(parts):
    result = "$"
    for part in parts:
        result += (f"[{part}]" if type(part) is int else "." + part if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", part)
                   else "[" + json.dumps(part, ensure_ascii=True) + "]")
    return result


def decode_boundary_json(text, *, allow_identical_core_duplicates=False):
    """Independently check the recorded complete-fence JSON recovery contract."""
    require(isinstance(text, str), "recovered review text is not a string")
    candidate, removed = text.strip(), []
    if candidate.startswith("```"):
        first, separator, remainder = candidate.partition("\n")
        require(separator and first.strip().lower() in ("```", "```json")
                and remainder.rstrip().endswith("```"), "invalid complete JSON fence")
        candidate = remainder.rstrip()[:-3].strip()
        removed = ["leading", "trailing"]
    elif candidate.endswith("```"):
        candidate = candidate[:-3].rstrip()
        removed = ["trailing"]

    identical = []

    def convert(value, path):
        if isinstance(value, _JSONPairs):
            groups = {}
            for key, child in value:
                groups.setdefault(key, []).append(convert(child, (*path, key)))
            result = {}
            for key, versions in groups.items():
                if len(versions) > 1:
                    review_item = len(path) == 2 and (path[0] == "rows" and type(path[1]) is int
                                                     or path[0] == "judgments" and isinstance(path[1], str))
                    require(allow_identical_core_duplicates and review_item
                            and key in {"rating", "finding", "scope", "image_indices", "observation"}
                            and all(_same_json(versions[0], other) for other in versions[1:]),
                            "unapproved or conflicting duplicate JSON key")
                    identical.append({"path": _json_path((*path, key)), "field": key, "occurrences": len(versions)})
                result[key] = versions[0]
            return result
        if isinstance(value, list):
            return [convert(child, (*path, i)) for i, child in enumerate(value)]
        return value

    parsed = convert(json.loads(candidate, object_pairs_hook=_JSONPairs), ())
    require(isinstance(parsed, dict), "recovered review is not a JSON object")
    diagnostic = {"error": None, "removed_fences": removed}
    if allow_identical_core_duplicates:
        diagnostic.update(identical_duplicates=identical, duplicate_errors=[])
    return parsed, diagnostic


def pairs(pool):
    result = {(p["task"], p["label_id"]) for p in pool["propositions"]}
    require(len(result) == len(pool["propositions"]), "duplicate pool candidates")
    for proposition in pool["propositions"]:
        task, label = proposition["task"], proposition["label_id"]
        require(task in TASKS[:4] and type(label) is int and 0 <= label < BOUNDS[task], "pool ID outside ontology")
        require(proposition["id"] == f"{task}_{label}", "misbound pool candidate ID")
    return result


def with_components(values):
    result = set(values)
    for task, value in list(values):
        if task == "ivt":
            result.update(COMPONENTS[value].items())
    return result


def audit_sources(output, plan, adapter):
    drift = {}
    require((output / "execution.lock").read_text(encoding="utf-8").strip() == sha(output / "plan.json"),
            "execution marker does not bind the frozen plan")
    require(sha(Path(plan["previous_budget"]) / "budget.json") == plan["previous_budget_sha256"],
            "previous budget ledger changed")
    for name, digest in plan["source_sha256"].items():
        require(sha(output / "frozen_source" / name) == digest, f"frozen source hash mismatch: {name}")
        current = ROOT / name
        if not current.exists() or sha(current) != digest:
            drift[name] = sha(current) if current.exists() else None
    selection = indexed(plan["selection"])
    require(bool(selection), "empty selection")
    require(plan["profile"] == "repair_revision_tolerant_distributed_v1", "unrecognized trial profile")
    require(plan["arms"] == list(ARMS) and plan["threshold"] == 4 and plan["round_cap"] == 3,
            "unexpected comparison policy")
    source = Path(plan["selection_source"])
    require(sha(source) == plan["selection_sha256"], "selection-source hash mismatch")
    manifest = read(source)
    require(manifest["no_gt_label_values_used_for_selection"] is True, "selection is not declared mask-only")
    stages = {}
    for stage, indices in (("development", (1, 6)), ("confirmation", (3, 8))):
        stages[stage] = [identity([r for r in manifest["selection"] if r["video_id"] == video][i])
                         for video in ("VID103", "VID23", "VID31", "VID96") for i in indices]
    require(list(selection) == stages[plan["stage"]], "targets differ from predeclared time-index selection")
    require(not set(stages["development"]) & set(stages["confirmation"]), "confirmation targets overlap development")
    for row in selection.values():
        if adapter is not None:
            require(adapter.entries[row["video_id"]].split is DatasetSplit.TRAINING, "non-Training video")
        else:
            require(row["video_id"] in {"VID103", "VID23", "VID31", "VID96"}, "unexpected simulated Training video")
        require(identity(row) == row["key"], "selection key mismatch")
        for item in row["images"]:
            require(sha(Path(item["path"])) == item["sha256"], "selected image hash mismatch")
        preflight = read(output / "h0_preflight" / f"{row['key']}.json")
        require(len(image_refs(preflight)) == len(row["causal_frame_ids"]), "image count mismatch")
        if adapter is not None:
            rebuilt = redact_images(gemini_h0_wire(build_gemini_base(adapter, row)))
            require(rebuilt == preflight, "H0 preflight differs from actual selected images/configuration")
    return {"current_source_differences": drift, "selection_confirmation": {
        "verified_disjoint": True, "predeclared_identity_sets": stages,
        "selection_uses_gt_label_values": False, "round_selection_uses_gt_scores": False,
        "primary_round_policy": "last actual immutable snapshot under the fixed 3-round stopping policy",
        "generalization_limit": "different times in the same four Training videos; not held-out-surgery/Test evidence"}}


def money(rows):
    result = {}
    for row in rows:
        account = result.setdefault(row["account"], {})
        kind = row["charge_kind"]
        account[kind] = str(Decimal(account.get(kind, "0")) + Decimal(row["charge"]))
    return result


def audit_calls(output, plan, budget, selection):
    calls = budget["calls"]
    cache_path = output / "cache_hits.json"
    cache_hits = read(cache_path) if cache_path.exists() else []
    reused = []
    reused_runs = {}
    for hit in cache_hits:
        source = Path(hit["source"])
        require(sha(source / "request.json") == hit["request_sha256"], "cached request hash mismatch")
        require(sha(source / "response.json") == hit["response_sha256"], "cached response hash mismatch")
        record = read(source / "record.json")
        require(record["status"].startswith("JSON_PARSED"), "cache reused an unsuccessful response")
        require(all(hit[k] == record[k] for k in ("target", "stage", "seat")), "cache hit points to another stage/seat")
        reused.append((record, source))
        run = source.parent.parent
        reused_runs[str(run.resolve())] = read(run / "budget.json")
    effective = [*calls, *(r for r, _ in reused)]
    require(len(calls) <= plan["max_calls"], "call cap exceeded")
    require([r["index"] for r in calls] == list(range(len(calls))), "call ledger indices changed")
    require(len({(r["target"], r["stage"], r["seat"]) for r in calls}) == len(calls), "duplicate paid stage/seat")
    require(len({(r["target"], r["stage"], r["seat"]) for r in effective}) == len(effective), "paid/cache duplicate stage/seat")
    reference = {key: image_refs(read(output / "h0_preflight" / f"{key}.json")) for key in selection}
    evidence_hashes = {}
    locations = [(row, output / "calls" / f"{row['index']:03d}_{row['target']}_{row['stage']}_{row['seat']}") for row in calls]
    locations.extend(reused)
    for row, folder in locations:
        require(row["target"] in selection, "call outside frozen sample set")
        body = read(folder / "request.json")
        evidence_hashes[str(folder / "request.json")] = sha(folder / "request.json")
        require(read(folder / "record.json") == row, "call record differs from budget ledger")
        require(image_refs(body) == reference[row["target"]], "API image bytes/order/detail changed")
        seat, stage = row["seat"], row["stage"]
        require(seat in (*SEATS, "base"), "unknown call seat")
        require(body["model"] == (plan["h0"] if seat == "base" else plan["models"][seat]), "request model mismatch")
        if seat in ROUTES:
            require(body["provider"]["only"] == [ROUTES[seat]] and body["provider"]["allow_fallbacks"] is False,
                    "provider route is not locked")
        record_path = None
        packet = json.loads(body["messages"][0]["content"][0]["text"]) if seat != "base" else None
        if packet is not None:
            require(next(iter(packet)) == "academic_context" and packet["academic_context"] == ACADEMIC,
                    "academic context missing or misplaced")
            require(not {"gt", "ground_truth", "current_prediction", "h0", "other_reviews"} & set(packet),
                    "review exposes answer origin/GT")
            require(all(not {"origin", "source", "h0", "gt", "ground_truth"} & set(p)
                        for p in packet["propositions"]), "review candidate exposes answer origin/GT")
            arm = next((a for a in ARMS if stage.startswith(a + "_review_")), None)
            require(arm is not None, "review stage does not identify an experimental arm")
            number = int(stage.rsplit("_", 1)[1])
            record_path = output / "targets" / row["target"] / f"{arm}_round_{number}.json"
            if record_path.exists():
                require(packet["propositions"] == read(record_path)["pool"]["propositions"], "review sees a different pool")
            require(("proposal_instructions" in packet) == (arm == ARMS[1]), "arm proposal contract mismatch")
            if body["response_format"]["type"] == "json_schema":
                require(body["response_format"]["json_schema"]["schema"] == packet["response_schema"],
                        "transport and textual response schema differ")
        response_path = folder / "response.json"
        usage = row.get("usage", {}) or {}
        if response_path.exists():
            response = read(response_path)
            evidence_hashes[str(response_path)] = sha(response_path)
            require(response["http_status"] == row["http_status"], "response status differs from ledger")
            if row["status"].startswith("JSON_PARSED"):
                raw = response["body"]
                require(raw["model"] == body["model"], "parsed response model mismatch")
                if seat in PROVIDERS:
                    require(raw["provider"] == PROVIDERS[seat], "parsed response provider mismatch")
                require(raw["choices"][0]["finish_reason"] == "stop" and not raw["choices"][0]["message"].get("refusal"),
                        "truncation/refusal was recorded as parsed")
                if record_path is not None and record_path.exists():
                    recovery_path = folder / "parse_recovery.json"
                    recovery = read(recovery_path) if recovery_path.exists() else None
                    compatibility = recovery is not None and "identical_duplicates" in recovery["diagnostic"]
                    parsed, parse_diagnostic = decode_boundary_json(
                        raw["choices"][0]["message"]["content"], allow_identical_core_duplicates=compatibility)
                    if recovery is not None:
                        require(recovery["response_sha256"] == sha(response_path), "parse-recovery hash mismatch")
                        require(parse_diagnostic == recovery["diagnostic"] and _same_json(parsed, recovery["parsed"]),
                                "recorded parse recovery differs from independent parse")
                    if row["status"] == "JSON_PARSED_FENCE_NORMALIZED":
                        require(recovery is not None and bool(parse_diagnostic["removed_fences"]), "fence normalization lacks provenance")
                    require(_same_json(parsed, read(record_path)["raw_reviews"][seat]), "saved review differs from paid response")
        charge = Decimal(row["charge"])
        if row["charge_kind"] == "native":
            expected = (Decimal(str(usage["cost_in_usd_ticks"])) / Decimal(10**10)
                        if seat == "grok" else Decimal(str(usage["cost"])))
        elif row["charge_kind"] == "conservative_estimate":
            require(seat == "qwen", "unknown token-estimate billing seat")
            inp, out = map(Decimal, plan["rates"][seat])
            expected = Decimal(usage["prompt_tokens"]) * inp + Decimal(usage["completion_tokens"]) * out
        else:
            require(row["charge_kind"] == "unknown_reserved", "unknown charge kind")
            expected = Decimal(row["reserve"])
        require(charge == expected and charge >= 0, "incorrect recorded charge")
    for account, occupied in budget["occupied"].items():
        total = sum((Decimal(r["charge"]) for r in calls if r["account"] == account), Decimal(0))
        require(Decimal(occupied) == Decimal(plan["carried_occupied"][account]) + total, "occupied costs do not reconcile")
    groups = {"shared_h0": [r for r in calls if r["stage"] == "h0"]}
    groups.update({a: [r for r in calls if r["stage"].startswith(a + "_")] for a in ARMS})
    require(sum(map(len, groups.values())) == len(calls), "unattributed paid call")
    related_calls = [*calls, *(r for ledger in reused_runs.values() for r in ledger["calls"])]
    report = {"post_calls": len(calls), "call_statuses": dict(Counter(r["status"] for r in calls)),
            "cached_successful_calls": len(reused), "effective_input_calls": len(effective),
            "cache_hits": cache_hits, "cache_source_budget_sha256": {path: sha(Path(path) / "budget.json") for path in reused_runs},
            "reused_evidence_original_costs": money([r for r, _ in reused]),
            "combined_new_and_cache_source_runs": {
                "calls": len(related_calls), "costs": money(related_calls),
                "includes_failed_source_calls": True, "source_runs_counted_once": list(reused_runs),
                "note": "Includes all calls in each exact-cache source run once, plus this run's new calls; excludes carried pre-experiment history."},
            "total_new_costs": money(calls), "accounting_note": "shared H0 counted once; historical carried charges excluded; currencies and charge kinds never summed",
            "cost_groups": {g: {"calls": len(rows), "costs": money(rows)} for g, rows in groups.items()},
            "cost_by_stage": {stage: {"calls": len(rows), "costs": money(rows)}
                              for stage in sorted({r["stage"] for r in calls})
                              if (rows := [r for r in calls if r["stage"] == stage])},
            "evidence_sha256": evidence_hashes}
    return report, effective


def audit_history(output, states, plan, calls):
    actual, incomplete = {}, []
    for key, state in states.items():
        actual[key] = {}
        for arm_name in ARMS:
            arm, previous = state["arms"][arm_name], state["h0"]
            previous_pool = with_components({(t, i) for t in TASKS[:4] for i in (previous[t] if previous else [])})
            history = arm["history"]
            require([r["round"] for r in history] == list(range(1, len(history) + 1)), "noncontiguous actual review rounds")
            for record in history:
                number = record["round"]
                attempted = {row["seat"] for row in calls if row["target"] == key
                             and row["stage"] == f"{arm_name}_review_{number}"}
                if attempted != set(SEATS):
                    incomplete.append({"target": key, "arm": arm_name, "round": number,
                                       "status": "INCOMPLETE_PANEL_DISPATCH", "missing_seats": sorted(set(SEATS) - attempted)})
                require(number <= plan["round_cap"], "round cap exceeded")
                require(read(output / "targets" / key / f"{arm_name}_round_{number}.json") == record,
                        "history differs from saved round record")
                require(record["before"] == previous, "arm state crossed trajectories or skipped a round")
                pool, next_pool = pairs(record["pool"]), pairs(record["next_pool"])
                if arm_name == ARMS[0]:
                    proposed = {(t, i) for t in TASKS[:4] for i in record["proposal"][t]}
                    require(pool == with_components(previous_pool | proposed), "single-proposer pool transition differs")
                    require(next_pool == pool, "single-proposer unexpectedly queues reviewer proposals")
                else:
                    require(pool == previous_pool, "new distributed proposal entered the same review round")
                    queued = record["pending"]["queued_ivt"]
                    require(len(queued) <= 4 and len(set(queued)) == len(queued), "distributed queue cap/uniqueness violated")
                    require(all(("ivt", i) not in pool for i in queued), "queued IVT was already reviewed")
                    suggested = set()
                    counts = Counter()
                    for seat in SEATS:
                        items = record["suggestions"][seat]["ivt"]
                        require(len(items) <= 2 and len(items) == len(set(items)), "seat proposal cap/uniqueness violated")
                        require(record["proposal_diagnostics"][seat]["proposal_status"] == "VALID" or not items,
                                "invalid proposal entered suggestions")
                        suggested.update(items)
                        counts.update(items)
                    require(set(queued) <= suggested, "queued proposal has no reviewer source")
                    require({str(k): v for k, v in counts.items()} == record["pending"]["proposer_count"],
                            "candidate-source counts differ from reviewer proposals")
                    require(next_pool == with_components(pool | {("ivt", i) for i in queued}), "next pool differs from queued proposals")
                require(record["after"]["phase"] == state["h0"]["phase"], "repair changed phase")
                require({(t, i) for t in TASKS[:4] for i in record["after"][t]} <= pool,
                        "unreviewed candidate entered accepted prediction")
                for pid, diagnostic in record["diagnostics"].items():
                    scores = diagnostic["scores"]
                    require(len(scores) == 5, "mean lacks five reviewer seats")
                    mean = None if any(v is None for v in scores) else sum(scores) / 5
                    require(same_number(mean, record["means"][pid]), "mean differs from five valid scores")
                if record["status"] == "MODEL_PASS":
                    require(bool(pool) and not record["issues"] and not record["pending"]["queued_ivt"], "false MODEL_PASS")
                    if arm_name == ARMS[1]:
                        require(all(d["proposal_status"] == "VALID" for d in record["proposal_diagnostics"].values()),
                                "missing proposal was treated as MODEL_PASS")
                previous, previous_pool = record["after"], next_pool
            require(arm["current"] == previous, "final arm state differs from last actual round")
            if history:
                require(pairs(arm["pool"]) == previous_pool, "final pending pool differs")
            actual[key][arm_name] = len(history)
            if (state["h0"] is None or arm["status"] not in FINAL_STATUSES or arm["active"]
                    or arm["status"] == "UNRESOLVED" and len(history) < plan["round_cap"]):
                incomplete.append({"target": key, "arm": arm_name, "status": arm["status"], "actual_rounds": len(history)})
    return {"actual_rounds": actual, "incomplete_trajectories": incomplete,
            "new_proposals_only_eligible_in_later_rounds": True,
            "last_round_unreviewed_proposals_never_accepted": True}


def pool_statistics(states, truths, arm, number):
    coverage = {t: Counter() for t in TASKS[:4]}
    rating_quality = {t: Counter() for t in TASKS[:4]}
    invalid = Counter()
    for key, state in states.items():
        row = truths[key]
        history = [r for r in state["arms"][arm]["history"] if r["round"] <= number]
        record = history[-1] if history else None
        pool = record["pool"]["propositions"] if record else []
        for task in TASKS[:4]:
            if row["mask"][task]:
                gt = set(row["gt"][task])
                available = {p["label_id"] for p in pool if p["task"] == task}
                coverage[task].update(valid_targets=1, gt_labels=len(gt), reviewed_pool_hits=len(gt & available),
                                      pool_candidates=len(available), h0_hits=len(gt & labels(row["h0"], task)))
        for candidate in pool:
            task, pid = candidate["task"], candidate["id"]
            if not row["mask"][task]:
                continue
            correct = candidate["label_id"] in row["gt"][task]
            mean = record["means"][pid]
            state_name = "invalid" if mean is None else "high" if mean >= 4 else "low" if mean <= 2 else "uncertain"
            rating_quality[task][state_name + ("_true" if correct else "_false")] += 1
        if record and record["round"] == number:
            for diag in record["diagnostics"].values():
                invalid.update(f"{seat}:{reason}" for seat, reason in diag["invalid"].items())
    return {"candidate_coverage": {t: dict(v) for t, v in coverage.items()},
            "rating_quality": {t: dict(v) for t, v in rating_quality.items()}, "invalid_evidence": dict(invalid)}


def audit_round(output, number, plan, states, adapter):
    path = output / f"round_{number}_predictions.json"
    snapshot = indexed(read(path))
    require(set(snapshot) == {r["key"] for r in plan["selection"]}, "snapshot omits selected targets")
    saved = read(output / "scores" / f"round_{number}.json")
    require(sha(path) == saved["prediction_sha256"], "prediction digest differs from score input")
    output_arms, truths = {}, {}
    for arm in ARMS:
        rows = read(output / "scores" / f"round_{number}_{arm}_truth.json")
        by_key = indexed(rows)
        require(set(by_key) == set(snapshot), "scored targets differ from snapshot")
        for key, row in by_key.items():
            snap = snapshot[key]
            require(row["h0"] == snap["h0"] == states[key]["h0"], "H0 changed between arms/rounds")
            require(row["final"] == snap["arms"][arm]["final"], "scored prediction differs from saved final")
            history = [r for r in states[key]["arms"][arm]["history"] if r["round"] <= number]
            require(snap["arms"][arm]["actual_rounds"] == len(history), "actual_rounds differs from review history")
            require(row["final"] == (history[-1]["after"] if history else row["h0"]), "snapshot chose an unsaved/best-GT round")
        if adapter is not None:
            _, fresh = score_saved(adapter, rows)
            require(indexed(fresh) == by_key, "saved GT/mask differs from fresh dataset")
        truth_only = {key: {k: row[k] for k in ("gt", "mask", "h0")} for key, row in by_key.items()}
        if truths:
            require(truth_only == truths, "arms use different GT/masks/H0")
        truths = truth_only
        h0_metrics, final_metrics = metrics(rows, "h0"), metrics(rows, "final")
        check_metrics(h0_metrics, saved["reports"][arm]["arms"]["h0"]["tasks"])
        check_metrics(final_metrics, saved["reports"][arm]["arms"]["final"]["tasks"])
        changes = change_counts(rows)
        for key, count in changes["frames"].items():
            require(count == saved["reports"][arm]["paired"]["h0_to_final"]["frames"].get(key, 0), "frame change counts mismatch")
        paired_rows = [row for row in rows if row["h0"] is not None and all(
            snapshot[identity(row)]["arms"][a]["actual_rounds"] > 0 for a in ARMS)]
        if paired_rows:
            paired_metrics = metrics(paired_rows, "final")
            check_metrics(paired_metrics, saved["paired_reviewed_subset"][arm]["arms"]["final"]["tasks"])
        else:
            paired_metrics = None
        output_arms[arm] = {"metrics": final_metrics, "changes_from_h0": changes,
                            "paired_reviewed_targets": len(paired_rows), "paired_reviewed_metrics": paired_metrics,
                            **pool_statistics(states, by_key, arm, number)}
    return {"prediction_sha256": sha(path), "h0_metrics": h0_metrics, "arms": output_arms,
            "coverage": saved["coverage"], "costs_this_round_excluding_shared_h0": {}}


def primary_success(round_result):
    """Predeclared final snapshot criterion, never a search over saved rounds."""
    candidate = round_result["arms"][ARMS[1]]
    comparator = round_result["arms"][ARMS[0]]
    references = {"h0": round_result["h0_metrics"], ARMS[0]: comparator["metrics"]}
    comparisons = {}
    for reference, values in references.items():
        comparisons[reference] = all(
            candidate["metrics"][task][metric] is not None and values[task][metric] is not None
            and candidate["metrics"][task][metric] >= values[task][metric]
            for task in ("target", "ivt") for metric in ("micro_f1", "exact_set_accuracy"))
    candidate_loss = sum(m["fp"] + m["fn"] for m in candidate["metrics"].values())
    references_loss = {name: sum(m["fp"] + m["fn"] for m in values.values()) for name, values in references.items()}
    comparisons["fewer_total_label_errors"] = all(candidate_loss < loss for loss in references_loss.values())
    comparisons["no_extra_frames_with_any_worsened_head_vs_single_proposer"] = (
        sum(candidate["changes_from_h0"]["frames"][k] for k in ("worsened", "mixed"))
        <= sum(comparator["changes_from_h0"]["frames"][k] for k in ("worsened", "mixed")))
    return {"checks": comparisons, "met_on_audited_final_snapshot": all(comparisons.values()),
            "candidate_total_label_errors": candidate_loss, "reference_total_label_errors": references_loss,
            "not_a_guarantee_of_generalization": True}


def audit(output, *, dataset_root=Path("D:/cholec_dataset"), saved_truth_only=False):
    output = Path(output)
    result = {"schema_version": "repair_revision_independent_audit_v1", "verified": False, "completed": False,
              "fresh_gt_verified": False, "saved_truth_only": saved_truth_only, "errors": [], "rounds": {},
              "comparison": ["h0", *ARMS], "new_arms_are_separate_closed_loops": True,
              "historical_strict_replay_is_not_a_new_closed_loop": True, "model_pass_is_not_gt_correctness": True}

    def section(name, function):
        try:
            value = function()
        except (AuditFailure, OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            result["errors"].append({"section": name, "type": type(exc).__name__, "detail": str(exc)})
            return None
        return value

    inputs = section("read_inputs", lambda: (read(output / "plan.json"), read(output / "budget.json"),
                                              read(output / "inference_states.json")))
    if inputs:
        plan, budget, states = inputs
        adapter = None if saved_truth_only else section("dataset", lambda: CholecTrack20DatasetAdapter(dataset_root, causal_window_size=3))
        selection = {r["key"]: r for r in plan["selection"]}
        result["plan_sha256"] = sha(output / "plan.json")
        result["input_sha256"] = {name: sha(output / name) for name in ("plan.json", "budget.json", "inference_states.json")}
        source = section("sources_selection", lambda: audit_sources(output, plan, adapter))
        if source:
            result.update(source)
        calls = section("calls", lambda: audit_calls(output, plan, budget, selection))
        if calls:
            result.update(calls[0])
        history = section("history", lambda: audit_history(output, states, plan, calls[1] if calls else budget["calls"]))
        if history:
            result.update(history)
        completion_path = output / "completion.json"
        completion = read(completion_path) if completion_path.exists() else None
        result["completion"] = completion
        round_numbers = sorted(int(p.stem.split("_")[1]) for p in output.glob("round_*_predictions.json"))
        for number in round_numbers:
            value = section(f"round_{number}", lambda n=number: audit_round(output, n, plan, states, adapter))
            if value:
                value["costs_this_round_excluding_shared_h0"] = {arm: money([
                    r for r in budget["calls"] if r["stage"].startswith(arm + "_") and r["stage"].endswith(f"_{number}")]) for arm in ARMS}
                result["rounds"][str(number)] = value
        if completion:
            section("completion_counts", lambda: require(
                completion["targets"] == len(selection) == len(states)
                and completion["post_calls"] == len(budget["calls"])
                and round_numbers == list(range(1, completion["last_snapshot"] + 1)), "completion counts/snapshot numbers differ"))
        result["verified"] = not result["errors"] and bool(result["rounds"])
        result["fresh_gt_verified"] = adapter is not None and result["verified"]
        result["completed"] = bool(result["verified"] and completion and budget["stopped"]
                                   and history and not history["incomplete_trajectories"])
        if result["rounds"]:
            last = str(max(map(int, result["rounds"])))
            result["primary_evaluation_snapshot"] = int(last)
            result["primary_success"] = primary_success(result["rounds"][last])
            result["primary_success"]["eligible_as_completed_experiment"] = result["completed"]
        if not result["completed"]:
            result["partial_notice"] = "Partial/failed or not fully audited; report coverage and failures, do not treat unobserved repairs as negative benefit labels."
    destination = output / "independent_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("verified", "completed", "fresh_gt_verified", "post_calls",
                      "primary_evaluation_snapshot", "primary_success", "errors")}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument("--saved-truth-only", action="store_true", help="fixtures only: do not claim fresh dataset/GT verification")
    arguments = parser.parse_args()
    audit(arguments.output, dataset_root=arguments.dataset_root, saved_truth_only=arguments.saved_truth_only)
