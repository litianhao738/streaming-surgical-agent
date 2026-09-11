"""Frozen-pool paired verifier prompt experiment; scoring is offline after closure."""
from __future__ import annotations

import argparse
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

from scripts import run_contact_first_candidate_trial as prior
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, PROVIDERS, RATES_V2, review_wire
from scripts.run_repair_revision_trial import normalize_five
from scripts.score_disputed_relation_trial import exact_same as same
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import recent_mean_panel as panel
from surgical_agent.research.verification.candidate_coordinator import SEATS

SOURCE = ROOT / "artifacts/preflight/contact_first_candidate_20260909_v2"
TEMPLATE = ROOT / "src/surgical_agent/research/verification/prompts/verb_visual_audit_v1.json"
ARMS = ("control", "verb_prompt")
TASKS = ("instrument", "verb", "target", "ivt", "phase")
LIMITS = {"openrouter_usd": "4", "xai_usd": "3", "aliyun_cny": "3"}
MAX_CALLS = 240
RULE = "Verb Precision strictly increases over fresh control and is >= H0; Verb Recall/F1 and all other head Precision/F1 do not decrease; at least one beneficial change, no selector failure, all planned panels dispatched. Previously inspected Training development cohort, not confirmation."


def append_audit(body):
    result = deepcopy(body)
    packet = json.loads(result["messages"][0]["content"][0]["text"])
    if "verb_visual_audit" in packet:
        raise ValueError("duplicate audit")
    packet["verb_visual_audit"] = read(TEMPLATE)
    result["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return result


def wire(seat, base, selected, pool, arm):
    body = review_wire(seat, base, selected, pool)
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return append_audit(body) if arm == "verb_prompt" else body


def verify(output):
    plan = read(output / "plan.json")
    for k, v in {"arms":list(ARMS), "limits":LIMITS, "max_calls":MAX_CALLS, "models":MODELS,
                 "providers":PROVIDERS, "rates":RATES_V2, "success_rule":RULE}.items():
        same(plan[k], v, "protocol")
    for n, h in plan["sources"].items():
        same(sha(ROOT / n), h, "source")
    for n, h in plan["inputs"].items():
        same(sha(output / n), h, "input")
    for n, h in plan["archive"].items():
        same(sha(SOURCE / n), h, "source archive")
    for s in plan["selection"]:
        for im in s["images"]:
            same(sha(im["path"]), im["sha256"], "image")
    return plan


def prepare(output, adapter):
    if output.exists():
        raise ValueError("fresh directory required")
    old, rows, records, _, done = prior.audit(SOURCE, adapter)
    same(len(rows), 24, "24 planned targets")
    selection = deepcopy(old["selection"])
    initials = []
    for s, row in zip(selection, rows, strict=True):
        initials.append({"key": row["key"], "h0": row["h0"], "pool": records[row["key"]]["control"]["pool"]})
    save(output / "initial_state.json", initials)
    # Use archive roundtrip order for both prepare and execute, including nested pool dictionaries.
    initials = read(output / "initial_state.json")
    preflight = {}
    view = prior.infrastructure.InferenceOnlyAdapter(adapter)
    for s, initial in zip(selection, initials, strict=True):
        base = build_gemini_base(view, s)
        preflight[s["key"]] = {}
        for arm in ARMS:
            preflight[s["key"]][arm] = {seat: fingerprint(wire(seat, base, s, initial["pool"], arm)) for seat in SEATS}
    save(output / "preflight.json", preflight)
    save(output / "model_metadata.json", prior.metadata_preflight())
    dependencies = {ROOT / n for n in old["source_sha256"]} | {Path(__file__).resolve(), TEMPLATE,
        ROOT / "tests/unit/test_verb_prompt_trial.py"}
    plan = {"created_utc": now(), "arms": list(ARMS), "selection": selection, "limits": LIMITS,
        "max_calls": MAX_CALLS, "models": MODELS, "providers": PROVIDERS, "rates": RATES_V2,
        "success_rule": RULE, "sources": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(dependencies)},
        "inputs": {n: sha(output / n) for n in ("initial_state.json", "preflight.json", "model_metadata.json")},
        "archive": {**done["inference_artifact_sha256"], "completion.json": sha(SOURCE / "completion.json")},
        "protocol": "Same cached H0 and control pools. Fresh five-seat review per arm; alternate arm order. Only append Verb check. No extra rounds, candidate generation, H0, Phase, Tracker or Gate calls. Same thresholds and IVT component dependencies; other head changes are possible. No GT during inference; all targets scored with masks after closure. No retry or sample substitution."}
    save(output / "plan.json", plan)
    for p in dependencies:
        dest = output / "frozen_source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    verify(output)
    print(json.dumps({"prepared": True, "targets": 24, "nonempty_pools": sum(bool(i['pool']['propositions']) for i in initials),
                      "max_calls": MAX_CALLS, "limits": LIMITS, "plan_sha256": sha(output / 'plan.json')}), flush=True)


class BoundCalls(TimedCalls):
    def __init__(self, output, plan):
        super().__init__(output, limits={k: Decimal(v) for k, v in LIMITS.items()}, rates=RATES_V2,
                         providers=PROVIDERS, max_calls=MAX_CALLS, reasoning_seats=("grok", "gemini"))
        self.expected = read(output / "preflight.json")
        self.attempted = set()

    def call(self, target, stage, seat, body):
        same(fingerprint(body), self.expected[target][stage][seat], "frozen request")
        with self.lock:
            ident = (target, stage, seat)
            if ident in self.attempted:
                raise ValueError("retry forbidden")
            self.attempted.add(ident)
        return super().call(target, stage, seat, body)


def run_arm(calls, base, selected, initial, arm):
    pool, h0 = initial["pool"], initial["h0"]
    if not pool["propositions"]:
        return {"prediction": deepcopy(h0), "status": "EMPTY_POOL_UNVERIFIED", "panel_seconds": 0}
    start = perf_counter()
    with ThreadPoolExecutor(max_workers=5) as workers:
        raw = dict(zip(SEATS, workers.map(lambda s: calls.call(selected["key"], arm, s,
                   wire(s, base, selected, pool, arm)), SEATS), strict=True))
    elapsed = perf_counter() - start
    normalized, formatting = normalize_five(raw, pool, len(base.images))
    means, diagnostics = panel.aggregate(normalized, pool, image_count=len(base.images))
    prediction = panel.select(h0, pool, means, threshold=4)
    issues = panel.unresolved(prediction, pool, means, diagnostics, threshold=4)
    same(prediction["phase"], h0["phase"], "Phase unchanged")
    return {"prediction": prediction, "status": "UNRESOLVED" if issues else "MODEL_PASS",
            "panel_seconds": elapsed, "raw": raw, "reviews": normalized, "formatting": formatting,
            "means": means, "diagnostics": diagnostics, "issues": issues}


def execute(output, adapter):
    plan = verify(output)
    with (output / "execution.lock").open("x") as f:
        f.write(sha(output / "plan.json"))
    initials = read(output / "initial_state.json")
    calls = BoundCalls(output, plan)
    calls.persist()
    view = prior.infrastructure.InferenceOnlyAdapter(adapter)
    rows = [{"key": s["key"], "video_id": s["video_id"], "frame_id": s["frame_id"], "h0": i["h0"],
             **{a: deepcopy(i['h0']) for a in ARMS}, "statuses": dict.fromkeys(ARMS, "NOT_ATTEMPTED")}
            for s, i in zip(plan["selection"], initials, strict=True)]
    start, fatal = perf_counter(), None
    try:
        for index, (s, initial, row) in enumerate(zip(plan["selection"], initials, rows, strict=True)):
            base = build_gemini_base(view, s)
            for arm in (ARMS if index % 2 == 0 else reversed(ARMS)):
                if calls.stopped:
                    row["statuses"][arm] = "STOPPED_FALLBACK_H0"
                    continue
                result = run_arm(calls, base, s, initial, arm)
                save(output / "targets" / s["key"] / f"{arm}.json", result)
                row[arm], row["statuses"][arm] = result["prediction"], result["status"]
                save(output / "predictions.json", rows)
                print(json.dumps({"target": s['key'], "arm": arm, "calls": len(calls.rows), "status": result['status']}), flush=True)
    except Exception as exc:
        fatal = type(exc).__name__
        raise
    finally:
        calls.stopped = True
        calls.persist()
        save(output / "predictions.json", rows)
        try:
            verify(output)
        except Exception:
            fatal = fatal or "FROZEN_SOURCE_CHANGED"
            raise
        finally:
            files = [output / n for n in ("plan.json", "execution.lock", "predictions.json", "budget.json")]
            files += [p for folder in ("targets", "calls") for p in (output / folder).rglob("*.json")]
            save(output / "completion.json", {"closed_utc": now(), "fatal_error": fatal,
                "elapsed_seconds": perf_counter() - start, "post_calls": len(calls.rows),
                "hashes": {p.relative_to(output).as_posix(): sha(p) for p in sorted(files)}})


def score(output, adapter):
    plan = verify(output)
    done, ledger, rows = (read(output / n) for n in ("completion.json", "budget.json", "predictions.json"))
    if done['fatal_error'] or not ledger['stopped']:
        raise ValueError("nonfatal closure required")
    for n, h in done['hashes'].items():
        same(sha(output / n), h, "closed artifact")
    calls = ledger['calls']
    same(len(calls), done['post_calls'], "call count")
    if len(calls) > MAX_CALLS or len({(c['target'], c['stage'], c['seat']) for c in calls}) != len(calls):
        raise ValueError("duplicate/excess calls")
    for account, amount in ledger['occupied'].items():
        charges = [Decimal(c['charge']) for c in calls if c['account'] == account]
        if any(not x.is_finite() or x < 0 for x in charges):
            raise ValueError("invalid charge")
        same(sum(charges), Decimal(amount), "cost sum")
        if Decimal(amount) > Decimal(LIMITS[account]):
            raise ValueError("budget exceeded")
    replay = ReplayCalls(output, calls)
    view = prior.infrastructure.InferenceOnlyAdapter(adapter)
    initials = read(output / 'initial_state.json')
    records = {}
    for s, i, row in zip(plan['selection'], initials, rows, strict=True):
        same((row['key'],row['video_id'],row['frame_id']), (s['key'],s['video_id'],s['frame_id']), 'identity')
        same(row['h0'], i['h0'], 'cached H0')
        base = build_gemini_base(view, s)
        for a in ARMS:
            path = output / 'targets' / s['key'] / f'{a}.json'
            if not path.exists():
                same(row['statuses'][a], 'STOPPED_FALLBACK_H0', 'unattempted')
                same(row[a], i['h0'], 'fallback')
                continue
            result = run_arm(replay, base, s, i, a)
            stored = read(path)
            same({k:v for k,v in result.items() if k!='panel_seconds'},
                 {k:v for k,v in stored.items() if k!='panel_seconds'}, 'raw replay')
            same(row[a], result['prediction'], 'scored output')
            records[s['key'],a] = stored
    same(len(replay.rows), len(calls), 'all calls replayed')
    _, truth = score_saved(adapter, [{**r, 'h1':None, 'final':r['verb_prompt']} for r in rows])
    truths = {(t['video_id'],t['frame_id']):t for t in truth}
    metrics = {a:compute_repair_comparison([{**truths[r['video_id'],r['frame_id']], 'h0':r['h0'],
               'h1':None,'final':r[a]} for r in rows])['arms']['final'] for a in ('h0',*ARMS)}
    pairs = [('h0','control'),('h0','verb_prompt'),('control','verb_prompt')]
    details = {a+'_to_'+b:[{'key':r['key'],**frame_delta(r[a],r[b],truths[r['video_id'],r['frame_id']]['gt'],
                         truths[r['video_id'],r['frame_id']]['mask'])} for r in rows] for a,b in pairs}
    comparisons = {k:summarize_deltas(v) for k,v in details.items()}
    costs = {a:{account:{kind:str(sum(Decimal(c['charge']) for c in calls if c['stage']==a and c['account']==account
             and c['charge_kind']==kind)) for kind in ('native','conservative_estimate','unknown_reserved')}
             for account in LIMITS} for a in ARMS}
    timing = {a:{'panel_seconds_sum':sum(v['panel_seconds'] for (k,arm),v in records.items() if arm==a),
                 'seats':{s:{'calls':sum(c['stage']==a and c['seat']==s for c in calls),
                 'seconds_sum':sum(c['elapsed_seconds'] for c in calls if c['stage']==a and c['seat']==s)} for s in SEATS}}
                 for a in ARMS}
    before,after = (metrics[a]['tasks'] for a in ARMS)
    checks = {'verb_precision_increases':after['verb']['micro_precision']>before['verb']['micro_precision'],
        'verb_precision_ge_H0':after['verb']['micro_precision']>=metrics['h0']['tasks']['verb']['micro_precision'],
        'verb_recall_not_decreased':after['verb']['micro_recall']>=before['verb']['micro_recall'],
        **{t+'_'+m+'_not_decreased':after[t][m]>=before[t][m] for t in TASKS for m in ('micro_precision','micro_f1')},
        'beneficial_edit_exists':comparisons['control_to_verb_prompt']['fixed_label_errors']>0,
        'all_panels_dispatched':len(calls)==10*sum(bool(i['pool']['propositions']) for i in initials)}
    report = {'metrics':metrics,'comparisons':comparisons,'costs_by_arm':costs,'timing':timing,
        'elapsed_seconds':done['elapsed_seconds'],'post_calls':len(calls),
        'transport_statuses':dict(Counter(c['status'] for c in calls)),
        'success_checks':checks,'success':all(checks.values()),'raw_replay_passed':True,
        'limitations':plan['protocol']+' Previously scored Training development targets, not independent validation.'}
    save(output / 'metrics.json',report)
    save(output / 'scored_truth.json',truth)
    save(output / 'frame_deltas.json',details)
    print(json.dumps({'success':report['success'],'costs':costs,'calls':len(calls)}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','execute','score'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path('D:/cholec_dataset'),causal_window_size=3)
    globals()[args.command](args.output,adapter)
