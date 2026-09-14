"""Fixed zero-API policies; see docs/SCHEME4_AMBIGUITY_AUDIT_PROTOCOL_2026-09-14.md."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.run_pgp_pipeline import CachedBackend
from surgical_agent.research.gate import pgp_runtime as runtime
from surgical_agent.research.gate.tracker_pipeline_v2 import output_modules, null_label_cleanup, current_classes
from surgical_agent.research.gate.pgp_tracker_gemini38 import FrozenTracker
from surgical_agent.research.verification.prior_gated_repair import phase_rate
from surgical_agent.research.verification.prior_panel import COMPONENTS
from surgical_agent.perception.final_only import final_only_schema
from tools.audit.gate_proposals_offline_20260913 import offline

TASKS = ('instrument', 'verb', 'target', 'ivt', 'phase')
POLICIES = ('baseline', 'qwen4_add', 'qwen5_add', 'consumed_panel_add', 'consumed_panel_release')
CAPS = {t: final_only_schema()['properties'][t]['properties']['selected_ids']['maxItems'] for t in TASKS[:-1]}
SOURCE = ROOT / 'artifacts/training/gate/full_official_reviewers_20260912_v1'
BASE = ROOT / 'artifacts/preflight/scheme4_output_v22_all_20260914'
TRUTH = ROOT / 'artifacts/training/gate/official_behavior_net_v2_20260913/primary_rows.json'
LIVE = ROOT / 'artifacts/experiments/scheme4_two_targets_20260914_r1'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def canonical(p):
    return {t: sorted(set(p[t])) for t in TASKS}


def finish(raw, base):
    pred, _ = output_modules(raw, base['_tracker_classes'], ontology_filter=True)
    pred, _ = null_label_cleanup(pred, base['probe_ratings'])
    pred['phase'] = base['prediction']['phase']
    return canonical(pred)


def apply_policy(name, base, pool, prior, before):
    """Inference only: no truth, frame-specific whitelist or unconsumed review access."""
    old = canonical(base['prediction'])
    out = deepcopy(old)
    if name == 'baseline':
        return out, False
    if name == 'consumed_panel_release':
        if base['compact_depth'] == 5:
            out = finish(before, base)
    else:
        ids = {p['label_id'] for p in pool['propositions'] if p['task'] == 'ivt'}
        for c in sorted((ids & runtime.AMB_IVTS) - set(old['ivt'])):
            comp = COMPONENTS[c]
            if comp['instrument'] not in old['instrument'] or comp['target'] not in old['target']:
                continue
            if name == 'consumed_panel_add':
                allowed = base['compact_depth'] == 5 and c in before['ivt']
            else:
                ratings = base['probe_ratings']
                value = ratings.get(f'ivt:{c}')
                minimum = 5 if name == 'qwen5_add' else 4
                allowed = value is not None and value >= minimum
                allowed = allowed and all(ratings.get(f'{t}:{v}') is not None and ratings[f'{t}:{v}'] >= 4
                                          for t, v in comp.items())
                allowed = allowed and phase_rate(prior, 'ivt', c, base['h0']['phase'][0]) >= .01
            if allowed:
                out['ivt'].append(c)
                out['verb'].append(comp['verb'])
    out = canonical(out)
    if any(len(out[t]) > cap for t, cap in CAPS.items()):
        return old, True
    return out, False


def reconstruct(base, record, prior):
    pool = runtime.make_pool(base['h0'])
    try:
        pool = runtime.make_pool(base['h0'], record['proposal_raw'], pool)
    except (ValueError, TypeError):
        pass
    assert pool == record['pool'], ('pool mismatch', base['key'])
    assert record['h0'] == base['h0'], ('H0 mismatch', base['key'])
    qindex = runtime.SEATS.index('qwen')
    qr = {f"{p['task']}:{p['label_id']}": record['diagnostics'][p['id']]['scores'][qindex]
          for p in pool['propositions']}
    assert qr == base['probe_ratings'], ('probe mismatch', base['key'])
    before = deepcopy(base['cheap'])
    if base['compact_depth'] == 5:
        assert base['gate_action'] == 1
        # All five seats are actually consumed by this fixed route. Stored means are sealed.
        before, _ = runtime.select_prior_gated(base['h0'], pool, record['means'], prior,
                                               phase=base['h0']['phase'][0], **runtime.GATE)
    protected = runtime.repair(base['cheap'], before) if base['gate_action'] else deepcopy(before)
    assert finish(protected, base) == base['prediction'], ('baseline reconstruction', base['key'])
    return pool, before, protected


def metrics(preds, truth):
    counts = {t: [0, 0, 0] for t in TASKS}
    for key, p in preds.items():
        for t in TASKS:
            a, b = set(p[t]), set(truth[key]['gt'][t])
            for j, n in enumerate((len(a & b), len(a-b), len(b-a))):
                counts[t][j] += n
    f1 = {t: 200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 100. for t, (tp, fp, fn) in counts.items()}
    return {'rows': len(preds), 'mean_f1': sum(f1.values()) / len(TASKS), 'f1': f1,
            'errors': sum(fp+fn for tp,fp,fn in counts.values()), 'counts_tp_fp_fn': counts}


def summarize(arms, truth, fallbacks):
    result = {}
    for name, preds in arms.items():
        result[name] = metrics(preds, truth)
        result[name]['per_video'] = {v: metrics({k:p for k,p in preds.items() if truth[k]['video_id']==v}, truth)
                                     for v in sorted({truth[k]['video_id'] for k in preds})}
        edits = Counter()
        for k,p in preds.items():
            old = arms['baseline'][k]
            previous = now = 0
            for t in TASKS:
                a,b,g = set(old[t]),set(p[t]),set(truth[k]['gt'][t])
                previous += len(a ^ g); now += len(b ^ g)
                if t == 'ivt':
                    edits['added_correct_ivts'] += len((b-a) & g)
                    edits['added_wrong_ivts'] += len((b-a)-g)
                    edits['removed_correct_ivts'] += len((a-b)&g)
                    edits['removed_wrong_ivts'] += len((a-b)-g)
            if p != old:
                edits['changed_frames'] += 1
                edits['better_frames' if now < previous else 'worse_frames' if now > previous else 'equal_error_frames'] += 1
        result[name]['edits'] = dict(edits)
        result[name]['capacity_fallbacks'] = fallbacks[name]
    base = result['baseline']
    for name, r in result.items():
        r['passes_development_screen'] = (r['mean_f1'] > base['mean_f1'] and r['f1']['ivt'] > base['f1']['ivt']
             and r['errors'] < base['errors'] and all(
                 s['mean_f1'] >= base['per_video'][v]['mean_f1'] and s['f1']['ivt'] >= base['per_video'][v]['f1']['ivt']
                 and s['errors'] <= base['per_video'][v]['errors'] for v,s in r['per_video'].items()))
    return result


def diagnose(stages, truth, bases):
    reports = {}
    for wanted in (None, 19):
        stage_hits, missed, qwen = Counter(), Counter(), Counter()
        breakdown = {}
        for k, st in stages.items():
            b = bases[k]
            gt = set(truth[k]['gt']['ivt'])
            if wanted is not None:
                gt &= {wanted}
            for c in gt:
                stage_hits['gt_instances'] += 1
                for name, values in st.items():
                    stage_hits[name] += c in values
                rating = b['probe_ratings'].get(f'ivt:{c}')
                qwen['not_in_pool' if c not in st['pool'] else 'invalid' if rating is None else f'rating_{rating}'] += 1
                if c in st['final']:
                    continue
                if c not in st['pool']:
                    reason = 'missing_candidate'
                elif c not in st['before_protection']:
                    reason = 'gate_skip' if not b['gate_action'] else 'early_stop' if b['compact_depth'] < 5 else 'panel_or_prior_not_selected'
                elif c not in st['after_protection']:
                    reason = 'ambiguity_rollback'
                else:
                    reason = 'output_filter'
                missed[reason] += 1
                breakdown.setdefault(c, Counter())[reason] += 1
        reports['all_ivts' if wanted is None else 'ivt_19'] = {
            'stage_true_instances': dict(stage_hits), 'final_miss_partition': dict(missed),
            'qwen_for_gt': dict(qwen),
            'largest_missing_classes': [{'ivt':c,'misses':sum(cnt.values()),'reasons':dict(cnt)}
                                        for c,cnt in sorted(breakdown.items(),key=lambda x:-sum(x[1].values()))[:15]]}
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    config = read(ROOT / 'DEFAULT_PIPELINE_VERSION.json')
    inventory_path = ROOT / config['replay_inventory']
    assert sha(inventory_path) == config['replay_inventory_sha256']
    sealed = read(inventory_path)
    receipt = read(BASE / 'receipt.json')
    assert receipt['state'] == 'PASS' and sha(BASE/'predictions.jsonl') == receipt['predictions_sha256']
    bases = {p['key']:p for p in map(json.loads,(BASE/'predictions.jsonl').read_text('utf-8').splitlines())}
    assert len(bases) == 6059 and set(bases) == set(sealed['keys'])
    tracker = FrozenTracker(ROOT/config['tracker_index'])
    selections = {s['key']:s for s in read(SOURCE/'plan.json')['selection']}
    priors = {v: read(SOURCE/'priors'/f'{v}.json') for v in ('VID103','VID23','VID31','VID96')}
    arms = {name:{} for name in POLICIES}; fallbacks = Counter(); stages = {}
    for n,(k,b) in enumerate(bases.items(),1):
        path = SOURCE/'targets'/k/'result.json'
        assert sha(path) == sealed['source_sha256'][k]
        prior = priors[b['video_id']]
        assert prior['excluded_video'] == b['video_id'] and b['video_id'] not in prior['fit_videos']
        selected = selections[k]
        b['_tracker_classes'] = current_classes(tracker.snapshot(selected), selected)
        pool,before,protected = reconstruct(b, read(path), prior)
        for name in POLICIES:
            arms[name][k], failed = apply_policy(name,b,pool,prior,before)
            fallbacks[name] += failed
        stages[k] = {'h0':b['h0']['ivt'], 'pool':[p['label_id'] for p in pool['propositions'] if p['task']=='ivt'],
                     'cheap':b['cheap']['ivt'], 'before_protection':before['ivt'],
                     'after_protection':protected['ivt'], 'final':b['prediction']['ivt']}
        if n % 1000 == 0:
            print(f'Validated {n}/6059 cached rows', flush=True)
    # GT is opened after every full-cache inference arm has been computed.
    truth = {r['sample_id']:r for r in read(TRUTH)}
    assert all(all(truth[k]['mask'][t] for t in TASKS) for k in bases)
    result = summarize(arms,truth,fallbacks)
    diagnosis = diagnose(stages,truth,bases)
    livearms = {name:{} for name in POLICIES}; livefb = Counter(); live_details = []
    assert sha(LIVE/'predictions.jsonl') == read(LIVE/'receipt.json')['predictions_sha256']
    for b in map(json.loads,(LIVE/'predictions.jsonl').read_text('utf-8').splitlines()):
        k=b['key']; folder=LIVE/'targets'/k
        selected = next(s for s in read(LIVE/'plan.json')['selection'] if s['key']==k)
        b['_tracker_classes'] = current_classes(tracker.snapshot(selected), selected)
        proposal = read(folder/'run/calls'/f'001_{k}_proposal_base/response.json')['body']['choices'][0]['message']['content']
        proposal = json.loads(proposal)
        raw = json.loads(read(folder/'changed/control_graph_qwen/response.json')['choices'][0]['message']['content'])
        pool = runtime.make_pool(b['h0']); pool=runtime.make_pool(b['h0'],proposal,pool)
        normalized,_ = CachedBackend.normalize_compact({s:raw if s=='qwen' else None for s in runtime.SEATS},pool,3)
        _,diag = runtime.aggregate(normalized,pool,image_count=3)
        b['probe_ratings'] = {f"{p['task']}:{p['label_id']}":diag[p['id']]['scores'][runtime.SEATS.index('qwen')] for p in pool['propositions']}
        b['prediction'] = finish(b['cheap'], b)
        prior = read(LIVE/'priors'/f"{b['video_id']}.json")
        for name in POLICIES:
            livearms[name][k],failed = apply_policy(name,b,pool,prior,b['cheap'])
            livefb[name] += failed
        live_details.append({'key':k,'predictions':{name:livearms[name][k] for name in POLICIES},
                             'probe_ratings':b['probe_ratings'], 'gate_action':b['gate_action'], 'logical_calls':b['logical_calls']})
    write(args.output/'report.json', {'scope':'post-hoc full-fit Training screen; no independent validation',
          'api_calls':0, 'additional_logical_calls':0, 'baseline_logical_calls':sum(b['logical_calls'] for b in bases.values()),
          'policies':result, 'diagnosis':diagnosis, 'live_two_targets':summarize(livearms,truth,livefb),
          'live_details':live_details})
    with (args.output/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for k in bases:
            f.write(json.dumps({'key':k,'policies':{name:arms[name][k] for name in POLICIES}},ensure_ascii=False)+'\n')
    write(args.output/'stages.json',stages)
    write(args.output/'receipt.json', {'state':'PASS','rows':len(bases),'api_calls':0,'unconsumed_reviewers_used':False,
          'baseline_reconstruction_mismatches':0,'elapsed_seconds':time.perf_counter()-start,
          'source_inventory_sha256':sha(inventory_path),'baseline_predictions_sha256':sha(BASE/'predictions.jsonl'),
          'annotations_sha256':sha(TRUTH), 'script_sha256':sha(Path(__file__)),
          'protocol_sha256':sha(ROOT/'docs/SCHEME4_AMBIGUITY_AUDIT_PROTOCOL_2026-09-14.md'),
          'report_sha256':sha(args.output/'report.json'),'predictions_sha256':sha(args.output/'predictions.jsonl')})
    for name,r in result.items():
        print(name,round(r['mean_f1'],4),round(r['f1']['ivt'],4),r['errors'],r['edits'],r['passes_development_screen'])
    print(json.dumps(diagnosis['ivt_19'],ensure_ascii=False))


if __name__ == '__main__':
    with offline():
        main()
