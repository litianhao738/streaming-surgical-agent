"""Offline comparison; preserve collection and original assessment artifacts."""
import sys
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import assess_tracker_scheme4 as s
from scripts.trial_tracker_schemes56 import DEFAULT, verify


def main():
    plan = verify(DEFAULT)
    rows = {r['sample_id']: r for r in s.read(s.ROWS)}
    ledger = s.read(s.BASE / 'official_behavior_net_v2_20260913/costs.json')['per_call_main_attempt']
    def cost(keys):
        out = Counter(logical_calls=len(keys))
        for key in keys:
            row = ledger[key]; account = row['account']
            out[account] += 1 if row['requests_only'] else float(row['original_currency_total']) / row['calls']
        return out
    with np.load(s.NPZ, allow_pickle=True) as z:
        ids = z['ids'].tolist(); index = {k: i for i,k in enumerate(ids)}
        route = z['qwen_hgb_route'].astype(bool)
        baseline = np.where(route[:, None, None], z['ambiguity'], z['cheap'])
        depths = z['depths'].copy()
    order = ('qwen', 'gpt', 'gemini', 'grok', 'deepseek')
    base_costs = []
    for i in range(len(ids)):
        keys = ['h0|base', 'proposal|base', 'control_graph|qwen']
        if route[i]:
            keys += ['control_graph|' + a for a in order[:depths[i,0]] if a != 'qwen']
            keys += ['phase_recommendation|base'] + ['joint_r1|' + a for a in order[:depths[i,1]]]
        base_costs.append(cost(keys))
    selected = [x['key'] for x in plan['selection']]
    ix = [index[k] for k in selected]
    def summed(costs):
        out = Counter()
        for c in costs: out.update(c)
        return dict(out)
    result = {'api_calls': 0, 'full_6059': {'original_no_tracker': {**s.quality(baseline), 'estimated_accounts': summed(base_costs)}},
              'same_16_historical_reference': {'original_no_tracker': {**s.quality(baseline[ix]), 'estimated_accounts': summed([base_costs[i] for i in ix])}}}
    all_preds = [json.loads(line) for line in (s.DEFAULT / 'predictions.jsonl').read_text().splitlines()]
    for variant, route_name in [('old_gate_outer','old_gate_route'), ('v2_outer','v2_outer_route')]:
        cc, costs, subc, subcosts = [], [], [], []
        for r in all_preds:
            count = s.counts(r['predictions'][variant], rows[r['key']]['gt'])
            c = cost(r['review_call_keys'] if r[route_name] else ['h0|base','proposal|base','control_graph|qwen'])
            cc.append(count); costs.append(c)
            if r['key'] in selected: subc.append(count); subcosts.append(c)
        result['full_6059']['scheme4_' + variant] = {**s.quality(np.array(cc)), 'estimated_accounts': summed(costs)}
        result['same_16_historical_reference']['scheme4_' + variant] = {**s.quality(np.array(subc)), 'estimated_accounts': summed(subcosts)}
    predictions = s.read(DEFAULT / 'predictions.json')
    control_raw, control_smoothed = [], []
    for r in predictions:
        a = r['arms']['control']; p = dict(a['before_output_modules'])
        control_raw.append(s.counts(p, rows[r['key']]['gt']))
        p['phase'] = a['prediction']['phase']
        control_smoothed.append(s.counts(p, rows[r['key']]['gt']))
    q = s.read(DEFAULT / 'quality_report.json')
    result['same_16_fresh_paired'] = {'no_tracker_interaction_raw_phase': s.quality(np.array(control_raw)),
        'no_tracker_interaction_same_phase_filter': s.quality(np.array(control_smoothed)),
        'scheme4_old_gate': q['arms']['control'], 'scheme5_on4': q['arms']['prior'], 'scheme6_on4': q['arms']['qwen']}
    result['same_16_fresh_paired']['no_tracker_interaction_same_phase_filter']['cost_accounts_per_inference_path'] = q['arms']['control']['cost_accounts_per_inference_path']
    records = []
    for p in DEFAULT.glob('targets/*/run/budget.json'): records.extend(s.read(p)['calls'])
    records.extend(s.read(p) for p in DEFAULT.glob('targets/*/changed/*/record.json'))
    timing = {}
    for label, rr in [('scheme5_including_fresh_control', [r for r in records if not r['target'].endswith('__qwen')]),
                      ('scheme6_incremental', [r for r in records if r['target'].endswith('__qwen')]), ('total', records)]:
        first = min(datetime.fromisoformat(r['started_utc']) for r in rr)
        last = max(datetime.fromisoformat(r['finished_utc']) for r in rr)
        occupied = Counter()
        for r in rr: occupied[r['account']] += float(r['charge'])
        timing[label] = {'wall_seconds_first_dispatch_to_last_response': (last-first).total_seconds(),
                         'physical_requests': len(rr), 'accounts': dict(occupied)}
    timing['scheme4_offline_replay_and_fit'] = {'wall_seconds': s.read(s.DEFAULT/'report.json')['elapsed_seconds'], 'new_api_calls':0, 'new_api_cost':0}
    result['experiment_time_and_cost'] = timing
    result['limitations'] = ['No matched live full-original baseline was collected. Historical baseline differs in response realization and Gate calibration.',
        'Fresh no-Tracker paired control retains interaction-only review and the same phase filter; it isolates M1/M2, not the entire old pipeline.',
        'Inference account estimates exclude Tracker GPU time, power and local compute. End-to-end online latency was not measured.',
        'Scheme5/6 are additions on scheme4 postprocessing, not standalone replacements; 16 reused Training frames are not independent validation.']
    out = DEFAULT / 'baseline_comparison.json'
    s.write(out, result)
    print(json.dumps({k: v for k,v in result.items() if k != 'same_16_fresh_paired'},ensure_ascii=False))
    print(json.dumps({k:{x:v[x] for x in ('f1','errors')} for k,v in result['same_16_fresh_paired'].items()},ensure_ascii=False))


if __name__ == '__main__': main()
