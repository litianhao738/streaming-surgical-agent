"""Independent arithmetic/transport audit; offline only after scored closure."""
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'artifacts/preflight/verb_prompt_20260909_v1'
HEADS = ('instrument', 'verb', 'target', 'ivt', 'phase')
ARMS = ('control', 'verb_prompt')


def read(name):
    return json.loads((OUTPUT / name).read_text(encoding='utf8'))


def main():
    report, done, plan = (read(n) for n in ('metrics.json', 'completion.json', 'plan.json'))
    assert done['closed_utc'] and not done['fatal_error']
    assert report['raw_replay_passed']
    ledger = read('budget.json')['calls']
    rows = read('predictions.json')
    truths = {(t['video_id'], t['frame_id']): t for t in read('scored_truth.json')}
    initials = {i['key']: i for i in read('initial_state.json')}
    assert len(rows) == len(truths) == len(initials) == 24
    frozen = {p.relative_to(OUTPUT).as_posix() for folder in ('targets', 'calls')
              for p in (OUTPUT / folder).rglob('*.json')}
    frozen |= {'plan.json', 'execution.lock', 'predictions.json', 'budget.json'}
    assert frozen == set(done['hashes'])
    for name, digest in done['hashes'].items():
        assert hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() == digest
    transport = Counter()
    for call in ledger:
        folder = f"calls/{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
        assert read(folder + '/record.json') == call
        response = read(folder + '/response.json')
        transport[str(response['http_status'])] += 1
        if call['status'] in ('JSON_PARSED', 'JSON_PARSED_FENCE_NORMALIZED'):
            body = response['body']
            assert response['http_status'] == 200
            assert body['model'] == plan['models'][call['seat']]
            if call['seat'] in plan['providers']:
                assert body['provider'] == plan['providers'][call['seat']]
            assert body['choices'][0]['finish_reason'] == 'stop'
            assert not body['choices'][0]['message'].get('refusal')
    counts = {}
    for arm in ('h0', *ARMS):
        counts[arm] = {}
        all_exact = 0
        for row in rows:
            t = truths[row['video_id'], row['frame_id']]
            all_exact += all(set(row[arm][h]) == set(t['gt'][h]) for h in HEADS if t['mask'][h])
        assert all_exact == report['metrics'][arm]['all_valid_heads_exact']['exact_matches']
        for h in HEADS:
            sets = [(set(row[arm][h]), set(truths[row['video_id'], row['frame_id']]['gt'][h]))
                    for row in rows if truths[row['video_id'], row['frame_id']]['mask'][h]]
            tp, fp, fn = (sum(len(op(a, g)) for a, g in sets) for op in
                          (lambda a, g: a & g, lambda a, g: a-g, lambda a, g: g-a))
            v = {'tp': tp, 'fp': fp, 'fn': fn, 'valid_targets': len(sets),
                 'exact_matches': sum(a == g for a, g in sets), 'micro_precision': tp/(tp+fp),
                 'micro_recall': tp/(tp+fn), 'micro_f1': 2*tp/(2*tp+fp+fn),
                 'exact_set_accuracy': sum(a == g for a, g in sets)/len(sets)}
            assert all(math.isclose(x, report['metrics'][arm]['tasks'][h][k], abs_tol=1e-12) for k, x in v.items())
            counts[arm][h] = v
    changes, validity, invalid_reasons = [], {a: Counter() for a in ARMS}, {a: Counter() for a in ARMS}
    for row in rows:
        gt = truths[row['video_id'], row['frame_id']]
        if not initials[row['key']]['pool']['propositions']:
            continue
        records = {a: read('targets/' + row['key'] + '/' + a + '.json') for a in ARMS}
        for a in ARMS:
            for pid, d in records[a]['diagnostics'].items():
                task = next(p['task'] for p in initials[row['key']]['pool']['propositions'] if p['id'] == pid)
                validity[a][task + '_valid_propositions'] += not bool(d['invalid'])
                validity[a][task + '_invalid_propositions'] += bool(d['invalid'])
                for reason in d['invalid'].values():
                    invalid_reasons[a][str(reason)] += 1
        for h in HEADS:
            if row['control'][h] == row['verb_prompt'][h]:
                continue
            changed_ids = set(row['control'][h]) ^ set(row['verb_prompt'][h])
            props = [p for p in initials[row['key']]['pool']['propositions'] if p['task'] == h and p['label_id'] in changed_ids]
            changes.append({'key': row['key'], 'task': h, 'before': row['control'][h],
                            'after': row['verb_prompt'][h], 'gt': gt['gt'][h], 'mask': gt['mask'][h],
                            'judgments': {p['id']: {a: {'mean': records[a]['means'][p['id']],
                             'diagnostics': records[a]['diagnostics'][p['id']],
                             'reviews': {s: r['judgments'].get(p['id']) for s, r in records[a]['reviews'].items()}}
                             for a in ARMS} for p in props}})
    amounts = {a: str(sum(Decimal(c['charge']) for c in ledger if c['account'] == a)) for a in plan['limits']}
    assert all(Decimal(v) == Decimal(read('budget.json')['occupied'][a]) for a, v in amounts.items())
    out = {'all_15_metric_tables_verified': True, 'all_closed_hashes_and_calls_verified': True,
           'transport_http': dict(transport), 'counts': counts, 'costs_occupied': amounts,
           'validity': {a: dict(v) for a, v in validity.items()},
           'invalid_reason_counts': {a: dict(v) for a, v in invalid_reasons.items()},
           'changed_head_rows': changes}
    (OUTPUT / 'independent_diagnostics.json').write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    print(json.dumps({'all_checks_passed': True, 'changed_head_rows': len(changes), 'http': dict(transport)}))


if __name__ == '__main__':
    main()
