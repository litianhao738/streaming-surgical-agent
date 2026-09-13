"""Read-only completion checks and descriptive diagnostics; no API transport."""
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.trial_tracker_schemes56 import DEFAULT, verify, study
from surgical_agent.research.verification.prior_panel import labels


def main():
    plan = verify(DEFAULT)
    records = []
    for p in DEFAULT.glob('targets/*/run/budget.json'):
        records.extend(study.read(p)['calls'])
    records.extend(study.read(p) for p in DEFAULT.glob('targets/*/changed/*/record.json'))
    identities = [(r['target'], r['stage'], r['seat']) for r in records]
    assert len(records) == len(set(identities)) == 208
    assert all(r.get('finished_utc') and r.get('http_status') is not None for r in records)
    by_seat = Counter(r['seat'] for r in records)
    assert by_seat == {'base': 32, 'qwen': 48, 'gpt': 32, 'gemini': 32, 'grok': 32, 'deepseek': 32}
    db = sqlite3.connect(f'file:{(DEFAULT / "budget.sqlite").as_posix()}?mode=ro', uri=True)
    states = dict(db.execute('select state,count(*) from calls group by state'))
    assert states == {'TERMINAL': 208}
    db.close()
    reused = 0
    for s in plan['selection']:
        c = study.read(DEFAULT / 'targets' / (s['key'] + '__control') / 'responses.json')
        q = study.read(DEFAULT / 'targets' / (s['key'] + '__qwen') / 'responses.json')
        assert c['pool'] == q['pool'] and c['proposal'] == q['proposal']
        assert all(c['reviews'][seat] == q['reviews'][seat] for seat in ('gpt', 'gemini', 'grok', 'deepseek'))
        reused += 1
    pred = study.read(DEFAULT / 'predictions.json')
    assert len(pred) == 16
    for row in pred:
        for a in row['arms'].values():
            assert labels(a['prediction']) == a['prediction']
    changes = {}
    for arm in ('prior', 'qwen'):
        changes[arm] = []
        for row in pred:
            c, t = row['arms']['control'], row['arms'][arm]
            edits = {head: {'removed': sorted(set(c['prediction'][head]) - set(t['prediction'][head])),
                            'added': sorted(set(t['prediction'][head]) - set(c['prediction'][head]))}
                     for head in c['prediction'] if c['prediction'][head] != t['prediction'][head]}
            if edits or c['gate_action'] != t['gate_action'] or c['call_keys'] != t['call_keys']:
                changes[arm].append({'key': row['key'], 'edits': edits,
                                     'gate_actions': [c['gate_action'], t['gate_action']],
                                     'logical_calls': [len(c['call_keys']), len(t['call_keys'])]})
    statuses, refusals, failures = {}, [], []
    for r in records:
        arm = r['target'].rsplit('__', 1)[1]
        key = arm + '/' + r['seat']
        statuses.setdefault(key, Counter())[r['status']] += 1
        err = r.get('provider_error') or {}
        if str(err.get('code')) == '1301' or err.get('code') == 'data_inspection_failed':
            refusals.append({'target': r['target'], 'seat': r['seat'], 'code': err.get('code')})
        if r['status'] != 'JSON_PARSED':
            failures.append({k: r.get(k) for k in ('target', 'seat', 'status', 'http_status', 'error_type', 'charge_kind')})
    starts = [r['started_utc'] for r in records]
    ends = [r['finished_utc'] for r in records]
    five_end = max(r['finished_utc'] for r in records if not r['target'].endswith('__qwen'))
    six_start = min(r['started_utc'] for r in records if r['target'].endswith('__qwen'))
    assert five_end <= six_start
    report = {'api_calls_from_this_audit': 0, 'verified_frozen_plan': True,
              'physical_dispatches': 208, 'unique_dispatches': 208, 'terminal_dispatches': 208,
              'requests_by_seat': dict(by_seat), 'scheme6_shared_response_parity_rows': reused,
              'started_utc': min(starts), 'finished_utc': max(ends), 'scheme5_finished_before_scheme6': True,
              'statuses_by_arm_and_seat': statuses, 'refusals': refusals, 'non_json_parsed': failures,
              'prediction_and_route_changes': changes,
              'note': 'Descriptive checks after collection; no new primary acceptance criteria or provider requests.'}
    study.write(DEFAULT / 'completion_audit.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
