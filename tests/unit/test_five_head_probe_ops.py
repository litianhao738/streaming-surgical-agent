"""Status and hard-interruption recovery for a five_head_probe collection run (synthetic run directory)."""
from decimal import Decimal
import json
import sqlite3

import pytest

from scripts import five_head_probe_collection_status as status
from scripts import recover_five_head_probe_interrupted as recover
from surgical_agent.research.gate.collection_budget import Budget

LIMITS = {'openrouter_usd': '60', 'aliyun_cny': '30', 'glm_requests': '10', 'deepseek_requests': '10'}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def finished(seat, account, charge, status_='JSON_PARSED'):
    return {'stage': 'five_head_v1', 'seat': seat, 'account': account, 'charge': charge, 'status': status_,
            'started_utc': 't0', 'finished_utc': 't1'}


def dispatched(seat, account, reserve):
    return {'stage': 'five_head_v1', 'seat': seat, 'account': account, 'charge': reserve, 'status': 'DISPATCHED',
            'started_utc': 't0'}


def changed_seat(run, target, record, response=True):
    folder = run / 'targets' / target / 'changed' / f"five_head_v1_{record['seat']}"
    write(folder / 'request.json', {'model': 'm'})
    write(folder / 'record.json', dict(record, target=target))
    if response:
        write(folder / 'response.json', {'choices': []})


def run_seat(run, target, rows):
    """gpt/gemini seats: per-target ledger plus indexed evidence folders."""
    ledger = [dict(r, target=target, index=i) for i, r in enumerate(rows)]
    write(run / 'targets' / target / 'run' / 'budget.json', {'calls': ledger, 'occupied': {}, 'limits': {}, 'stopped': False})
    for r in ledger:
        folder = run / 'targets' / target / 'run' / 'calls' / f"{r['index']:03d}_{target}_{r['stage']}_{r['seat']}"
        write(folder / 'request.json', {})
        write(folder / 'record.json', r)
        if r['status'] == 'JSON_PARSED':
            write(folder / 'response.json', {})


@pytest.fixture
def run(tmp_path):
    run = tmp_path / 'run'
    write(run / 'plan.json', {'profile': 'five_head_probe_training_collection_v2', 'review_mode': 'five_head_probe',
                              'workers': 8, 'limits': LIMITS, 'selection': [{'key': k} for k in ('A', 'B', 'C', 'D', 'R', 'E')],
                              'reused_pilot': {'directory': 'pilot'}})
    budget = Budget(run / 'budget.sqlite', LIMITS, 'plan-hash')
    # A: complete fresh target (qwen + gpt settled).
    budget.reserve(Budget.key('A', 'five_head_v1', 'qwen'), 'aliyun_cny', Decimal('0.01'))
    budget.settle(Budget.key('A', 'five_head_v1', 'qwen'), Decimal('0.002'))
    budget.reserve(Budget.key('A', 'five_head_v1', 'gpt'), 'openrouter_usd', Decimal('0.01'))
    budget.settle(Budget.key('A', 'five_head_v1', 'gpt'), Decimal('0.003'))
    changed_seat(run, 'A', finished('qwen', 'aliyun_cny', '0.002'))
    run_seat(run, 'A', [finished('gpt', 'openrouter_usd', '0.003')])
    write(run / 'results' / 'A.json', {'sample_id': 'A', 'joint_depth': 2,
                                       'request_timings': [{'seat': 'qwen', 'response_present': True}, {'seat': 'gpt', 'response_present': False}]})
    write(run / 'progress' / 'A.json', {'sample_id': 'A', 'calls': []})
    # R: reused pilot row.
    write(run / 'results' / 'R.json', {'sample_id': 'R', 'joint_depth': 3, 'reused_from': 'pilot', 'request_timings': []})
    # B: soft stop; qwen finished, no result yet (replays for free).
    budget.reserve(Budget.key('B', 'five_head_v1', 'qwen'), 'aliyun_cny', Decimal('0.01'))
    budget.settle(Budget.key('B', 'five_head_v1', 'qwen'), Decimal('0.002'))
    changed_seat(run, 'B', finished('qwen', 'aliyun_cny', '0.002'))
    write(run / 'progress' / 'B.json', {'sample_id': 'B', 'calls': [{'seat': 'qwen', 'seconds': 1, 'response_present': True}]})
    write(run / 'errors' / 'B.json', {'error_type': 'BudgetStop', 'error': 'another worker stopped the run'})
    # C: hard kill while gpt was in flight (qwen settled, gpt reserved and DISPATCHED).
    budget.reserve(Budget.key('C', 'five_head_v1', 'qwen'), 'aliyun_cny', Decimal('0.01'))
    budget.settle(Budget.key('C', 'five_head_v1', 'qwen'), Decimal('0.002'))
    budget.reserve(Budget.key('C', 'five_head_v1', 'gpt'), 'openrouter_usd', Decimal('0.01'))
    changed_seat(run, 'C', finished('qwen', 'aliyun_cny', '0.002'))
    run_seat(run, 'C', [dispatched('gpt', 'openrouter_usd', '0.01')])
    write(run / 'progress' / 'C.json', {'sample_id': 'C', 'calls': [{'seat': 'qwen', 'seconds': 1, 'response_present': True}]})
    # D: hard kill while qwen was in flight (reserved, DISPATCHED, no response).
    budget.reserve(Budget.key('D', 'five_head_v1', 'qwen'), 'aliyun_cny', Decimal('0.01'))
    changed_seat(run, 'D', dispatched('qwen', 'aliyun_cny', '0.01'), response=False)
    # E: never started (no journal at all).
    budget.close()
    write(run / 'failure.json', {'error_type': 'KeyboardInterrupt', 'error': ''})
    return run


def occupied(run):
    db = sqlite3.connect(str(run / 'budget.sqlite'))
    try:
        return dict(db.execute('SELECT name,occupied FROM accounts').fetchall()), \
            db.execute('SELECT key,state FROM calls ORDER BY key').fetchall()
    finally:
        db.close()


def test_status_classifies_results_journals_and_reservations(run):
    s = status.scan(run)
    assert s['results'] == {'done': 2, 'reused_pilot': 1, 'fresh': 1, 'remaining': 4,
                            'joint_depths': {2: 1}, 'missing_responses': 1}
    assert s['budget']['states'] == {'TERMINAL': 4, 'RESERVED': 2}
    assert {p['target'] for p in s['budget']['pending']} == {'C', 'D'}
    j = s['journaled_without_result']
    assert j['replayable_on_resume'] == 1
    assert [x['target'] for x in j['uncertain_needs_recover']] == ['C', 'D']
    assert any('DISPATCHED' in r for r in j['uncertain_needs_recover'][0]['reasons'])
    assert s['errors'] == {'BudgetStop: another worker stopped the run': 1}
    assert s['failure']['error_type'] == 'KeyboardInterrupt'
    assert s['needs_recover'] is True
    text = status.render(s)
    assert 'UNCERTAIN C' in text and 'UNCERTAIN D' in text and 'run recover_five_head_probe_interrupted.py' in text


def test_status_handles_missing_and_complete_runs(tmp_path, run):
    assert status.scan(tmp_path / 'nowhere') == {'run': str(tmp_path / 'nowhere'), 'exists': False}
    write(run / 'receipt.json', {'state': 'PASS', 'rows': 6})
    assert status.scan(run)['complete'] is True
    assert 'COMPLETE' in status.render(status.scan(run))


def test_recovery_dry_run_changes_nothing(run):
    before = occupied(run)
    report = recover.plan_recovery(run)
    assert sorted(report['targets']) == ['C', 'D']
    assert report['deleted_call_rows'] == 3
    assert report['abandoned_charges_kept_as_occupied'] == {'aliyun_cny': '0.012', 'openrouter_usd': '0.01'}
    assert 'budget reservation without settled outcome' in report['targets']['C']['reasons']
    assert occupied(run) == before
    assert (run / 'targets' / 'C').exists() and (run / 'failure.json').exists()


def test_recovery_apply_quarantines_uncertain_targets_and_keeps_occupied(run):
    before_accounts, before_calls = occupied(run)
    record = recover.apply_recovery(run, recover.plan_recovery(run))
    after_accounts, after_calls = occupied(run)
    assert after_accounts == before_accounts
    assert [k for k, _ in after_calls] == [Budget.key('A', 'five_head_v1', 'gpt'), Budget.key('A', 'five_head_v1', 'qwen'),
                                           Budget.key('B', 'five_head_v1', 'qwen')]
    assert all(state == 'TERMINAL' for _, state in after_calls)
    quarantine = run / 'quarantine' / record['stamp']
    assert not (run / 'targets' / 'C').exists() and (quarantine / 'targets' / 'C' / 'run' / 'budget.json').exists()
    assert not (run / 'targets' / 'D').exists() and (quarantine / 'targets' / 'D').exists()
    assert (run / 'targets' / 'A').exists() and (run / 'targets' / 'B').exists()
    assert not (run / 'progress' / 'C.json').exists() and (run / 'progress' / 'B.json').exists()
    assert not (run / 'failure.json').exists() and (quarantine / 'failure.json').exists()
    assert not (run / 'errors' / 'B.json').exists() and (quarantine / 'errors_previous_attempt' / 'B.json').exists()
    saved = json.loads((run / 'recovery' / (record['stamp'] + '.json')).read_text('utf-8'))
    assert saved['deleted_call_rows'] == 3 and saved['occupied_before'] == saved['occupied_after']
    after = status.scan(run)
    assert after['needs_recover'] is False and after['budget']['pending'] == []
    assert after['journaled_without_result']['uncertain_needs_recover'] == []
    assert after['recoveries'] == [record['stamp'] + '.json']
    # Re-reserving the abandoned identities must now be possible.
    budget = Budget(run / 'budget.sqlite', LIMITS, 'plan-hash')
    budget.reserve(Budget.key('C', 'five_head_v1', 'gpt'), 'openrouter_usd', Decimal('0.01'))
    budget.close()


def test_recovery_refuses_reservation_for_completed_target(run):
    budget = Budget(run / 'budget.sqlite', LIMITS, 'plan-hash')
    budget.reserve(Budget.key('A', 'five_head_v1', 'gemini'), 'openrouter_usd', Decimal('0.01'))
    budget.close()
    with pytest.raises(RuntimeError, match='already have results'):
        recover.plan_recovery(run)


def test_recovery_reports_nothing_after_soft_stop(run):
    recover.apply_recovery(run, recover.plan_recovery(run))
    report = recover.plan_recovery(run)
    assert report['targets'] == {} and report['deleted_call_rows'] == 0


def test_recovery_skips_complete_run(run):
    write(run / 'receipt.json', {'state': 'PASS'})
    assert recover.plan_recovery(run) == {'run': str(run), 'complete': True, 'targets': {}}
