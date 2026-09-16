import json
import sqlite3

import pytest

from scripts.retry_testing_half_reports import reusable_rows
from scripts import retry_testing_half_reports as recovery


def make_source(tmp_path, pending=False):
    reports = tmp_path / 'reports'
    (reports / 'results').mkdir(parents=True)
    for arm in ('h0', 'full'):
        (reports / 'results' / f'{arm}_reports.jsonl').write_text(
            json.dumps({'judge_status': 'invalid_response', 'judge_cache_key': 'bad_judge'}) + '\n',
            encoding='utf-8')
    with sqlite3.connect(reports / 'cache.sqlite') as db:
        db.execute('CREATE TABLE calls (key TEXT PRIMARY KEY, stage TEXT, request TEXT, status TEXT, response TEXT, seconds REAL, mock INTEGER)')
        rows = [('good_report', 'report_generation', 'COMPLETE', '{"report":"A valid report"}'),
                ('empty_report', 'report_generation', 'COMPLETE', ''),
                ('limited', 'report_generation', 'TERMINAL_HTTP_FAILURE', ''),
                ('good_judge', 'offline_evaluation', 'COMPLETE', '{}'),
                ('bad_judge', 'offline_evaluation', 'COMPLETE', '{}')]
        if pending:
            rows.append(('pending', 'report_generation', 'PENDING', ''))
        db.executemany('INSERT INTO calls VALUES (?,?,?,?,?,?,?)',
                       [(key, stage, '{}', status, json.dumps({'text': text}), 1, 0)
                        for key, stage, status, text in rows])
    return tmp_path


def test_reuse_success_only_without_mutating_original(tmp_path):
    source = make_source(tmp_path)
    before = (source / 'reports/cache.sqlite').read_bytes()
    keep, retry = reusable_rows(source)
    assert {r[0] for r in keep} == {'good_report', 'good_judge'}
    assert {r[0] for r in retry} == {'empty_report', 'limited', 'bad_judge'}
    assert (source / 'reports/cache.sqlite').read_bytes() == before


def test_unresolved_dispatch_is_not_automatically_resent(tmp_path):
    with pytest.raises(ValueError, match='Unresolved dispatch'):
        reusable_rows(make_source(tmp_path, pending=True))


def test_gate_default_drift_allowed_but_report_code_drift_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    gate = tmp_path / 'DEFAULT_PGP_GATE_VERSION.json'
    report = tmp_path / 'report.py'
    gate.write_text('old gate')
    report.write_text('frozen report')
    plan = {'source_sha256': {p.name: recovery.workflow.sha(p) for p in (gate, report)}}
    gate.write_text('new phase gate')
    audit = recovery.validate_report_runtime(tmp_path, plan)
    assert audit[gate.name]['current'] != audit[gate.name]['frozen']
    report.write_text('changed report')
    with pytest.raises(ValueError, match='Report recovery runtime changed'):
        recovery.validate_report_runtime(tmp_path, plan)


def test_core_verification_uses_seals_and_rejects_changed_predictions(tmp_path):
    write = recovery.write_artifact
    sha = recovery.workflow.sha
    write(tmp_path / 'plan.json', {'scope': 'historical scope'})
    plan_hash = sha(tmp_path / 'plan.json')
    write(tmp_path / 'prepared.json', {'plan_sha256': plan_hash})
    receipt = {'state': 'PASS', 'plan_sha256': plan_hash}
    for name in ('predictions', 'continuous_predictions'):
        write(tmp_path / (name + '.json'), [])
        receipt[name + '_sha256'] = sha(tmp_path / (name + '.json'))
    write(tmp_path / 'receipt.json', receipt)
    write(tmp_path / 'scores_detail.json', {})
    original = {'core_plan_sha256': plan_hash}
    origin_receipt = {'core_receipt_sha256': sha(tmp_path / 'receipt.json'),
                      'scores_sha256': sha(tmp_path / 'scores_detail.json')}
    recovery.verify_sealed_core(tmp_path, original, origin_receipt)
    write(tmp_path / 'predictions.json', ['changed'])
    with pytest.raises(ValueError, match='Core predictions changed'):
        recovery.verify_sealed_core(tmp_path, original, origin_receipt)
