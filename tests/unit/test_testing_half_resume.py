from decimal import Decimal

import pytest
from scripts import testing_half_resume as resume
from scripts import run_testing_half_complete as complete


def fixture_run(tmp_path, monkeypatch, terminal):
    core = tmp_path/'core'
    complete.write(core/'plan.json', {'runtime_sha256': {}})
    complete.write(tmp_path/'plan.json', {'source_sha256': {}, 'core_output': str(core), 'report_arms': ['H0', 'FULL']})
    for path in (tmp_path/'failure.json', core/'failure.json', tmp_path/'execution.lock', core/'execution_started.json'):
        complete.write(path, {})
    complete.write(core/'results/VID01_100.json', {'prediction': 'preserved'})
    budget = complete.Budget(core/'budget.sqlite', {'openrouter_usd': '5'}, resume.sha(core/'plan.json'))
    key = budget.key('VID01_100', 'proposal', 'base')
    budget.reserve(key, 'openrouter_usd', Decimal('1'))
    if terminal:
        budget.settle(key, Decimal('.1'))
    budget.close()
    if terminal:
        complete.write(core/'targets/VID01_100/run/calls/000_VID01_100_proposal_base/record.json',
                       {'target': 'VID01_100', 'stage': 'proposal', 'seat': 'base', 'status': 'JSON_PARSED', 'finished_utc': 'done'})
    monkeypatch.setattr(complete, 'verify_core', lambda *a, **kw: {})
    return core


def test_resume_preserves_plan_budget_and_results_archives_only_control_files(tmp_path, monkeypatch):
    core = fixture_run(tmp_path, monkeypatch, True)
    original = {p: p.read_bytes() for p in (tmp_path/'plan.json', core/'plan.json', core/'results/VID01_100.json')}
    check = resume.prepare_resume(tmp_path, apply=False)
    assert check['preserved_requests'] == 1 and (tmp_path/'failure.json').exists()
    result = resume.prepare_resume(tmp_path)
    assert result['api_calls'] == 0 and result['preserved_targets'] == 1
    assert all(p.read_bytes() == data for p, data in original.items())
    assert not (tmp_path/'execution.lock').exists() and not (core/'execution_started.json').exists()
    resume.validate_runtime(core, resume.read(core/'plan.json'), 'runtime_sha256')
    budget = complete.Budget(core/'budget.sqlite', {'openrouter_usd': '5'}, resume.sha(core/'plan.json'))
    assert budget.summary()['dispatches'] == 1 and budget.summary()['accounts']['openrouter_usd']['occupied'] == '0.1'
    budget.close()


def test_resume_rejects_unresolved_calls_without_removing_stop_markers(tmp_path, monkeypatch):
    fixture_run(tmp_path, monkeypatch, False)
    with pytest.raises(ValueError, match='unresolved'):
        resume.prepare_resume(tmp_path)
    assert (tmp_path/'failure.json').exists() and (tmp_path/'execution.lock').exists()
