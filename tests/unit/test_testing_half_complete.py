from copy import deepcopy
from decimal import Decimal
import json
import threading
import time

import pytest
from scripts import run_testing_half_complete as run
from scripts.score_pipeline_reports import mock_caller


@pytest.fixture
def plan():
    config, policies = run.report_config()
    return dict(report_config=config, policies=policies, max_report_dispatches=32,
                rates={s: {'prompt': '0.000001', 'completion': '0.000002'} for s in policies})


def predictions():
    labels = {'instrument': [0], 'verb': [1], 'target': [0], 'ivt': [0], 'phase': [0]}
    return [dict(key=f'VID01_{f}', video_id='VID01', frame_id=f,
                 h0=deepcopy(labels), prediction=deepcopy(labels)) for f in (100, 125)]


def scores(rows):
    return {'details': [{'key': r['key'], 'h0': r['h0'], 'pipeline': r['prediction'],
                        'gt': deepcopy(r['h0']), 'mask': {'ivt': r['frame_id'] == 100, 'phase': True}}
                       for r in rows]}


def test_full_reports_then_v2_judge_gt_isolation_and_dedup(tmp_path, plan):
    observed = []
    def caller(request):
        observed.append(deepcopy(request))
        if 'structured_prediction_hash' in request:
            assert 'ground_truth' not in request and 'ground_truth' not in request['prompt']
            assert 'VID01' not in request['prompt']
        else:
            assert (tmp_path/'generation_seal.json').exists()
            assert request['prompt_version'] == 'gsr_report_v1_judge_2'
            assert request['wire_policy']['reasoning'] == {'effort': 'low'}
            assert request['omit_temperature']
        time.sleep(.005)
        return mock_caller(request)
    stop = threading.Event()
    cache = run.ConcurrentCache(tmp_path/'cache.sqlite', caller, plan, stop, mock=True)
    try:
        run.generate_reports(predictions(), tmp_path, plan['report_config'], cache, 8, stop)
        assert len(observed) == 1  # identical H0/FULL and adjacent states share one request
        summary = run.evaluate_reports(scores(predictions()), tmp_path, plan['report_config'], cache, 8, stop)
        assert len(observed) == 2
        assert summary['total_frames'] == 2 and summary['eligible_frames'] == 1
        assert summary['status'] == 'COMPLETE' and summary['mock']
        for arm in ('H0', 'FULL'):
            values = summary['methods'][arm]
            assert values['gsr_score'] == pytest.approx(.8*values['rule_score']+.2*values['llm_judge_score'])
        assert len((tmp_path/'full_reports.jsonl').read_text().splitlines()) == 2
    finally:
        cache.close()


def test_concurrent_cache_failure_not_resent_and_success_survives_reopen(tmp_path, plan):
    count = [0]
    def fail(request):
        count[0] += 1
        time.sleep(.005)
        raise TimeoutError('unknown outcome')
    path = tmp_path/'cache.sqlite'
    cache = run.ConcurrentCache(path, fail, plan, threading.Event())
    def task(_): cache.call('report_generation', {'prompt': 'same'})
    with pytest.raises(TimeoutError):
        run.core.bounded_map(task, range(8), 8, threading.Event())
    assert count[0] == 1
    cache.close()
    reopened = run.ConcurrentCache(path, fail, plan, threading.Event())
    try:
        with pytest.raises(RuntimeError, match='no automatic retry'):
            reopened.call('report_generation', {'prompt': 'same'})
        assert count[0] == 1
        response = {'text': '{}'}
        reopened.caller = lambda _: response
        reopened.call('report_generation', {'prompt': 'different'})
    finally:
        reopened.close()
    reopened = run.ConcurrentCache(path, fail, plan, threading.Event())
    try:
        assert reopened.call('report_generation', {'prompt': 'different'})[2] is True
        assert count[0] == 1
    finally:
        reopened.close()


@pytest.mark.parametrize('failure', [False, True])
def test_report_http_wire_and_budget_unknown_outcomes(tmp_path, plan, monkeypatch, failure):
    monkeypatch.setattr(run, 'load_api_key_file', lambda _: type('Secret', (), {'reveal': lambda _: 'fake-only'})())
    wires = []
    def post(url, **kwargs):
        wires.append(kwargs['json'])
        assert kwargs['allow_redirects'] is False
        if failure:
            raise TimeoutError('uncertain')
        body = {'model': 'openai/gpt-5.6-luna', 'usage': {'cost': .001},
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
        return type('Response', (), {'status_code': 200, 'json': lambda _: body})()
    monkeypatch.setattr(run.requests, 'post', post)
    budget = run.Budget(tmp_path/'budget.sqlite', {'openrouter_usd': '1'}, 'test')
    stop = threading.Event()
    caller = run.ReportCaller(tmp_path, plan, budget, stop)
    req = dict(model_config=plan['report_config']['judge'], prompt='synthetic',
               **plan['policies']['offline_evaluation'])
    try:
        if failure:
            with pytest.raises(TimeoutError): caller(req)
            assert len(budget.pending()) == 1 and stop.is_set()
        else:
            caller(req)
            assert not budget.pending()
            assert Decimal(budget.summary()['accounts']['openrouter_usd']['occupied']) == Decimal('.001')
        assert len(wires) == 1
        assert 'temperature' not in wires[0]
        assert wires[0]['reasoning'] == {'effort': 'low'}
        assert wires[0]['max_tokens'] == 4096
        assert wires[0]['provider']['only'] == ['openai']
    finally:
        budget.close()


def test_budget_blocks_http_before_dispatch(tmp_path, plan, monkeypatch):
    monkeypatch.setattr(run, 'load_api_key_file', lambda _: type('Secret', (), {'reveal': lambda _: 'fake-only'})())
    monkeypatch.setattr(run.requests, 'post', lambda *a, **k: pytest.fail('must not dispatch'))
    budget = run.Budget(tmp_path/'budget.sqlite', {'openrouter_usd': '0'}, 'test')
    try:
        caller = run.ReportCaller(tmp_path, plan, budget, threading.Event())
        with pytest.raises(run.BudgetStop):
            caller(dict(model_config=plan['report_config']['judge'], prompt='test',
                        **plan['policies']['offline_evaluation']))
        assert budget.summary()['dispatches'] == 0
    finally:
        budget.close()


def test_unsealed_or_modified_report_cannot_be_evaluated(tmp_path, plan):
    stop = threading.Event()
    cache = run.ConcurrentCache(tmp_path/'cache.sqlite', mock_caller, plan, stop, mock=True)
    try:
        with pytest.raises(FileNotFoundError):
            run.evaluate_reports(scores(predictions()), tmp_path, plan['report_config'], cache, 8, stop)
        run.generate_reports(predictions(), tmp_path, plan['report_config'], cache, 8, stop)
        path = next((tmp_path/'generated').glob('*.json'))
        run.write(path, {'modified': True})
        with pytest.raises(ValueError, match='changed before evaluation'):
            run.evaluate_reports(scores(predictions()), tmp_path, plan['report_config'], cache, 8, stop)
    finally:
        cache.close()


def test_complete_execute_orders_all_stages_and_reuses_core(tmp_path, plan, monkeypatch):
    out, core_out = tmp_path/'complete', tmp_path/'existing_core'
    records = predictions()
    run.write(core_out/'plan.json', {})
    run.write(core_out/'receipt.json', {'state': 'PASS'})
    run.write(core_out/'predictions.json', records)
    run.write(core_out/'continuous_predictions.json', records)
    full_plan = {**plan, 'profile': run.PROFILE, 'source_sha256': {}, 'core_output': str(core_out),
                 'core_plan_sha256': run.sha(core_out/'plan.json'), 'report_limits': {'openrouter_usd': '1'}}
    run.write(out/'plan.json', full_plan)
    run.write(out/'prepared.json', {'plan_sha256': run.sha(out/'plan.json')})
    monkeypatch.setattr(run, 'verify_core', lambda *a, **k: {})
    monkeypatch.setattr(run.core, 'execute', lambda *a: pytest.fail('completed core must not rerun'))
    def score(core_dir):
        assert (out/'reports/generation_seal.json').exists()
        run.write(core_dir/'scores_detail.json', scores(records))
    monkeypatch.setattr(run.core, 'score', score)
    monkeypatch.setattr(run, 'ReportCaller', lambda *args: mock_caller)
    run.execute(out, 8, True)
    receipt = run.read(out/'receipt.json')
    assert receipt['state'] == 'PASS' and receipt['eligible_gsr_frames'] == 1
    assert (out/'reports/results/summary.json').exists()
    with pytest.raises(FileExistsError): run.execute(out, 8, True)


def test_paid_flag_is_required_before_any_work(tmp_path):
    with pytest.raises(ValueError, match='allow-paid'): run.execute(tmp_path, 8, False)
    with pytest.raises(ValueError, match='workers'): run.execute(tmp_path, 9, True)


def test_complete_prepare_reuses_failed_local_directory(tmp_path, plan, monkeypatch):
    out = tmp_path/'complete'
    (out/'core/priors').mkdir(parents=True)
    called = []
    def prepare(core_out, scope, limits, *, resume=False, streaming=False):
        assert streaming
        called.append((core_out, resume))
        run.write(core_out/'plan.json', {'runtime_sha256': {}})
    monkeypatch.setattr(run.core, 'prepare', prepare)
    monkeypatch.setattr(run, 'verify_core', lambda _: {'runtime_sha256': {}})
    monkeypatch.setattr(run, 'endpoint_rates', lambda *a: (plan['rates'], {}))
    monkeypatch.setattr(run, 'load_api_key_file', lambda _: type('Secret', (), {'reveal': lambda _: 'fake-only'})())
    with pytest.raises(ValueError, match='resume-prepare'):
        run.prepare(out, None, '30', tmp_path/'limits.json')
    run.prepare(out, None, '30', tmp_path/'limits.json', resume=True)
    assert called == [(out/'core', True)]
    assert run.read(out/'prepared.json')['api_posts'] == 0
    with pytest.raises(ValueError, match='sealed/unknown'):
        run.prepare(out, None, '30', tmp_path/'limits.json', resume=True)
