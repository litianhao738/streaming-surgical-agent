import threading
from decimal import Decimal

import pytest
from scripts import testing_half_transport as http
from scripts import run_testing_half_complete as complete


@pytest.mark.parametrize('status,error,expected', [
    (400, {'code': 'invalid_parameter_error', 'message': 'URL invalid'}, False),
    (400, {'code': 'data_inspection_failed'}, False),
    (429, {'message': 'rate limit'}, False),
    (401, {'message': 'invalid api key'}, False),
    (402, {}, True), (400, {'code': 'Arrearage'}, True),
    (400, {'code': '1113'}, True), (403, {'message': 'insufficient balance'}, True),
])
def test_billing_classification(status, error, expected):
    assert http.billing_failure(status, error) is expected


@pytest.mark.parametrize('billing', [False, True])
def test_real_changed_transport_keeps_400_record_and_continues_next_target(tmp_path, monkeypatch, billing):
    t = http.transport
    monkeypatch.setattr(t.routes, 'route_body', lambda seat, body, config: body)
    monkeypatch.setattr(t.routes, 'credentials', lambda *args: ('https://unit.test', 'fake'))
    posted = []
    error = {'code': 'Arrearage' if billing else 'invalid_parameter_error', 'message': 'failed'}
    def post(*args, **kwargs):
        posted.append(kwargs)
        return type('Response', (), {'status_code': 400, 'ok': False, 'json': lambda _: {'error': error}})()
    monkeypatch.setattr(t.requests, 'post', post)
    budget = complete.Budget(tmp_path/'budget.sqlite', {'glm_requests': '10'}, 'test')
    stop = threading.Event()
    plan = {'reviewer_config': {}, 'endpoints': {'grok': 'https://unit.test'}}
    body = {'model': 'unit-model', 'messages': []}
    try:
        caller = http.GuardedCalls(tmp_path, plan, {'key': 'VID01_100'}, budget, stop)
        if billing:
            with pytest.raises(complete.BudgetStop, match='balance'):
                caller.call('VID01_100', 'control_graph', 'grok', body)
            assert stop.is_set()
        else:
            assert caller.call('VID01_100', 'control_graph', 'grok', body) is None
            assert not stop.is_set()
            again = http.GuardedCalls(tmp_path, plan, {'key': 'VID01_100'}, budget, stop)
            assert again.call('VID01_100', 'control_graph', 'grok', body) is None
            assert len(posted) == 1  # terminal error is replayed, not resent
            other = http.GuardedCalls(tmp_path, plan, {'key': 'VID01_125'}, budget, stop)
            assert other.call('VID01_125', 'control_graph', 'grok', body) is None
            assert len(posted) == 2 and not stop.is_set()
        assert not budget.pending()
    finally:
        budget.close()


def test_classifier_does_not_clear_another_workers_stop():
    shared = threading.Event()
    proxy = http.ClassifiedStop(shared)
    proxy.set()
    assert not shared.is_set()
    shared.set()
    assert proxy.is_set()


def test_report_http_400_continues_and_missing_judge_is_not_a_score(tmp_path, monkeypatch):
    config, policies = complete.report_config()
    plan = dict(report_config=config, policies=policies, max_report_dispatches=10,
                rates={s: {'prompt': '.000001', 'completion': '.000001'} for s in policies})
    monkeypatch.setattr(complete, 'load_api_key_file', lambda _: type('Secret', (), {'reveal': lambda _: 'fake'})())
    posts = []
    def post(*args, **kwargs):
        posts.append(kwargs)
        return type('Response', (), {'status_code': 400, 'json': lambda _: {'error': {'code': 'invalid_parameter'}}})()
    monkeypatch.setattr(complete.requests, 'post', post)
    budget = complete.Budget(tmp_path/'budget.sqlite', {'openrouter_usd': '5'}, 'test')
    stop = threading.Event()
    cache = complete.ConcurrentCache(tmp_path/'cache.sqlite', complete.ReportCaller(tmp_path, plan, budget, stop), plan, stop)
    request = dict(model_config=config['generator'], prompt='test', structured_prediction_hash='test')
    try:
        first = cache.call('report_generation', request)[0]
        assert first['text'] == '' and first['transport_error']['http_status'] == 400
        assert not stop.is_set()
        assert cache.call('report_generation', request)[2]
        assert len(posts) == 1
        assert cache.db.execute('SELECT status FROM calls').fetchone()[0] == 'TERMINAL_HTTP_FAILURE'
    finally:
        cache.close(); budget.close()
