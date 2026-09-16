from copy import deepcopy
from decimal import Decimal
import json

import pytest

from scripts import complete_testing_half_judges as finish
from surgical_agent.research.reporting.contracts import GroundTruth, ModelConfig
from surgical_agent.research.reporting.judge_selection import build_judge
from surgical_agent.research.reporting.llm_judge_calibrated import CHECKS, PROMPT as OLD_PROMPT
from surgical_agent.research.reporting.llm_judge_strict import PROMPT, PROMPT_VERSION, build_strict_judge
from surgical_agent.research.reporting.transport import CachedCalls


def valid_response():
    return {'text': json.dumps({'audit': dict.fromkeys(CHECKS, 'match'), 'errors': [],
                               'factual_consistency': 80, 'clarity': 90, 'coherence': 90,
                               'overall': 85, 'major_error': False})}


def config():
    return {'judge_version': 'v2_low', 'judge': ModelConfig(
        'openrouter', 'openai/gpt-5.6-luna', 'https://example.invalid', 0., 4096).to_dict()}


def test_invalid_or_transport_response_cannot_change_existing_rows():
    rows = [{'report': 'liver', 'rule_score': 40, 'llm_judge_score': None,
             'gsr_score': None, 'diagnostics': {'original': True}}]
    before = deepcopy(rows)
    assert not finish.apply_response(rows, {'text': 'invalid'})
    assert rows == before
    assert not finish.apply_response(rows, {**valid_response(), 'transport_error': {'http_status': 502}})
    assert rows == before
    assert finish.apply_response(rows, valid_response())
    assert rows[0]['gsr_score'] == pytest.approx(49)
    assert rows[0]['diagnostics']['original'] is True


def test_summary_roundoff_tolerance_does_not_accept_missing_or_changed_scores(monkeypatch):
    summary = {'status': 'COMPLETE', 'total_frames': 1, 'eligible_frames': 1,
               'excluded_partial_labels': 0, 'paired_complete_frames': 1, 'mock': False,
               'methods': {'H0': {'gsr_score': 49.54442905324019}}}
    monkeypatch.setattr(finish, 'summarize', lambda _: summary)
    baseline = deepcopy(summary)
    baseline['methods']['H0']['gsr_score'] += 1e-12
    finish.verify_summary([], baseline)
    for changed in (None, 49.55):
        baseline['methods']['H0']['gsr_score'] = changed
        with pytest.raises(ValueError, match='summary mismatch'):
            finish.verify_summary([], baseline)


def test_strict_prompt_has_new_cache_identity_but_same_parser_and_formula(tmp_path):
    seen = []
    def caller(request):
        seen.append(request)
        return valid_response()
    cache = CachedCalls(tmp_path / 'cache.sqlite', caller, max_calls=2, mock=True)
    try:
        report, truth = {'report': 'liver'}, GroundTruth((29,), 5, True, True)
        old = build_judge(config(), cache).evaluate(report, truth)
        new = build_strict_judge({**config(), 'judge_version': 'v2_low_strict'}, cache).evaluate(report, truth)
        again = build_strict_judge({**config(), 'judge_version': 'v2_low_strict'}, cache).evaluate(report, truth)
        assert len(seen) == 2 and again['judge_cache_hit']
        assert old['judge_cache_key'] != new['judge_cache_key']
        assert old['llm_judge_score'] == new['llm_judge_score'] == 85
        assert seen[0]['prompt'].startswith(OLD_PROMPT)
        assert seen[1]['prompt'].startswith(PROMPT)
        assert new['judge_prompt_version'] == PROMPT_VERSION
        assert seen[1]['prompt_version'] == PROMPT_VERSION
        assert seen[0]['wire_policy'] == seen[1]['wire_policy']
        assert 'only for that coverage evidence entry' in PROMPT
        assert 'For omissions only, report_quote may be empty.' in OLD_PROMPT
    finally:
        cache.close()


def make_source(tmp_path, monkeypatch):
    source, core = tmp_path / 'source', tmp_path / 'core'
    source.mkdir(); core.mkdir()
    labels = {'instrument': [0], 'verb': [1], 'target': [0], 'ivt': [0], 'phase': [0]}
    scores = {'details': [{'key': 'VID01_100', 'h0': labels, 'pipeline': labels,
                          'gt': labels, 'mask': {'ivt': True, 'phase': True}}]}
    finish.write_artifact(core / 'scores_detail.json', scores)
    cfg = config()
    plan = {'source_sha256': {}, 'core_output': str(core), 'report_config': cfg,
            'max_report_dispatches': 10, 'policies': {'offline_evaluation': {
                'wire_policy': {'reasoning': {'effort': 'low'}, 'response_format': {'type': 'json_object'}, 'stream': False},
                'omit_temperature': True, 'provider_tag': 'openai'}}}
    finish.write_artifact(source / 'plan.json', plan)
    finish.write_artifact(source / 'prepared.json', {'plan_sha256': finish.w.sha(source / 'plan.json')})
    finish.write_artifact(source / 'receipt.json', {})
    monkeypatch.setattr(finish, 'verify_sealed_core', lambda *args: None)
    monkeypatch.setattr(finish, 'evaluate_rule', lambda raw, truth: {'rule_score': 40})
    cache = CachedCalls(source / 'reports/cache.sqlite', lambda req: {'text': 'invalid'}, max_calls=1)
    rows = []
    for h, f, truth in finish.load_pairs(scores):
        for item in (h, f):
            report = {'video_id': item.video_id, 'frame_id': item.frame_id,
                      'source_prediction': item.source_prediction, 'canonical_appendix': item.appendix(),
                      'report': 'liver', 'raw_response': '{}', 'rule_score': 40,
                      'report_valid': True, 'gsr_score': None, 'diagnostics': {}}
            rows.append({**report, **build_judge(cfg, cache).evaluate(report, truth)})
    cache.close()
    finish.write_artifacts(source / 'reports/results', rows, {}, {})
    return source


@pytest.mark.parametrize('rounds,success', [(1, False), (3, True)])
def test_bounded_recovery_deduplicates_frames_and_shares_total_budget(tmp_path, monkeypatch, rounds, success):
    source = make_source(tmp_path, monkeypatch)
    original = (source / 'reports/cache.sqlite').read_bytes()
    calls = []
    def caller_factory(out, plan, budget, stop):
        def caller(request):
            calls.append(request)
            identity = budget.key('test', 'judge', str(len(calls)))
            budget.reserve(identity, 'openrouter_usd', Decimal('.1'))
            budget.settle(identity, Decimal('.1'))
            return valid_response() if len(calls) >= 2 else {'text': 'invalid'}
        return caller
    monkeypatch.setattr(finish.w, 'ReportCaller', caller_factory)
    out = tmp_path / 'completed'
    args = ['--source', str(source), '--output', str(out), '--allow-paid',
            '--max-rounds', str(rounds), '--interval-seconds', '0', '--budget-usd', '1']
    if success:
        finish.main(args)
    else:
        with pytest.raises(RuntimeError, match='exhausted'):
            finish.main(args)
    receipt = finish.w.read(out / 'receipt.json')
    assert receipt['state'] == ('COMPLETE' if success else 'INCOMPLETE')
    assert len(calls) == (2 if success else 1)  # H0/FULL share one Judge request.
    assert float(receipt['new_budget_occupied_usd']) == pytest.approx(.1 * len(calls))
    assert (source / 'reports/cache.sqlite').read_bytes() == original
    if success:
        second = finish.w.read(out / 'attempts/round_02/receipt.json')
        assert second['budget']['accounts']['openrouter_usd']['cap'] == '0.9'
        summary = finish.w.read(out / 'reports/results/summary.json')
        assert summary['paired_complete_frames'] == summary['eligible_frames'] == 1


def test_completion_dry_run_never_constructs_paid_caller(tmp_path, monkeypatch):
    source = make_source(tmp_path, monkeypatch)
    monkeypatch.setattr(finish.w, 'ReportCaller', lambda *a: pytest.fail('No paid caller in dry run'))
    out = tmp_path / 'dry'
    finish.main(['--source', str(source), '--output', str(out), '--dry-run', '--allow-paid'])
    assert not out.exists()


def test_existing_retry_command_routes_judge_only_to_bounded_recovery(tmp_path, monkeypatch):
    from scripts import retry_testing_half_reports as retry
    source = tmp_path / 'source'
    source.mkdir()
    monkeypatch.setattr(retry, 'reusable_rows', lambda _: ([], [('key', 'offline_evaluation')]))
    seen = []
    monkeypatch.setattr(finish, 'main', lambda args: seen.append(args))
    monkeypatch.setattr(retry.sys, 'argv', ['retry', '--source', str(source), '--output', str(tmp_path / 'out'),
                                          '--max-rounds', '7', '--budget-usd', '1.5', '--dry-run'])
    retry.main()
    assert len(seen) == 1
    assert seen[0][seen[0].index('--max-rounds') + 1] == '7'
    assert seen[0][seen[0].index('--budget-usd') + 1] == '1.5'
    assert '--dry-run' in seen[0] and '--allow-paid' not in seen[0]
