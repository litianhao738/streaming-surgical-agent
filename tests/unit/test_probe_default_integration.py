import json
from types import SimpleNamespace
import pytest
from scripts import probe_replay_backend as replay
from scripts import run_pipeline as dispatch


def test_probe_replay_uses_five_head_responses_and_fails_on_uncollected_seat():
    original = dict(h0_raw={'h0':1}, proposal_raw={}, review_raw={}, joint_raw={},
                    phase_recommendation={'raw':None})
    backend = replay.ProbeReplayBackend(original, {'five_head_raw':{'qwen':{'phase':[5]}, 'gpt':None}})
    response = backend.five_head('qwen')
    response['phase'][0] = 1
    assert backend.five_head('qwen')['phase'] == [5]
    assert backend.five_head('gpt') is None  # actually collected invalid response
    with pytest.raises(ValueError, match='uncollected'):
        backend.five_head('glm')


def test_collection_rejects_unsealed_or_testing_data(tmp_path):
    receipt = {'state':'PASS', 'review_mode':'five_head_probe', 'rows_sha256':'sealed', 'rows':1}
    (tmp_path/'rows.jsonl').write_text(json.dumps({'sample_id':'a', 'source_split':'Testing'})+'\n')
    with pytest.raises(ValueError, match='not sealed'):
        replay.load_collection(tmp_path, lambda p:'changed', lambda p:receipt)
    with pytest.raises(ValueError, match='Testing'):
        replay.load_collection(tmp_path, lambda p:'sealed', lambda p:receipt)


def test_complete_dispatch_forwards_scope_budget_and_paid_flag(monkeypatch, tmp_path):
    manifest = tmp_path/'default.json'
    manifest.write_text(json.dumps({'testing_entrypoint':'scripts/run_testing_half_complete.py'}))
    monkeypatch.setattr(dispatch, 'DEFAULT_MANIFEST', manifest)
    called = []
    monkeypatch.setattr(dispatch.subprocess, 'run', lambda cmd, **kw: called.append(cmd) or SimpleNamespace(returncode=0))
    assert dispatch.main(['testing-run', '--output', str(tmp_path/'out'), '--scope', str(tmp_path/'scope'),
                          '--budget-limits', str(tmp_path/'caps.json'), '--allow-paid']) == 0
    cmd = called[0]
    assert cmd[4] == 'run'
    assert cmd[cmd.index('--scope')+1] == str((tmp_path/'scope').resolve())
    assert '--core-budget-limits' in cmd and '--allow-paid' in cmd


def test_complete_dispatch_never_silently_ignores_training_source(monkeypatch, tmp_path):
    manifest = tmp_path/'default.json'
    manifest.write_text('{}')
    monkeypatch.setattr(dispatch, 'DEFAULT_MANIFEST', manifest)
    with pytest.raises(SystemExit):
        dispatch.main(['testing-run', '--output', str(tmp_path/'out'), '--source', str(tmp_path)])
