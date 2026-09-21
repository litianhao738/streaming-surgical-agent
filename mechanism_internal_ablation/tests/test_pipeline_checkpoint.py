from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
from pathlib import Path
import pytest
from scripts.pipeline_checkpoint import Checkpoint
from scripts import testing_half_resume as qwen
from scripts import run_testing_half_complete as complete


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_legacy_names_import_then_no_scan_or_payload_read(tmp_path, monkeypatch):
    selection = [{'key':'a','stage':'pipeline'}, {'key':'b','stage':'warmup'}, {'key':'c','stage':'pipeline'}]
    save(tmp_path/'results/a.json', {'prediction': {}})
    save(tmp_path/'h0/b.json', {})
    save(tmp_path/'h0/c.json', {})  # H0 is not a completed pipeline result.
    cp = Checkpoint(tmp_path, selection, 'plan')
    assert cp.done == {'a', 'b'}
    cp.close()
    def forbidden(*args, **kwargs):
        raise AssertionError('historical scan/read')
    monkeypatch.setattr(Path, 'glob', forbidden)
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    cp = Checkpoint(tmp_path, selection, 'plan')
    assert cp.done == {'a', 'b'} and cp.recover_tail() == 0
    cp.close()


def test_atomic_result_before_checkpoint_commit_recovers_only_tail(tmp_path):
    selection = [{'key':k, 'stage':'pipeline'} for k in ('a','b')]
    cp = Checkpoint(tmp_path, selection, 'plan')
    cp.start('a'); cp.start('b')
    save(tmp_path/'results/a.json', {'prediction': {}})
    cp.close()
    cp = Checkpoint(tmp_path, selection, 'plan')
    assert cp.recover_tail() == 1 and cp.done == {'a'}
    cp.close()
    with pytest.raises(ValueError, match='another plan'):
        Checkpoint(tmp_path, selection, 'different')


def test_eight_workers_persist_all_completions(tmp_path):
    selection = [{'key':str(i), 'stage':'pipeline'} for i in range(50)]
    cp = Checkpoint(tmp_path, selection, 'plan')
    def work(s):
        cp.start(s['key'])
        save(tmp_path/'results'/(s['key']+'.json'), {'prediction': {}})
        cp.finish(s['key'])
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, selection))
    cp.close()
    cp = Checkpoint(tmp_path, selection, 'plan')
    assert len(cp.done) == 50
    cp.close()


@pytest.mark.parametrize('module,driver', [(qwen,complete)])
@pytest.mark.parametrize('terminal', [True,False])
def test_fast_resume_no_request_tree_scan_preserves_budget(tmp_path, monkeypatch, module, driver, terminal):
    core = tmp_path/'core'
    selection = [{'key':'a', 'stage':'pipeline'}]
    core_plan = {'runtime_sha256':{}, 'selection':selection}
    save(core/'plan.json', core_plan)
    save(core/'results/a.json', {'prediction':{}})
    save(tmp_path/'plan.json', {'source_sha256':{}, 'core_output':str(core),
          'report_arms':['H0','FULL'], 'core_plan_sha256':module.sha(core/'plan.json')})
    save(tmp_path/'prepared.json', {'plan_sha256':module.sha(tmp_path/'plan.json')})
    save(tmp_path/'failure.json', {})
    save(core/'failure.json', {})
    budget = complete.Budget(core/'budget.sqlite', {'openrouter_usd':'5'}, module.sha(core/'plan.json'))
    budget.reserve('request', 'openrouter_usd', Decimal('1'))
    if terminal:
        budget.settle('request', Decimal('.1'))
    budget.close()
    monkeypatch.setattr(driver, 'verify_core', lambda *a, **k: core_plan)
    monkeypatch.setattr(module, 'reconcile_completed_reservations', lambda *a: 0)
    def no_scan(*args, **kwargs):
        raise AssertionError('must not scan request tree')
    monkeypatch.setattr(Path, 'rglob', no_scan)
    if not terminal:
        with pytest.raises(ValueError, match='unresolved'):
            module.prepare_resume(tmp_path)
        assert (tmp_path/'failure.json').exists()
        return
    preview = module.prepare_resume(tmp_path, apply=False)
    assert preview['preserved_requests'] == 1 and not (core/'checkpoint.sqlite').exists()
    result = module.prepare_resume(tmp_path)
    assert result['preserved_targets'] == 1 and result['mode'] == 'CHECKPOINT'
    assert not (tmp_path/'failure.json').exists()
    module.validate_runtime(core, core_plan, 'runtime_sha256')
    budget = complete.Budget(core/'budget.sqlite', {'openrouter_usd':'5'}, module.sha(core/'plan.json'))
    assert budget.summary()['dispatches'] == 1
    assert budget.summary()['accounts']['openrouter_usd']['occupied'] == '0.1'
    budget.close()
