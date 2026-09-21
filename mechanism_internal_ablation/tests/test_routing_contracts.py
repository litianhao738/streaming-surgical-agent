"""Frozen-threshold routing and bounded full-pipeline retry contracts."""
import sys
from pathlib import Path
import threading
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'mechanism_internal_ablation/src')]
from fixed_threshold_gate import choose, uncertainty_score
from mechanisms import rule_score
from scripts import run_testing_half_pipeline as pipeline
from surgical_agent.research.gate.collection_budget import BudgetStop


def snapshot():
    return {'h0': {'instrument': [0], 'verb': [], 'target': [], 'ivt': [], 'phase': [0]},
            'pool': {'propositions': [{'id': 'p', 'task': 'instrument', 'label_id': 0}]},
            'probe_ratings_all': {'p': 1, **{f'phase_{i}': 5 if i == 0 else 1 for i in range(7)}},
            'gate_score': 0.5}


def test_definite_refutation_is_rule_signal_not_uncertainty():
    s = snapshot()
    assert uncertainty_score(s) == 0
    assert rule_score(s) == 1
    s['probe_ratings_all']['p'] = None
    assert uncertainty_score(s) == 1
    assert rule_score(s) == 0


def test_independent_random_is_order_invariant_and_not_learned_budget():
    values = {str(i): snapshot() for i in range(100)}
    cfg = {'uncertainty_threshold': .5, 'rule_threshold': .5, 'learned_threshold': .5,
           'random_probability': .5, 'random_seeds': [0, 1]}
    selected, _ = choose(values, cfg)
    reversed_selected, _ = choose(dict(reversed(list(values.items()))), cfg)
    assert selected == reversed_selected
    assert len(selected['learned']) == 100
    assert 0 < len(selected['random_seed0']) < 100
    assert selected['random_seed0'] != selected['random_seed1']
    cfg['random_probability'] = 0
    assert choose(values, cfg)[0]['random_seed0'] == []
    cfg['random_probability'] = 1
    assert len(choose(values, cfg)[0]['random_seed0']) == 100


def test_deferred_errors_drain_healthy_targets_and_have_bounded_retries(tmp_path):
    seen = []
    def work(item):
        seen.append(item['key'])
        if item['key'] == 'bad':
            raise ValueError('terminal content failure')
    with pytest.raises(RuntimeError, match='remain'):
        pipeline.deferred_map(work, [{'key': 'bad'}, {'key': 'good'}], 1,
                              threading.Event(), tmp_path, retry_rounds=2)
    assert seen.count('good') == 1
    assert seen.count('bad') == 3
    assert seen.index('good') < len(seen) - 1


def test_budget_stop_is_not_deferred(tmp_path):
    stop = threading.Event()
    def work(item):
        raise BudgetStop('budget exhausted')
    with pytest.raises(BudgetStop):
        pipeline.deferred_map(work, [{'key': 'a'}], 1, stop, tmp_path)
    assert stop.is_set()
    assert not (tmp_path / 'deferred_errors').exists()
