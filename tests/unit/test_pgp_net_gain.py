from copy import deepcopy
import numpy as np
import pytest
from tests.unit.test_pgp_training import sample_data
from surgical_agent.research.gate import pgp_net_gain as net
from surgical_agent.research.gate import pgp_training as ref
from surgical_agent.research.gate import pgp_default


def test_sign_targets_do_not_mix_error_count_into_harm():
    a = np.tile([1, 1, 1], (2, 5, 1))
    b = a.copy(); b[0] = [2, 2, 2]; b[1, 4] = [2, 0, 0]
    t = net.branch_targets(a, b)
    assert not t[0].any()
    assert t[1, 1, 0] and not t[1, 1, 1]


def test_all_289_policies_match_direct_replay():
    d = sample_data(); s = np.random.default_rng(31).uniform(size=(8, 2, 2))
    result = net.select_policy(s, d['actions_counts'], d['action_costs'], d['videos'])
    assert len(result['candidates']) == 289
    for c in result['candidates']:
        r = ref.measure(d, net.select_actions(s, c['thresholds']))
        assert np.isclose(c['f1'], r['original']['five_head_mean_f1'])
        assert c['errors'] == r['original']['total_errors']
        assert c['calls'] == r['logical_calls']
        assert c['feasible'] == ref.acceptance(d, r)['passed']


def test_held_labels_do_not_change_routing():
    a = sample_data(); b = deepcopy(a); held = b['videos'] == 'VID96'
    b['cheap_merged'][held] = [0, 2, 2]
    b['full_merged'][held] = [2, 0, 0]
    b['actions_counts'][held] = [1, 0, 0]
    saved = [{}, {}]; outputs = []
    for i, d in enumerate((a, b)):
        outputs.append(net.nested_fit(d, lambda _: None, lambda n, m: saved[i].update({n: deepcopy(m)})))
    assert saved[0]['outer_VID96'] == saved[1]['outer_VID96']
    np.testing.assert_array_equal(outputs[0][2][held], outputs[1][2][held])


def test_noop_is_infeasible_and_prefix_is_charged():
    d = sample_data(); d['actions_counts'][:] = d['cheap'][:, None]
    s = np.zeros((8, 2, 2)); r = net.select_policy(s, d['actions_counts'], d['action_costs'], d['videos'])
    assert r['status'] == 'INFEASIBLE_CONSERVATIVE_FALLBACK'
    assert ref.measure(d, net.select_actions(s, r['thresholds']))['logical_calls'] == 40


def test_default_is_pinned_and_tampering_rejected(tmp_path):
    import json
    manifest, model = pgp_default.load_default()
    assert manifest['automatic_experiment_promotion'] is False
    p = tmp_path / manifest['model_artifact']; p.parent.mkdir(parents=True)
    p.write_text(json.dumps(model), encoding='utf-8')
    (tmp_path / 'DEFAULT_PGP_GATE_VERSION.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='hash changed'):
        pgp_default.load_default(tmp_path)
    with pytest.raises(ValueError, match='dual-threshold'):
        net.predict_actions(model, np.zeros((1, 866)), model['feature_names'])
