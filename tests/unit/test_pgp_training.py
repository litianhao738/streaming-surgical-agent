from copy import deepcopy

import numpy as np

from surgical_agent.research.gate import pgp_gate as core
from surgical_agent.research.gate import pgp_training as training


def sample_data():
    n = 8
    cheap = np.tile(np.array([1, 1, 1]), (n, 5, 1))
    full = cheap.copy()
    full[::2, :4] = [2, 0, 0]
    full[1::2, :4] = [0, 2, 2]
    full[::2, 4] = [2, 0, 0]
    full[1::2, 4] = [1, 1, 1]
    counts = np.repeat(cheap[:, None], 4, axis=1)
    counts[:, 1, :4] = full[:, :4]
    counts[:, 2, 4] = full[:, 4]
    counts[:, 3] = full
    costs = np.tile([5., .01, 0., 0., 2.], (n, 4, 1))
    costs[:, 1] += [4, .02, 1, 1, 4]
    costs[:, 2] += [4, .02, 1, 1, 4]
    costs[:, 3] += [8, .04, 2, 2, 8]
    return {'ids': np.array([f's{i}' for i in range(n)]),
            'videos': np.repeat(['VID103', 'VID23', 'VID31', 'VID96'], 2),
            'X': np.column_stack([np.arange(n) % 2, np.arange(n) / n]),
            'feature_names': ['h0_count', 'qwen_support'],
            'cheap': cheap, 'full': full, 'cheap_merged': cheap.copy(), 'full_merged': full.copy(),
            'actions_counts': counts, 'actions_merged_counts': counts.copy(), 'action_costs': costs,
            'rows': [{'frame_id': i} for i in range(n)]}


def test_vectorized_selector_matches_direct_four_action_replay():
    data = sample_data()
    rng = np.random.default_rng(31)
    scores = rng.uniform(size=(8, 2, 2))
    result = training.select_policy(scores, data['actions_counts'], data['action_costs'], data['videos'])
    assert len(result['candidates']) == 625
    for candidate in result['candidates']:
        actions = core.select_actions(scores, candidate['thresholds'])
        direct = training.measure(data, actions)
        expected = training.acceptance(data, direct)
        assert np.isclose(candidate['f1'], direct['original']['five_head_mean_f1'])
        assert candidate['errors'] == direct['original']['total_errors']
        assert candidate['calls'] == direct['logical_calls']
        assert candidate['feasible'] == expected['passed']


def test_all_skip_or_only_noop_branches_are_not_success():
    data = sample_data()
    for key in ('full', 'full_merged'):
        data[key] = data['cheap'].copy()
    data['actions_counts'][:] = data['cheap'][:, None]
    data['actions_merged_counts'][:] = data['cheap'][:, None]
    result = training.select_policy(np.zeros((8, 2, 2)), data['actions_counts'], data['action_costs'], data['videos'])
    assert result['status'] == 'INFEASIBLE_CONSERVATIVE_FALLBACK'
    assert result['feasible_candidates'] == 0
    actions = core.select_actions(np.zeros((8, 2, 2)), result['thresholds'])
    assert np.all(actions == 0)
    m = training.measure(data, actions)
    assert m['logical_calls'] == 40  # five paid prerequisite logical calls per row
    assert not training.acceptance(data, m)['passed']


def test_outer_labels_cannot_change_that_folds_model_threshold_or_actions():
    first = sample_data()
    second = deepcopy(first)
    held = second['videos'] == 'VID96'
    second['cheap'][held] = [0, 1, 4]
    second['full'][held] = [4, 0, 0]
    second['cheap_merged'][held] = second['cheap'][held]
    second['full_merged'][held] = second['full'][held]
    for a in range(4):
        second['actions_counts'][held, a] = second['cheap'][held]
        if a & 1:
            second['actions_counts'][held, a, :4] = second['full'][held, :4]
        if a & 2:
            second['actions_counts'][held, a, 4] = second['full'][held, 4]
    second['actions_merged_counts'] = second['actions_counts'].copy()
    saved = [{}, {}]
    outputs = []
    for i, data in enumerate((first, second)):
        result, scores, actions = training.nested_fit(data, 'semantic_benefit_original_harm', lambda _: None,
                                                     lambda name, model: saved[i].update({name: deepcopy(model)}))
        outputs.append((result, scores, actions))
    assert saved[0]['outer_VID96'] == saved[1]['outer_VID96']
    np.testing.assert_array_equal(outputs[0][1][held], outputs[1][1][held])
    np.testing.assert_array_equal(outputs[0][2][held], outputs[1][2][held])
    assert saved[0]['final_research_model']['deployable'] is False
    assert saved[0]['final_research_model']['tracker_enabled'] is False


def test_random_control_is_reproducible_and_does_not_exceed_call_cap():
    data = sample_data()
    actions = np.tile([1, 2], 4)
    ref = training.measure(data, actions)
    a, x = training.random_control(data, actions, draws=5)
    b, y = training.random_control(data, actions, draws=5)
    assert a == b
    np.testing.assert_array_equal(x, y)
    assert a['calls_min_max'][1] <= ref['logical_calls']
