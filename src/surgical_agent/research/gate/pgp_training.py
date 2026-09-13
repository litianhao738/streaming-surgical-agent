"""Video-separated fitting and fixed-policy selection for offline PGP-Gate.

This module has no API, Tracker, checkpoint loading, or dataset discovery path.
Its inputs are validated, observed Training-only counterfactual arrays.
"""
from __future__ import annotations

from itertools import product
import numpy as np

from surgical_agent.research.gate import official_net_training as metrics
from surgical_agent.research.gate import pgp_gate as core

SEED = 3407
MIN_HELP = (-0.01, 0.25, 0.5, 0.75, 1.01)
MAX_HARM = (-0.01, 0.25, 0.5, 0.75, 1.01)
BRANCH_THRESHOLDS = tuple(product(MIN_HELP, MAX_HARM))
POLICIES = tuple((a, b) for a in BRANCH_THRESHOLDS for b in BRANCH_THRESHOLDS)
SKIP = ((1.01, -0.01), (1.01, -0.01))
EPS = 1e-10
COST_FIELDS = ('logical_calls', 'known_usd', 'glm_requests', 'deepseek_requests', 'reviewer_calls')


def quality(counts):
    return metrics.pooled(np.asarray(counts))


def measure(data, actions, indices=None):
    ix = np.arange(len(data['ids'])) if indices is None else np.asarray(indices)
    actions = np.asarray(actions, dtype=int)
    if actions.shape != (len(ix),) or np.any((actions < 0) | (actions > 3)):
        raise ValueError('one valid branch action per observed target required')
    counts = data['actions_counts'][ix, actions]
    merged = data['actions_merged_counts'][ix, actions]
    costs = data['action_costs'][ix, actions]
    result = {'original': quality(counts), 'merged': quality(merged),
              'actions': {str(a): int(np.sum(actions == a)) for a in range(4)},
              'reviewed_targets': int(np.count_nonzero(actions)),
              'interaction_selected': int(np.count_nonzero(actions & 1)),
              'phase_selected': int(np.count_nonzero(actions & 2)),
              **dict(zip(COST_FIELDS, costs.sum(axis=0).tolist())), 'by_video': {}}
    for v in sorted(set(data['videos'][ix])):
        mask = data['videos'][ix] == v
        result['by_video'][str(v)] = {
            'n': int(mask.sum()), 'original': quality(counts[mask]), 'merged': quality(merged[mask]),
            'actions': {str(a): int(np.sum(actions[mask] == a)) for a in range(4)},
            **dict(zip(COST_FIELDS, costs[mask].sum(axis=0).tolist()))}
    return result


def acceptance(data, result, indices=None):
    ix = np.arange(len(data['ids'])) if indices is None else np.asarray(indices)
    cheap, full = quality(data['cheap'][ix]), quality(data['full'][ix])
    observed = result['original']
    per_video = {}
    for v in sorted(set(data['videos'][ix])):
        ref = quality(data['cheap'][ix[data['videos'][ix] == v]])
        actual = result['by_video'][str(v)]['original']
        per_video[str(v)] = {
            'f1_delta': actual['five_head_mean_f1'] - ref['five_head_mean_f1'],
            'errors_delta': actual['total_errors'] - ref['total_errors'],
            'passed': bool(actual['five_head_mean_f1'] + EPS >= ref['five_head_mean_f1']
                           and actual['total_errors'] <= ref['total_errors'])}
    checks = {
        'pooled_f1_at_least_cheap_and_full': observed['five_head_mean_f1'] + EPS >= max(cheap['five_head_mean_f1'], full['five_head_mean_f1']),
        'pooled_errors_at_most_cheap_and_full': observed['total_errors'] <= min(cheap['total_errors'], full['total_errors']),
        'every_video_no_harm_vs_cheap': all(x['passed'] for x in per_video.values()),
        'actual_positive_gain': result['reviewed_targets'] > 0 and (
            observed['five_head_mean_f1'] > cheap['five_head_mean_f1'] + EPS or observed['total_errors'] < cheap['total_errors']),
        'calls_below_same_probe_same_stop_always': result['logical_calls'] < data['action_costs'][ix, 3, 0].sum()}
    return {'passed': bool(all(checks.values())), 'checks': {k: bool(v) for k, v in checks.items()},
            'per_video_vs_cheap': per_video}


def select_policy(scores, action_counts, action_costs, videos):
    """Select on the supplied inner OOF population only, using 625 fixed policies.

    The function deliberately accepts no full dataset or outer index. Branch
    counts are additive, but the complete policy's pooled and per-video risks
    are evaluated jointly; there is no independence assumption about errors.
    """
    scores, action_counts, action_costs, videos = map(np.asarray, (scores, action_counts, action_costs, videos))
    n = len(videos)
    if scores.shape != (n, 2, 2) or action_counts.shape != (n, 4, 5, 3) or action_costs.shape != (n, 4, 5):
        raise ValueError('invalid policy-selection shapes')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('finite classifier scores in [0,1] required')
    for a in (action_counts, action_costs):
        if not np.allclose(a[:, 3], a[:, 1] + a[:, 2] - a[:, 0], atol=1e-8):
            raise ValueError('branch composition is not additive; cannot use this selector')
    enabled = [np.column_stack([(scores[:, b, 0] >= h) & (scores[:, b, 1] <= d)
                               for h, d in BRANCH_THRESHOLDS]) for b in range(2)]
    groups = {'ALL': np.ones(n, dtype=bool), **{str(v): videos == v for v in sorted(set(videos))}}
    totals, cost_totals = {}, {}
    for v, mask in groups.items():
        base = action_counts[mask, 0].sum(axis=0)
        increments = [enabled[b][mask].T.astype(np.int64) @
                      (action_counts[mask, 1 << b] - action_counts[mask, 0]).reshape(mask.sum(), 15)
                      for b in range(2)]
        totals[v] = base[None, None] + increments[0].reshape(25, 1, 5, 3) + increments[1].reshape(1, 25, 5, 3)
        cb = action_costs[mask, 0].sum(axis=0)
        ci = [enabled[b][mask].T.astype(float) @ (action_costs[mask, 1 << b] - action_costs[mask, 0]) for b in range(2)]
        cost_totals[v] = cb[None, None] + ci[0][:, None] + ci[1][None]
    cheap, full = quality(action_counts[:, 0]), quality(action_counts[:, 3])
    references = {v: quality(action_counts[mask, 0]) for v, mask in groups.items() if v != 'ALL'}
    table = []
    for k, thresholds in enumerate(POLICIES):
        i, j = divmod(k, 25)
        q = quality(totals['ALL'][i, j])
        deltas = {v: {'f1': quality(totals[v][i, j])['five_head_mean_f1'] - references[v]['five_head_mean_f1'],
                      'errors': quality(totals[v][i, j])['total_errors'] - references[v]['total_errors']}
                  for v in references}
        checks = {
            'pooled_f1': q['five_head_mean_f1'] + EPS >= max(cheap['five_head_mean_f1'], full['five_head_mean_f1']),
            'pooled_errors': q['total_errors'] <= min(cheap['total_errors'], full['total_errors']),
            'per_video': all(x['f1'] >= -EPS and x['errors'] <= 0 for x in deltas.values()),
            'positive_gain': q['five_head_mean_f1'] > cheap['five_head_mean_f1'] + EPS or q['total_errors'] < cheap['total_errors'],
            'cost_saving': cost_totals['ALL'][i, j, 0] < action_costs[:, 3, 0].sum()}
        table.append({'policy_id': k, 'thresholds': thresholds, 'f1': q['five_head_mean_f1'],
                      'errors': q['total_errors'], 'calls': float(cost_totals['ALL'][i, j, 0]),
                      'known_usd': float(cost_totals['ALL'][i, j, 1]),
                      'minimum_video_f1_delta': min(x['f1'] for x in deltas.values()),
                      'per_video_deltas': deltas, 'checks': {a: bool(b) for a, b in checks.items()},
                      'feasible': bool(all(checks.values()))})
    viable = [r for r in table if r['feasible']]
    if viable:
        chosen = min(viable, key=lambda r: (r['calls'], -r['minimum_video_f1_delta'], -r['f1'], r['errors'], r['policy_id']))
        thresholds = chosen['thresholds']
    else:
        thresholds = SKIP
        chosen = table[POLICIES.index(SKIP)]
    return {'status': 'FEASIBLE' if viable else 'INFEASIBLE_CONSERVATIVE_FALLBACK',
            'thresholds': thresholds, 'feasible_candidates': len(viable), 'candidate_count': len(table),
            'selected': chosen, 'candidates': table}


def nested_fit(data, supervision, emit, save_model):
    """Evaluate without outer-label tuning, then fit a research-only final model."""
    X, vids = data['X'], data['videos']
    targets = core.branch_targets(data['cheap'], data['full'], data['cheap_merged'], data['full_merged'], supervision=supervision)
    ix = np.arange(len(vids)); unique = sorted(set(vids))
    scores = np.full((len(ix), 2, 2), np.nan)
    actions = np.zeros(len(ix), dtype=np.int64)
    folds, memo = [], {}

    def model_for(fit):
        key = tuple(sorted(set(vids[fit])))
        if key not in memo:
            emit('Fitting fresh Gate heads on ' + ','.join(key))
            model = core.fit_model(X[fit], targets[fit], vids[fit], data['feature_names'])
            model['fit_video_ids'] = list(key)
            model['fit_sample_count'] = len(fit)
            model['supervision'] = supervision
            memo[key] = model
        return memo[key]

    for held_video in unique:
        fit, held = ix[vids != held_video], ix[vids == held_video]
        inner = np.full((len(fit), 2, 2), np.nan)
        inner_bindings = []
        for inner_video in sorted(set(vids[fit])):
            train = fit[vids[fit] != inner_video]
            valid = fit[vids[fit] == inner_video]
            assert held_video not in set(vids[train]) and inner_video not in set(vids[train])
            model = model_for(train)
            inner[vids[fit] == inner_video] = core.predict_scores(model, X[valid], data['feature_names'])
            inner_bindings.append({'held': str(inner_video), 'fit': sorted(set(vids[train]))})
        selected = select_policy(inner, data['actions_counts'][fit], data['action_costs'][fit], vids[fit])
        model = model_for(fit)
        scores[held] = core.predict_scores(model, X[held], data['feature_names'])
        actions[held] = core.select_actions(scores[held], selected['thresholds'])
        held_report = measure(data, actions[held], held)
        save_model('outer_' + str(held_video), {**model, 'thresholds': selected['thresholds'], 'selection_status': selected['status'],
                    'held_video': str(held_video), 'deployable': False, 'tracker_enabled': False})
        folds.append({'held_video': str(held_video), 'fit_videos': sorted(set(vids[fit])),
                      'inner_partitions': inner_bindings, 'inner_selection': selected, 'held_result': held_report})
        emit(f'Held {held_video}: {selected["status"]}; F1={held_report["original"]["five_head_mean_f1"]:.4f}; actions={held_report["actions"]}')
    assert np.isfinite(scores).all()
    result = measure(data, actions)
    result['quality_cost_acceptance'] = acceptance(data, result)
    result['all_inner_feasible'] = all(f['inner_selection']['status'] == 'FEASIBLE' for f in folds)
    result['conditional_development_pass'] = result['all_inner_feasible'] and result['quality_cost_acceptance']['passed']
    result['folds'] = folds
    result['supervision'] = supervision
    result['target_counts'] = {b: {k: int(targets[:, j, c].sum()) for c, k in enumerate(('help', 'harm'))}
                               for j, b in enumerate(('interaction', 'phase'))}
    # These OOF scores come from models fitted on the other three videos.
    # Select one research operating point without calling it independent testing.
    selected = select_policy(scores, data['actions_counts'], data['action_costs'], vids)
    final_model = model_for(ix)
    final_model.update({'thresholds': selected['thresholds'], 'selection_status': selected['status'],
                        'tracker_enabled': False, 'deployable': False, 'paper_final': False,
                        'conditional_development_pass': result['conditional_development_pass'],
                        'purpose': 'Frozen Training-only research model; independent validation absent',
                        'validation_scope': 'Gate-only nested LOVO; upstream not fully nested; four reused development videos'})
    if not np.allclose(core.predict_scores(final_model, X, data['feature_names']),
                       core.predict_scores(__import__('json').loads(__import__('json').dumps(final_model)), X, data['feature_names'])):
        raise ValueError('research model serialization drift')
    save_model('final_research_model', final_model)
    result['final_training_oof_selection'] = selected
    return result, scores, actions


def random_control(data, reference_actions, draws=300):
    """Conditional block-random branch benchmark under each video's call cap."""
    rng = np.random.default_rng(SEED)
    n = len(reference_actions); frames = np.array([r['frame_id'] for r in data['rows']])
    inc = np.stack([data['action_costs'][:, a, 0] - data['action_costs'][:, 0, 0] for a in (1, 2)], axis=1)
    raw = []
    for _ in range(draws):
        chosen = np.zeros(n, dtype=int)
        for v in sorted(set(data['videos'])):
            ix = np.flatnonzero(data['videos'] == v)
            ordered = ix[np.argsort(frames[ix], kind='stable')]
            blocks = [ordered[k:k + 30] for k in range(0, len(ordered), 30)]
            sequence = np.concatenate([blocks[k] for k in rng.permutation(len(blocks))])
            pairs = [(int(i), int(b)) for i in sequence for b in rng.permutation(2)]
            budget = (data['action_costs'][ix, reference_actions[ix], 0] - data['action_costs'][ix, 0, 0]).sum()
            used = 0.
            for i, b in pairs:
                if used + inc[i, b] > budget + EPS:
                    break
                chosen[i] |= 1 << b
                used += inc[i, b]
        cost = data['action_costs'][np.arange(n), chosen].sum(axis=0)
        q = quality(data['actions_counts'][np.arange(n), chosen])
        raw.append([q['five_head_mean_f1'], q['total_errors'], cost[0], cost[1]])
    arr = np.asarray(raw)
    return {'draws': draws, 'seed': SEED, 'block_targets': 30,
            'scope': 'Conditional offline reference; same per-video logical-call cap, not matched dollars or independent-video confidence',
            'prefix_can_underspend': True, 'f1_mean': float(arr[:, 0].mean()), 'f1_p975': float(np.quantile(arr[:, 0], .975)),
            'errors_mean': float(arr[:, 1].mean()), 'errors_p025': float(np.quantile(arr[:, 1], .025)),
            'calls_min_max': [float(arr[:, 2].min()), float(arr[:, 2].max())],
            'known_usd_min_max': [float(arr[:, 3].min()), float(arr[:, 3].max())]}, arr
