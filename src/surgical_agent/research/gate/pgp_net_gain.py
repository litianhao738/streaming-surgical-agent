"""Experimental net-gain PGP; never changes the pinned dual-threshold default."""
from __future__ import annotations

from itertools import product
import json
import numpy as np

from surgical_agent.research.gate import pgp_gate as classifier
from surgical_agent.research.gate import pgp_training as reference

RULE = 'branch_net_gain_threshold_v1'
SUPERVISION = 'merged_branch_frame_f1_sign'
# Fixed before fitting: old whole-frame useful range plus explicit endpoints.
THRESHOLDS = (-1.01, -1., -.75, -.5, -.25, -.1, -.05, 0., .025, .05, .1, .2, .4, .6, .8, 1., 1.01)
POLICIES = tuple(product(THRESHOLDS, repeat=2))
SKIP = (1.01, 1.01)


def branch_targets(cheap_merged, full_merged):
    """Literal same-view help/harm, defined by branch frame-F1 signs.

    Both heads use the same semantic label view. Original-label error constraints
    remain in policy calibration/evaluation, not mixed into one score target.
    """
    a, b = (np.asarray(x, dtype=float) for x in (cheap_merged, full_merged))
    if a.ndim != 3 or a.shape[1:] != (5, 3) or b.shape != a.shape:
        raise ValueError('matching (N,5,3) counts required')
    if any(not np.isfinite(x).all() or np.any(x < 0) or np.any(x != np.floor(x)) for x in (a, b)):
        raise ValueError('finite nonnegative integer counts required')
    def f1(x):
        den = 2 * x[..., 0] + x[..., 1] + x[..., 2]
        return np.divide(2 * x[..., 0], den, out=np.ones_like(den), where=den != 0)
    delta = f1(b) - f1(a)
    utility = np.column_stack((delta[:, :4].mean(axis=1), delta[:, 4]))
    return np.stack((utility > classifier.EPS, utility < -classifier.EPS), axis=2)


def net_scores(scores):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (2, 2) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('finite (N,2,2) help/harm scores in [0,1] required')
    return values[:, :, 0] - values[:, :, 1]


def select_actions(scores, thresholds):
    net = net_scores(scores)
    limits = np.asarray(thresholds, dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all():
        raise ValueError('one finite net threshold per branch required')
    enabled = net >= limits
    return enabled[:, 0].astype(np.uint8) + 2 * enabled[:, 1].astype(np.uint8)


def predict_actions(model, X, feature_names):
    if model.get('decision_rule') != RULE:
        raise ValueError('net-gain decoder cannot apply a dual-threshold model')
    scores = classifier.predict_scores(model, X, feature_names)
    return scores, select_actions(scores, model['thresholds'])


def select_policy(scores, action_counts, action_costs, videos):
    """Jointly evaluate the 289 full policies on only supplied inner OOF rows."""
    net = net_scores(scores)
    counts, costs, vids = map(np.asarray, (action_counts, action_costs, videos))
    n, m = len(net), len(THRESHOLDS)
    if vids.shape != (n,) or counts.shape != (n, 4, 5, 3) or costs.shape != (n, 4, 5):
        raise ValueError('policy input shapes do not agree')
    if not n:
        raise ValueError('empty calibration population')
    for x in (counts, costs):
        if not np.isfinite(x).all() or np.any(x < 0) or not np.allclose(x[:, 3], x[:, 1] + x[:, 2] - x[:, 0], atol=1e-8):
            raise ValueError('invalid or nonadditive cached actions')
    enabled = [net[:, b, None] >= np.asarray(THRESHOLDS)[None] for b in range(2)]
    groups = {'ALL': np.ones(n, bool), **{str(v): vids == v for v in sorted(set(vids))}}
    metrics, all_costs, delta = {}, None, {}
    for name, mask in groups.items():
        base = counts[mask, 0].sum(axis=0)
        inc = [enabled[b][mask].T.astype(np.int64) @ (counts[mask, 1 << b] - counts[mask, 0]).reshape(mask.sum(), 15)
               for b in range(2)]
        total = base[None, None] + inc[0].reshape(m, 1, 5, 3) + inc[1].reshape(1, m, 5, 3)
        den = 2 * total[..., 0] + total[..., 1] + total[..., 2]
        f1 = np.divide(2 * total[..., 0], den, out=np.ones_like(den, dtype=float), where=den != 0).mean(axis=-1) * 100
        errors = total[..., 1:].sum(axis=(-1, -2))
        metrics[name] = (f1, errors)
        q = reference.quality(counts[mask, 0])
        delta[name] = (f1 - q['five_head_mean_f1'], errors - q['total_errors'])
        if name == 'ALL':
            cb = costs[:, 0].sum(axis=0)
            ci = [enabled[b].T.astype(float) @ (costs[:, 1 << b] - costs[:, 0]) for b in range(2)]
            all_costs = cb[None, None] + ci[0][:, None] + ci[1][None]
    cheap, full = reference.quality(counts[:, 0]), reference.quality(counts[:, 3])
    f1, errors = metrics['ALL']
    check_arrays = {
        'pooled_f1': f1 + reference.EPS >= max(cheap['five_head_mean_f1'], full['five_head_mean_f1']),
        'pooled_errors': errors <= min(cheap['total_errors'], full['total_errors']),
        'per_video': np.logical_and.reduce([(delta[v][0] >= -reference.EPS) & (delta[v][1] <= 0) for v in groups if v != 'ALL']),
        'positive_gain': (delta['ALL'][0] > reference.EPS) | (delta['ALL'][1] < 0),
        'cost_saving': all_costs[:, :, 0] < costs[:, 3, 0].sum()}
    table = []
    for k, thresholds in enumerate(POLICIES):
        i, j = divmod(k, m)
        dv = {v: {'f1': float(delta[v][0][i, j]), 'errors': int(delta[v][1][i, j])} for v in groups if v != 'ALL'}
        checks = {name: bool(x[i, j]) for name, x in check_arrays.items()}
        table.append({'policy_id': k, 'thresholds': thresholds, 'f1': float(f1[i, j]), 'errors': int(errors[i, j]),
                      'calls': float(all_costs[i, j, 0]), 'known_usd': float(all_costs[i, j, 1]),
                      'minimum_video_f1_delta': min(d['f1'] for d in dv.values()), 'per_video_deltas': dv,
                      'checks': checks, 'feasible': all(checks.values())})
    viable = [x for x in table if x['feasible']]
    chosen = min(viable, key=lambda x: (x['calls'], -x['minimum_video_f1_delta'], -x['f1'], x['errors'], x['policy_id'])) if viable else table[POLICIES.index(SKIP)]
    return {'status': 'FEASIBLE' if viable else 'INFEASIBLE_CONSERVATIVE_FALLBACK', 'thresholds': chosen['thresholds'],
            'feasible_candidates': len(viable), 'candidate_count': len(table), 'selected': chosen, 'candidates': table}


def nested_fit(data, emit, save_model):
    X, videos = data['X'], data['videos']
    targets = branch_targets(data['cheap_merged'], data['full_merged'])
    ix = np.arange(len(videos)); scores = np.full((len(ix), 2, 2), np.nan)
    actions = np.zeros(len(ix), dtype=np.uint8)
    models, folds = {}, []
    def fit_model(indices):
        key = tuple(sorted(set(videos[indices])))
        if key not in models:
            emit('Fitting fresh net-gain heads on ' + ','.join(key))
            model = classifier.fit_model(X[indices], targets[indices], videos[indices], data['feature_names'])
            model.update({'fit_video_ids': list(key), 'fit_sample_count': len(indices), 'supervision': SUPERVISION,
                          'decision_rule': RULE, 'deployable': False, 'paper_final': False,
                          'independent_validation': False, 'upstream_fully_nested': False})
            models[key] = model
        return models[key]
    for outer in sorted(set(videos)):
        fit, held = ix[videos != outer], ix[videos == outer]
        inner_scores = np.full((len(fit), 2, 2), np.nan)
        partitions = []
        for inner in sorted(set(videos[fit])):
            train, valid = fit[videos[fit] != inner], fit[videos[fit] == inner]
            if outer in set(videos[train]) or inner in set(videos[train]):
                raise ValueError('video leakage in fitting indices')
            model = fit_model(train)
            inner_scores[videos[fit] == inner] = classifier.predict_scores(model, X[valid], data['feature_names'])
            partitions.append({'held': str(inner), 'fit': sorted(set(videos[train]))})
        selected = select_policy(inner_scores, data['actions_counts'][fit], data['action_costs'][fit], videos[fit])
        model = fit_model(fit)
        scores[held] = classifier.predict_scores(model, X[held], data['feature_names'])
        actions[held] = select_actions(scores[held], selected['thresholds'])
        save_model('outer_' + str(outer), {**model, 'held_video': str(outer), 'thresholds': selected['thresholds'],
                                         'selection_status': selected['status']})
        measured = reference.measure(data, actions[held], held)
        folds.append({'held_video': str(outer), 'fit_videos': sorted(set(videos[fit])), 'inner_partitions': partitions,
                      'inner_selection': selected, 'held_result': measured})
        emit(f'Held {outer}: {selected["status"]}; F1={measured["original"]["five_head_mean_f1"]:.4f}; actions={measured["actions"]}')
    result = reference.measure(data, actions)
    result.update({'supervision': SUPERVISION, 'decision_rule': RULE, 'folds': folds,
                   'quality_cost_acceptance': reference.acceptance(data, result),
                   'all_inner_feasible': all(x['inner_selection']['status'] == 'FEASIBLE' for x in folds),
                   'target_counts': {b: {name: int(targets[:, j, k].sum()) for k, name in enumerate(('help', 'harm'))}
                                     for j, b in enumerate(('interaction', 'phase'))}})
    result['conditional_development_pass'] = result['all_inner_feasible'] and result['quality_cost_acceptance']['passed']
    final_selection = select_policy(scores, data['actions_counts'], data['action_costs'], videos)
    final_model = {**fit_model(ix), 'thresholds': final_selection['thresholds'], 'selection_status': final_selection['status'],
                   'conditional_development_pass': result['conditional_development_pass'],
                   'purpose': 'Experimental research model; cannot promote or replace the selected default'}
    first_scores, first_actions = predict_actions(final_model, X, data['feature_names'])
    reloaded_scores, reloaded_actions = predict_actions(json.loads(json.dumps(final_model)), X, data['feature_names'])
    if not np.array_equal(first_actions, reloaded_actions) or not np.array_equal(first_scores, reloaded_scores):
        raise ValueError('serialized net model prediction drift')
    save_model('final_research_model', final_model)
    result['final_training_oof_selection'] = final_selection
    return result, scores, actions, targets
