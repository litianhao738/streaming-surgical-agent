"""Isolated Training-only comparison of repair admission and Claude cascades.

No existing artifact is overwritten. Never imports old executable probes.
run: .venv-p2/Scripts/python.exe -B -X utf8 tools/audit/gate_proposals_offline_20260913.py
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from surgical_agent.research.gate import official_net_training as old
from surgical_agent.research.verification.prior_panel import COMPONENTS
from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.data.constants import TASK_ID_BOUNDS

SOURCE = ROOT / 'artifacts/training/gate/official_behavior_net_v2_20260913'
RAW = ROOT / 'artifacts/training/gate/full_official_reviewers_20260912_v1/targets'
TASKS = old.TASKS
LABELS = [(t, c) for t in TASKS for c in range(TASK_ID_BOUNDS[t][1] + 1)]
ORDER = list(SEATS)
QORDER = ['qwen', *[s for s in ORDER if s != 'qwen']]
DEFAULT = ROOT / 'artifacts/research/gate_proposals_offline_20260913_r1'
VERSION = 'gate-proposals-comparison-20260913-v1'
CONFIGS = {
    'own_base_net': ('base', 'anchored', 'net', False),
    'own_qwen1_net': ('qwen1', 'anchored', 'net', False),
    'own_qwen2_net': ('qwen2', 'anchored', 'net', False),
    'claude1_change': ('base', 'full', 'change', False),
    'claude1_stream_base': ('base', 'full', 'change', True),
    'claude1_stream_temporal': ('temporal', 'full', 'change', True),
    'claude2_pool_only': ('pool', 'full', 'change', False),
    'claude2_qwen1_change': ('qwen1', 'full', 'change', False),
    'claude2_qwen2_change': ('qwen2', 'full', 'change', False),
    'claude2_qwen2_net_original': ('qwen2', 'full', 'net', False),
    'claude3_base_merged_net': ('base', 'full', 'merged_net', False),
    'claude3_qwen2_merged_net': ('qwen2', 'full', 'merged_net', False),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def write(path, obj):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


@contextmanager
def offline():
    def deny(*a, **kw):
        raise RuntimeError('Network and API forbidden in Training-only offline comparison')
    with patch('socket.socket.connect', deny), patch('socket.socket.connect_ex', deny), \
         patch('socket.create_connection', deny), patch('requests.sessions.Session.request', deny), \
         patch('urllib.request.urlopen', deny):
        yield


def anchored_admission(cheap, full):
    """Reject ONLY new unanchored verb/target additions; no GT or video id.

    Keep independent instruments, old cheap labels surviving the full selector,
    full IVTs and full phase. Never invent a relation or prune an old cheap label.
    """
    out = {t: sorted(full[t]) for t in TASKS}
    for t in ('verb', 'target'):
        anchors = {COMPONENTS[c][t] for c in full['ivt']}
        out[t] = sorted(set(full[t]) & (set(cheap[t]) | anchors))
    return out


def settled(scores, invalid, present, queried):
    if any(ORDER[i] in invalid for i in queried):
        return True
    subtotal = sum(scores[i] for i in queried)
    remaining = 5 - len(queried)
    return subtotal + remaining > 10 if present else subtotal + 5 * remaining < 20


def phase_settled(phases, current, queried):
    def void(p):
        return any(ORDER[i] in phases[p]['invalid'] for i in queried)
    if void(current):
        return True
    remaining = 5 - len(queried)
    lower = sum(phases[current]['scores'][i] for i in queried) + remaining
    for p in range(7):
        if p == current or void(p):
            continue
        upper = sum(phases[p]['scores'][i] for i in queried) + 5 * remaining
        if upper >= 20 and upper > lower:
            return False
    return True


def stop_plan(result, h0, order):
    indices = [ORDER.index(s) for s in order]
    props, diags = result['pool']['propositions'], result['diagnostics']
    phases = [result['joint_diagnostics'][f'phase_{p}'] for p in range(7)]
    kc = next((k for k in range(5) if all(settled(diags[p['id']]['scores'], diags[p['id']]['invalid'],
              p['label_id'] in h0[p['task']], indices[:k]) for p in props)), 5)
    kj = next((k for k in range(5) if phase_settled(phases, h0['phase'][0], indices[:k])), 5)
    return kc, kj


def qwen_features(result, cheap):
    """Only pool membership and Qwen's own scores/invalid bit, never panel means.

    Absent pool and invalid response have separate encodings. Full/GT deliberately
    absent from this signature. Joint features require paid phase recommendation.
    """
    index = ORDER.index('qwen')
    pool = {(p['task'], p['label_id']): p['id'] for p in result['pool']['propositions']}
    membership, compact, joint = [], [], []
    for t, c in LABELS:
        pid = pool.get((t, c))
        membership.extend([float(pid is not None), float(c in cheap[t])])
        d = result['diagnostics'].get(pid, {})
        scores = d.get('scores')
        valid = scores is not None and 'qwen' not in d.get('invalid', {}) and scores[index] is not None
        compact.extend([float(scores[index]) / 5 if valid else 0., float(valid)])
        jd = result['joint_diagnostics'].get(f'{t}_{c}', {})
        js = jd.get('scores')
        jvalid = js is not None and 'qwen' not in jd.get('invalid', {}) and js[index] is not None
        joint.extend([float(js[index]) / 5 if jvalid else 0., float(jvalid)])
    return membership, compact, joint


def cue(rows, i, last, changed):
    if last is None:
        return [0., 0., 0., 5., 10.]
    age = (rows[i]['frame_id'] - rows[last]['frame_id']) / 25.
    if age > 10:
        return [0., 0., 0., 5., 10.]
    diff = sum(set(rows[i]['cheap_labels'][t]) != set(rows[last]['cheap_labels'][t]) for t in TASKS)
    return [1., float(changed[last]), float(diff == 0), float(diff), age]


def counts_outputs(rows, outputs, merged=False):
    projected = [{'gt': r['gt'], 'mask': r['mask'], 'out': o} for r, o in zip(rows, outputs)]
    return old.counts(projected, 'out', merged)


def prepare(rows, out, source_bindings):
    n = len(rows)
    vids = np.array([r['video_id'] for r in rows])
    assert set(vids) == set(old.VIDEOS) and all(r['source_split'] == 'Training' for r in rows)
    assert all(r['features_postcheap'][k] == 0 for r in rows for k in old.FEATURE_ORDER if k.startswith('tracker_'))
    data = {'rows': rows, 'videos': vids}
    for name, field in (('cheap', 'cheap_labels'), ('full', 'final_labels'), ('h0', 'h0_labels')):
        data[name] = old.counts(rows, field)
        data[name + '_merged'] = old.counts(rows, field, True)
    outputs = [anchored_admission(r['cheap_labels'], r['final_labels']) for r in rows]
    data['anchored'] = counts_outputs(rows, outputs)
    data['anchored_merged'] = counts_outputs(rows, outputs, True)
    data['anchored_outputs'] = outputs
    data['changed'] = np.array([any(set(r['cheap_labels'][t]) != set(r['final_labels'][t]) for t in TASKS) for r in rows])
    X = np.array([[r['features_postcheap'][k] for k in old.FEATURE_ORDER] for r in rows], float)
    pool, q1, q2, stops, bindings = [], [], [], [], {}
    for i, r in enumerate(rows):
        path = (RAW / r['sample_id'] / 'result.json').resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source_bindings[str(path)]:
            raise ValueError('Unsealed original reviewer result: ' + str(path))
        result = json.loads(raw)
        assert result['h0'] == r['h0_labels']
        assert result['predictions']['gated_control_jointphase'] == r['final_labels']
        bindings[str(path)] = digest
        a, b, c = qwen_features(result, r['cheap_labels'])
        pool.append(a); q1.append(b); q2.append(c)
        plans = [stop_plan(result, r['h0_labels'], order) for order in (ORDER, QORDER)]
        for kc, kj in plans:
            if kc < 5:
                assert all(r['final_labels'][t] == r['cheap_labels'][t] for t in TASKS[:4])
            if kj < 5:
                assert r['final_labels']['phase'] == r['h0_labels']['phase']
        stops.append(plans)
        if (i + 1) % 1000 == 0:
            log(f'Verified cached evidence {i + 1}/{n}')
    data['X'] = {'base': X, 'pool': np.column_stack([X, pool]),
                 'qwen1': np.column_stack([X, pool, q1]), 'qwen2': np.column_stack([X, pool, q1, q2])}
    temporal = np.zeros((n, 5))
    data['order'] = {}
    for v in old.VIDEOS:
        ix = np.array(sorted(np.flatnonzero(vids == v), key=lambda i: rows[i]['frame_id']))
        data['order'][v] = ix
        last = None
        for i in ix:
            temporal[i] = cue(rows, i, last, data['changed'])
            last = i
    data['X']['temporal'] = np.column_stack([X, temporal])
    data['stops'] = np.array(stops)
    data['costs'] = make_costs(rows, data['stops'], read(SOURCE / 'costs.json')['per_call_main_attempt'])
    write(out / 'reviewer_source_sha256.json', bindings)
    np.savez_compressed(out / 'prepared_arrays.npz', **{k: v for k, v in data.items() if isinstance(v, np.ndarray)},
                        **{'X_' + k: v for k, v in data['X'].items()})
    return data


def make_costs(rows, stops, ledger):
    # Columns: logical calls, known USD, GLM requests, DS requests, reviewer calls.
    n = len(rows)
    def cost(keys):
        return np.array([len(keys), sum(ledger[k].get('usd_per_call', 0) for k in keys),
                         sum(k.endswith('|grok') for k in keys), sum(k.endswith('|deepseek') for k in keys),
                         sum(k.startswith(('control_graph|', 'joint_r1|')) for k in keys)], float)
    basenames = ['h0|base', 'proposal|base', 'phase_recommendation|base']
    full = cost(basenames + ['control_graph|' + s for s in ORDER] + ['joint_r1|' + s for s in ORDER])
    out = {'h0': np.tile(cost(['h0|base']), (n, 1)), 'full': np.tile(full, (n, 1))}
    for view in ('base', 'pool', 'qwen1', 'qwen2'):
        base, reached = [], []
        for i, r in enumerate(rows):
            keys = ['h0|base']
            if view != 'base' or r['features_postcheap']['proposal_rule']:
                keys.append('proposal|base')
            if view in ('qwen1', 'qwen2'):
                keys.append('control_graph|qwen')
            if view == 'qwen2':
                keys += ['phase_recommendation|base', 'joint_r1|qwen']
            base.append(cost(keys))
            which = int(view in ('qwen1', 'qwen2'))
            order = QORDER if which else ORDER
            kc, kj = stops[i, which]
            kc = max(kc, int(which)); kj = max(kj, int(view == 'qwen2'))
            reached.append(cost(basenames + ['control_graph|' + s for s in order[:kc]] + ['joint_r1|' + s for s in order[:kj]]))
        out[view] = (np.array(base), np.array(reached))
        assert np.all(out[view][1] >= out[view][0] - 1e-10)
    out['temporal'] = out['base']
    return out


def fit_predict(X, y, vids, fit, test, kind):
    assert not (set(vids[fit]) & set(vids[test]))
    vf = vids[fit]
    weights = np.array([len(fit) / (len(set(vf)) * np.sum(vf == v)) for v in vf])
    scaler = StandardScaler().fit(X[fit], sample_weight=weights)
    if kind == 'change':
        if len(np.unique(y[fit])) == 1:
            return np.full(len(test), float(y[fit][0])), None
        model = LogisticRegression(C=1., solver='liblinear', max_iter=2000, random_state=3407)
    else:
        model = Ridge(alpha=100., solver='cholesky')
    model.fit(scaler.transform(X[fit]), y[fit], sample_weight=weights)
    w = np.atleast_2d(model.coef_)[0] / scaler.scale_
    bias = float(np.asarray(model.intercept_).ravel()[0] - w @ scaler.mean_)
    values = X[test] @ w + bias
    if kind == 'change':
        values = 1 / (1 + np.exp(-np.clip(values, -700, 700)))
    return values, (w, bias)


def policy(data, ix, score, threshold, config, model=None):
    view, repair, target, streaming = config
    selected = score >= threshold
    source = np.where(selected, ix, -1)
    if streaming:
        for v in sorted(set(data['videos'][ix])):
            local = sorted(np.flatnonzero(data['videos'][ix] == v), key=lambda k: data['rows'][ix[k]]['frame_id'])
            last = None
            for k in local:
                i = ix[k]
                c = cue(data['rows'], i, last, data['changed'])
                s = score[k]
                if view == 'temporal' and model is not None:
                    w, bias = model
                    z = np.r_[data['X']['base'][i], c] @ w + bias
                    s = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
                review = last is None or c[4] >= 10 or s >= threshold
                selected[k] = review
                if review:
                    last = i; source[k] = i
                elif c[2] and c[4] <= 5:
                    source[k] = last
                else:
                    source[k] = -1
    return selected, source


def chosen_counts(data, ix, source, repair, merged=False):
    suffix = '_merged' if merged else ''
    result = data['cheap' + suffix][ix].copy()
    own = source == ix
    result[own] = data[repair + suffix][ix[own]]
    # Reused prediction MUST be scored against CURRENT GT, never copied TP/FP/FN.
    for k in np.flatnonzero((source >= 0) & ~own):
        i, previous = ix[k], source[k]
        pred = data['rows'][previous]['final_labels'] if repair == 'full' else data['anchored_outputs'][previous]
        result[k] = counts_outputs([data['rows'][i]], [pred], merged)[0]
    return result


def measure(data, ix, selected, source, config):
    view, repair, _, _ = config
    counts = chosen_counts(data, ix, source, repair)
    merged = chosen_counts(data, ix, source, repair, True)
    base, reached = data['costs'][view]
    costs = np.where(selected[:, None], reached[ix], base[ix])
    fidelity = []
    for i, previous in zip(ix, source):
        pred = data['rows'][i]['cheap_labels'] if previous < 0 else (
            data['rows'][previous]['final_labels'] if repair == 'full' else data['anchored_outputs'][previous])
        fidelity.append(all(set(pred[t]) == set(data['rows'][i]['final_labels'][t]) for t in TASKS))
    return {'original': old.pooled(counts), 'merged': old.pooled(merged), 'reviewed': int(selected.sum()),
            'logical_calls': int(costs[:, 0].sum()), 'known_usd': float(costs[:, 1].sum()),
            'glm_requests': int(costs[:, 2].sum()), 'deepseek_requests': int(costs[:, 3].sum()),
            'reviewer_calls': int(costs[:, 4].sum()), 'fidelity': float(np.mean(fidelity)),
            'by_video': {v: {'original': old.pooled(counts[data['videos'][ix] == v]),
                            'merged': old.pooled(merged[data['videos'][ix] == v])} for v in sorted(set(data['videos'][ix]))}}


def feasible(data, ix, result, objective, merged):
    suffix = '_merged' if merged else ''
    metric = 'merged' if merged else 'original'
    cheap, full = [old.pooled(data[k + suffix][ix]) for k in ('cheap', 'full')]
    if objective == 'fidelity95':
        return result['fidelity'] >= .95 - 1e-12
    r = result[metric]
    if objective == 'restore_f1':
        return r['five_head_mean_f1'] >= max(cheap['five_head_mean_f1'], full['five_head_mean_f1']) - 1e-10
    if r['five_head_mean_f1'] < max(cheap['five_head_mean_f1'], full['five_head_mean_f1']) - 1e-10:
        return False
    if r['total_errors'] > min(cheap['total_errors'], full['total_errors']):
        return False
    for v in set(data['videos'][ix]):
        vi = ix[data['videos'][ix] == v]
        c = old.pooled(data['cheap' + suffix][vi]); actual = result['by_video'][v][metric]
        if actual['five_head_mean_f1'] < c['five_head_mean_f1'] - 1e-10 or actual['total_errors'] > c['total_errors']:
            return False
    return True


def random_controls(data, route, config, draws=300):
    view, repair, _, streaming = config
    if streaming:
        return {'status': 'NOT_APPLICABLE: streaming reuse has action-dependent outputs; see separate causal random stream'}
    base, reached = data['costs'][view]
    increments = reached[:, 0] - base[:, 0]
    rng = np.random.default_rng(3407)
    allix = np.arange(len(route))
    raw = []
    for _ in range(draws):
        mask = np.zeros(len(route), bool)
        for v, ix in data['order'].items():
            budget = increments[ix][route[ix]].sum()
            blocks = [ix[j:j + 30] for j in range(0, len(ix), 30)]
            seq = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
            # Largest randomized block-prefix within same per-video logical budget.
            selected = seq[np.cumsum(increments[seq]) <= budget]
            mask[selected] = True
        cs = np.where(mask[:, None, None], data[repair], data['cheap'])
        costs = np.where(mask[:, None], reached, base).sum(0)
        m = old.pooled(cs)
        raw.append([m['five_head_mean_f1'], m['total_errors'], costs[0], costs[1]])
    arr = np.array(raw)
    return {'draws': draws, 'seed': 3407, 'block_targets': 30, 'budget': 'same per-video logical-call upper bound; whole prefix may underspend < max increment',
            'f1_mean': float(arr[:, 0].mean()), 'f1_p975': float(np.quantile(arr[:, 0], .975)),
            'errors_mean': float(arr[:, 1].mean()), 'errors_p025': float(np.quantile(arr[:, 1], .025)),
            'logical_calls_min_max': [float(arr[:, 2].min()), float(arr[:, 2].max())],
            'known_usd_min_max': [float(arr[:, 3].min()), float(arr[:, 3].max())]}


def evaluate_config(data, name, config, out):
    view, repair, target, streaming = config
    n = len(data['rows']); ixall = np.arange(n)
    X = data['X'][view]
    suffix = '_merged' if target == 'merged_net' else ''
    y = data['changed'].astype(float) if target == 'change' else (
        data['cheap' + suffix][:, :, 1:].sum((1, 2)) - data[repair + suffix][:, :, 1:].sum((1, 2))).astype(float)
    thresholds = [-1., *np.linspace(0, 1, 21).tolist(), 1.01] if target == 'change' else [-1e6, -1., -.5, -.25, -.1, -.05, 0., .025, .05, .1, .2, .4, .8, 1.6, 1e6]
    objectives = ['balanced', 'restore_f1'] + (['fidelity95'] if target == 'change' else [])
    predictions = np.zeros(n)
    routes = {k: np.zeros(n, bool) for k in objectives}
    sources = {k: np.full(n, -1, int) for k in objectives}
    folds = {k: [] for k in objectives}
    curves = {str(t): {'route': np.zeros(n, bool), 'source': np.full(n, -1, int)} for t in thresholds}
    for outer in old.VIDEOS:
        fit = ixall[data['videos'] != outer]; test = ixall[data['videos'] == outer]
        inner_policy = {t: (np.zeros(len(fit), bool), np.full(len(fit), -1, int)) for t in thresholds}
        for inner in sorted(set(data['videos'][fit])):
            train = fit[data['videos'][fit] != inner]; held = fit[data['videos'][fit] == inner]
            score, model = fit_predict(X, y, data['videos'], train, held, target)
            positions = np.flatnonzero(data['videos'][fit] == inner)
            for t in thresholds:
                a, b = policy(data, held, score, t, config, model)
                inner_policy[t][0][positions] = a; inner_policy[t][1][positions] = b
        candidates = {t: measure(data, fit, *inner_policy[t], config) for t in thresholds}
        score, model = fit_predict(X, y, data['videos'], fit, test, target)
        predictions[test] = score
        for t in thresholds:
            a, b = policy(data, test, score, t, config, model)
            curves[str(t)]['route'][test] = a; curves[str(t)]['source'][test] = b
        for objective in objectives:
            accepted = [t for t, m in candidates.items() if feasible(data, fit, m, objective, target == 'merged_net')]
            metric = 'merged' if target == 'merged_net' else 'original'
            if accepted:
                if objective == 'fidelity95':
                    chosen = min(accepted, key=lambda t: (candidates[t]['logical_calls'], -candidates[t]['fidelity'], -t))
                else:
                    chosen = min(accepted, key=lambda t: (candidates[t]['logical_calls'], -candidates[t][metric]['five_head_mean_f1'], candidates[t][metric]['total_errors'], -t))
            else:
                chosen = thresholds[-1]
            a, b = policy(data, test, score, chosen, config, model)
            routes[objective][test] = a; sources[objective][test] = b
            folds[objective].append({'held_out': outer, 'fit_videos': sorted(set(data['videos'][fit])),
                                     'feasible_candidates': len(accepted), 'threshold': chosen,
                                     'status': 'FEASIBLE' if accepted else 'INFEASIBLE_CONSERVATIVE_FALLBACK',
                                     'inner_selected': candidates[chosen], 'inner_candidates': {str(t): m for t, m in candidates.items()}})
        log(f'{name}: held out {outer}, fit only other videos')
    result = {'config': config, 'objectives': {}, 'fixed_threshold_oof_curves_diagnostic_only': {}}
    for objective in objectives:
        report = measure(data, ixall, routes[objective], sources[objective], config)
        report['folds'] = folds[objective]
        report['all_inner_feasible'] = all(f['status'] == 'FEASIBLE' for f in folds[objective])
        report['outer_objective_met'] = feasible(data, ixall, report, objective, target == 'merged_net')
        report['original_balanced_quality_met'] = feasible(data, ixall, report, 'balanced', False)
        report['random_block_control'] = random_controls(data, routes[objective], config)
        control = report['random_block_control']
        report['beats_block_random_both_metrics'] = (report['original']['five_head_mean_f1'] > control.get('f1_p975', 1e6)
                                                   and report['original']['total_errors'] < control.get('errors_p025', -1))
        report['development_passed'] = bool(report['all_inner_feasible'] and report['original_balanced_quality_met']
            and report['beats_block_random_both_metrics'] and report['reviewed'] > 0
            and report['logical_calls'] < 13 * n)
        result['objectives'][objective] = report
    for t, p in curves.items():
        result['fixed_threshold_oof_curves_diagnostic_only'][t] = measure(data, ixall, p['route'], p['source'], config)
    if target == 'change':
        result['change_auc'] = float(roc_auc_score(y, predictions))
        result['change_auc_by_video'] = {v: float(roc_auc_score(y[data['videos'] == v], predictions[data['videos'] == v])) for v in old.VIDEOS}
    else:
        result['net_gain_diagnostics'] = {v: {'predicted_mean': float(predictions[data['videos'] == v].mean()),
            'observed_mean': float(y[data['videos'] == v].mean())} for v in old.VIDEOS}
    write(out / (name + '.json'), result)
    np.savez_compressed(out / (name + '_routes.npz'), ids=np.array([r['sample_id'] for r in data['rows']]),
                        predictions=predictions, **{'route_' + k: v for k, v in routes.items()},
                        **{'source_' + k: v for k, v in sources.items()})
    return result


def diagnosis(rows):
    outputs = [anchored_admission(r['cheap_labels'], r['final_labels']) for r in rows]
    result = {}
    for v in ['ALL', *old.VIDEOS]:
        pairs = [(r, o) for r, o in zip(rows, outputs) if v == 'ALL' or r['video_id'] == v]
        selected = [r for r, _ in pairs]
        corrected = Counter(); corrupted = Counter(); removed_good = Counter(); removed_bad = Counter(); mixed = 0
        for r, o in pairs:
            good = bad = 0
            for t in TASKS:
                a, b, g = [set(r[k][t]) for k in ('cheap_labels', 'final_labels', 'gt')]
                helps = len((a - b) - g) + len((b - a) & g)
                hurts = len((b - a) - g) + len((a - b) & g)
                corrected[t] += helps; corrupted[t] += hurts; good += helps; bad += hurts
                removed = b - set(o[t]); removed_good[t] += len(removed & g); removed_bad[t] += len(removed - g)
            mixed += int(good > 0 and bad > 0)
        result[v] = {'rows': len(selected), 'mixed_good_bad_frames': mixed, 'corrected_bits': dict(corrected),
            'corrupted_bits': dict(corrupted), 'anchor_rejected_correct_additions': dict(removed_good),
            'anchor_rejected_wrong_additions': dict(removed_bad),
            'cheap': old.pooled(old.counts(selected, 'cheap_labels')), 'full': old.pooled(old.counts(selected, 'final_labels')),
            'anchored': old.pooled(counts_outputs(selected, [o for _, o in pairs])),
            'merged_cheap': old.pooled(old.counts(selected, 'cheap_labels', True)),
            'merged_full': old.pooled(old.counts(selected, 'final_labels', True)),
            'merged_anchored': old.pooled(counts_outputs(selected, [o for _, o in pairs], True))}
    return result


def prepare_free_sensitivity(rows):
    """No mixing initial reviewer answers with retry outcomes. No stop savings credited."""
    assert all(r['source_split'] == 'Training' and r['video_id'] in old.VIDEOS for r in rows)
    data = {'rows': rows, 'videos': np.array([r['video_id'] for r in rows]), 'order': {}}
    for name, field in (('cheap', 'cheap_labels'), ('full', 'final_labels')):
        data[name] = old.counts(rows, field)
        data[name + '_merged'] = old.counts(rows, field, True)
    outputs = [anchored_admission(r['cheap_labels'], r['final_labels']) for r in rows]
    data['anchored_outputs'] = outputs
    data['anchored'] = counts_outputs(rows, outputs)
    data['anchored_merged'] = counts_outputs(rows, outputs, True)
    data['changed'] = np.array([any(set(r['cheap_labels'][t]) != set(r['final_labels'][t]) for t in TASKS) for r in rows])
    data['X'] = {'base': np.array([[r['features_postcheap'][k] for k in old.FEATURE_ORDER] for r in rows], float)}
    for v in old.VIDEOS:
        data['order'][v] = np.array(sorted(np.flatnonzero(data['videos'] == v), key=lambda i: rows[i]['frame_id']))
    data['costs'] = make_costs(rows, np.full((len(rows), 2, 2), 5), read(SOURCE / 'costs.json')['per_call_main_attempt'])
    return data


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args(); out = args.output.resolve()
    if out.parent != DEFAULT.parent or not out.name.startswith('gate_proposals_offline_'):
        raise ValueError('Use a new direct child artifacts/research/gate_proposals_offline_*')
    out.mkdir(exist_ok=False)
    start = time.time()
    with offline():
        from scripts.train_gate_accuracy_value_v4 import source_bindings
        preserved = source_bindings()
        for directory in (SOURCE, ROOT / 'artifacts/training/gate/accuracy_value_v4_20260913_r1'):
            preserved.update({str(p.resolve()): sha(p) for p in directory.rglob('*') if p.is_file()})
        preserved.update({str(p.resolve()): sha(p) for p in (
            ROOT / 'tools/audit/exact_sequential_stop_probe.py', ROOT / 'tools/audit/when_to_verify_probe.py',
            ROOT / 'artifacts/research/exact_sequential_stop_probe_20260913.json',
            ROOT / 'artifacts/research/when_to_verify_probe_20260913.json')})
        recipe = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(), 'configs': CONFIGS,
            'primary_variant': 'own_qwen1_net', 'primary_objective': 'balanced',
            'primary_rows': 6059, 'outer': 'leave one Training video out', 'inner': 'leave one of other three videos out',
            'models': 'video-weighted standardized Ridge alpha=100 net error reduction; logistic C=1 natural class weights for change',
            'thresholds': 'fixed grids in code; minimum calls among inner-feasible policies; no held-out quantiles',
            'balanced': 'pooled F1>=max(cheap,full), errors<=min(cheap,full), every video F1>=cheap and errors<=cheap',
            'restore_f1': 'secondary: pooled F1>=max(cheap,full); DOES NOT ensure per-video precision or fewer errors',
            'fidelity95': 'Claude change target secondary: inner empirical agreement>=95%, not a conformal guarantee',
            'infeasible': 'maximum threshold; streaming still retains predeclared force/reuse. Mark infeasible, never claim success.',
            'own_repair': 'new verb/target additions require a surviving full IVT anchor; keep other frozen decisions; no GT/video id',
            'probe_cost': 'Qwen1 requires proposal; Qwen2 additionally phase recommendation. Count all base and reviewer calls, cached or not.',
            'exact_stop': 'fixed default order, or Qwen then remaining default order; replay no-change bounds; existing cached outputs verified',
            'temporal_variant': 'faithful Claude teacher-forced training; actual action-dependent causal histories at evaluation; mismatch disclosed',
            'sensitivity': '7053/7008 fixed-repair replay and free-feature own-net/Claude-merged-net Gates. No exact-stop savings credited on retries. Qwen outcomes only first-attempt 6059, never pair retry final with original opinions.',
            'limitations': ['4 videos, development only; earlier exploratory same-data analyses informed fixed repair',
                'upstream priors not fully nested across Gate fitting; no independent confirmation',
                'known USD excludes unpriced GLM/DS', 'exact stop preserves archived realizations; latency/new live stochastic outputs untested'],
            'api_calls': 0, 'Testing_access': False, 'VID110_access': False,
            'script_sha256': sha(__file__), 'old_artifacts_sha256': preserved}
        write(out / 'recipe.json', recipe)
        datasets = {d: read(SOURCE / (d + '_rows.json')) for d in ('primary', 'sensitivity', 'strict_complete')}
        write(out / 'diagnosis.json', {d: diagnosis(r) for d, r in datasets.items()})
        log('Diagnosis and fixed recipe saved before fitting')
        data = prepare(datasets['primary'], out, read(SOURCE / 'preparation_receipt.json')['source_sha256'])
        allix = np.arange(len(data['rows'])); allon = np.ones(len(allix), bool)
        summary = {'baselines': {}, 'variants': {}, 'api_calls': 0}
        for repair in ('full', 'anchored'):
            summary['baselines'][repair + '_exact_stop'] = measure(data, allix, allon, allix, ('base', repair, 'net', False))
            for view in ('qwen1', 'qwen2'):
                summary['baselines'][repair + '_' + view + '_always'] = measure(data, allix, allon, allix, (view, repair, 'net', False))
        for name in ('cheap', 'h0', 'full'):
            costs = data['costs']['base'][0] if name == 'cheap' else data['costs'][name]
            summary['baselines'][name] = {'original': old.pooled(data[name]), 'merged': old.pooled(data[name + '_merged']),
                'logical_calls': int(costs[:, 0].sum()), 'known_usd': float(costs[:, 1].sum())}
        for name, config in CONFIGS.items():
            log('Starting ' + name)
            r = evaluate_config(data, name, config, out)
            summary['variants'][name] = {k: {a: b for a, b in v.items() if a != 'folds'} for k, v in r['objectives'].items()}
            summary['variants'][name]['change_auc'] = r.get('change_auc')
        summary['sensitivity'] = {}
        for dataset in ('sensitivity', 'strict_complete'):
            folder = out / dataset; folder.mkdir()
            free = prepare_free_sensitivity(datasets[dataset])
            summary['sensitivity'][dataset] = {'reviewer_evidence': 'not read; original-attempt and retries never mixed',
                'cost': 'no exact stop; 13 logical calls for each escalated frame', 'variants': {}}
            for name in ('own_base_net', 'claude3_base_merged_net'):
                log(f'Starting {dataset}: {name}')
                r = evaluate_config(free, name, CONFIGS[name], folder)
                summary['sensitivity'][dataset]['variants'][name] = {k: {a: b for a, b in v.items() if a != 'folds'} for k, v in r['objectives'].items()}
        write(out / 'summary.json', summary)
        # Fixed source and code preservation check after every fit.
        for path, expected in preserved.items():
            if sha(path) != expected:
                raise ValueError('Existing artifact changed: ' + path)
        write(out / 'receipt.json', {'state': 'COMPLETED', 'elapsed_seconds': time.time() - start,
            'api_calls': 0, 'Testing_access': False, 'VID110_access': False, 'old_bindings_unchanged': len(preserved),
            'output_sha256': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()},
            'development_passed_variants': [name for name, x in summary['variants'].items() if x['balanced']['development_passed']]})
        log('COMPLETE: ' + str(out))


if __name__ == '__main__':
    main()
