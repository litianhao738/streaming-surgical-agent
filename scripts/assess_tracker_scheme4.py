"""Independent, network-disabled 6,059-row scheme-4 replay and Gate v2 assessment."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.run_pgp_pipeline import CachedBackend
from surgical_agent.research.gate.pgp_ambiguity import repair
from surgical_agent.research.gate.tracker_pipeline_v2 import (
    VERSION, VERB_RULES, CausalPhaseFilter, output_modules, run_interaction, current_classes,
)
from surgical_agent.research.gate.pgp_tracker_gemini38 import FrozenTracker
from surgical_agent.research.gate.final_only_training import canonical_labels
from tools.audit.gate_proposals_offline_20260913 import offline

TASKS = ('instrument', 'verb', 'target', 'ivt', 'phase')
BASE = ROOT / 'artifacts/training/gate'
ROWS = BASE / 'official_behavior_net_v2_20260913/primary_rows.json'
NPZ = BASE / 'pgp_ambiguity_assessment_20260913_r2/predictions.npz'
SOURCE = BASE / 'full_official_reviewers_20260912_v1'
TRACKER = BASE.parent / 'tracker_clip_v2_oof5_20260906/oof'
DEFAULT = ROOT / 'artifacts/research/tracker_scheme4_20260914_r2'
PROTOCOL = ROOT / 'docs/TRACKER_SCHEME_4_5_6_TRIAL_PROTOCOL_2026-09-14.md'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def counts(pred, truth):
    return np.array([(len(set(pred[t]) & set(truth[t])), len(set(pred[t]) - set(truth[t])),
                      len(set(truth[t]) - set(pred[t]))) for t in TASKS], dtype=np.int64)


def quality(c):
    sums = c.sum(axis=0)
    tp, fp, fn = sums.T
    den = 2 * tp + fp + fn
    f = np.divide(200 * tp, den, out=np.full(len(tp), 100.), where=den != 0)
    return {'f1': float(f.mean()), 'errors': int((fp + fn).sum()),
            'by_head': {t: {'f1': float(f[j]), 'tp': int(tp[j]), 'fp': int(fp[j]), 'fn': int(fn[j])}
                        for j, t in enumerate(TASKS[:len(tp)])}}


def load_trackers(videos):
    index = read(TRACKER / 'index.json')
    result, hashes = {}, {}
    for v in videos:
        rel = index['video_to_artifact'][v]
        path = TRACKER / rel
        digest = sha(path)
        assert digest == index['artifacts'][rel]
        hashes[str(path)] = digest
        frames = read(path)['videos'][v]
        assert frames['source_split'].lower() == 'training'
        result[v] = {int(f['frame_id']): {int(t['instrument_id']) for t in f['tracks'] if t['score'] >= .35}
                     for f in frames['frames']}
    return result, hashes


def assess(out):
    out.mkdir(exist_ok=False)
    (out / 'models').mkdir()
    start = time.perf_counter()
    paths = [ROWS, NPZ, TRACKER / 'index.json', PROTOCOL, Path(__file__),
             ROOT / 'src/surgical_agent/research/gate/tracker_pipeline_v2.py',
             ROOT / 'DEFAULT_PIPELINE_VERSION.json', ROOT / 'DEFAULT_PGP_GATE_VERSION.json']
    bindings = {str(p): sha(p) for p in paths}
    write(out / 'protocol_receipt.json', {'source_sha256': bindings, 'api_calls': 0, 'scope': 'Training-only development replay'})
    rows = read(ROWS)
    assert len(rows) == 6059 and all(r['source_split'] == 'Training' and all(r['mask'].values()) for r in rows)
    ids = np.array([r['sample_id'] for r in rows]); vid = np.array([r['video_id'] for r in rows])
    fid = np.array([r['frame_id'] for r in rows]); videos = sorted(set(vid)); n = len(rows)
    assert set(videos) == {'VID103', 'VID23', 'VID31', 'VID96'}
    with np.load(NPZ, allow_pickle=True) as z:
        pos = {s: i for i, s in enumerate(z['ids'].tolist())}
        order = np.array([pos[s] for s in ids])
        data = {k: z[k][order] for k in ('X', 'cheap', 'ambiguity', 'costs', 'depths', 'qwen_hgb_route')}
    old_route = data['qwen_hgb_route'].astype(bool)
    model = read(BASE / 'pgp_ambiguity_assessment_20260913_r2/models/final_research_model.json')
    feature_names = model['feature_names']; X = data['X']
    trackers, tracker_hashes = load_trackers(videos)
    tracker_provider = FrozenTracker(TRACKER / 'index.json')
    inventory = {s['key']: s for s in read(SOURCE / 'plan.json')['selection']}
    sealed = read(BASE / 'pgp_gate_no_tracker_20260913_r1/replay/pgp_reviewer_source_sha256.json')
    priors = {v: read(SOURCE / 'priors' / (v + '.json')) for v in videos}
    cheap, full, logs = [], [], []
    call_all = np.zeros(n, int)
    print('Replaying original compact decisions, features and paths', flush=True)
    with offline():
        for i, row in enumerate(rows):
            path = SOURCE / 'targets' / row['sample_id'] / 'result.json'
            assert sha(path) == sealed[str(path.resolve())]
            frame = row['frame_id']
            selected = {'key': row['sample_id'], 'video_id': row['video_id'], 'frame_id': frame,
                        'causal_frame_ids': [frame - 50, frame - 25, frame], 'source_split': 'Training'}
            selected.update({k: inventory[row['sample_id']][k] for k in ('images', 'alignment_version')})
            actual_classes = current_classes(tracker_provider.snapshot(selected), selected)
            assert actual_classes == trackers[row['video_id']].get(frame), ('Tracker adapter mismatch', row['sample_id'])
            replay = run_interaction(CachedBackend(read(path)), selected, priors[row['video_id']], lambda _: (1., 1))
            expected = repair(row['cheap_labels'], row['final_labels'])
            expected['phase'] = row['cheap_labels']['phase']
            assert replay['prediction'] == canonical_labels(expected), ('prediction', row['sample_id'])
            assert replay['cheap'] == row['cheap_labels']
            assert np.array_equal([replay['features'][k] for k in feature_names], X[i])
            assert all(k.startswith(('h0|', 'proposal|', 'control_graph|')) for k in replay['call_keys'])
            assert replay['logical_calls'] == 3 + max(int(data['depths'][i, 0]) - 1, 0)
            call_all[i] = replay['logical_calls']
            cheap.append(canonical_labels(replay['cheap'])); full.append(replay['prediction'])
            logs.append({'key': row['sample_id'], 'call_keys_review': replay['call_keys']})
            if (i + 1) % 500 == 0:
                print(f'Runtime parity {i + 1}/{n}', flush=True)

    CC = np.array([counts(p, r['gt']) for p, r in zip(cheap, rows)])
    CR = np.array([counts(p, r['gt']) for p, r in zip(full, rows)])
    assert np.array_equal(CC, data['cheap'])
    assert np.array_equal(CR[:, :4], data['ambiguity'][:, :4])
    baseline = np.where(old_route[:, None, None], data['ambiguity'], CC)
    assert abs(quality(baseline)['f1'] - 61.69890226009785) < 1e-5
    assert quality(baseline)['errors'] == 36150

    FC, FR, fallback = [], [], Counter()
    edit_counts = Counter()
    for i, row in enumerate(rows):
        ts = trackers[row['video_id']].get(row['frame_id'])
        pc, lc = output_modules(cheap[i], ts)
        pr, lr = output_modules(full[i], ts)
        FC.append(pc); FR.append(pr)
        selected_log = lr if old_route[i] else lc
        for key in ('m1_capacity_fallback', 'm2_capacity_fallback'):
            fallback[key] += int(selected_log[key])
        fallback['tracker_unavailable'] += int(ts is None)
        deleted = set(selected_log['deleted_ivts']); truth = set(row['gt']['ivt'])
        edit_counts['deleted_correct_ivt'] += len(deleted & truth)
        edit_counts['deleted_incorrect_ivt'] += len(deleted - truth)
    CC_T = np.array([counts(p, r['gt']) for p, r in zip(FC, rows)])
    CR_T = np.array([counts(p, r['gt']) for p, r in zip(FR, rows)])
    y = np.array([any(a[t] != b[t] for t in TASKS[:4]) for a, b in zip(FC, FR)], int)

    phase_predictions, phase_counts = {}, {}
    for window in (0, 5, 10, 20, 30, 60):
        filt = CausalPhaseFilter(window)
        pred = np.zeros(n, int)
        for i in np.lexsort((fid, vid)):
            pred[i] = filt.apply(str(vid[i]), int(fid[i]), cheap[i]['phase'][0])
        c = CC[:, 4:].copy()
        for i in range(n):
            ok = int(pred[i] == rows[i]['gt']['phase'][0]); c[i, 0] = (ok, 1 - ok, 1 - ok)
        phase_predictions[window], phase_counts[window] = pred, c
    window_choice, nested_phase, nested_phase_pred = {}, CC[:, 4:].copy(), np.zeros(n, int)
    for v in videos:
        fit = vid != v; held = vid == v
        best = max(phase_counts, key=lambda w: (quality(phase_counts[w][fit])['f1'], -w))
        window_choice[v] = best
        nested_phase[held] = phase_counts[best][held]; nested_phase_pred[held] = phase_predictions[best][held]
    final_window = max(phase_counts, key=lambda w: (quality(phase_counts[w])['f1'], -w))
    print('Output modules replayed; fitting Gate v2', flush=True)

    phase_usd = np.array([.003483 + sum([.000390, .002854, .004462, 0., 0.][:int(k)]) for k in data['depths'][:, 1]])
    skip_usd, review_usd = data['costs'][:, 1, 0, 1], data['costs'][:, 1, 1, 1] - phase_usd

    def choose(scores, ix):
        no, yes = CC_T[ix, :4], CR_T[ix, :4]
        q0, q1 = quality(no), quality(yes)
        candidates = []
        for frac in np.arange(.05, 1.0001, .05):
            threshold = float(np.quantile(scores[ix], 1 - frac))
            route = scores[ix] >= threshold
            c = np.where(route[:, None, None], yes, no); q = quality(c)
            per = all(quality(c[vid[ix] == v])['f1'] + 1e-12 >= quality(no[vid[ix] == v])['f1'] and
                      quality(c[vid[ix] == v])['errors'] <= quality(no[vid[ix] == v])['errors'] for v in set(vid[ix]))
            good = per and q['errors'] <= q0['errors'] and q['f1'] - q0['f1'] + 1e-12 >= .9 * max(0., q1['f1'] - q0['f1'])
            candidates.append({'fraction': round(float(frac), 2), 'threshold': threshold, 'feasible': bool(good),
                               'calls': int(np.where(route, call_all[ix], 3).sum())})
        feasible = [c for c in candidates if c['feasible']]
        selected = min(feasible, key=lambda c: (c['calls'], c['fraction'])) if feasible else {'threshold': 1.01, 'feasible': False}
        return {'selected': selected, 'candidates': candidates}

    cache = {}
    def fit(indices):
        key = tuple(sorted(set(vid[indices])))
        if key not in cache:
            pos = y[indices].mean()
            if pos in (0, 1):
                raise ValueError('constant fit labels require explicit constant-model handling')
            weights = np.where(y[indices], .5 / pos, .5 / (1 - pos))
            cache[key] = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15, random_state=3407).fit(X[indices], y[indices], sample_weight=weights)
        return cache[key]

    route_v2 = np.zeros(n, bool); oof = np.zeros(n); folds = {}
    for v in videos:
        train = np.where(vid != v)[0]; held = np.where(vid == v)[0]; inner = np.zeros(n)
        for u in videos:
            if u == v:
                continue
            tr = np.where((vid != v) & (vid != u))[0]; va = np.where(vid == u)[0]
            inner[va] = fit(tr).predict_proba(X[va])[:, 1]
        selection = choose(inner, train)
        estimator = fit(train); oof[held] = estimator.predict_proba(X[held])[:, 1]
        route_v2[held] = oof[held] >= selection['selected']['threshold']
        folds[v] = selection
        joblib.dump(estimator, out / 'models' / ('outer_' + v + '.joblib'))
        print('Gate outer fold ' + v, flush=True)
    calibration = choose(oof, np.arange(n))
    estimator = fit(np.arange(n)); estimator_path = out / 'models/final_estimator.joblib'
    joblib.dump(estimator, estimator_path)
    scores_final = estimator.predict_proba(X)[:, 1]
    assert np.array_equal(scores_final, joblib.load(estimator_path).predict_proba(X)[:, 1])
    fullfit_route = scores_final >= calibration['selected']['threshold']
    write(out / 'models/final_model.json', {'version': VERSION, 'feature_names': feature_names,
          'estimator_file': estimator_path.name, 'estimator_sha256': sha(estimator_path),
          'threshold': calibration['selected']['threshold'], 'selection_feasible': calibration['selected']['feasible'],
          'action_rule': '1 if score >= threshold else 0', 'phase_window_seconds': int(final_window),
          'deployable': False, 'scope': 'Training research; full-fit calibration transfer requires separate validation'})

    def measure(route, *, tracker=True, phase=nested_phase):
        c = np.where(route[:, None, None], CR_T if tracker else CR, CC_T if tracker else CC).copy()
        c[:, 4:] = phase
        q = quality(c)
        q.update(calls=int(np.where(route, call_all, 3).sum()), known_usd_estimate=float(np.where(route, review_usd, skip_usd).sum()),
                 reviewed=int(route.sum()), per_video={v: quality(c[vid == v]) for v in videos})
        return q

    none, allr = np.zeros(n, bool), np.ones(n, bool)
    variants = {'original_pgp': {**quality(baseline), 'calls': int(data['costs'][np.arange(n), 1, old_route.astype(int), 0].sum()),
                               'per_video': {v: quality(baseline[vid == v]) for v in videos}},
                'm1_m4_old_gate_outer': measure(old_route), 'm1_m5_gate_v2_outer': measure(route_v2),
                'same_probe_always_review': measure(allr), 'same_probe_no_review': measure(none),
                'gate_v2_fullfit_training_replay_fixed_window': measure(fullfit_route, phase=phase_counts[final_window])}
    no, yes = variants['same_probe_no_review'], variants['same_probe_always_review']
    for name in ('m1_m4_old_gate_outer', 'm1_m5_gate_v2_outer'):
        q = variants[name]
        q['f1_gain_retention'] = (q['f1'] - no['f1']) / (yes['f1'] - no['f1']) if yes['f1'] != no['f1'] else None
        q['error_reduction_retention'] = (no['errors'] - q['errors']) / (no['errors'] - yes['errors']) if no['errors'] != yes['errors'] else None
    ablation = {}
    for tracker in (False, True):
        for pgp in (False, True):
            for smooth in (False, True):
                q = measure(old_route if pgp else none, tracker=tracker, phase=nested_phase if smooth else phase_counts[0])
                if not pgp:
                    q['calls'] = int(data['costs'][:, 0, 0, 0].sum())
                    q['known_usd_estimate'] = float(data['costs'][:, 0, 0, 1].sum())
                ablation[f'T{int(tracker)}_PGP{int(pgp)}_Phase{int(smooth)}'] = q

    latency = {}
    for name, pp in (('raw_cheap', phase_predictions[0]), ('nested_filter', nested_phase_pred), ('fixed_window_filter', phase_predictions[final_window])):
        delays, gaps, missed = [], [], 0
        for v in videos:
            seq = np.where(vid == v)[0]; seq = seq[np.argsort(fid[seq])]
            truth = [rows[i]['gt']['phase'][0] for i in seq]
            transitions = [j for j in range(1, len(seq)) if truth[j] != truth[j - 1]]
            for j in transitions:
                end = next((k for k in range(j + 1, len(seq)) if truth[k] != truth[j]), len(seq))
                hit = next((k for k in range(j, end) if pp[seq[k]] == truth[j]), None)
                gaps.append(float((fid[seq[j]] - fid[seq[j - 1]]) / 25))
                if hit is None:
                    missed += 1
                else:
                    delays.append(float((fid[seq[hit]] - fid[seq[j]]) / 25))
        latency[name] = {'observed_transitions': len(gaps), 'matched_segments': len(delays), 'unmatched_segments': missed,
                         'median_seconds_from_first_observed_new_gt': float(np.median(delays)) if delays else None,
                         'p90_seconds_from_first_observed_new_gt': float(np.percentile(delays, 90)) if delays else None,
                         'max_boundary_observation_gap_seconds': max(gaps, default=None)}
    with (out / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for i, row in enumerate(rows):
            predictions = {}
            for name, route, phase in (('old_gate_outer', old_route, nested_phase_pred), ('v2_outer', route_v2, nested_phase_pred),
                                        ('v2_fullfit_training', fullfit_route, phase_predictions[final_window])):
                p = dict(FR[i] if route[i] else FC[i]); p['phase'] = [int(phase[i])]
                predictions[name] = canonical_labels(p)
            stream.write(json.dumps({'key': row['sample_id'], 'video_id': row['video_id'], 'frame_id': row['frame_id'],
                         'predictions': predictions, 'old_gate_route': bool(old_route[i]), 'v2_outer_route': bool(route_v2[i]),
                         'v2_fullfit_route': bool(fullfit_route[i]), 'review_call_keys': logs[i]['call_keys_review']}) + '\n')
    np.savez_compressed(out / 'predictions.npz', ids=ids, videos=vid, X=X, labels=y, old_route=old_route,
                        v2_outer_route=route_v2, oof_scores=oof, fullfit_scores=scores_final,
                        fullfit_route=fullfit_route, phase_nested=nested_phase_pred, phase_fixed=phase_predictions[final_window])
    report = {'version': VERSION, 'rows': n, 'api_calls': 0, 'default_changed': False,
              'runtime_parity_rows': n, 'prediction_feature_call_mismatches': 0,
              'real_frozen_tracker_adapter_parity_rows': n,
              'scope': 'Reused four Training development videos; fixed upstream not fully nested; no independent validation.',
              'fallbacks_old_gate': dict(fallback), 'ivt_deletions_old_gate': dict(edit_counts),
              'phase': {'outer_choices': window_choice, 'final_fit_window_seconds': int(final_window),
                        'nested_f1': quality(nested_phase)['f1'],
                        'fixed_window_f1': {str(w): quality(c)['f1'] for w, c in phase_counts.items()},
                        'latency_observation_only': latency,
                        'latency_limit': 'Sparse cached observations; true transition lies within preceding observation gap; these are not dense-stream deployment latency estimates.'},
              'gate': {'changed_frames': int(y.sum()), 'oof_auc': float(roc_auc_score(y, oof)), 'outer_folds': folds, 'final_calibration': calibration},
              'variants': variants, 'ablation_fixed_old_gate_routes': ablation,
              'source_sha256': bindings, 'tracker_sha256': tracker_hashes,
              'predictions_sha256': sha(out / 'predictions.jsonl'), 'elapsed_seconds': time.perf_counter() - start}
    for p, digest in bindings.items():
        assert sha(p) == digest, 'frozen source changed during assessment: ' + p
    write(out / 'report.json', report)
    print(json.dumps({'output': str(out), 'variants': {k: {a: q[a] for a in ('f1', 'errors', 'calls')} for k, q in variants.items()},
                      'fallbacks': dict(fallback), 'phase_choices': window_choice}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    with offline():
        assess(args.output)
