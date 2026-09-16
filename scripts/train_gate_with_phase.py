"""Offline five-head Gate training from existing Training caches; no API calls."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.assess_tracker_scheme4 import ROWS, DEFAULT, TASKS, load_trackers, counts, quality
from surgical_agent.research.gate.pgp_ambiguity import repair
from surgical_agent.research.gate.tracker_pipeline_v2 import output_modules
from tools.audit.gate_proposals_offline_20260913 import offline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--unified-cache', type=Path, help='Sealed shared-panel replay directory from build_unified_gate_cache.py or a completed five_head_probe collection')
    parser.add_argument('--drop-phase-probe-features', action='store_true',
                        help='Ablation on a five_head_probe cache: fit the 42 historical features only, same labels and protocol')
    parser.add_argument('--label', choices=('change', 'harm'), default='change',
                        help='change: any head differs after review (default). harm: review strictly reduces fp+fn over the five heads (GT-derived training label only)')
    parser.add_argument('--selection-rule', choices=('strict', 'overall'), default='strict',
                        help='strict: every video must not regress vs skip_review (default). overall: only pooled errors and 90%% of the full-review gain are required')
    args = parser.parse_args()
    if args.drop_phase_probe_features and not args.unified_cache:
        parser.error('--drop-phase-probe-features requires a five_head_probe --unified-cache')
    started = time.perf_counter()
    rows = json.loads(ROWS.read_text('utf-8'))
    assert rows and all(r['source_split'] == 'Training' and all(r['mask'].values()) for r in rows)
    ids = np.array([r['sample_id'] for r in rows])
    videos = np.array([r['video_id'] for r in rows])
    frames = np.array([r['frame_id'] for r in rows])
    unique = sorted(set(videos)); n = len(rows)
    assert len(set(ids)) == n and len(unique) >= 4
    with np.load(DEFAULT / 'predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(z['ids'], ids)
        X = z['X'].copy()
    manifest = json.loads((DEFAULT / 'models/final_model.json').read_text('utf-8'))
    names = manifest['feature_names']
    assert X.shape == (n, len(names)) and np.isfinite(X).all()
    trackers, tracker_hashes = load_trackers(unique)
    cheap, reviewed = [], []
    cache_mode = 'unified'
    if args.unified_cache:
        receipt = json.loads((args.unified_cache/'receipt.json').read_text('utf-8'))
        path = args.unified_cache/'rows.jsonl'
        cache_mode = receipt['review_mode']
        assert receipt['state'] == 'PASS' and cache_mode in ('unified', 'five_head_probe') and receipt['rows'] == n
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['rows_sha256']
        cached = [json.loads(line) for line in path.read_text('utf-8').splitlines()]
        assert [r['sample_id'] for r in cached] == ids.tolist()
        assert all(r['source_split'] == 'Training' for r in cached)
        if cache_mode == 'five_head_probe':
            from surgical_agent.research.gate.unified_review import PHASE_PROBE_NAMES
            base_names = [k for k in names if not k.startswith('qwen_')]
            assert np.array_equal(X[:, [names.index(k) for k in base_names]],
                np.array([[r['features'][k] for k in base_names] for r in cached]))
            assert all(k in r['features'] for r in cached for k in PHASE_PROBE_NAMES)
            if not args.drop_phase_probe_features:
                names = names + list(PHASE_PROBE_NAMES)
            X = np.array([[r['features'][k] for k in names] for r in cached], dtype=float)
            assert X.shape == (n, 42 if args.drop_phase_probe_features else 54) and np.isfinite(X).all()
        else:
            if args.drop_phase_probe_features:
                parser.error('--drop-phase-probe-features requires a five_head_probe cache')
            assert np.array_equal(X, np.array([[r['features'][k] for k in names] for r in cached]))
        cheap = [r['cheap'] for r in cached]; reviewed = [r['reviewed'] for r in cached]
        review_calls = np.array([r['logical_calls'] for r in cached])
    else:
        for row in rows:
            ts = trackers[row['video_id']].get(row['frame_id'])
            a, _ = output_modules(row['cheap_labels'], ts)
            b, _ = output_modules(repair(row['cheap_labels'], row['final_labels']), ts)
            a['phase'] = row['cheap_labels']['phase']
            b['phase'] = row['final_labels']['phase']
            cheap.append(a); reviewed.append(b)
    cc = np.array([counts(p, r['gt']) for p, r in zip(cheap, rows)])
    cr = np.array([counts(p, r['gt']) for p, r in zip(reviewed, rows)])
    # Preserve the existing change-prediction objective, extended to phase.
    # GT is used for offline threshold evaluation, never an inference feature.
    y = np.array([any(a[t] != b[t] for t in TASKS) for a, b in zip(cheap, reviewed)], int)
    if args.label == 'harm':
        y = (cr[:, :, 1:].sum(axis=(1, 2)) < cc[:, :, 1:].sum(axis=(1, 2))).astype(int)
    raw0 = np.array([p['phase'][0] for p in cheap])
    raw1 = np.array([p['phase'][0] for p in reviewed])
    truth = np.array([r['gt']['phase'][0] for r in rows])
    sequences = [np.where(videos == v)[0][np.argsort(frames[videos == v])] for v in unique]
    lefts = [np.searchsorted(frames[ix], frames[ix] - 60 * 25, side='right') for ix in sequences]

    def evaluate(route):
        result = np.where(route[:, None, None], cr, cc).copy()
        raw = np.where(route, raw1, raw0)
        smooth = raw.copy()
        for ix, left in zip(sequences, lefts):
            votes = np.eye(7, dtype=int)[raw[ix]]
            cumulative = np.vstack([np.zeros((1, 7), int), votes.cumsum(axis=0)])
            window = cumulative[np.arange(len(ix)) + 1] - cumulative[left]
            tied = window == window.max(axis=1)[:, None]
            smooth[ix] = np.where(tied.sum(axis=1) == 1, window.argmax(axis=1), raw[ix])
        ok = smooth == truth
        result[:, 4] = np.column_stack([ok, ~ok, ~ok])
        return result

    baseline = evaluate(np.zeros(n, bool))
    full = evaluate(np.ones(n, bool))
    assert abs(quality(baseline[:, 4:])['f1'] - 85.34411619079056) < 1e-8
    if not args.unified_cache:
        assert abs(quality(full[:, 4:])['f1'] - 85.50915992738075) < 1e-8
    print(f'Cache checks passed: {n} frames, {len(unique)} videos, {len(names)} features; API calls: 0', flush=True)
    if args.check_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cache = {}

    def fit(ix):
        key = tuple(sorted(set(videos[ix])))
        if key not in cache:
            fraction = y[ix].mean()
            if fraction in (0, 1):
                raise ValueError('Training fold has constant labels')
            weights = np.where(y[ix], .5 / fraction, .5 / (1 - fraction))
            cache[key] = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05,
                max_leaf_nodes=15, random_state=3407).fit(X[ix], y[ix], sample_weight=weights)
        return cache[key]

    def choose(scores, ix):
        q0, q1 = quality(baseline[ix]), quality(full[ix])
        candidates = []
        for frac in np.arange(.05, 1.0001, .05):
            threshold = float(np.quantile(scores[ix], 1 - frac))
            route = scores >= threshold
            c = evaluate(route)
            q = quality(c[ix])
            per = all(quality(c[j])['f1'] + 1e-12 >= quality(baseline[j])['f1'] and
                      quality(c[j])['errors'] <= quality(baseline[j])['errors']
                      for v in set(videos[ix]) for j in [ix[videos[ix] == v]])
            feasible = (per or args.selection_rule == 'overall') and q['errors'] <= q0['errors'] and q['f1'] - q0['f1'] + 1e-12 >= .9 * max(0., q1['f1'] - q0['f1'])
            candidates.append({'threshold': threshold, 'fraction': float(frac), 'feasible': bool(feasible),
                               'review_frames': int(route[ix].sum()), 'metrics': q})
        valid = [c for c in candidates if c['feasible']]
        selected = min(valid, key=lambda c: (c['review_frames'], c['fraction'])) if valid else {
            'threshold': 1.01, 'feasible': False, 'review_frames': 0}
        return {'selected': selected, 'candidates': candidates}

    oof = np.zeros(n); routed = np.zeros(n, bool); folds = {}
    for v in unique:
        train = np.where(videos != v)[0]; held = np.where(videos == v)[0]
        inner = np.zeros(n)
        for u in unique:
            if u == v:
                continue
            tr = np.where((videos != v) & (videos != u))[0]
            val = np.where(videos == u)[0]
            inner[val] = fit(tr).predict_proba(X[val])[:, 1]
        selection = choose(inner, train)
        oof[held] = fit(train).predict_proba(X[held])[:, 1]
        routed[held] = oof[held] >= selection['selected']['threshold']
        folds[v] = selection
        print(f'Completed video fold: {v}', flush=True)
    calibration = choose(oof, np.arange(n))
    estimator = fit(np.arange(n))
    model_path = args.output / 'estimator.joblib'
    joblib.dump(estimator, model_path)
    assert np.array_equal(estimator.predict_proba(X), joblib.load(model_path).predict_proba(X))
    model = {'version': 'five-head-unified-gate-research-v1' if args.unified_cache else 'five-head-phase-gate-research-v1', 'feature_names': names,
             'estimator_file': model_path.name, 'estimator_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
             'threshold': calibration['selected']['threshold'], 'selection_feasible': calibration['selected']['feasible'],
             'action_rule': '1 if score >= threshold else 0', 'phase_window_seconds': 60,
             'label_definition': ('Review strictly reduces fp+fn summed over the five heads (GT-derived training label; inference uses features only)'
                                  if args.label == 'harm' else 'Any of five heads changes after historical review and tracker M1/M2, before smoothing'),
             'label': args.label, 'selection_rule': args.selection_rule,
             'selection_rule_definition': ('Pooled errors <= skip_review and pooled gain >= 90% of full-review gain; per-video non-regression NOT required'
                                           if args.selection_rule == 'overall' else 'Every video non-regressing vs skip_review, pooled errors <= skip_review, pooled gain >= 90% of full-review gain'),
             'deployable': False, 'runtime_requirement': 'Route phase review together with interaction review, then smooth routed phase stream',
             'scope': 'Reused Training development videos; fixed upstream not fully nested; runtime integration and independent validation required'}
    if args.unified_cache:
        model.update(review_mode=cache_mode, output_modules='v2.2',
            runtime_requirement='One post-Gate joint panel for all five heads, then v2.2 Tracker output modules and routed phase smoothing',
            training_cache_sha256=receipt['rows_sha256'])
        if cache_mode == 'five_head_probe':
            model.update(version='five-head-probe-gate-research-v1', feature_set='historical_42_plus_phase_probe_12',
                runtime_requirement='Five-head Qwen probe -> 54-feature Gate -> remaining seats on same pool; no phase recommendation or second Qwen call')
            if args.drop_phase_probe_features:
                model.update(version='five-head-probe-gate-base42-ablation-v1', feature_set='historical_42_only',
                    ablation='Same five_head_probe cache and labels without the 12 phase probe features; isolates their contribution from the protocol change',
                    runtime_requirement='Five-head Qwen probe -> 42-feature Gate (phase probe ratings ignored) -> remaining seats on same pool')
    report = {'rows': n, 'videos': unique, 'api_calls': 0, 'default_changed': False,
              'feature_count': len(names), 'feature_set': model.get('feature_set', 'historical_42'),
              'label': args.label, 'selection_rule': args.selection_rule,
              'label_positives': int(y.sum()), 'outer_folds': folds, 'calibration': calibration,
              'variants': {'skip_review': quality(baseline), 'all_review': quality(full),
                           'nested_gate': quality(evaluate(routed))},
              'nested_review_frames': int(routed.sum()), 'elapsed_seconds': time.perf_counter() - started,
              'tracker_sha256': tracker_hashes,
              'sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [ROWS, DEFAULT / 'predictions.npz', DEFAULT / 'models/final_model.json', Path(__file__)]},
              'limitations': model['scope'], 'selection_objective': 'Fewest reviewed frames meeting five-head quality constraints; not monetary cost optimization'}
    if args.unified_cache:
        report.update(unified_cache=str(args.unified_cache), unified_cache_sha256=receipt['rows_sha256'],
            nested_logical_calls=int(np.where(routed, review_calls, 3).sum()),
            all_review_logical_calls=int(review_calls.sum()), output_modules='v2.2')
    for filename, data in [('model.json', model), ('report.json', report)]:
        (args.output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(args.output / 'predictions.npz', ids=ids, labels=y, oof_scores=oof, nested_route=routed)
    print(json.dumps({'output': str(args.output), 'variants': report['variants'],
                      'selection_feasible': model['selection_feasible']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with offline():
        main()
