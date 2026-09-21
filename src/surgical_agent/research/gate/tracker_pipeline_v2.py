"""Independent Training research runtime: interaction-only PGP and local output modules.

No network or GT is accessed here. The caller owns the backend, frozen predictions,
Gate model, and per-stream phase state. The existing PGP runtime stays unchanged.

Output-module versions (routing and the Gate are identical across them):
- v2.1: M1 Tracker instrument fusion + IVT filter, M2 verb rules, M3 phase filter.
- v2.2: v2.1 plus an ontology filter (only instruments that occur in some IVT component are
  output; the Tracker's specimen_bag is a triplet target, never an instrument) and null-label
  cleanup (a null IVT the Qwen probe validly rated 1 is removed; null_verb/null_target are kept
  only together with a null IVT, as in every Training annotation).
"""
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
import math

from surgical_agent.perception.final_only import final_only_schema
from surgical_agent.research.gate.final_only_training import canonical_labels
from surgical_agent.research.gate import pgp_runtime as original
from surgical_agent.research.verification.candidate_coordinator import _TASK_NAMES
from surgical_agent.research.verification.prior_panel import COMPONENTS

VERSION = 'tracker-output-interaction-pgp-research-v2.1'
VERB_RULES = tuple((_TASK_NAMES['instrument'].index(i), _TASK_NAMES['verb'].index(v))
                   for i, v in (('clipper', 'clip'), ('hook', 'dissect'), ('scissors', 'cut')))
CAPS = {t: final_only_schema()['properties'][t]['properties']['selected_ids']['maxItems']
        for t in ('instrument', 'verb')}
TRIPLET_INSTRUMENTS = frozenset(c['instrument'] for c in COMPONENTS.values())
NULL_VERB = _TASK_NAMES['verb'].index('null_verb')
NULL_TARGET = _TASK_NAMES['target'].index('null_target')
NULL_IVTS = frozenset(c for c, comp in COMPONENTS.items() if comp['verb'] == NULL_VERB and comp['target'] == NULL_TARGET)
OUTPUT_MODULES = {
    'v2.1': {'version': 'scheme4-output-modules-v2.1', 'ontology_filter': False, 'null_cleanup': False},
    'v2.2': {'version': 'scheme4-output-modules-v2.2', 'ontology_filter': True, 'null_cleanup': True},
}
DEFAULT_OUTPUT_MODULES = 'v2.2'


def current_classes(snapshot, selected):
    """Unavailable/misaligned packets yield None; valid empty detections yield set()."""
    if not isinstance(snapshot, Mapping) or snapshot.get('status') != 'AVAILABLE':
        return None
    if snapshot.get('video_id') != selected['video_id'] or snapshot.get('source_max_frame_id') != selected['frame_id']:
        return None
    frames = snapshot.get('frames')
    if not isinstance(frames, Sequence) or not frames or not isinstance(frames[-1], Mapping) or frames[-1].get('frame_id') != selected['frame_id']:
        return None
    result = set()
    try:
        for track in frames[-1]['tracks']:
            inst, score = track['instrument_id'], track['score']
            if type(inst) is not int or not 0 <= inst < 7 or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
                return None
            if score >= .35:
                result.add(inst)
    except (KeyError, TypeError, ValueError):
        return None
    return result


def output_modules(prediction, tracker_classes, *, fusion=True, verb_rules=True, ontology_filter=False):
    """M1/M2 with explicit schema-capacity fallback, never GT-based truncation."""
    out = canonical_labels(prediction)
    log = {'tracker_available': tracker_classes is not None, 'm1_applied': False,
           'm1_capacity_fallback': False, 'm2_capacity_fallback': False, 'deleted_ivts': [], 'added_verbs': []}
    if ontology_filter:
        log['excluded_instruments'] = [i for i in out['instrument'] if i not in TRIPLET_INSTRUMENTS]
        out['instrument'] = [i for i in out['instrument'] if i in TRIPLET_INSTRUMENTS]
    if tracker_classes is None or not fusion:
        return canonical_labels(out), log
    ts = set(tracker_classes)
    if any(type(i) is not int or not 0 <= i < 7 for i in ts):
        raise ValueError('invalid instrument class')
    if ontology_filter:
        log['excluded_tracker_classes'] = sorted(ts - TRIPLET_INSTRUMENTS)
        ts &= TRIPLET_INSTRUMENTS
    if len(ts) > CAPS['instrument']:
        log['m1_capacity_fallback'] = True
        return canonical_labels(out), log
    out['instrument'] = sorted(ts)
    log['deleted_ivts'] = [c for c in out['ivt'] if COMPONENTS[c]['instrument'] not in ts]
    out['ivt'] = [c for c in out['ivt'] if c not in log['deleted_ivts']]
    log['m1_applied'] = True
    if verb_rules:
        added = {v for i, v in VERB_RULES if i in ts} - set(out['verb'])
        if len(set(out['verb']) | added) > CAPS['verb']:
            log['m2_capacity_fallback'] = True
        else:
            out['verb'] = sorted(set(out['verb']) | added)
            log['added_verbs'] = sorted(added)
    return canonical_labels(out), log


def null_label_cleanup(prediction, probe_ratings):
    """Drop null IVTs the Qwen probe validly rated 1; keep null verb/target only with a null IVT.

    `probe_ratings` maps 'ivt:<id>' to the Qwen probe's rating (None when invalid or unrated);
    it holds only responses already paid for before the Gate decision.
    """
    out = canonical_labels(prediction)
    removed = [c for c in out['ivt'] if c in NULL_IVTS and probe_ratings.get(f'ivt:{c}') == 1]
    out['ivt'] = [c for c in out['ivt'] if c not in removed]
    log = {'removed_null_ivts': removed, 'removed_null_verb': False, 'removed_null_target': False}
    if not NULL_IVTS & set(out['ivt']):
        log['removed_null_verb'] = NULL_VERB in out['verb']
        log['removed_null_target'] = NULL_TARGET in out['target']
        out['verb'] = [v for v in out['verb'] if v != NULL_VERB]
        out['target'] = [t for t in out['target'] if t != NULL_TARGET]
    return canonical_labels(out), log


class CausalPhaseFilter:
    """M3: raw votes in (t-W,t], ties keep current raw phase; no recursive votes."""
    def __init__(self, seconds=60, fps=25):
        if type(seconds) is not int or seconds < 0 or type(fps) is not int or fps <= 0:
            raise ValueError('invalid window or frame rate')
        self.seconds, self.fps = seconds, fps
        self.history, self.last = {}, {}

    def apply(self, video, frame, raw_phase):
        if not isinstance(video, str) or not video or type(frame) is not int or type(raw_phase) is not int or not 0 <= raw_phase < 7:
            raise ValueError('invalid stream observation')
        if video in self.last and frame <= self.last[video]:
            raise ValueError('duplicate or noncausal stream observation')
        self.last[video] = frame
        history = self.history.setdefault(video, deque())
        if not self.seconds:
            return raw_phase
        while history and history[0][0] <= frame - self.seconds * self.fps:
            history.popleft()
        history.append((frame, raw_phase))
        votes = Counter(p for _, p in history)
        best = [p for p, count in votes.items() if count == max(votes.values())]
        return best[0] if len(best) == 1 else raw_phase


def run_interaction(backend, selected, prior, predict_gate, *, inference_split='Training'):
    """M4. Same prefix, compact aggregation and exact stop as frozen PGP; actions 0/1."""
    if inference_split not in ('Training', 'Testing') or selected.get('source_split') != inference_split or selected['video_id'] == 'VID110':
        raise ValueError('explicit matching Training/Testing inference scope required')
    if any(k in selected for k in ('gt', 'ground_truth', 'labels', 'mask')):
        raise ValueError('truth must not enter inference')
    if prior['excluded_video'] != selected['video_id'] or selected['video_id'] in prior['fit_videos']:
        raise ValueError('query-video prior leakage')
    frames = selected['causal_frame_ids']
    if len(frames) != 3 or sorted(set(frames)) != frames or frames[-1] != selected['frame_id']:
        raise ValueError('three distinct causal frames required')
    calls = []

    def query(stage, seat, fn):
        key = stage + '|' + seat
        if key in calls:
            raise ValueError('duplicate request')
        calls.append(key)
        return fn()

    h0 = backend.parse_h0(query('h0', 'base', backend.h0))
    pool = original.make_pool(h0)
    proposal = query('proposal', 'base', lambda: backend.proposal(h0, pool, prior))
    if proposal is not None:
        try:
            pool = original.make_pool(h0, proposal, pool)
        except ValueError:
            pass
    bf = original.base_features.extract_features(h0, target_frame_id=selected['frame_id'], causal_frame_ids=frames, tracker_snapshot=None)
    cheap, features = original.cheap_features.extract_features(bf, h0, pool, prior, video_id=selected['video_id'], gate=original.GATE, tracker=False)
    features = {k: float(v) for k, v in features.items() if not k.startswith('tracker_')}
    props = pool['propositions']
    observed, compact_raw = {p['id']: [] for p in props}, {}

    def compact(seat):
        compact_raw[seat] = query('control_graph', seat, lambda: backend.compact(seat, h0, pool))
        normalized, _ = backend.normalize_compact({s: compact_raw.get(s) for s in original.SEATS}, pool, 3)
        if props:
            _, diag = original.aggregate(normalized, pool, image_count=3)
            for p in props:
                d = diag[p['id']]
                observed[p['id']].append(None if seat in d['invalid'] else d['scores'][original.SEATS.index(seat)])

    review_mode = getattr(predict_gate, 'review_mode', 'separate')
    five_head_probe = review_mode == 'five_head_probe'
    qwen_joint_raw = None
    if five_head_probe:
        joint_pool = original.joint_pool(pool)
        qwen_joint_raw = query('five_head_v1', 'qwen', lambda: backend.five_head('qwen', h0, joint_pool))
        normalized, _ = original.normalize_five_heads(
            {s: qwen_joint_raw if s == 'qwen' else None for s in original.SEATS}, joint_pool, image_count=3)
        _, diagnostics = original.aggregate_five_heads(normalized, joint_pool, image_count=3)
        for p in props:
            d = diagnostics[p['id']]
            observed[p['id']].append(None if 'qwen' in d['invalid'] else d['scores'][original.SEATS.index('qwen')])
        from surgical_agent.research.gate.unified_review import phase_probe_features
        features.update(phase_probe_features(diagnostics, h0, original))
    else:
        compact('qwen')
    probe_ratings = {f"{p['task']}:{p['label_id']}": observed[p['id']][0] for p in props}
    features.update(original.probe_features(props, h0, observed))
    score, action = predict_gate(features)
    if action not in (0, 1):
        raise ValueError('interaction Gate requires action 0 or 1')
    phase_review_enabled = bool(getattr(predict_gate, 'phase_review_enabled', False))
    if review_mode not in ('separate', 'unified', 'five_head_probe') or review_mode in ('unified', 'five_head_probe') and not phase_review_enabled:
        raise ValueError('invalid verification mode')
    out, depth, phase_depth = deepcopy(cheap), 0 if five_head_probe else 1, 1 if five_head_probe else 0
    phase_decision = {'status': 'gate_skipped' if phase_review_enabled else 'disabled'}
    if action and review_mode in ('unified', 'five_head_probe'):
        from surgical_agent.research.gate.unified_review import verify
        out, phase_depth, phase_decision = verify(backend, h0, cheap, pool, prior, query, original,
            reuse_probe=five_head_probe, probe_raw=qwen_joint_raw)
    elif action:
        for seat in original.ORDER[1:]:
            if all(original.ambiguous(p) or original.settled(observed[p['id']], p['label_id'] in h0[p['task']]) for p in props):
                break
            compact(seat)
            depth += 1
        if depth == 5:
            normalized, _ = backend.normalize_compact({s: compact_raw.get(s) for s in original.SEATS}, pool, 3)
            means, _ = original.aggregate(normalized, pool, image_count=3)
            out, _ = original.select_prior_gated(h0, pool, means, prior, phase=h0['phase'][0], **original.POST_REVIEW_GATE)
        if phase_review_enabled:
            rec = query('phase_recommendation', 'base', lambda: backend.phase_recommendation(h0, pool, prior))
            joint_pool = original.joint_pool(pool)
            phase_observed, phase_raw = [[] for _ in range(7)], {}
            for seat in original.ORDER:
                if original.phase_settled(phase_observed, h0['phase'][0]):
                    break
                phase_raw[seat] = query('joint_r1', seat, lambda: backend.joint(seat, h0, pool, rec))
                normalized, _ = original.normalize_five_heads(
                    {s: phase_raw.get(s) for s in original.SEATS}, joint_pool, image_count=3)
                _, diagnostics = original.aggregate_five_heads(normalized, joint_pool, image_count=3)
                for p in range(7):
                    d = diagnostics[f'phase_{p}']
                    phase_observed[p].append(None if seat in d['invalid'] else d['scores'][original.SEATS.index(seat)])
                phase_depth += 1
            phase_decision = {'status': 'exact_early_stop', 'retained_phase': h0['phase'][0]}
            if phase_depth == 5:
                normalized, _ = original.normalize_five_heads(phase_raw, joint_pool, image_count=3)
                means, _ = original.aggregate_five_heads(normalized, joint_pool, image_count=3)
                out['phase'], phase_decision = original.decide_phase(h0, means)
        out = original.repair(cheap, out)
    return {'key': selected['key'], 'h0': h0, 'cheap': cheap, 'prediction': canonical_labels(out),
            'features': features, 'gate_score': float(score), 'gate_action': int(action),
            'logical_calls': len(calls), 'call_keys': calls, 'compact_depth': depth, 'phase_depth': phase_depth,
            'repair_policy_version': original.REPAIR_POLICY_VERSION, 'post_review_prior_enabled': False,
            'phase_review_enabled': phase_review_enabled, 'phase_decision': phase_decision,
            'review_mode': review_mode, 'joint_depth': phase_depth if review_mode in ('unified', 'five_head_probe') else 0,
            'candidate_pool_tasks': ['instrument', 'verb', 'target', 'ivt', 'phase'] if five_head_probe else ['instrument', 'verb', 'target', 'ivt'],
            'phase_before_smoothing': out['phase'][0],
            'probe_ratings': probe_ratings}


def run_target(backend, selected, prior, predict_gate, *, tracker_snapshot, phase_filter, output=DEFAULT_OUTPUT_MODULES, inference_split='Training'):
    options = OUTPUT_MODULES[output]
    result = run_interaction(backend, selected, prior, predict_gate, inference_split=inference_split)
    pred, log = output_modules(result['prediction'], current_classes(tracker_snapshot, selected),
                               ontology_filter=options['ontology_filter'])
    if options['null_cleanup']:
        pred, null_log = null_label_cleanup(pred, result['probe_ratings'])
        log.update(null_log)
    pred['phase'] = [phase_filter.apply(selected['video_id'], selected['frame_id'], result['phase_before_smoothing'])]
    result.update(prediction=canonical_labels(pred), output_modules=log, version=VERSION,
                  output_modules_version=options['version'])
    if result['phase_review_enabled']:
        result['version'] = ('tracker-five-head-unified-pipeline-v1' if result['review_mode'] == 'unified'
                             else 'tracker-five-head-phase-pipeline-v1')
    if result['review_mode'] == 'five_head_probe':
        result['version'] = 'tracker-five-head-probe-pipeline-v1'
    return result
