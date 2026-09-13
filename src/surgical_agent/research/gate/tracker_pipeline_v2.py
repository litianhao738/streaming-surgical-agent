"""Independent Training research runtime: interaction-only PGP and local output modules.

No network or GT is accessed here. The caller owns the backend, frozen predictions,
Gate model, and per-stream phase state. The existing PGP runtime stays unchanged.
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


def output_modules(prediction, tracker_classes, *, fusion=True, verb_rules=True):
    """M1/M2 with explicit schema-capacity fallback, never GT-based truncation."""
    out = canonical_labels(prediction)
    log = {'tracker_available': tracker_classes is not None, 'm1_applied': False,
           'm1_capacity_fallback': False, 'm2_capacity_fallback': False, 'deleted_ivts': [], 'added_verbs': []}
    if tracker_classes is None or not fusion:
        return out, log
    ts = set(tracker_classes)
    if any(type(i) is not int or not 0 <= i < 7 for i in ts):
        raise ValueError('invalid instrument class')
    if len(ts) > CAPS['instrument']:
        log['m1_capacity_fallback'] = True
        return out, log
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


def run_interaction(backend, selected, prior, predict_gate):
    """M4. Same prefix, compact aggregation and exact stop as frozen PGP; actions 0/1."""
    if selected.get('source_split') != 'Training' or selected['video_id'] == 'VID110':
        raise ValueError('Training research only')
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

    compact('qwen')
    features.update(original.probe_features(props, h0, observed))
    score, action = predict_gate(features)
    if action not in (0, 1):
        raise ValueError('interaction Gate requires action 0 or 1')
    out, depth = deepcopy(cheap), 1
    if action:
        for seat in original.ORDER[1:]:
            if all(original.ambiguous(p) or original.settled(observed[p['id']], p['label_id'] in h0[p['task']]) for p in props):
                break
            compact(seat)
            depth += 1
        if depth == 5:
            normalized, _ = backend.normalize_compact({s: compact_raw.get(s) for s in original.SEATS}, pool, 3)
            means, _ = original.aggregate(normalized, pool, image_count=3)
            out, _ = original.select_prior_gated(h0, pool, means, prior, phase=h0['phase'][0], **original.GATE)
        out = original.repair(cheap, out)
    return {'key': selected['key'], 'h0': h0, 'cheap': cheap, 'prediction': canonical_labels(out),
            'features': features, 'gate_score': float(score), 'gate_action': int(action),
            'logical_calls': len(calls), 'call_keys': calls, 'compact_depth': depth, 'phase_depth': 0}


def run_target(backend, selected, prior, predict_gate, *, tracker_snapshot, phase_filter):
    result = run_interaction(backend, selected, prior, predict_gate)
    pred, log = output_modules(result['prediction'], current_classes(tracker_snapshot, selected))
    pred['phase'] = [phase_filter.apply(selected['video_id'], selected['frame_id'], result['cheap']['phase'][0])]
    result.update(prediction=canonical_labels(pred), output_modules=log, version=VERSION)
    return result
