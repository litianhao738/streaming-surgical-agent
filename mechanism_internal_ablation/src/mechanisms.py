"""Deterministic mechanism ablations. No transport, credentials or ground truth."""
from copy import deepcopy
from hashlib import sha256

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.research.gate import pgp_runtime as original
from surgical_agent.research.gate.final_only_training import canonical_labels

SLOTS = ('qwen', 'gpt', 'gemini', 'grok', 'deepseek')
HEADS = ('instrument', 'verb', 'target', 'ivt', 'phase')


def digest_key(*parts):
    return sha256('|'.join(map(str, parts)).encode('utf-8')).hexdigest()


def aggregate(pool, raw):
    if set(raw) != set(SLOTS):
        raise ValueError('All five observed terminal slots are required')
    joint = original.joint_pool(pool)
    normalized, formats = original.normalize_five_heads(raw, joint, image_count=3)
    means, diagnostics = original.aggregate_five_heads(normalized, joint, image_count=3)
    return {'means': means, 'diagnostics': diagnostics, 'formats': formats}


def probe_ratings(pool, probe):
    panel = aggregate(pool, {s: probe if s == 'qwen' else None for s in SLOTS})
    idx = original.SEATS.index('qwen')
    return {pid: None if 'qwen' in d['invalid'] else d['scores'][idx]
            for pid, d in panel['diagnostics'].items()}


def rule_score(snapshot):
    ratings = snapshot['probe_ratings_all']
    present, absent = [], []
    for p in snapshot['pool']['propositions']:
        if original.ambiguous(p) or ratings[p['id']] is None:
            continue
        values = present if p['label_id'] in snapshot['h0'][p['task']] else absent
        values.append(ratings[p['id']])
    score = sum(v <= 2 for v in present) / max(1, len(present))
    score += sum(v >= 4 for v in absent) / max(1, len(absent))
    phase = [ratings[f'phase_{p}'] for p in range(7)]
    if all(v is not None for v in phase):
        old = snapshot['h0']['phase'][0]
        best = max(v for p, v in enumerate(phase) if p != old)
        if best >= 4:
            score += max(0, best - phase[old]) / 4
    return score


def selections(snapshots, threshold, seeds=range(20)):
    keys = sorted(snapshots)
    learned = sorted(k for k in keys if snapshots[k]['gate_score'] >= threshold)
    scores = {k: rule_score(snapshots[k]) for k in keys}
    rule = sorted(keys, key=lambda k: (-scores[k], digest_key('rule_v1', k)))[:len(learned)]
    random = {str(seed): sorted(keys, key=lambda k: digest_key('random_v1', seed, k))[:len(learned)]
              for seed in seeds}
    return {'learned': learned, 'rule': rule, 'random': random, 'rule_scores': scores,
            'K': len(learned), 'N': len(keys), 'threshold': threshold}


def difference(before, after):
    return {h: {'added': sorted(set(after[h]) - set(before[h])),
                'removed': sorted(set(before[h]) - set(after[h]))} for h in HEADS}


def decide(snapshot, panel, prior, *, prior_after=True, rollback=True):
    """H0 remains selector input; cheap is rollback and evaluation reference."""
    h0, cheap, pool = (deepcopy(snapshot[k]) for k in ('h0', 'cheap', 'pool'))
    means = panel['means']
    four_means = {p['id']: means[p['id']] for p in pool['propositions']}
    fallback = None
    try:
        out, prior_log = original.select_prior_gated(
            h0, pool, four_means, prior, phase=h0['phase'][0], threshold=4.0,
            veto_rate=.01 if prior_after else None, add_rate=.7 if prior_after else None,
            prune=(), universe_add=False)
    except ApiSchemaError as exc:
        original.make_pool(h0)
        out, prior_log = deepcopy(h0), None
        fallback = {'error_type': type(exc).__name__, 'message': str(exc),
                    'fallback': 'validated_initial_labels'}
    out['phase'], phase_log = original.decide_phase(h0, means)
    before = deepcopy(out)
    if rollback:
        out = original.repair(cheap, out)
    out = canonical_labels(out)
    return out, {'prior': prior_log, 'phase': phase_log, 'schema_fallback': fallback,
                 'before_rollback': before, 'rollback_difference': difference(before, out)}
