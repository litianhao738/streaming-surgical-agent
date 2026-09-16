"""One shared post-Gate panel for interaction and phase admission."""
from copy import deepcopy


PHASE_PROBE_NAMES = tuple(f'qwen_phase_{i}_rating' for i in range(7)) + (
    'qwen_phase_valid_count', 'qwen_phase_current_rating', 'qwen_phase_alternative_max',
    'qwen_phase_alternative_margin', 'qwen_phase_settled')


def phase_probe_features(diagnostics, h0, original):
    seat_index = original.SEATS.index('qwen')
    values = [None if 'qwen' in diagnostics[f'phase_{p}']['invalid'] else
              diagnostics[f'phase_{p}']['scores'][seat_index] for p in range(7)]
    current = values[h0['phase'][0]]
    alternatives = [v for p, v in enumerate(values) if p != h0['phase'][0] and v is not None]
    best = max(alternatives, default=None)
    numbers = [float(v) if v is not None else -1. for v in values]
    numbers += [float(sum(v is not None for v in values)), float(current) if current is not None else -1.,
                float(best) if best is not None else -1.,
                float(best-current) if current is not None and best is not None else 0.,
                float(original.phase_settled([[v] for v in values], h0['phase'][0]))]
    return dict(zip(PHASE_PROBE_NAMES, numbers))


def verify(backend, h0, cheap, pool, prior, query, original, *, reuse_probe=False, probe_raw=None):
    # Keep the historical request unchanged so existing joint responses remain usable.
    rec = None if reuse_probe else query('phase_recommendation', 'base', lambda: backend.phase_recommendation(h0, pool, prior))
    joint_pool = original.joint_pool(pool)
    observed = {p['id']: [] for p in pool['propositions']}
    phase_observed = [[] for _ in range(7)]
    raw, depth = {}, 0
    means = {}
    def observe(seat, response):
        nonlocal means
        raw[seat] = response
        normalized, _ = original.normalize_five_heads(
            {s: raw.get(s) for s in original.SEATS}, joint_pool, image_count=3)
        means, diagnostics = original.aggregate_five_heads(normalized, joint_pool, image_count=3)
        for pid, values in observed.items():
            d = diagnostics[pid]
            values.append(None if seat in d['invalid'] else d['scores'][original.SEATS.index(seat)])
        for p in range(7):
            d = diagnostics[f'phase_{p}']
            phase_observed[p].append(None if seat in d['invalid'] else d['scores'][original.SEATS.index(seat)])
    if reuse_probe:
        observe('qwen', probe_raw)
        depth = 1
    for seat in (original.ORDER[1:] if reuse_probe else original.ORDER):
        interaction_done = all(original.ambiguous(p) or original.settled(
            observed[p['id']], p['label_id'] in h0[p['task']]) for p in pool['propositions'])
        phase_done = original.phase_settled(phase_observed, h0['phase'][0])
        if interaction_done and phase_done:
            break
        response = (query('five_head_v1', seat, lambda: backend.five_head(seat, h0, joint_pool)) if reuse_probe
                    else query('joint_r1', seat, lambda: backend.joint(seat, h0, pool, rec)))
        observe(seat, response)
        depth += 1
    out = deepcopy(cheap)
    decision = {'status': 'joint_exact_early_stop', 'retained_phase': h0['phase'][0]}
    if depth == 5:
        # Both heads consume the same validated responses. The prior retains its H0 bucket.
        out, _ = original.select_prior_gated(h0, pool,
            {p['id']: means[p['id']] for p in pool['propositions']}, prior,
            phase=h0['phase'][0], **original.GATE)
        out['phase'], decision = original.decide_phase(h0, means)
    return original.repair(cheap, out), depth, decision
