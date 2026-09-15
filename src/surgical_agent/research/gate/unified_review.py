"""One shared post-Gate panel for interaction and phase admission."""
from copy import deepcopy


def verify(backend, h0, cheap, pool, prior, query, original):
    # Keep the historical request unchanged so existing joint responses remain usable.
    rec = query('phase_recommendation', 'base', lambda: backend.phase_recommendation(h0, pool, prior))
    joint_pool = original.joint_pool(pool)
    observed = {p['id']: [] for p in pool['propositions']}
    phase_observed = [[] for _ in range(7)]
    raw, depth = {}, 0
    for seat in original.ORDER:
        interaction_done = all(original.ambiguous(p) or original.settled(
            observed[p['id']], p['label_id'] in h0[p['task']]) for p in pool['propositions'])
        phase_done = original.phase_settled(phase_observed, h0['phase'][0])
        if interaction_done and phase_done:
            break
        raw[seat] = query('joint_r1', seat, lambda: backend.joint(seat, h0, pool, rec))
        normalized, _ = original.normalize_five_heads(
            {s: raw.get(s) for s in original.SEATS}, joint_pool, image_count=3)
        means, diagnostics = original.aggregate_five_heads(normalized, joint_pool, image_count=3)
        for pid, values in observed.items():
            d = diagnostics[pid]
            values.append(None if seat in d['invalid'] else d['scores'][original.SEATS.index(seat)])
        for p in range(7):
            d = diagnostics[f'phase_{p}']
            phase_observed[p].append(None if seat in d['invalid'] else d['scores'][original.SEATS.index(seat)])
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
