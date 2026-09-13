"""Gate after H0 + proposal + prior admission; no reviews or GT in features."""
from surgical_agent.research.gate import mainline_training as base
from surgical_agent.research.gate import terminal_utility_training as terminal
from surgical_agent.research.verification.prior_gated_repair import is_null, phase_rate, select_prior_gated
from surgical_agent.research.verification.prior_panel import labels

FEATURE_VERSION = 'mainline_postcheap_features_v1'
LABEL_VERSION = 'mainline_escalation_utility_v1'
EXTRA_FEATURES = ('cheap_vetoed', 'cheap_added', 'cheap_changed', 'pool_new_total',
                  'pool_new_instrument', 'pool_new_verb', 'pool_new_target', 'pool_new_ivt',
                  'h0_ivt_prior_min', 'h0_ivt_prior_mean', 'new_ivt_prior_max')
FEATURE_ORDER = (*base.FEATURE_ORDER, *EXTRA_FEATURES)


def cheap_state(h0, pool, prior, *, video_id, gate=None):
    if prior['excluded_video'] != video_id or video_id in prior['fit_videos']:
        raise ValueError('query-video prior leakage')
    h0 = labels(h0)
    phase = h0['phase'][0]
    kw = {} if gate is None else {k: gate[k] for k in ('veto_rate', 'add_rate', 'prune')}
    cheap, log = select_prior_gated(h0, pool, None, prior, phase=phase, **kw)
    new = [p for p in pool['propositions'] if p['label_id'] not in h0[p['task']]]
    hr = [phase_rate(prior, 'ivt', c, phase) for c in h0['ivt'] if not is_null(c)]
    nr = [phase_rate(prior, 'ivt', p['label_id'], phase) for p in new
          if p['task'] == 'ivt' and not is_null(p['label_id'])]
    extra = dict(cheap_vetoed=len(log['vetoed']), cheap_added=len(log['prior_added']),
                 cheap_changed=float(bool(log['vetoed'] or log['prior_added'])), pool_new_total=len(new),
                 **{f'pool_new_{t}': sum(p['task'] == t for p in new) for t in base.TASKS if t != 'phase'},
                 h0_ivt_prior_min=min(hr) if hr else 1.0,
                 h0_ivt_prior_mean=sum(hr)/len(hr) if hr else 1.0,
                 new_ivt_prior_max=max(nr) if nr else 0.0)
    assert set(extra) == set(EXTRA_FEATURES)
    return labels(cheap), extra


def extract_features(h0_features, h0, pool, prior, *, video_id, gate=None, tracker=False):
    base.validate_features(h0_features)
    cheap, extra = cheap_state(h0, pool, prior, video_id=video_id, gate=gate)
    old = dict(h0_features) if tracker else base.mask_tracker(h0_features)
    return cheap, {**old, **extra}


def label_outcome(cheap, final, *, gt, mask):
    result = terminal.label_outcome(cheap, final, gt=gt, mask=mask)
    result['label_version'] = LABEL_VERSION
    result['reference_arm'] = 'cheap_h0_proposal_prior'
    result['cheap_counts_by_task'] = result.pop('h0_counts_by_task')
    return result
