"""Rule-aware cheap tier. Features simulate only information available to Gate."""
from surgical_agent.research.gate import escalation_training as v1
from surgical_agent.research.verification.prior_gated_repair import phase_rate

base = v1.base
LABEL_VERSION = v1.LABEL_VERSION
FEATURE_VERSION = 'mainline_postcheap_rule_features_v2'
FEATURE_ORDER = (*v1.FEATURE_ORDER, 'proposal_rule')
label_outcome = v1.label_outcome


def proposal_needed(h0, prior, *, video_id, gate=None):
    if prior['excluded_video'] != video_id or video_id in prior['fit_videos']:
        raise ValueError('query-video prior leakage')
    h0=base.canonical_labels(h0)
    rate=.7 if gate is None else gate['add_rate']
    if rate is None:return False
    return any(c not in h0['ivt'] and phase_rate(prior,'ivt',c,h0['phase'][0])>=rate for c in range(94))


def extract_features(h0_features,h0,pool,prior,*,video_id,gate=None,tracker=False):
    rule=proposal_needed(h0,prior,video_id=video_id,gate=gate)
    # Never inspect proposal outputs on the negative branch, even during replay.
    available=pool if rule else {'propositions':[]}
    if available is None:raise ValueError('rule-positive branch requires the observed candidate pool')
    cheap,values=v1.extract_features(h0_features,h0,available,prior,video_id=video_id,gate=gate,tracker=tracker)
    values['proposal_rule']=float(rule)
    return cheap,values


def cost(route, rules, *, h0_usd, proposal_usd, full_usd):
    if len(route)!=len(rules):raise ValueError('route/rule length mismatch')
    # A reviewed negative-rule frame must fetch its proposal after Gate.
    calls=sum(13 if r else 1+int(p) for r,p in zip(route,rules))
    usd=sum(full_usd if r else h0_usd+int(p)*proposal_usd for r,p in zip(route,rules))
    return {'calls':calls,'usd_equivalent_estimate':usd,'usd_share_full':usd/(len(route)*full_usd)}
