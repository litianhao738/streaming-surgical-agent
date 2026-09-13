"""Causal single-probe PGP orchestration shared by cached and live transports.

Only this module decides which calls happen. Backends never receive ground truth
or future responses. A missing observed answer is invalid; unqueried seats are
not evidence. Tracker is intentionally absent from the model interface.
"""
from copy import deepcopy
from surgical_agent.research.gate import mainline_training as base_features
from surgical_agent.research.gate import escalation_rule_training as cheap_features
from surgical_agent.research.gate.pgp_ambiguity import AMB_VERBS, AMB_IVTS, repair
from surgical_agent.research.verification.prior_gated_joint import joint_pool, decide_phase
from surgical_agent.research.verification.prior_gated_repair import select_prior_gated
from surgical_agent.research.verification.recent_mean_panel import aggregate
from surgical_agent.research.verification.five_head_repair import normalize_five_heads, aggregate_five_heads
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool

ORDER=('qwen','gpt','gemini','grok','deepseek')
PROBE_NAMES=('qwen_nonamb_items','qwen_invalid_nonamb','qwen_present_min','qwen_present_le2','qwen_present_le3',
             'qwen_absent_max','qwen_absent_ge4','qwen_absent_ge5','qwen_settled_fraction','qwen_all_settled')
GATE={'veto_rate':.01,'add_rate':.7,'prune':[]}


def ambiguous(p):
    return p['task']=='verb' and p['label_id'] in AMB_VERBS or p['task']=='ivt' and p['label_id'] in AMB_IVTS


def settled(values,present):
    if any(v is None for v in values): return True
    left=5-len(values)
    return sum(values)+left>10 if present else sum(values)+5*left<20


def phase_settled(observed,current):
    if any(v is None for v in observed[current]): return True
    left=5-len(observed[current]); lower=sum(observed[current])+left
    return not any(p!=current and None not in s and sum(s)+5*left>=20 and sum(s)+5*left>lower for p,s in enumerate(observed))


def probe_features(props,h0,observed):
    nonamb=[p for p in props if not ambiguous(p)]
    valid=[(observed[p['id']][0],p['label_id'] in h0[p['task']]) for p in nonamb if observed[p['id']][0] is not None]
    pres=[v for v,p in valid if p]; absent=[v for v,p in valid if not p]
    done=[ambiguous(p) or settled(observed[p['id']],p['label_id'] in h0[p['task']]) for p in props]
    values=[len(nonamb),sum(observed[p['id']][0] is None for p in nonamb),min(pres,default=5),sum(v<=2 for v in pres),sum(v<=3 for v in pres),
            max(absent,default=1),sum(v>=4 for v in absent),sum(v>=5 for v in absent),sum(done)/max(1,len(props)),int(all(done))]
    return dict(zip(PROBE_NAMES,map(float,values)))


def run_target(backend, selected, prior, predict_gate):
    """backend supplies H0/proposal/review methods; predict_gate maps 42 features to score/action."""
    if selected.get('source_split')!='Training' or selected['video_id']=='VID110':
        raise ValueError('this release accepts Training inference only')
    if any(k in selected for k in ('gt','ground_truth','labels','mask')): raise ValueError('truth must not enter inference')
    if prior['excluded_video']!=selected['video_id'] or selected['video_id'] in prior['fit_videos']:
        raise ValueError('query-video prior leakage')
    frames=selected['causal_frame_ids']
    if len(frames)!=3 or sorted(set(frames))!=frames or frames[-1]!=selected['frame_id']:
        raise ValueError('three distinct ordered causal frames ending at target required')
    calls=[]
    def query(stage,seat,fn):
        key=stage+'|'+seat
        if key in calls: raise ValueError('duplicate request')
        calls.append(key)
        return fn()
    raw=query('h0','base',backend.h0)
    h0=backend.parse_h0(raw); pool=make_pool(h0)
    proposal=query('proposal','base',lambda:backend.proposal(h0,pool,prior))
    if proposal is not None:
        try: pool=make_pool(h0,proposal,pool)
        except ValueError: pass # same frozen first-attempt fallback, no retry
    bf=base_features.extract_features(h0,target_frame_id=selected['frame_id'],causal_frame_ids=frames,tracker_snapshot=None)
    cheap,feat=cheap_features.extract_features(bf,h0,pool,prior,video_id=selected['video_id'],gate=GATE,tracker=False)
    feat={k:float(v) for k,v in feat.items() if not k.startswith('tracker_')}
    props=pool['propositions']; observed={p['id']:[] for p in props}; compact_raw={}
    def compact(seat):
        raw=query('control_graph',seat,lambda:backend.compact(seat,h0,pool))
        compact_raw[seat]=raw
        normalized,_=backend.normalize_compact({s:compact_raw.get(s) for s in SEATS},pool,3)
        if props:
            _,diag=aggregate(normalized,pool,image_count=3)
            for p in props:
                d=diag[p['id']]; observed[p['id']].append(None if seat in d['invalid'] else d['scores'][SEATS.index(seat)])
    compact('qwen')
    feat.update(probe_features(props,h0,observed))
    score,action=predict_gate(feat)
    if action not in (0,3): raise ValueError('whole-frame Gate requires action 0 or 3')
    out=deepcopy(cheap); phase_observed=[[] for _ in range(7)]; phase_raw={}
    compact_depth=1; phase_depth=0
    if action==3:
        for seat in ORDER[1:]:
            if all(ambiguous(p) or settled(observed[p['id']],p['label_id'] in h0[p['task']]) for p in props): break
            compact(seat); compact_depth+=1
        if compact_depth==5:
            normalized,_=backend.normalize_compact({s:compact_raw.get(s) for s in SEATS},pool,3)
            means,_=aggregate(normalized,pool,image_count=3)
            out,_=select_prior_gated(h0,pool,means,prior,phase=h0['phase'][0],**GATE)
        rec=query('phase_recommendation','base',lambda:backend.phase_recommendation(h0,pool,prior))
        jp=joint_pool(pool)
        for seat in ORDER:
            if phase_settled(phase_observed,h0['phase'][0]): break
            phase_raw[seat]=query('joint_r1',seat,lambda:backend.joint(seat,h0,pool,rec))
            norm,_=normalize_five_heads({s:phase_raw.get(s) for s in SEATS},jp,image_count=3)
            _,diag=aggregate_five_heads(norm,jp,image_count=3)
            for p in range(7):
                d=diag[f'phase_{p}']; phase_observed[p].append(None if seat in d['invalid'] else d['scores'][SEATS.index(seat)])
            phase_depth+=1
        if phase_depth==5:
            norm,_=normalize_five_heads(phase_raw,jp,image_count=3)
            means,_=aggregate_five_heads(norm,jp,image_count=3)
            out['phase'],_=decide_phase(h0,means)
        out=repair(cheap,out)
    return {'key':selected['key'],'h0':h0,'cheap':cheap,'prediction':out,'features':feat,'gate_score':float(score),
            'gate_action':int(action),'logical_calls':len(calls),'call_keys':calls,'compact_depth':compact_depth,
            'phase_depth':phase_depth,'tracker_enabled':False}
