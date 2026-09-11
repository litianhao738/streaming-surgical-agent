"""Explicit edit evidence and dependency-aware patches; no GT, network or defaults."""
from copy import deepcopy

from jsonschema import Draft202012Validator

from surgical_agent.research.verification.candidate_coordinator import SEATS
from surgical_agent.research.verification.prior_panel import COMPONENTS, TASKS, labels
from surgical_agent.research.verification.semantic_coordinator import item_error

VERSION = 'explicit_edit_transactions_v1'
MIN_VOTES = 3


def claims(h0, pool):
    h0 = labels(h0)
    ids = {p['id'] for p in pool['propositions']}
    if len(ids) != len(pool['propositions']):
        raise ValueError('duplicate pool ID')
    required = {f'{q}_{v}' for q in TASKS for v in h0[q]}
    if not required <= ids:
        raise ValueError('pool omits existing labels')
    result = []
    for p in pool['propositions']:
        q, v = p['task'], p['label_id']
        if p['id'] != f'{q}_{v}':
            raise ValueError('noncanonical proposition')
        if q == 'ivt' and not {f'{t}_{c}' for t,c in COMPONENTS[v].items()} <= ids:
            raise ValueError('IVT components missing from pool')
        result.append({**deepcopy(p),'operation':'REMOVE' if v in h0[q] else 'ADD',
            'related_ivt_ids':[x['id'] for x in pool['propositions'] if x['task']=='ivt'
                               and (x['label_id']==v if q=='ivt' else COMPONENTS[x['label_id']][q]==v)]})
    return result


def item_schema(image_count):
    props = {'decision':{'type':'string','enum':['SUPPORTED','OPPOSED','UNCLEAR']},
             'scope':{'type':'string','enum':['WHOLE_FRAME','LOCAL_REGION','UNCERTAIN']},
             'image_indices':{'type':'array','items':{'type':'integer','minimum':0,'maximum':image_count-1},
                              'maxItems':image_count},
             'observation':{'type':'string','minLength':1,'maxLength':1000}}
    return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}


def response_schema(edits, image_count=3):
    props = {e['id']:item_schema(image_count) for e in edits}
    return {'type':'object','properties':{'assessments':{'type':'object','properties':props,
        'required':list(props),'additionalProperties':False}},'required':['assessments'],'additionalProperties':False}


def presence_for(decision, operation):
    if decision=='UNCLEAR':
        return 'UNKNOWN'
    return 'PRESENT' if (decision=='SUPPORTED') == (operation=='ADD') else 'ABSENT'


def decode_item(item, edit, image_count=3):
    if not Draft202012Validator(item_schema(image_count)).is_valid(item):
        return 'UNKNOWN','SCHEMA_INVALID'
    indices=item['image_indices']
    if any(type(i) is not int for i in indices) or len(indices)!=len(set(indices)):
        return 'UNKNOWN','INVALID_IMAGE_INDICES'
    if not item['observation'].strip():
        return 'UNKNOWN','EMPTY_OBSERVATION'
    state=presence_for(item['decision'],edit['operation'])
    if state!='UNKNOWN' and image_count-1 not in indices:
        return 'UNKNOWN','NO_CURRENT_FRAME_EVIDENCE'
    if state=='ABSENT' and item['scope']!='WHOLE_FRAME':
        return 'UNKNOWN','LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL'
    if state=='PRESENT' and item['scope']=='UNCERTAIN':
        return 'UNKNOWN','UNCERTAIN_SUPPORT'
    return state,None


def aggregate_states(per_seat, ids):
    if set(per_seat)!=set(SEATS):
        raise ValueError('five named seats required')
    out={}
    for pid in ids:
        votes={s:per_seat[s][pid] for s in SEATS}
        yes=sum(v['state']=='PRESENT' and v['error'] is None for v in votes.values())
        no=sum(v['state']=='ABSENT' and v['error'] is None for v in votes.values())
        state=('CONFLICT' if yes and no else 'PRESENT' if yes>=MIN_VOTES else
               'ABSENT' if no>=MIN_VOTES else 'UNKNOWN')
        out[pid]={'state':state,'present_votes':yes,'absent_votes':no,'votes':votes}
    return out


def assess(raw, edits, image_count=3):
    ids={e['id'] for e in edits}
    if set(raw)!=set(SEATS):
        raise ValueError('five named seats required')
    per_seat={}
    for seat in SEATS:
        reply=raw[seat]
        outer=(isinstance(reply,dict) and set(reply)=={'assessments'} and
               isinstance(reply['assessments'],dict) and not set(reply['assessments'])-ids)
        items=reply['assessments'] if outer else {}
        per_seat[seat]={}
        for edit in edits:
            item=items.get(edit['id'])
            state,error=decode_item(item,edit,image_count)
            per_seat[seat][edit['id']]={'state':state,'error':error,'item':deepcopy(item)}
    return aggregate_states(per_seat,ids)


def assess_legacy(reviews, pool, image_count=3):
    """Use original judgments to isolate selector changes; do not invent edit reviews."""
    per_seat={}
    ids={p['id'] for p in pool['propositions']}
    for seat in SEATS:
        items=reviews[seat].get('judgments',{}) if isinstance(reviews.get(seat),dict) else {}
        per_seat[seat]={}
        for p in pool['propositions']:
            item=items.get(p['id'])
            error=item_error(item,p['task'],image_count)
            state=('UNKNOWN' if error or item['rating']==3 else 'PRESENT' if item['rating']>=4 else 'ABSENT')
            per_seat[seat][p['id']]={'state':state,'error':error,'item':deepcopy(item)}
    return aggregate_states(per_seat,ids)


def apply(h0, pool, evidence):
    """Independent positives survive missing IVTs; whole-frame negatives propagate.

    Propagation never overrides any positive IVT vote. Such inconsistencies stay
    pending with explicit reasons. An accepted IVT can support missing components
    if they have no explicit counterevidence. Omitted fields never mean deletion.
    """
    initial=labels(h0)
    edits=claims(initial,pool)
    if set(evidence)!={e['id'] for e in edits}:
        raise ValueError('evidence must cover exact pool')
    out=deepcopy(initial)
    log=[]
    state=lambda pid:evidence[pid]['state']
    for i in list(out['ivt']):
        if state(f'ivt_{i}')=='ABSENT':
            out['ivt'].remove(i)
            log.append({'operation':'REMOVE','id':f'ivt_{i}','reason':'DIRECT_FRAME_ABSENCE'})
    for q in TASKS[:3]:
        for v in list(initial[q]):
            pid=f'{q}_{v}'
            if state(pid)!='ABSENT':
                continue
            dependencies=[i for i in out['ivt'] if COMPONENTS[i][q]==v]
            conflicts=[i for i in dependencies if evidence[f'ivt_{i}']['present_votes']>0]
            if conflicts:
                log.append({'operation':'KEEP','id':pid,'reason':'COMPONENT_RELATION_EVIDENCE_CONFLICT','ivt':conflicts})
                continue
            # A supported whole-frame absence entails absence of every IVT using it.
            for i in dependencies:
                out['ivt'].remove(i)
                log.append({'operation':'REMOVE','id':f'ivt_{i}','reason':'ENTAILED_BY_FRAME_COMPONENT_ABSENCE','source':pid})
            out[q].remove(v)
            log.append({'operation':'REMOVE','id':pid,'reason':'DIRECT_FRAME_ABSENCE'})
    for e in edits:
        q,v=e['task'],e['label_id']
        if e['operation']=='ADD' and q!='ivt' and state(e['id'])=='PRESENT':
            out[q].append(v)
            log.append({'operation':'ADD','id':e['id'],'reason':'INDEPENDENT_FRAME_PRESENCE'})
    for e in edits:
        if e['task']!='ivt' or e['operation']!='ADD' or state(e['id'])!='PRESENT':
            continue
        comp=COMPONENTS[e['label_id']]
        opposed=[f'{q}_{v}' for q,v in comp.items() if evidence[f'{q}_{v}']['absent_votes']>0]
        if opposed:
            log.append({'operation':'KEEP','id':e['id'],'reason':'COMPONENT_COUNTEREVIDENCE','components':opposed})
            continue
        out['ivt'].append(e['label_id'])
        log.append({'operation':'ADD','id':e['id'],'reason':'SAME_RELATION_PRESENCE'})
        for q,v in comp.items():
            if v not in out[q]:
                out[q].append(v)
                log.append({'operation':'ADD','id':f'{q}_{v}','reason':'ENTAILED_BY_ACCEPTED_IVT','source':e['id']})
    out=labels(out)
    if out['phase']!=initial['phase']:
        raise AssertionError('Phase changed')
    return {'prediction':out,'transactions':log,
            'pending':{pid:e for pid,e in evidence.items() if e['state'] in ('UNKNOWN','CONFLICT')}}
