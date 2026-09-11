from copy import deepcopy

import pytest

from surgical_agent.research.verification import transactional_review as tx
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool


def initial():
    return {'instrument':[0],'verb':[1],'target':[0],'ivt':[17],'phase':[1]}


def evidence(pool):
    return {p['id']:{'state':'UNKNOWN','present_votes':0,'absent_votes':0,'votes':{}} for p in pool['propositions']}


def set_state(e,pid,state):
    e[pid].update(state=state,present_votes=3 if state=='PRESENT' else 0,absent_votes=3 if state=='ABSENT' else 0)


@pytest.mark.parametrize(('decision','operation','state'),[
    ('SUPPORTED','ADD','PRESENT'),('SUPPORTED','REMOVE','ABSENT'),
    ('OPPOSED','ADD','ABSENT'),('OPPOSED','REMOVE','PRESENT'),('UNCLEAR','REMOVE','UNKNOWN')])
def test_operation_semantics(decision,operation,state):
    assert tx.presence_for(decision,operation)==state


def test_local_absence_and_missing_current_are_not_votes():
    edit={'operation':'REMOVE'}
    item={'decision':'SUPPORTED','scope':'LOCAL_REGION','image_indices':[2],'observation':'local view'}
    assert tx.decode_item(item,edit)[1]=='LOCAL_ABSENCE_CANNOT_DELETE_FRAME_LABEL'
    item.update(scope='WHOLE_FRAME',image_indices=[1])
    assert tx.decode_item(item,edit)[1]=='NO_CURRENT_FRAME_EVIDENCE'


def test_conflict_not_averaged_away_and_invalid_not_a_vote():
    votes={s:{'x':{'state':'PRESENT','error':None}} for s in SEATS}
    votes[SEATS[0]]['x']['state']='ABSENT'
    assert tx.aggregate_states(votes,['x'])['x']['state']=='CONFLICT'
    votes[SEATS[0]]['x']['error']='INVALID'
    assert tx.aggregate_states(votes,['x'])['x']['state']=='PRESENT'
    votes[SEATS[1]]['x']['state']='UNKNOWN'
    votes[SEATS[2]]['x']['state']='UNKNOWN'
    assert tx.aggregate_states(votes,['x'])['x']['state']=='UNKNOWN'


def test_frame_absence_can_remove_related_unknown_ivt_atomically():
    h0=initial();pool=make_pool(h0);e=evidence(pool);set_state(e,'verb_1','ABSENT')
    result=tx.apply(h0,pool,e)
    assert result['prediction']=={**h0,'verb':[],'ivt':[]}
    assert h0==initial()
    assert any(x['reason']=='ENTAILED_BY_FRAME_COMPONENT_ABSENCE' for x in result['transactions'])


def test_component_absence_cannot_override_positive_relation_evidence():
    h0=initial();pool=make_pool(h0);e=evidence(pool);set_state(e,'verb_1','ABSENT')
    e['ivt_17']['present_votes']=1
    result=tx.apply(h0,pool,e)
    assert result['prediction']==h0
    assert result['transactions'][0]['reason']=='COMPONENT_RELATION_EVIDENCE_CONFLICT'


def test_ivt_deletion_preserves_independent_components():
    h0=initial();pool=make_pool(h0);e=evidence(pool);set_state(e,'ivt_17','ABSENT')
    assert tx.apply(h0,pool,e)['prediction']=={**h0,'ivt':[]}


def test_independent_verb_does_not_require_an_ivt():
    h0=initial();pool=make_pool(h0,{'instrument':[],'verb':[0],'target':[],'ivt':[]});e=evidence(pool)
    set_state(e,'verb_0','PRESENT')
    final=tx.apply(h0,pool,e)['prediction']
    assert final['verb']==[0,1] and final['ivt']==[17]


def test_accepted_relation_can_resolve_unclear_components_but_not_refuted_ones():
    h0=initial();pool=make_pool(h0,{'instrument':[],'verb':[],'target':[],'ivt':[7]});e=evidence(pool)
    set_state(e,'ivt_7','PRESENT')
    final=tx.apply(h0,pool,e)['prediction']
    assert final['ivt']==[7,17] and final['verb']==[0,1]
    e['verb_0']['absent_votes']=1
    assert tx.apply(h0,pool,e)['prediction']==h0


def test_two_tools_sharing_component_prevent_unjustified_global_deletion():
    h0=initial();h0['ivt']=[7,17];h0['verb']=[0,1]
    pool=make_pool(h0);e=evidence(pool);set_state(e,'ivt_7','ABSENT')
    result=tx.apply(h0,pool,e)['prediction']
    assert result['ivt']==[17] and result['instrument']==[0] and result['target']==[0]


def test_missing_assessment_does_not_invalidate_other_candidates():
    h0=initial();pool=make_pool(h0);edits=tx.claims(h0,pool)
    item={'decision':'OPPOSED','scope':'LOCAL_REGION','image_indices':[2],'observation':'visible tool'}
    raw={s:{'assessments':{'instrument_0':deepcopy(item)}} for s in SEATS}
    result=tx.assess(raw,edits)
    assert result['instrument_0']['state']=='PRESENT'
    assert result['ivt_17']['state']=='UNKNOWN'
    assert result['ivt_17']['votes'][SEATS[0]]['error']=='SCHEMA_INVALID'


def test_no_evidence_preserves_h0_and_phase():
    h0=initial();pool=make_pool(h0)
    assert tx.apply(h0,pool,evidence(pool))['prediction']==h0
