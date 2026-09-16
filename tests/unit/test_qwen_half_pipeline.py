from copy import deepcopy
from decimal import Decimal
import json
import threading
import pytest
from scripts import testing_qwen_h0 as qwen
from scripts import run_testing_half_pipeline as core
from scripts import reviewer_routes_official as routes
from surgical_agent.research.gate.collection_budget import Budget, BudgetStop, AmbiguousDispatch


def test_qwen_wire_changes_only_model_and_generation_policy():
    original = {'model':'google/gemini-3.8-flash','provider':{'only':['google']},
                'reasoning':{'effort':'low'},'messages':[{'role':'user','content':'same image prompt'}]}
    wire = qwen.wire_body(original)
    assert wire['model']=='qwen3.8-max' and wire['enable_thinking'] is False
    assert 'provider' not in wire and 'reasoning' not in wire
    assert wire['messages']==original['messages']
    assert original['model']=='google/gemini-3.8-flash'


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(routes,'credentials',lambda *args:(qwen.ENDPOINT.removesuffix('/chat/completions'),'test-key'))
    calls=[]
    prediction={'schema_version':'joint_perception_final_only_v1',
                'phase':{'selected_id':0}, **{k:{'selected_ids':[0]} for k in ('instrument','verb','target','ivt')}}
    raw={'model':qwen.MODEL,'choices':[{'finish_reason':'stop','message':{'content':json.dumps(prediction)}}],
         'usage':{'prompt_tokens':5000,'completion_tokens':150}}
    class Response:
        status_code=200
        ok=True
        def json(self):return deepcopy(raw)
    monkeypatch.setattr(qwen.requests,'post',lambda *args,**kwargs:calls.append(kwargs['json']) or Response())
    budget=Budget(tmp_path/'budget.sqlite',{'aliyun_cny':'10'},'plan')
    yield dict(out=tmp_path,plan={'qwen_h0':deepcopy(qwen.CONFIG),'reviewer_config':{}},
               selected={'key':'VID01_100'},budget=budget,stop=threading.Event(),
               body={'messages':[{'role':'user','content':'prompt'}]}),calls,raw
    budget.close()


def test_paid_h0_journal_replays_without_second_post(setup):
    args,calls,raw=setup
    a=qwen.call_h0(**args)
    b=qwen.call_h0(**args)
    assert a==b and len(calls)==1
    assert Decimal(args['budget'].summary()['accounts']['aliyun_cny']['occupied'])==Decimal('0.0654')
    record=core.read(args['out']/'targets/VID01_100/qwen_h0/record.json')
    assert record['seat']=='qwen_h0' and record['status']=='JSON_PARSED'


def test_wrong_model_never_becomes_qwen_prediction_or_automatic_retry(setup):
    args,calls,raw=setup
    raw['model']='google/gemini-3.8-flash'
    with pytest.raises(ValueError,match='identity'):
        qwen.call_h0(**args)
    assert args['stop'].is_set()
    with pytest.raises(AmbiguousDispatch):qwen.call_h0(**args)
    assert len(calls)==1


def test_budget_failure_makes_no_request_and_leaves_no_target_journal(setup):
    args,calls,raw=setup
    args['budget'].reserve('other','aliyun_cny',Decimal('10'))
    with pytest.raises(BudgetStop):qwen.call_h0(**args)
    assert not calls and not (args['out']/'targets/VID01_100/qwen_h0').exists()


def test_finished_qwen_response_recovers_pending_reservation_without_post(setup):
    from scripts.testing_half_resume import reconcile_completed_reservations
    args,calls,raw=setup
    qwen.call_h0(**args)
    core.write(args['out']/'plan.json',{'limits':{'aliyun_cny':'10'}})
    with args['budget'].db:
        args['budget'].db.execute("UPDATE calls SET state='RESERVED'")
        args['budget'].db.execute("UPDATE meta SET value=? WHERE key='plan'",(core.sha(args['out']/'plan.json'),))
    assert reconcile_completed_reservations(args['out'])==1
    assert len(calls)==1 and not args['budget'].pending()


def test_qwen_scope_cannot_change_evaluation_frames(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new'
    old.mkdir();new.mkdir()
    core.write(old/'run_scope.json',{})
    row=dict(key='v_100',video_id='v',frame_id=100,stage='pipeline',evaluation_target=True,causal_frame_ids=[50,75,100])
    (old/'frame_inventory.jsonl').write_text(json.dumps(row)+'\n')
    (new/'frame_inventory.jsonl').write_text(json.dumps({**row,'evaluation_target':False})+'\n')
    core.write(new/'run_scope.json',{'base_model':qwen.MODEL,'parent_scope_path':str(old),
                                   'parent_scope_sha256':core.sha(old/'run_scope.json')})
    with pytest.raises(ValueError,match='timelines differ'):core.scope_rows(new)
