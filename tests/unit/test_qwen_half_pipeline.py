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


def test_authorized_historical_403_retries_without_erasing_charge(setup, monkeypatch):
    args,calls,raw=setup
    good_post=qwen.requests.post
    class Denied:
        status_code=403
        ok=False
        def json(self):
            return {'error':{'code':'AccessDenied.Unpurchased'}}
    monkeypatch.setattr(qwen.requests,'post',lambda *a,**k:Denied())
    with pytest.raises(ValueError):
        qwen.call_h0(**args)
    path=args['out']/'targets/VID01_100/qwen_h0/record.json'
    before=path.read_bytes()
    reserved=Decimal(json.loads(before)['charge'])
    core.write(path.with_name('retry_authorized.json'),{'record_sha256':core.sha(path)})
    args['stop'].clear()
    monkeypatch.setattr(qwen.requests,'post',good_post)
    qwen.call_h0(**args)
    assert path.read_bytes()==before
    assert args['budget'].summary()['dispatches']==2
    assert Decimal(args['budget'].summary()['accounts']['aliyun_cny']['occupied'])==reserved+Decimal('.0654')


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


def test_proposal_uses_distinct_journal_and_does_not_parse_as_h0(setup):
    args,calls,raw=setup
    raw['choices'][0]['message']['content']=json.dumps({'candidate_proposals':[]})
    result=qwen.call_base(**args,stage='proposal')
    assert result=={'candidate_proposals':[]}
    record=core.read(args['out']/'targets/VID01_100/qwen_proposal/record.json')
    assert record['stage']=='proposal' and record['seat']=='qwen_proposal'
    assert qwen.call_base(**args,stage='proposal')==result
    assert len(calls)==1


def test_proposal_wrapper_preserves_reviewer_routes(setup):
    args,calls,raw=setup
    class Delegate:
        def call(self,*a):return a
    wrapper=qwen.ProposalCalls(Delegate(),*(args[k] for k in ('out','plan','selected','budget','stop')))
    raw['choices'][0]['message']['content']='{}'
    assert wrapper.call('VID01_100','proposal','base',args['body'])=={}
    assert calls[0]['model']=='qwen3.8-max'
    assert wrapper.call('VID01_100','five_head_v1','gpt',{})==('VID01_100','five_head_v1','gpt',{})


def test_long_proposal_reserves_more_preserves_prompt_and_settles_actual_usage(setup, monkeypatch):
    args,calls,raw=setup
    args['body']['messages'][0]['content']='x'*40000
    original=deepcopy(args['body'])
    raw['choices'][0]['message']['content']='{"candidate_proposals":[]}'
    reserved=[]
    reserve=args['budget'].reserve
    def capture(identity,account,amount):
        reserved.append(amount)
        return reserve(identity,account,amount)
    monkeypatch.setattr(args['budget'],'reserve',capture)
    qwen.call_base(**args,stage='proposal')
    record=core.read(args['out']/'targets/VID01_100/qwen_proposal/record.json')
    assert record['reserved_input_tokens']==41984
    assert reserved==[qwen.charge({'prompt_tokens':41984,'completion_tokens':4096},qwen.CONFIG)]
    assert calls[0]['messages']==original['messages'] and args['body']==original
    assert Decimal(args['budget'].summary()['accounts']['aliyun_cny']['occupied'])==Decimal('0.0654')
    # Replay must not estimate input again, or dispatch/charge a second time.
    monkeypatch.setattr(qwen,'reservation_input_tokens',lambda *a:pytest.fail('cached request re-estimated'))
    qwen.call_base(**args,stage='proposal')
    assert len(calls)==1


def test_dynamic_reservation_still_blocks_before_dispatch(setup):
    args,calls,raw=setup
    args['body']['messages'][0]['content']='x'*40000
    args['budget'].reserve('other','aliyun_cny',Decimal('9.4'))
    with pytest.raises(BudgetStop):qwen.call_base(**args,stage='proposal')
    assert not calls and not (args['out']/'targets/VID01_100/qwen_proposal').exists()


def test_reservation_accounts_for_utf8_and_images():
    import base64,io
    from PIL import Image
    buf=io.BytesIO();Image.new('RGB',(1400,1400)).save(buf,format='PNG')
    wire={'messages':[{'content':[{'type':'text','text':'手'*10000},
          {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}}]}]}
    assert qwen.reservation_input_tokens(wire,qwen.CONFIG)==41984


def test_bad_schema_is_deferred_and_retry_has_separate_cost_and_journal(setup):
    from surgical_agent.api.errors import ApiSchemaError
    args,calls,raw=setup
    valid = raw['choices'][0]['message']['content']
    raw['choices'][0]['message']['content'] = json.dumps({
        'schema_version':'wrong', 'instrument':[0], 'verb':[0], 'target':[0], 'ivt':[0]})
    with pytest.raises(ApiSchemaError):
        qwen.call_h0(**args)
    assert not args['stop'].is_set()
    original = (args['out']/'targets/VID01_100/qwen_h0/record.json').read_bytes()
    raw['choices'][0]['message']['content'] = valid
    result = qwen.call_h0(**args)
    assert qwen.call_h0(**args) == result
    assert len(calls) == 2
    assert (args['out']/'targets/VID01_100/qwen_h0/record.json').read_bytes() == original
    assert (args['out']/'targets/VID01_100/qwen_h0_retry_1/record.json').exists()
    assert args['budget'].summary()['dispatches'] == 2
    assert Decimal(args['budget'].summary()['accounts']['aliyun_cny']['occupied']) == Decimal('.1308')
    assert not args['budget'].pending()


def test_deferred_frames_run_after_healthy_frames(tmp_path):
    seen = []
    def task(item):
        seen.append(item['key'])
        if item['key'] == 'bad' and seen.count('bad') == 1:
            raise ValueError('invalid frame response')
    core.deferred_map(task, [{'key':'bad'}, {'key':'good'}], 1, threading.Event(), tmp_path)
    assert seen == ['bad', 'good', 'bad']
    assert not list((tmp_path/'deferred_errors').glob('*.json'))


def test_persistent_frame_errors_are_bounded_without_losing_healthy_work(tmp_path):
    seen = []
    def task(item):
        seen.append(item['key'])
        if item['key'] == 'bad':
            raise OSError('local frame unavailable')
    with pytest.raises(RuntimeError, match='3 deferred retry'):
        core.deferred_map(task, [{'key':'bad'}, {'key':'good'}], 1, threading.Event(), tmp_path)
    assert seen == ['bad', 'good', 'bad', 'bad', 'bad']
    assert core.read(tmp_path/'deferred_errors/bad.json')['state'] == 'PENDING_RETRY'


def test_deferred_map_stops_immediately_on_api_or_budget_stop(tmp_path):
    seen = []
    stop = threading.Event()
    def task(item):
        seen.append(item['key'])
        raise BudgetStop('API balance exhausted')
    with pytest.raises(BudgetStop):
        core.deferred_map(task, [{'key':'bad'}, {'key':'good'}], 1, stop, tmp_path)
    assert seen == ['bad'] and stop.is_set()


@pytest.mark.parametrize('status,error,retry', [
    (400, {'code':'invalid_parameter'}, True), (413, {}, True),
    (401, {}, False), (403, {}, False), (429, {}, False), (503, {}, False),
    (400, {'code':'insufficient_quota'}, False),
])
def test_qwen_http_failure_classification(status, error, retry):
    assert qwen.retryable_record(dict(status='FAILED', finished_utc='done',
        http_status=status, provider_error=error, response_sha256='saved', error_type='ValueError')) is retry
