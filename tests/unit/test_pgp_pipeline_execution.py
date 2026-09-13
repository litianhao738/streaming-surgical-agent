from copy import deepcopy
from argparse import Namespace
import json
import hashlib
from pathlib import Path
import pytest
from tests.unit.test_gate_ready_mainline import inputs, RecordingMock, ready
from scripts import pgp_pipeline_execution as execution


def test_prepare_execute_with_recording_transport_and_no_real_requests(tmp_path,monkeypatch,inputs):
    base,selected,prior,_=inputs
    selected={**selected,'video_id':'VID103','key':'VID103_100','anchor_frame_id':100,'alignment_version':'synthetic'}
    prior={**prior,'excluded_video':'VID103'}
    from surgical_agent.research.retrieval.prior_candidates import digest
    prior['table_sha256']=digest({k:v for k,v in prior.items() if k!='table_sha256'})
    source=tmp_path/'source'; (source/'priors').mkdir(parents=True)
    images=[]
    for fid in selected['causal_frame_ids']:
        p=source/f'{fid}.png'; p.write_bytes(b'synthetic-only')
        images.append({'path':str(p),'frame_id':fid,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    selected['images']=images
    limits={'openrouter_usd':'0','aliyun_cny':'0','glm_requests':'0','deepseek_requests':'0','xai_usd':'0'}
    (source/'plan.json').write_text(json.dumps({'selection':[selected],'limits':limits}),encoding='utf-8')
    (source/'priors/VID103.json').write_text(json.dumps(prior),encoding='utf-8')
    caps=tmp_path/'caps.json'; caps.write_text(json.dumps(limits),encoding='utf-8')
    args=Namespace(output=tmp_path/'out',source=source,limit=1,budget_limits=caps,allow_paid=False)
    execution.prepare(args)
    with pytest.raises(ValueError,match='allow-paid'): execution.execute(args)
    assert not (args.output/'execution_started.json').exists()
    import scripts.full_official_reviewer_transport as transport
    import scripts.collect_gate_escalation_v1 as collector
    mocks=[]
    class FakeCalls(RecordingMock):
        def __init__(self,*a,**kw): super().__init__(); mocks.append(self)
        def close(self): pass
    monkeypatch.setattr(transport,'Calls',FakeCalls)
    monkeypatch.setattr(collector,'load_base',lambda _:base)
    monkeypatch.setattr(execution,'predictor',lambda:(lambda _: (.1,0),{},{}))
    import requests
    monkeypatch.setattr(requests,'post',lambda *a,**kw:pytest.fail('paid request in test'))
    args.allow_paid=True
    with ready.old.joint.roster.lightweight_protocol(): execution.execute(args)
    result=json.loads((args.output/'targets/VID103_100/pgp_result.json').read_text('utf-8'))
    assert result['logical_calls']==3 and result['gate_action']==0
    with pytest.raises(FileExistsError): execution.execute(args)
