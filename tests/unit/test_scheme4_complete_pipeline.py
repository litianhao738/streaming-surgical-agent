from argparse import Namespace
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import pytest
from tests.unit.test_gate_ready_mainline import inputs, RecordingMock, ready
from scripts import run_tracker_scheme4_pipeline as app


@pytest.mark.parametrize('action',[0,1])
def test_prepare_execute_score_without_network(tmp_path,monkeypatch,inputs,action):
    base, selected, prior, _ = inputs
    selected = {**selected,'key':'VID103_100','video_id':'VID103','anchor_frame_id':100,
                'source_split':'Training','alignment_version':'synthetic'}
    from surgical_agent.research.verification.prior_panel import digest
    prior = {**prior,'excluded_video':'VID103'}
    prior['table_sha256'] = digest({k:v for k,v in prior.items() if k!='table_sha256'})
    source = tmp_path/'source'; (source/'priors').mkdir(parents=True)
    app.write(source/'priors/VID103.json',prior)
    images = []
    for frame in selected['causal_frame_ids']:
        path = source/f'{frame}.png'; path.write_bytes(b'synthetic')
        images.append({'path':str(path),'frame_id':frame,'sha256':app.sha(path)})
    selected['images'] = images
    index = tmp_path/'index.json'; app.write(index,{})
    class Tracker:
        def __init__(self,*args): pass
        def snapshot(self,s):
            return {'status':'AVAILABLE','video_id':s['video_id'],'source_max_frame_id':s['frame_id'],
                    'frames':[{'frame_id':s['frame_id'],'tracks':[{'instrument_id':2,'score':1.0}]}]}
    monkeypatch.setattr(app,'FrozenTracker',Tracker)
    real_predict,manifest,model = app.load_gate()
    monkeypatch.setattr(app,'load_gate',lambda:(lambda _: (.5,action),manifest,model))
    caps = dict.fromkeys(('openrouter_usd','aliyun_cny','glm_requests','deepseek_requests','xai_usd'),'0')
    monkeypatch.setattr(app,'inventory',lambda *a:({'limits':caps},[selected]))
    limits = tmp_path/'limits.json'; app.write(limits,caps)
    args = Namespace(source=source,output=tmp_path/'out',limit=1,budget_limits=limits,tracker_index=index,allow_paid=False)
    app.prepare(args)
    with pytest.raises(ValueError,match='allow-paid'): app.execute(args)
    assert not (args.output/'execution_started.json').exists()
    mocks = []
    class Calls(RecordingMock):
        def __init__(self,*a): super().__init__(); mocks.append(self)
        def close(self): pass
    import scripts.scheme4_transport as transport
    import scripts.collect_gate_escalation_v1 as collector
    monkeypatch.setattr(transport,'GuardedCalls',Calls)
    monkeypatch.setattr(collector,'load_base',lambda _:base)
    monkeypatch.setattr(app.frozen.joint,'credential_context',lambda _:nullcontext())
    import requests
    monkeypatch.setattr(requests,'post',lambda *a,**k:pytest.fail('network in mocked test'))
    args.allow_paid=True
    app.execute(args)
    result = json.loads((args.output/'predictions.jsonl').read_text())
    assert result['prediction']['instrument']==[2]
    assert 2 in result['prediction']['verb']
    assert result['gate_action']==action and 3<=result['logical_calls']<=7
    assert all(r['stage'] not in ('joint_r1','phase_recommendation') for r in mocks[0].rows)
    with pytest.raises(FileExistsError): app.execute(args)
    args.annotations = tmp_path/'truth.json'
    app.write(args.annotations,[{'sample_id':selected['key'],'video_id':'VID103','source_split':'Training',
                               'mask':dict.fromkeys(result['prediction'],True),'gt':result['prediction']}])
    app.score(args)
    assert app.read(args.output/'scores.json')['errors']==0


def test_loader_rejects_model_tampering(tmp_path):
    manifest = app.read(app.ROOT/'DEFAULT_PGP_GATE_VERSION.json')
    model_path = tmp_path/manifest['model_artifact']; model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b'{}')
    app.write(tmp_path/'DEFAULT_PGP_GATE_VERSION.json',manifest)
    with pytest.raises(ValueError,match='hash'): app.load_gate(tmp_path)


def test_default_dispatches_scheme4(monkeypatch,tmp_path):
    from scripts import run_pipeline
    commands=[]
    monkeypatch.setattr(run_pipeline.subprocess,'run',lambda command,**kw:(commands.append(command) or Namespace(returncode=0)))
    assert run_pipeline.main(['preflight','--output',str(tmp_path/'out'),'--limit','2'])==0
    assert any('run_tracker_scheme4_pipeline.py' in p for p in commands[0])
    assert '--limit' in commands[0]
    with pytest.raises(SystemExit):
        run_pipeline.main(['replay','--output',str(tmp_path/'out'),'--tracker','off'])
