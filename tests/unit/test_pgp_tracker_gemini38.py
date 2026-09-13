from copy import deepcopy
import pytest
from tests.unit.test_gate_ready_mainline import inputs, RecordingMock, ready
from scripts.run_pgp_tracker_gemini38 import Gemini38Backend
from surgical_agent.research.gate import pgp_tracker_gemini38 as profile


class Tracker:
    def snapshot(self,s):
        return {'status':'AVAILABLE','video_id':s['video_id'],'source_max_frame_id':s['frame_id'],
            'frames':[{'frame_id':f,'tracks':[{'track_id':'track1','instrument_id':5,'score':.9,'bbox_tlwh':[.1,.1,.2,.2],'age':j+1}]} for j,f in enumerate(s['causal_frame_ids'])]}


@pytest.mark.parametrize('tracker',[None,Tracker()])
@pytest.mark.parametrize('action',[0,3])
def test_tracker_reaches_gate_and_gemini38_is_first_probe(inputs,tracker,action):
    base,s,prior,_=inputs; s={**s,'source_split':'Training'}; calls=RecordingMock()
    captured=[]
    def gate(features):
        captured.append(features)
        assert len(features)==53 and set(features)==set(profile.FEATURE_NAMES)
        assert features['tracker_available']==float(tracker is not None)
        if tracker: assert features['tracker_tool_count']==1
        assert all(not k.startswith('qwen_') for k in features)
        assert list(calls.wires)[-1]==(s['key'],'control_graph','gemini')
        assert len(calls.wires)==3
        return .5,action
    with ready.old.joint.roster.lightweight_protocol():
        result=profile.run(Gemini38Backend(calls,base,s),s,prior,gate,tracker=tracker)
    import json
    probe=json.loads(calls.wires[(s['key'],'control_graph','gemini')])
    assert probe['model']=='google/gemini-3.8-flash'
    assert probe['provider']['only']==['google-ai-studio']
    assert result['tracker_enabled']==(tracker is not None)
    assert result['call_keys'][:3]==['h0|base','proposal|base','control_graph|gemini']
    assert result['call_keys'].count('control_graph|gemini')==1
    assert len(result['call_keys'])<=13
    if action==0: assert len(result['call_keys'])==3


def test_reject_old_gate_before_loading_estimator():
    from pathlib import Path
    path=Path('artifacts/training/gate/pgp_ambiguity_assessment_20260913_r2/models/final_research_model.json')
    with pytest.raises(ValueError,match='retrained'): profile.load_predictor(path,True)


def test_bad_tracker_rejected_before_requests(inputs):
    base,s,prior,_=inputs; s={**s,'source_split':'Training'}; calls=RecordingMock()
    class BadTracker(Tracker):
        def snapshot(self,s):
            x=super().snapshot(s); x['video_id']='wrong'; return x
    with pytest.raises(ValueError,match='video mismatch'):
        profile.run(Gemini38Backend(calls,base,s),s,prior,lambda _: (.5,0),tracker=BadTracker())
    assert not calls.wires
