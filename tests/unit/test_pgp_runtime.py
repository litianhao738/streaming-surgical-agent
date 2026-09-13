from copy import deepcopy
import pytest
from tests.unit.test_gate_ready_mainline import inputs, RecordingMock, ready
from scripts.run_pgp_pipeline import WireBackend
from surgical_agent.research.gate.pgp_runtime import run_target, phase_settled, settled
from surgical_agent.research.gate.pgp_ambiguity import repair
from scripts import run_prior_gated_joint_mainline as full


@pytest.mark.parametrize('action',[0,3])
@pytest.mark.parametrize('replacements',[{}, {('control_graph','qwen'):None}, {('joint_r1','qwen'):None}, {('proposal','base'):None}])
def test_actual_wire_backend_matches_frozen_output_and_skips_calls(inputs,action,replacements):
    base,selected,prior,gate=inputs; selected={**selected,'source_split':'Training'}
    calls=RecordingMock(replacements); old=RecordingMock(replacements)
    seen=[]
    def predict(features):
        assert len(features)==42 and not any('tracker' in k for k in features)
        assert set(stage for _,stage,_ in calls.wires)=={'h0','proposal','control_graph'}
        assert len(calls.wires)==3
        seen.append(deepcopy(features)); return .5,action
    with ready.old.joint.roster.lightweight_protocol():
        result=run_target(WireBackend(calls,base,selected),selected,prior,predict)
        reference=full.run_target(old,base,selected,prior,gate)
    expected=repair(result['cheap'],reference['predictions'][full.PRIMARY]) if action else result['cheap']
    assert result['prediction']==expected
    assert result['logical_calls']==len(calls.wires)<=13
    assert sum(stage=='control_graph' and seat=='qwen' for _,stage,seat in calls.wires)==1
    for key,wire in calls.wires.items(): assert wire==old.wires[key]
    if not action: assert result['logical_calls']==3


def test_future_or_truth_rejected_before_any_request(inputs):
    base,s,prior,_=inputs; calls=RecordingMock(); backend=WireBackend(calls,base,s)
    for selected in ({**s,'source_split':'Testing'}, {**s,'source_split':'Training','gt':{}}, {**s,'source_split':'Training','causal_frame_ids':[50,100,125]}):
        with pytest.raises(ValueError): run_target(backend,selected,prior,lambda _: (.5,0))
    assert not calls.wires


def test_stop_bounds_do_not_replace_equality_with_pass():
    assert not settled([5,5,5,1],False)
    assert settled([None],True)
    assert phase_settled([[5]*3]+[[1]*3 for _ in range(6)],0)
    assert not phase_settled([[5]*3,[5]*3]+[[1]*3 for _ in range(5)],0)
