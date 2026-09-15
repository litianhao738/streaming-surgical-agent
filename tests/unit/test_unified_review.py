from copy import deepcopy
from types import SimpleNamespace
import pytest
from surgical_agent.research.gate import pgp_runtime as original
from surgical_agent.research.gate.unified_review import verify


@pytest.mark.parametrize('interaction,old_phase,alternative,depth', [
    (5, 5, 1, 5),  # phase settled, interaction still needs the remaining seats
    (1, 1, 5, 5),  # interaction settled, phase still needs the remaining seats
    (1, 5, 1, 2),  # both decisions provably retain their current values
    (None, None, None, 1),  # invalid evidence cannot produce a change
])
def test_shared_panel_waits_for_both_decisions(interaction, old_phase, alternative, depth):
    h0 = {'instrument':[0], 'verb':[], 'target':[], 'ivt':[], 'phase':[0]}
    pool = {'propositions':[{'id':'instrument_1', 'task':'instrument', 'label_id':1}]}
    calls = []
    def query(stage, seat, fn):
        assert (stage, seat) not in calls
        calls.append((stage,seat)); return fn()
    def aggregate(reviews, jp, **kwargs):
        diagnostics = {}; means = {}
        for p in jp['propositions']:
            pid = p['id']
            scores = [reviews[s].get(pid) if reviews[s] else None for s in original.SEATS]
            diagnostics[pid] = {'scores':scores, 'invalid':{s:['invalid'] for s,v in zip(original.SEATS,scores) if v is None}}
            means[pid] = None if any(v is None for v in scores) else sum(scores)/5
        return means, diagnostics
    def select(current, sub, means, prior, **kwargs):
        assert set(means) == {'instrument_1'}
        out = deepcopy(current)
        if means['instrument_1'] is not None and means['instrument_1'] >= 4: out['instrument'].append(1)
        return out, {}
    def decide(current, means):
        assert all(f'phase_{i}' in means for i in range(7))
        return ([1] if means['phase_1'] is not None and means['phase_1'] >= 4 and means['phase_1'] > means['phase_0'] else [0]), {}
    values = {'instrument_1':interaction, **{f'phase_{i}':(old_phase if i == 0 else alternative if i == 1 else 1) for i in range(7)}}
    backend = SimpleNamespace(phase_recommendation=lambda *a: None, joint=lambda *a: deepcopy(values))
    api = SimpleNamespace(**{**vars(original),
        'joint_pool':lambda p: {'propositions':p['propositions']+[{'id':f'phase_{i}'} for i in range(7)]},
        'normalize_five_heads':lambda reviews, *a, **k:(reviews, {}),
        'aggregate_five_heads':aggregate, 'select_prior_gated':select, 'decide_phase':decide})
    out, actual_depth, _ = verify(backend, h0, h0, pool, {}, query, api)
    assert actual_depth == depth and len(calls) == depth + 1
    assert all(stage == 'joint_r1' for stage, _ in calls[1:])
    assert (1 in out['instrument']) == (interaction == 5)
    assert out['phase'] == ([1] if alternative == 5 else [0])
