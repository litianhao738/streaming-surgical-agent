"""Check isolation, invalid evidence, routing and metric denominators."""
import sys
from pathlib import Path
from copy import deepcopy
import unittest
from unittest.mock import patch

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parent
sys.path[:0] = [str(EXPERIMENT/'src'), str(ROOT/'src')]
from mechanisms import aggregate, decide, selections, rule_score, SLOTS, original
from metrics import evaluate


def snapshot():
    h0 = {'instrument':[0], 'verb':[], 'target':[], 'ivt':[], 'phase':[0]}
    pool = original.make_pool(h0)
    return {'h0':h0, 'cheap':deepcopy(h0), 'pool':pool,
            'probe_ratings_all':{**{p['id']:5 for p in pool['propositions']},
                                 **{f'phase_{p}':3 for p in range(7)}}, 'gate_score':0.5}


class MechanismContracts(unittest.TestCase):
    def test_unobserved_slot_cannot_be_confused_with_invalid_terminal(self):
        s = snapshot()
        with self.assertRaises(ValueError): aggregate(s['pool'], {'qwen':None})
        panel = aggregate(s['pool'], {seat:None for seat in SLOTS})
        self.assertTrue(all(mean is None for mean in panel['means'].values()))
        result,_ = decide(s,panel,{},prior_after=False,rollback=False)
        self.assertEqual(result,s['h0'])

    def test_prior_switch_is_local_and_current_stays_h0(self):
        s = snapshot();s['cheap']['instrument']=[1]
        panel=aggregate(s['pool'],{seat:None for seat in SLOTS})
        global_gate=deepcopy(original.GATE)
        with patch.object(original,'select_prior_gated',return_value=(deepcopy(s['h0']),{})) as select:
            decide(s,panel,{},prior_after=False,rollback=False)
            self.assertEqual(select.call_args.args[0],s['h0'])
            self.assertIsNone(select.call_args.kwargs['veto_rate'])
            self.assertIsNone(select.call_args.kwargs['add_rate'])
            self.assertEqual(select.call_args.kwargs['phase'],0)
        self.assertEqual(original.GATE,global_gate)
        self.assertEqual(s['cheap']['instrument'],[1])

    def test_ambiguity_rollback_uses_cheap_and_changes_only_verb_ivt(self):
        s=snapshot(); ambiguous=sorted(original.AMB_VERBS)
        s['cheap']['verb']=[ambiguous[0]]
        reviewed=deepcopy(s['h0']);reviewed['verb']=[ambiguous[-1]]
        panel=aggregate(s['pool'],{seat:None for seat in SLOTS})
        def fake_select(*args,**kwargs): return deepcopy(reviewed),{}
        with patch.object(original,'select_prior_gated',side_effect=fake_select):
            on,_=decide(s,panel,{},rollback=True)
            off,_=decide(s,panel,{},rollback=False)
        self.assertEqual(on['verb'],s['cheap']['verb'])
        self.assertEqual(off['verb'],reviewed['verb'])
        for h in ('instrument','target','phase'):self.assertEqual(on[h],off[h])

    def test_rule_gate_missing_phase_not_an_opposing_vote(self):
        s=snapshot();s['probe_ratings_all']['phase_1']=5
        self.assertEqual(rule_score(s),0.5)
        s['probe_ratings_all']['phase_2']=None
        self.assertEqual(rule_score(s),0)

    def test_equal_budget_and_order_independent_random_draws(self):
        snapshots={str(i):snapshot() for i in range(30)}
        for i,s in enumerate(snapshots.values()):s['gate_score']=i/30
        a=selections(snapshots,0.5)
        b=selections(dict(reversed(list(snapshots.items()))),0.5)
        self.assertEqual(a,b)
        self.assertEqual(a['K'],15)
        for group in [a['rule'],*a['random'].values()]:self.assertEqual(len(set(group)),15)
        self.assertGreater(len({tuple(v) for v in a['random'].values()}),1)

    def test_masks_and_partial_repairs_use_declared_denominators(self):
        empty={h:[] for h in ('instrument','verb','target','ivt','phase')}
        before={'a':{**empty,'instrument':[0]},'b':{**empty,'instrument':[0,1]},'c':empty}
        after={'a':{**empty,'instrument':[1]},'b':{**empty,'instrument':[0]},'c':empty}
        truth={k:{'gt':{**empty,'instrument':[0]},'mask':{h:False for h in empty}} for k in before}
        for k in ('a','b'):truth[k]['mask']['instrument']=True
        result=evaluate(list(before),before,after,truth)
        counts=result['repair']
        self.assertEqual((counts['n'],counts['C'],counts['F'],counts['H'],counts['D']),(2,1,1,1,2))
        self.assertEqual(counts['harm_percent'],100)
        self.assertEqual(counts['net_gain_pp'],0)
        self.assertIsNone(result['by_head']['phase']['accuracy'])


if __name__=='__main__':unittest.main()
