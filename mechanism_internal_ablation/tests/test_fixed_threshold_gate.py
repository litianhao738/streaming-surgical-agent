import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
from fixed_threshold_gate import selection_metrics, uncertainty_score, HEADS


class FixedThresholdMetrics(unittest.TestCase):
    def test_unknown_heads_and_unscored_selected_frames_do_not_enter_denominator(self):
        cheap={k:{h:[] for h in HEADS} for k in ('a','b','unscored')}
        full={k:{h:[] for h in HEADS} for k in cheap}
        full['a']['instrument']=[1]
        truth={k:{'gt':{h:[1] for h in HEADS},'mask':{h:h=='instrument' for h in HEADS}} for k in ('a','b')}
        result=selection_metrics(['a','b'],['a','unscored'],cheap,full,truth)
        self.assertEqual(result['error_recall'],50.)
        self.assertEqual(result['bvr'],100.)
        self.assertEqual(result['reviewed_eligible_frames'],1)
        empty=selection_metrics(['a','b'],[],cheap,full,truth)
        self.assertIsNone(empty['bvr'])

    def test_phase_tie_is_uncertain_and_clear_gap_is_certain(self):
        ratings={f'phase_{i}':1 for i in range(7)}
        ratings['phase_0']=5
        ratings['p']=5
        snapshot={'pool':{'propositions':[{'id':'p','task':'instrument','label_id':0}]},'probe_ratings_all':ratings}
        self.assertEqual(uncertainty_score(snapshot),0.)
        ratings['phase_1']=5
        self.assertEqual(uncertainty_score(snapshot),1.)
        ratings['phase_1']=None
        self.assertEqual(uncertainty_score(snapshot),1.)


if __name__=='__main__':unittest.main()
