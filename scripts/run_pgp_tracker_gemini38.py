"""Tracker/Gemini-3.8 PGP entry. No implicit reuse of the Qwen model or responses."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.run_pgp_pipeline import WireBackend, frozen
from scripts.compare_mainline_backbones import base_body
from surgical_agent.research.gate import pgp_tracker_gemini38 as profile


class Gemini38Backend(WireBackend):
    probe_model=profile.MODEL
    def compact(self,seat,h0,pool):
        wire=frozen.joint.roster.review_wire(seat,self.base,self.selected,pool)
        if seat=='gemini':
            wire=base_body(wire,'gemini')
            wire['reasoning']={'effort':'low'}
            if wire['model']!=profile.MODEL: raise ValueError('Gemini 3.8 route drift')
        frozen.check_requests(h0,[wire]); return self.call('control_graph',seat,wire)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['info','check-model'],nargs='?',default='info')
    p.add_argument('--gate-model',type=Path); p.add_argument('--tracker',choices=['on','off'],default='on')
    args=p.parse_args()
    if args.command=='info':
        print(json.dumps({'profile':profile.PROFILE,'probe_model':profile.MODEL,'tracker':args.tracker,
            'feature_count':len(profile.FEATURE_NAMES),'compact_order':['gemini','qwen','gpt','grok','deepseek'],
            'trained_model_available':False,'status':'IMPLEMENTED_REQUIRES_MATCHING_PROBE_DATA_AND_RETRAINING',
            'api_calls':0,'Testing_access':False},indent=2)); return
    if args.gate_model is None: p.error('--gate-model required')
    profile.load_predictor(args.gate_model,args.tracker=='on')
    print('Compatible model loaded; no API calls')


if __name__=='__main__': main()
