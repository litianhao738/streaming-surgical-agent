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


def information(tracker='on', gate_model=None):
    config=json.loads((ROOT/'configs/pgp_tracker_gemini38.json').read_text('utf-8'))
    selected=gate_model or config.get('gate_models',{}).get(tracker) or config.get('gate_model')
    path=(ROOT/selected).resolve() if selected else None
    available=path is not None and path.is_file()
    metadata=json.loads(path.read_text('utf-8')) if available else {}
    compatible=False; error=None
    if available:
        try:
            profile.load_predictor(path,tracker=='on');compatible=True
        except (ValueError,KeyError,FileNotFoundError) as exc:error=str(exc)
    return {'profile':profile.PROFILE,'probe_model':profile.MODEL,'tracker':tracker,
        'feature_count':len(profile.FEATURE_NAMES),'compact_order':['gemini','qwen','gpt','grok','deepseek'],
        'trained_model_available':available,'compatible_model':compatible,
        'gate_model':str(path) if path else None,'evaluation_pass':metadata.get('evaluation_pass',False),
        'status':config['status'],'validation_error':error,'selected_as_default':False,
        'api_calls':0,'Testing_access':False,'deployable':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['info','check-model'],nargs='?',default='info')
    p.add_argument('--gate-model',type=Path); p.add_argument('--tracker',choices=['on','off'],default='on')
    args=p.parse_args()
    if args.command=='info':
        print(json.dumps(information(args.tracker,args.gate_model),indent=2)); return
    if args.gate_model is None: p.error('--gate-model required')
    profile.load_predictor(args.gate_model,args.tracker=='on')
    print('Compatible model loaded; no API calls')


if __name__=='__main__': main()
