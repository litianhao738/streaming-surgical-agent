"""Complete PGP inference orchestration. Offline info/preflight/replay by default.

Live execution uses an explicitly supplied frozen Training plan and the existing
durable, budgeted transport. It requires --allow-paid; never enabled implicitly.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import json
import sys
from pathlib import Path
from copy import deepcopy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from surgical_agent.research.gate.pgp_runtime import run_target
from surgical_agent.research.gate.pgp_default import load_default
from surgical_agent.research.gate.pgp_ambiguity import predict
from scripts import run_prior_gated_joint_confirmation as frozen


class CachedBackend:
    """Responses are revealed one call at a time; no final/GT access by engine."""
    normalize_compact=staticmethod(frozen.common.normalize_five)
    parse_h0=staticmethod(frozen.gated.h0_from_raw)
    def __init__(self,record):
        self.responses={'h0':deepcopy(record['h0_raw']), 'proposal':deepcopy(record['proposal_raw']),
            'compact':deepcopy(record['review_raw']), 'joint':deepcopy(record['joint_raw']),
            'phase_recommendation':deepcopy(record['phase_recommendation']['raw'])}
    def h0(self): return deepcopy(self.responses['h0'])
    def proposal(self,*args): return deepcopy(self.responses['proposal'])
    def compact(self,seat,*args): return deepcopy(self.responses['compact'].get(seat))
    def phase_recommendation(self,*args): return deepcopy(self.responses['phase_recommendation'])
    def joint(self,seat,*args): return deepcopy(self.responses['joint'].get(seat))


class WireBackend:
    normalize_compact=staticmethod(frozen.common.normalize_five)
    parse_h0=staticmethod(frozen.gated.h0_from_raw)
    def __init__(self,calls,base,selected): self.calls,self.base,self.selected=calls,base,selected
    def call(self,stage,seat,wire): return self.calls.call(self.selected['key'],stage,seat,wire)
    def h0(self): return self.call('h0','base',frozen.gemini_h0_wire(self.base))
    def proposal(self,h0,pool,prior):
        hints=frozen.retrieve_candidate_hints(h0,prior,video_id=self.selected['video_id'])
        return self.call('proposal','base',frozen.proposal_wire(self.base,self.selected,h0,pool,hints['packet']))
    def compact(self,seat,h0,pool):
        wire=frozen.joint.roster.review_wire(seat,self.base,self.selected,pool)
        frozen.check_requests(h0,[wire]); return self.call('control_graph',seat,wire)
    def phase_recommendation(self,h0,pool,prior):
        hints=frozen.retrieve_candidate_hints(h0,prior,video_id=self.selected['video_id'])
        wire=frozen.phase_recommendation_wire(self.base,self.selected,h0,pool,hints)
        frozen.check_requests(h0,[wire]); return self.call('phase_recommendation','base',wire)
    def joint(self,seat,h0,pool,rec):
        error=frozen.joint.phase_choice_error(rec,3)
        wire=frozen.joint_review_wires(self.base,self.selected,h0,pool,None if error else rec['phase_id'])[seat]
        frozen.check_requests(h0,[wire]); return self.call('joint_r1',seat,wire)


def predictor():
    import joblib
    import hashlib
    manifest,model=load_default()
    # Validate and load once per run; avoid deserializing the estimator per frame.
    path=ROOT/model['estimator_artifact']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=model['estimator_sha256']: raise ValueError('estimator changed')
    estimator=joblib.load(path)
    def call(features):
        if set(features)!=set(model['feature_names']): raise ValueError('runtime feature schema mismatch')
        x=np.array([[features[k] for k in model['feature_names']]])
        if not np.isfinite(x).all(): raise ValueError('nonfinite runtime input')
        score=float(estimator.predict_proba(x)[0,1]); return score,3*int(score>=model['threshold'])
    return call,manifest,model


def replay(source,output,limit=None):
    from tools.audit.gate_proposals_offline_20260913 import offline
    from surgical_agent.research.gate.pgp_ambiguity import repair
    from surgical_agent.research.verification.prior_panel import labels
    from scripts.train_pgp_gate_no_tracker import sha,write
    output.mkdir(exist_ok=False)
    call,manifest,model=predictor()
    # Target inventory belongs to this explicitly named Training cache only.
    rows=json.loads((ROOT/'artifacts/training/gate/official_behavior_net_v2_20260913/primary_rows.json').read_text('utf-8'))
    sealed=json.loads((ROOT/'artifacts/training/gate/pgp_gate_no_tracker_20260913_r1/replay/pgp_reviewer_source_sha256.json').read_text('utf-8'))
    prior={v:json.loads((source/'priors'/f'{v}.json').read_text('utf-8')) for v in ('VID103','VID23','VID31','VID96')}
    results=[];total=0
    with offline(), (output/'predictions.jsonl').open('x',encoding='utf-8') as out:
        for r in rows[:limit]:
            path=source/'targets'/r['sample_id']/'result.json'
            if sha(path)!=sealed[str(path.resolve())]: raise ValueError('unsealed cache')
            record=json.loads(path.read_text('utf-8')); frame=r['frame_id']
            selected={'key':r['sample_id'],'video_id':r['video_id'],'frame_id':frame,
                      'causal_frame_ids':[frame-50,frame-25,frame],'source_split':'Training'}
            result=run_target(CachedBackend(record),selected,prior[r['video_id']],call)
            assert result['cheap']==r['cheap_labels'], 'cheap mismatch '+r['sample_id']
            expected=repair(r['cheap_labels'],r['final_labels']) if result['gate_action'] else r['cheap_labels']
            assert labels(result['prediction'])==labels(expected), 'prediction mismatch '+r['sample_id']
            for k,v in r['features_postcheap'].items():
                if not k.startswith('tracker_'): assert result['features'][k]==v, (r['sample_id'],k)
            out.write(json.dumps(result,ensure_ascii=False)+'\n'); total+=result['logical_calls']; results.append(result)
            if len(results)%500==0: print('Verified end-to-end targets',len(results),flush=True)
    write(output/'receipt.json',{'state':'PASS','rows':len(results),'logical_calls':total,'api_calls':0,
        'Testing_access':False,'default_version':manifest['version'],'tracker_enabled':False,'prediction_mismatches':0,
        'feature_mismatches':0,'prediction_sha256':sha(output/'predictions.jsonl')})
    print(json.dumps({'state':'PASS','rows':len(results),'logical_calls':total,'api_calls':0}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['info','preflight','replay','prepare','execute','score'],nargs='?',default='info')
    p.add_argument('--source',type=Path,default=ROOT/'artifacts/training/gate/full_official_reviewers_20260912_v1')
    p.add_argument('--output',type=Path); p.add_argument('--limit',type=int)
    p.add_argument('--allow-paid',action='store_true'); p.add_argument('--dataset-root',type=Path)
    p.add_argument('--budget-limits',type=Path,help='JSON account caps for a new paid execution plan')
    p.add_argument('--annotations',type=Path,help='Explicit Training annotations; used only by score after inference')
    p.add_argument('--variant',choices=['qwen','tracker-gemini38'],default='qwen')
    p.add_argument('--tracker',choices=['on','off'],default='on')
    p.add_argument('--gate-model',type=Path)
    args=p.parse_args()
    if args.command=='info':
        print(json.dumps(load_default()[0],indent=2)); return
    if args.command in ('replay','preflight'):
        if args.variant!='qwen':
            raise ValueError('old Qwen/Gemini-3.5 caches cannot replay the Gemini-3.8 variant; matching reviewer caches required')
        if args.output is None: p.error('--output fresh directory required')
        if args.limit is not None and args.limit<1: p.error('--limit must be positive')
        replay(args.source.resolve(),args.output.resolve(),args.limit if args.command=='replay' else (args.limit or 8)); return
    from scripts.pgp_pipeline_execution import dispatch
    dispatch(args)


if __name__=='__main__': main()
