"""Explicitly budgeted Training execution adapter; no network during preparation."""
from copy import deepcopy
import json
from pathlib import Path
import threading
from scripts.run_pgp_pipeline import ROOT, WireBackend, predictor
from surgical_agent.research.gate.pgp_runtime import run_target
from scripts.train_pgp_gate_no_tracker import sha,write


def prepare(args):
    if args.output is None or args.limit is None or args.limit<1 or args.budget_limits is None:
        raise ValueError('prepare requires fresh --output, positive --limit and explicit --budget-limits JSON')
    source=args.source.resolve(); original=json.loads((source/'plan.json').read_text('utf-8'))
    caps=json.loads(args.budget_limits.read_text('utf-8'))
    from decimal import Decimal
    if set(caps)!=set(original['limits']) or any(not Decimal(str(v)).is_finite() or Decimal(str(v))<0 for v in caps.values()):
        raise ValueError('explicit finite nonnegative caps for every original account required')
    selection=[]
    for s in original['selection']:
        if s['video_id'] not in ('VID103','VID23','VID31','VID96'): raise ValueError('Training-only plan required')
        t={k:deepcopy(s[k]) for k in ('key','video_id','frame_id','anchor_frame_id','causal_frame_ids','images','alignment_version')}
        t['source_split']='Training'; selection.append(t)
        if len(selection)==args.limit: break
    if len(selection)!=args.limit: raise ValueError('limit exceeds Training inventory')
    for t in selection:
        if len(t['images'])!=3: raise ValueError('three causal images required')
        for image,fid in zip(t['images'],t['causal_frame_ids'],strict=True):
            if image['frame_id']!=fid or sha(Path(image['path']))!=image['sha256']: raise ValueError('image mismatch')
    plan=deepcopy(original); plan['selection']=selection; plan['limits']={k:str(v) for k,v in caps.items()}
    plan['profile']='pgp-ambiguity-single-probe-training-v1'; plan['maximum_paid_calls']=13*len(selection)
    plan['pgp_default_sha256']=sha(ROOT/'DEFAULT_PGP_GATE_VERSION.json')
    plan['pgp_model']=json.loads((ROOT/'DEFAULT_PGP_GATE_VERSION.json').read_text('utf-8'))['model_artifact']
    plan['pgp_runtime_sha256']={p:sha(ROOT/p) for p in ('scripts/run_pgp_pipeline.py','scripts/pgp_pipeline_execution.py',
        'src/surgical_agent/research/gate/pgp_runtime.py','src/surgical_agent/research/gate/pgp_ambiguity.py','src/surgical_agent/research/gate/pgp_default.py')}
    plan['api_execution_validated']=False; plan['Testing_access']=False; plan['tracker_enabled']=False
    # This execution uses freshly submitted requests, never imports old response caches.
    plan['source_plan_sha256']=sha(source/'plan.json')
    out=args.output.resolve(); out.mkdir(exist_ok=False); (out/'priors').mkdir()
    plan['pgp_prior_sha256']={}
    for v in sorted({s['video_id'] for s in selection}):
        raw=(source/'priors'/f'{v}.json').read_bytes(); prior=json.loads(raw)
        if prior['excluded_video']!=v or v in prior['fit_videos']: raise ValueError('prior leakage')
        p=out/'priors'/f'{v}.json'; p.write_bytes(raw); plan['pgp_prior_sha256'][v]=sha(p)
    write(out/'plan.json',plan)
    print(json.dumps({'prepared':str(out),'targets':len(selection),'maximum_calls':plan['maximum_paid_calls'],'api_calls':0}))


def execute(args):
    if not args.allow_paid: raise ValueError('paid execution requires explicit --allow-paid; no calls made')
    if args.output is None: raise ValueError('--output prepared directory required')
    out=args.output.resolve(); plan=json.loads((out/'plan.json').read_text('utf-8'))
    if plan['profile']!='pgp-ambiguity-single-probe-training-v1': raise ValueError('wrong prepared profile')
    if sha(ROOT/'DEFAULT_PGP_GATE_VERSION.json')!=plan['pgp_default_sha256']: raise ValueError('default changed after prepare')
    for p,h in plan['pgp_runtime_sha256'].items():
        if sha(ROOT/p)!=h: raise ValueError('runtime changed after prepare')
    for v,h in plan['pgp_prior_sha256'].items():
        if sha(out/'priors'/f'{v}.json')!=h: raise ValueError('prior changed')
    for s in plan['selection']:
        if s.get('source_split')!='Training' or s['video_id'] not in ('VID103','VID23','VID31','VID96'): raise ValueError('Training-only execution')
        for image in s['images']:
            if sha(Path(image['path']))!=image['sha256']: raise ValueError('image changed')
    decide,_,_=predictor()
    # Exclusive marker prevents concurrent or repeated paid executions. Interrupted
    # records remain intact for explicit reconciliation, never blind retry.
    write(out/'execution_started.json',{'state':'STARTED','automatic_retry':False})
    from scripts.full_official_reviewer_transport import Calls
    from scripts.collect_gate_escalation_v1 import load_base
    from surgical_agent.research.gate.collection_budget import Budget
    budget=Budget(out/'budget.sqlite',plan['limits'],sha(out/'plan.json')); stop=threading.Event()
    completed=0
    try:
        for s in plan['selection']:
            (out/'targets'/s['key']).mkdir(parents=True,exist_ok=True)
            transport=Calls(out,plan,s,budget,stop)
            try:
                prior=json.loads((out/'priors'/f"{s['video_id']}.json").read_text('utf-8'))
                result=run_target(WireBackend(transport,load_base(s),s),s,prior,decide)
                write(out/'targets'/s['key']/'pgp_result.json',result); completed+=1
            finally: transport.close()
            if stop.is_set(): break
        write(out/'execution_receipt.json',{'state':'COMPLETED' if completed==len(plan['selection']) else 'STOPPED',
              'completed_targets':completed,'budget':budget.summary(),'Testing_access':False,'tracker_enabled':False})
    finally: budget.close()


def dispatch(args):
    if args.command=='prepare': return prepare(args)
    if args.command=='execute': return execute(args)
    if args.command=='score':
        if args.output is None or args.annotations is None: raise ValueError('score requires --output and explicit --annotations Training JSON')
        from surgical_agent.research.gate.official_net_training import counts, pooled
        rows=json.loads(args.annotations.read_text('utf-8'))
        if any(r.get('source_split')!='Training' or r.get('video_id') not in ('VID103','VID23','VID31','VID96') for r in rows):
            raise ValueError('only explicit Training annotations accepted')
        truth={r['sample_id']:r for r in rows}; out=args.output.resolve()
        if (out/'predictions.jsonl').exists(): predictions=[json.loads(x) for x in (out/'predictions.jsonl').read_text('utf-8').splitlines()]
        else: predictions=[json.loads(p.read_text('utf-8')) for p in sorted((out/'targets').glob('*/pgp_result.json'))]
        if not predictions: raise ValueError('no saved inference predictions')
        joined=[{**truth[r['key']],'pred':r['prediction']} for r in predictions]
        result={'original':pooled(counts(joined,'pred',False)),'merged':pooled(counts(joined,'pred',True)),
                'rows':len(joined),'scope':'post-inference Training evaluation; final fitted model, not outer OOF',
                'api_calls':0,'Testing_access':False}
        write(out/'scores.json',result); print(json.dumps(result)); return
    raise ValueError('unsupported command')
