"""Default scheme-4 Training pipeline. Inference never loads scoring annotations."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import threading
from decimal import Decimal
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.run_pgp_pipeline import CachedBackend, WireBackend, frozen
from surgical_agent.research.gate.tracker_pipeline_v2 import run_target, CausalPhaseFilter, VERSION
from surgical_agent.research.gate.pgp_tracker_gemini38 import FrozenTracker
from surgical_agent.research.gate.final_only_training import canonical_labels

PROFILE = 'tracker_scheme4_pipeline_v1'
VIDEOS = {'VID103', 'VID23', 'VID31', 'VID96'}
SOURCE = ROOT / 'artifacts/training/gate/full_official_reviewers_20260912_v1'


def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def load_gate(root=ROOT):
    manifest = read(root / 'DEFAULT_PGP_GATE_VERSION.json')
    if manifest.get('schema_version') != 'tracker_scheme4_gate_default_v1':
        raise ValueError('scheme4 default Gate manifest required')
    path = (root / manifest['model_artifact']).resolve()
    if not path.is_relative_to((root / 'artifacts/training/gate').resolve()) or sha(path) != manifest['model_sha256']:
        raise ValueError('Gate model location/hash mismatch')
    model = read(path)
    if model['version'] != VERSION or model['deployable'] or not model['selection_feasible']:
        raise ValueError('unsupported scheme4 Gate model')
    estimator_path = (path.parent / model['estimator_file']).resolve()
    if not estimator_path.is_relative_to(path.parent) or sha(estimator_path) != model['estimator_sha256']:
        raise ValueError('Gate estimator hash mismatch')
    names = model['feature_names']; threshold = model['threshold']
    if len(names) != 42 or len(set(names)) != 42 or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid feature schema/threshold')
    estimator = joblib.load(estimator_path)
    def predict(features):
        if set(features) != set(names): raise ValueError('Gate feature schema mismatch')
        x = np.array([[features[k] for k in names]], dtype=float)
        if not np.isfinite(x).all(): raise ValueError('nonfinite Gate input')
        score = float(estimator.predict_proba(x)[0, 1])
        return score, int(score >= threshold)
    return predict, manifest, model


def inventory(source, limit=None):
    config = read(ROOT / 'DEFAULT_PIPELINE_VERSION.json')
    inv_path = ROOT / config['replay_inventory']
    if sha(inv_path) != config['replay_inventory_sha256']: raise ValueError('inventory changed')
    keys = set(read(inv_path)['keys'])
    original = read(source / 'plan.json')
    selected = []
    for row in original['selection']:
        if row['key'] not in keys: continue
        if row['video_id'] not in VIDEOS: raise ValueError('unsupported video')
        s = {k: deepcopy(row[k]) for k in ('key','video_id','frame_id','anchor_frame_id','causal_frame_ids','images','alignment_version')}
        s['source_split'] = 'Training'
        if s['causal_frame_ids'] != [s['frame_id']-50,s['frame_id']-25,s['frame_id']]: raise ValueError('unaligned causal window')
        selected.append(s)
    if len(selected) != len(keys) or len({s['key'] for s in selected}) != len(keys): raise ValueError('incomplete/duplicate source inventory')
    selected.sort(key=lambda s: (s['video_id'],s['frame_id']))
    if limit is not None and not 1 <= limit <= len(selected): raise ValueError('invalid limit')
    return original, selected[:limit]


def tracker_index(args):
    return (args.tracker_index or ROOT / read(ROOT/'DEFAULT_PIPELINE_VERSION.json')['tracker_index']).resolve()


def validate_images(selection):
    for s in selection:
        if len(s['images']) != 3: raise ValueError('three images required')
        for im, frame in zip(s['images'],s['causal_frame_ids'],strict=True):
            if im['frame_id'] != frame or sha(im['path']) != im['sha256']: raise ValueError('image alignment/hash mismatch')


def infer_stream(selection, get_backend, priors, decide, tracker, window, sink):
    phase = CausalPhaseFilter(seconds=window)
    total = 0
    for n, s in enumerate(selection, 1):
        backend, close = get_backend(s)
        try:
            result = run_target(backend, s, priors[s['video_id']], decide,
                                tracker_snapshot=tracker.snapshot(s), phase_filter=phase)
            result.update(video_id=s['video_id'], frame_id=s['frame_id'], source_split='Training')
            sink.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n'); sink.flush()
            total += result['logical_calls']
        finally: close()
        if n % 500 == 0: print(f'Completed {n}/{len(selection)}', flush=True)
    return total


def replay(args):
    from tools.audit.gate_proposals_offline_20260913 import offline
    start = time.perf_counter()
    decide, manifest, model = load_gate()
    _, selection = inventory(args.source, args.limit if args.command=='replay' else (args.limit or 8))
    tracker = FrozenTracker(tracker_index(args))
    priors = {v:read(args.source/'priors'/f'{v}.json') for v in {s['video_id'] for s in selection}}
    sealed = read(ROOT / read(ROOT/'DEFAULT_PIPELINE_VERSION.json')['replay_inventory'])['source_sha256']
    for s in selection:
        if sha(args.source/'targets'/s['key']/'result.json') != sealed[s['key']]: raise ValueError('unsealed source response')
    args.output.mkdir(parents=True,exist_ok=False)
    def backend(s): return CachedBackend(read(args.source/'targets'/s['key']/'result.json')), lambda:None
    with offline(), (args.output/'predictions.jsonl').open('x',encoding='utf-8') as sink:
        calls = infer_stream(selection, backend, priors, decide, tracker, model['phase_window_seconds'], sink)
    receipt = {'state':'PASS','profile':PROFILE,'rows':len(selection),'logical_calls':calls,'api_calls':0,
        'gate_version':manifest['version'],'phase_window_seconds':model['phase_window_seconds'],
        'phase_initialization':'empty per-video state; chronological inventory prefix',
        'elapsed_seconds':time.perf_counter()-start,'predictions_sha256':sha(args.output/'predictions.jsonl'),
        'model_sha256':manifest['model_sha256'],'scope':'full-fit Training replay; not outer OOF or online validation'}
    write(args.output/'receipt.json', receipt); print(json.dumps(receipt),flush=True)


def prepare(args):
    if args.limit is None or args.budget_limits is None: raise ValueError('prepare needs --limit and --budget-limits')
    _, manifest, model = load_gate()
    original, selection = inventory(args.source,args.limit)
    validate_images(selection)
    index = tracker_index(args); tracker = FrozenTracker(index)
    for s in selection: tracker.snapshot(s)
    caps = read(args.budget_limits)
    if set(caps) != set(original['limits']) or any(not Decimal(str(v)).is_finite() or Decimal(str(v))<0 for v in caps.values()):
        raise ValueError('explicit finite nonnegative caps for all accounts required')
    plan = deepcopy(original)
    plan.update(profile=PROFILE,selection=selection,limits={k:str(v) for k,v in caps.items()},
        maximum_paid_calls=7*len(selection),tracker_index=str(index),tracker_index_sha256=sha(index),
        phase_window_seconds=model['phase_window_seconds'],api_execution_validated=False,Testing_access=False,
        tracker_enabled=True,automatic_retry=False)
    bound = ['DEFAULT_PIPELINE_VERSION.json','DEFAULT_PGP_GATE_VERSION.json','scripts/run_tracker_scheme4_pipeline.py',
        'scripts/run_pgp_pipeline.py','scripts/scheme4_transport.py','src/surgical_agent/research/gate/tracker_pipeline_v2.py',
        'scripts/assess_tracker_review_evidence_r3.py','scripts/full_official_reviewer_transport.py',
        'scripts/reviewer_routes_official.py','scripts/collect_gate_escalation_v1.py',
        'src/surgical_agent/research/gate/collection_budget.py','src/surgical_agent/research/gate/pgp_tracker_gemini38.py',
        'src/surgical_agent/research/gate/pgp_runtime.py',manifest['model_artifact'],
        str(Path(manifest['model_artifact']).parent/model['estimator_file'])]
    plan['runtime_sha256'] = {p:sha(ROOT/p) for p in bound}
    args.output.mkdir(parents=True,exist_ok=False); (args.output/'priors').mkdir()
    plan['prior_sha256'] = {}
    for v in sorted({s['video_id'] for s in selection}):
        prior = read(args.source/'priors'/f'{v}.json')
        if prior['excluded_video'] != v or v in prior['fit_videos']: raise ValueError('prior leakage')
        path = args.output/'priors'/f'{v}.json'; write(path,prior); plan['prior_sha256'][v] = sha(path)
    write(args.output/'plan.json',plan)
    write(args.output/'prepared.json',{'plan_sha256':sha(args.output/'plan.json'),'api_calls':0})
    print(json.dumps({'prepared':str(args.output),'rows':len(selection),'maximum_paid_calls':plan['maximum_paid_calls'],'api_calls':0}))


def execute(args):
    if not args.allow_paid: raise ValueError('execute requires explicitly approved budget and --allow-paid')
    out = args.output; plan = read(out/'plan.json')
    if plan['profile'] != PROFILE or sha(out/'plan.json') != read(out/'prepared.json')['plan_sha256']: raise ValueError('plan changed')
    for p,h in plan['runtime_sha256'].items():
        if sha(ROOT/p) != h: raise ValueError('runtime changed: '+p)
    for v,h in plan['prior_sha256'].items():
        if sha(out/'priors'/f'{v}.json') != h: raise ValueError('prior changed')
    if sha(plan['tracker_index']) != plan['tracker_index_sha256']: raise ValueError('tracker changed')
    selection = plan['selection']
    if any(s.get('source_split')!='Training' or s['video_id'] not in VIDEOS for s in selection): raise ValueError('Training only')
    if selection != sorted(selection,key=lambda s:(s['video_id'],s['frame_id'])): raise ValueError('noncausal selection order')
    validate_images(selection)
    decide, _, model = load_gate(); tracker = FrozenTracker(plan['tracker_index'])
    for s in selection: tracker.snapshot(s)
    from scripts.scheme4_transport import GuardedCalls
    from scripts.collect_gate_escalation_v1 import load_base
    from surgical_agent.research.gate.collection_budget import Budget
    # Exclusive durable marker prevents concurrent execution and blind resubmission.
    write(out/'execution_started.json',{'state':'STARTED','automatic_retry':False})
    budget = Budget(out/'budget.sqlite',plan['limits'],sha(out/'plan.json')); stop = threading.Event()
    priors = {v:read(out/'priors'/f'{v}.json') for v in plan['prior_sha256']}
    def backend(s):
        calls = GuardedCalls(out,plan,s,budget,stop)
        return WireBackend(calls,load_base(s),s), calls.close
    start = time.perf_counter()
    try:
        with frozen.joint.credential_context(plan), frozen.joint.roster.lightweight_protocol(), (out/'predictions.jsonl').open('x',encoding='utf-8') as sink:
            calls = infer_stream(selection,backend,priors,decide,tracker,model['phase_window_seconds'],sink)
        write(out/'receipt.json',{'state':'PASS','profile':PROFILE,'rows':len(selection),'logical_calls':calls,
            'budget':budget.summary(),'elapsed_seconds':time.perf_counter()-start,'predictions_sha256':sha(out/'predictions.jsonl'),
            'scope':'Training fresh execution; no independent accuracy validation'})
    except Exception as exc:
        write(out/'failure.json',{'state':'STOPPED','error_type':type(exc).__name__,'budget':budget.summary(),'automatic_retry':False})
        raise
    finally: budget.close()


def score(args):
    if args.annotations is None: raise ValueError('score requires explicit Training --annotations')
    receipt = read(args.output/'receipt.json')
    if receipt['state']!='PASS' or sha(args.output/'predictions.jsonl')!=receipt['predictions_sha256']: raise ValueError('incomplete/changed predictions')
    rows = read(args.annotations)
    if any(r.get('source_split')!='Training' or r['video_id'] not in VIDEOS for r in rows): raise ValueError('Training annotations only')
    truth = {r['sample_id']:r for r in rows}
    if len(truth)!=len(rows): raise ValueError('duplicate truth keys')
    predictions = [json.loads(line) for line in (args.output/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(predictions)!=receipt['rows'] or len({r['key'] for r in predictions})!=len(predictions): raise ValueError('prediction inventory mismatch')
    tasks = ('instrument','verb','target','ivt','phase'); counts = np.zeros((5,3),dtype=int)
    for r in predictions:
        p = canonical_labels(r['prediction']); row = truth[r['key']]
        if not all(row['mask'].values()): raise ValueError('complete five-head annotation required')
        for j,t in enumerate(tasks):
            a,b = set(p[t]),set(row['gt'][t]); counts[j] += [len(a&b),len(a-b),len(b-a)]
    tp,fp,fn = counts.T; den = 2*tp+fp+fn
    f = np.divide(200*tp,den,out=np.full(5,100.),where=den!=0)
    result = {'rows':len(predictions),'f1':float(f.mean()),'errors':int((fp+fn).sum()),
        'by_head':dict(zip(tasks,f.tolist())),'api_calls':0,'scope':'full-fit Training evaluation, not nested outer result'}
    write(args.output/'scores.json',result); print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',nargs='?',default='info',choices=['info','preflight','replay','prepare','execute','score'])
    p.add_argument('--source',type=Path,default=SOURCE); p.add_argument('--output',type=Path)
    p.add_argument('--limit',type=int); p.add_argument('--tracker-index',type=Path)
    p.add_argument('--budget-limits',type=Path); p.add_argument('--annotations',type=Path)
    p.add_argument('--allow-paid',action='store_true'); args = p.parse_args()
    if args.command=='info':
        _,manifest,model = load_gate()
        print(json.dumps({'pipeline':read(ROOT/'DEFAULT_PIPELINE_VERSION.json'),'gate':manifest,'threshold':model['threshold']},ensure_ascii=False,indent=2)); return
    if args.output is None: p.error('--output required')
    if args.limit is not None and args.limit<1: p.error('--limit must be positive')
    if args.command in ('replay','preflight'): replay(args)
    elif args.command=='prepare': prepare(args)
    elif args.command=='execute': execute(args)
    else: score(args)


if __name__=='__main__': main()
