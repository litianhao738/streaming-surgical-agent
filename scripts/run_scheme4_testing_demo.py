"""Fixed eight-video Testing demonstration: cached Gemini H0, unchanged scheme4 policy.

prepare is local only; execute is single-use and budgeted; score joins GT afterwards.
This entry point explicitly opts into Testing. The default Training CLI is unchanged.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import base64
from collections import Counter
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts import run_tracker_scheme4_pipeline as app
from scripts.scheme4_transport import GuardedCalls
from surgical_agent.research.gate.collection_budget import Budget, BudgetStop
from surgical_agent.research.gate.tracker_pipeline_v2 import (
    run_target, CausalPhaseFilter, output_modules, null_label_cleanup, current_classes)
from surgical_agent.research.verification.prior_panel import fit_prior, video_counts
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.schemas import DatasetSplit

INDEX = ROOT/'artifacts/h0_import/gemini_testing_delivery_20260914_r2/h0_index.jsonl'
TRACKER = ROOT/'artifacts/training/tracker_clip_v2_oof5_20260906/full'
DATASET = Path('D:/cholec_dataset')
CAPS = {'openrouter_usd':'0.75', 'aliyun_cny':'0.25', 'glm_requests':'8',
        'deepseek_requests':'8', 'xai_usd':'0'}
PROFILE = 'scheme4_testing_cached_h0_demo_v1'
PROTOCOL = ROOT/'docs/SCHEME4_TESTING_DEMO_PROTOCOL_2026-09-14.md'

def read(p): return json.loads(Path(p).read_bytes())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stream_sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def choose(index):
    """Selection depends only on video/frame and a complete three-image window."""
    selected=[]
    for v in sorted({r['video_id'] for r in index}):
        rows=sorted((r for r in index if r['video_id']==v and
                     r['causal_frame_ids']==[r['frame_id']-50,r['frame_id']-25,r['frame_id']]),
                    key=lambda r:r['frame_id'])
        selected.append(deepcopy(rows[len(rows)//2]))
    if len(selected)!=8: raise ValueError('expected eight Testing videos')
    return selected

def body_digest_matches(body, expected):
    # Delivery versions used compact or standard canonical JSON; identify exact bytes.
    for ascii_flag in (False,True):
        for separators in ((',',':'),None):
            encoded=json.dumps(body,sort_keys=True,ensure_ascii=ascii_flag,separators=separators).encode()
            if hashlib.sha256(encoded).hexdigest()==expected: return True
    return False

def extract_inputs(selected, out):
    wanted={s['custom_id']:s for s in selected}
    for directory in sorted({s['prepared_dir'] for s in selected}):
        folder=Path(directory); pending={k for k,s in wanted.items() if s['prepared_dir']==directory}
        manifest=read(folder/'manifest.json')
        for c in manifest['chunks']:
            if not pending: break
            path=folder/c['path'].replace('\\','/')
            with path.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    item=json.loads(line); cid=item['custom_id']
                    if cid not in pending: continue
                    s=wanted[cid]; body=item['body']
                    s['legacy_body_hash_reproduced']=body_digest_matches(body,s['input_body_sha256'])
                    # The delivery's per-request hash recipe is unavailable. Bind the original
                    # complete JSONL bytes to its manifest instead of silently accepting drift.
                    if stream_sha(path)!=c['sha256']:raise ValueError('delivered chunk hash mismatch')
                    s['input_chunk_sha256']=c['sha256']
                    blocks=[b for m in body['messages'] if isinstance(m['content'],list)
                            for b in m['content'] if b.get('type')=='image_url']
                    if len(blocks)!=3: raise ValueError('three original JPEGs required')
                    s['images']=[]
                    for frame,b in zip(s['causal_frame_ids'],blocks,strict=True):
                        prefix,data=b['image_url']['url'].split(',',1)
                        if prefix!='data:image/jpeg;base64': raise ValueError('expected delivered JPEG')
                        content=base64.b64decode(data,validate=True)
                        p=out/'images'/s['key']/f'{frame}.jpg';p.parent.mkdir(parents=True,exist_ok=True)
                        with p.open('xb') as f:f.write(content)
                        s['images'].append({'frame_id':frame,'path':str(p),'sha256':sha(p),
                                            'mime_type':'image/jpeg','detail':b['image_url'].get('detail','auto')})
                    s['input_chunk']=str(path);pending.remove(cid)
                    print('Extracted exact H0 images: '+s['key'],flush=True)
                    if not pending: break
        if pending: raise ValueError('missing delivered inputs')

def make_base(s):
    images=[]
    for im in s['images']:
        if sha(im['path'])!=im['sha256']: raise ValueError('image changed')
        images.append(SimpleNamespace(content=Path(im['path']).read_bytes(),mime_type=im['mime_type']))
    return SimpleNamespace(images=images,payload={'image_details':[im['detail'] for im in s['images']]})

def prepare(out):
    from PIL import Image
    import torch
    from torchvision.transforms.functional import to_tensor
    from surgical_agent.tracking.config import TrackerTrainingConfig
    from surgical_agent.tracking.detector import build_instrument_detector, load_tracker_checkpoint, decode_detections
    from surgical_agent.tracking.associator import CausalHungarianAssociator
    from scripts.run_prior_panel_trial import truth_row
    started=time.perf_counter()
    index=[json.loads(line) for line in INDEX.read_text('utf-8').splitlines()]
    if sha(INDEX)!=read(INDEX.parent/'audit.json')['h0_index_sha256']: raise ValueError('H0 index changed')
    selected=choose(index)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'selection_frozen.json',{'rule':'median rank of complete-three-image H0 targets per Testing video; no GT/masks/scores',
                                      'keys':[s['key'] for s in selected]})
    print('Frozen targets: '+', '.join(s['key'] for s in selected),flush=True)
    extract_inputs(selected,out)
    adapter=CholecTrack20DatasetAdapter(DATASET)
    counts={}
    for v,e in sorted(adapter.entries.items()):
        if e.split is DatasetSplit.TRAINING:
            counts[v]=video_counts(truth_row(r) for r in adapter.iter_video(v))
    print('Training-only prior statistics ready',flush=True)
    manifest=read(TRACKER/'training_manifest.json')
    if sha(TRACKER/'checkpoint.pt')!=manifest['checkpoint_sha256']: raise ValueError('Tracker hash mismatch')
    if any(adapter.entries[v].split is not DatasetSplit.TRAINING for v in manifest['training_video_ids']):
        raise ValueError('Tracker training provenance leakage')
    config=TrackerTrainingConfig(**manifest['config'])
    torch.set_num_threads(4)
    model=build_instrument_detector(config,use_pretrained=False)
    load_tracker_checkpoint(TRACKER/'checkpoint.pt',model=model,map_location='cpu');model.eval()
    decide, gate_manifest, gate_model=app.load_gate()
    tracker_seconds=0
    for s in selected:
        if adapter.entries[s['video_id']].split is not DatasetSplit.TESTING: raise ValueError('Testing only')
        if sha(s['record_path'])!=s['record_sha256']: raise ValueError('H0 record changed')
        s['cached_h0']=s.pop('h0');s['alignment_version']='delivered_testing_jpeg_exact_body_v1'
        # cheap_state leaves Phase equal to H0, so existing raw H0 supplies the identical M3 history.
        s['phase_history']=[{'frame_id':r['frame_id'],'phase':r['h0']['phase'][0]} for r in index
                            if r['video_id']==s['video_id'] and s['frame_id']-1500<r['frame_id']<s['frame_id']]
        prior=fit_prior(counts,s['video_id']);write(out/'priors'/f"{s['video_id']}.json",prior)
        s['prior_sha256']=sha(out/'priors'/f"{s['video_id']}.json")
        associator=CausalHungarianAssociator(iou_threshold=config.association_iou_threshold,max_age=config.max_age,max_frame_id_gap=25)
        associator.reset(s['video_id']);frames=[];tick=time.perf_counter()
        for im in s['images']:
            with Image.open(im['path']) as image:
                rgb=image.convert('RGB'); width,height=rgb.size;tensor=to_tensor(rgb)
            with torch.inference_mode(): detected=model([tensor])[0]
            detections=decode_detections(detected,width=width,height=height,score_threshold=config.score_threshold)
            tracks=associator.update(im['frame_id'],detections)
            frames.append({'frame_id':im['frame_id'],'tracks':[t.as_mapping() for t in tracks]})
        s['tracker_seconds']=time.perf_counter()-tick;tracker_seconds+=s['tracker_seconds']
        snapshot={'status':'AVAILABLE','video_id':s['video_id'],'source_split':'Testing',
                  'source_max_frame_id':s['frame_id'],'frames':frames,
                  'checkpoint_sha256':manifest['checkpoint_sha256'],
                  'association_scope':'three real causal frames; M1/M2 consume current detected classes only'}
        write(out/'tracker'/f"{s['key']}.json",snapshot)
        s['tracker_sha256']=sha(out/'tracker'/f"{s['key']}.json")
        print('Tracker ready: '+s['key']+' classes='+str(sorted(current_classes(snapshot,s))),flush=True)
    plan=deepcopy(read(ROOT/'artifacts/experiments/scheme4_two_targets_20260914_r1/plan.json'))
    plan.update(profile=PROFILE,selection=selected,limits=CAPS,maximum_paid_calls=48,Testing_access=True,
                automatic_retry=False,output_modules='v2.2',h0_reuse=True,gate_training=False,
                phase_window_seconds=60,gate_version=gate_manifest['version'],model_sha256=gate_manifest['model_sha256'],
                tracker_checkpoint_sha256=manifest['checkpoint_sha256'],tracker_training_videos=manifest['training_video_ids'],
                protocol_sha256=sha(PROTOCOL),h0_index_sha256=sha(INDEX),
                preparation_seconds=time.perf_counter()-started,tracker_seconds=tracker_seconds)
    bound={Path(__file__),PROTOCOL,ROOT/'DEFAULT_PIPELINE_VERSION.json',ROOT/'DEFAULT_PGP_GATE_VERSION.json',
           ROOT/gate_manifest['model_artifact'],ROOT/Path(gate_manifest['model_artifact']).parent/gate_model['estimator_file']}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path:
            p=Path(path).resolve()
            if p.is_relative_to(ROOT) and p.suffix=='.py' and p.is_file(): bound.add(p)
    plan['demo_runtime_sha256']={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(bound)}
    write(out/'plan.json',plan);write(out/'prepared.json',{'plan_sha256':sha(out/'plan.json'),'api_calls':0})
    print(json.dumps({'prepared':str(out),'rows':8,'limits':CAPS,'local_seconds':plan['preparation_seconds']}),flush=True)

class Backend(app.WireBackend):
    parse_h0=staticmethod(deepcopy)
    def h0(self): return deepcopy(self.selected['cached_h0'])
    def proposal(self,*args):
        self.proposal_raw=super().proposal(*args);return self.proposal_raw

class BoundedCalls:
    def __init__(self,delegate,counter):self.delegate,self.counter=delegate,counter
    def call(self,target,stage,seat,body):
        if stage not in ('proposal','control_graph'): raise ValueError('H0 and extra stages forbidden')
        if self.counter[0]>=48: raise BudgetStop('48-request cap')
        self.counter[0]+=1
        return self.delegate.call(target,stage,seat,body)

def execute(out,allow_paid):
    if not allow_paid: raise ValueError('--allow-paid required')
    plan=read(out/'plan.json')
    if plan['profile']!=PROFILE or sha(out/'plan.json')!=read(out/'prepared.json')['plan_sha256']:raise ValueError('plan changed')
    if plan['limits']!=CAPS or len(plan['selection'])!=8: raise ValueError('unexpected scope')
    for p,h in plan['demo_runtime_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('runtime changed: '+p)
    for s in plan['selection']:
        if sha(s['record_path'])!=s['record_sha256']:raise ValueError('H0 changed')
        for p,h in ((out/'priors'/f"{s['video_id']}.json",s['prior_sha256']),
                    (out/'tracker'/f"{s['key']}.json",s['tracker_sha256'])):
            if sha(p)!=h:raise ValueError('prepared input changed')
        make_base(s)
    decide,_,_=app.load_gate()
    write(out/'execution_started.json',{'state':'STARTED','automatic_retry':False})
    budget=Budget(out/'budget.sqlite',plan['limits'],sha(out/'plan.json'));stop=threading.Event()
    tick=time.perf_counter();counter=[0];results=[]
    try:
        with app.frozen.joint.credential_context(plan),app.frozen.joint.roster.lightweight_protocol():
            for s in plan['selection']:
                phase=CausalPhaseFilter(60)
                for h in s['phase_history']:phase.apply(s['video_id'],h['frame_id'],h['phase'])
                delegate=GuardedCalls(out,plan,s,budget,stop)
                backend=Backend(BoundedCalls(delegate,counter),make_base(s),s);before=counter[0];start=time.perf_counter()
                try:
                    result=run_target(backend,s,read(out/'priors'/f"{s['video_id']}.json"),decide,
                                      tracker_snapshot=read(out/'tracker'/f"{s['key']}.json"),
                                      phase_filter=phase,output='v2.2',inference_split='Testing')
                finally:delegate.close()
                assert result['cheap']['phase']==result['h0']['phase']
                # Reconstruct the no-T arm using only the very same consumed responses.
                class Replay:
                    parse_h0=staticmethod(deepcopy)
                    normalize_compact=staticmethod(app.frozen.common.normalize_five)
                    def h0(self):return deepcopy(result['h0'])
                    def proposal(self,*args):return deepcopy(backend.proposal_raw)
                    def compact(self,seat,*args):return deepcopy(observed[seat])
                # Raw reviews are captured by the backend below, never fetched again.
                observed=backend.review_raw
                without=run_target(Replay(),s,read(out/'priors'/f"{s['video_id']}.json"),decide,
                                   tracker_snapshot=None,phase_filter=CausalPhaseFilter(0),
                                   output='v2.2',inference_split='Testing')
                assert without['features']==result['features'] and without['call_keys']==result['call_keys']
                result.update(video_id=s['video_id'],frame_id=s['frame_id'],source_split='Testing',
                              without_tracker=without['prediction'],proposal_raw=backend.proposal_raw,
                              review_raw=observed,paid_calls=counter[0]-before,
                              downstream_seconds=time.perf_counter()-start,cached_h0=True,
                              phase_history_rows=len(s['phase_history']))
                write(out/'results'/f"{s['key']}.json",result);results.append(result)
                print(json.dumps({'target':s['key'],'gate_action':result['gate_action'],
                                  'depth':result['compact_depth'],'paid_calls':result['paid_calls'],
                                  'seconds':round(result['downstream_seconds'],2)}),flush=True)
        write(out/'predictions.json',results)
        write(out/'receipt.json',{'state':'PASS','rows':8,'actual_dispatches':counter[0],
              'budget':budget.summary(),'downstream_seconds':time.perf_counter()-tick,
              'predictions_sha256':sha(out/'predictions.json'),'plan_sha256':sha(out/'plan.json'),
              'cached_h0_rows':8,'new_h0_calls':0,'scope':'fixed eight-frame Testing demonstration; not full-test estimate'})
    except Exception as exc:
        write(out/'failure.json',{'state':'STOPPED','error_type':type(exc).__name__,
                                  'actual_dispatches':counter[0],'budget':budget.summary()});raise
    finally:budget.close()

def _compact(self,seat,*args):
    if not hasattr(self,'review_raw'):self.review_raw={}
    raw=app.WireBackend.compact(self,seat,*args);self.review_raw[seat]=deepcopy(raw);return raw
Backend.compact=_compact

def load_testing_truth(predictions):
    # Runtime adapter intentionally exposes NO Testing supervision. Scoring must use
    # the existing explicit offline annotation reader, only after prediction sealing.
    from surgical_agent.data.parser import parse_annotation_file
    from surgical_agent.evaluation.frame_ground_truth import aggregate_frame_target
    attrs={'instrument':'instrument_ids','verb':'verb_ids','target':'target_ids','ivt':'triplet_ids','phase':'phase_id'}
    truth={};sources={};wanted={r['key'] for r in predictions}
    for v in sorted({r['video_id'] for r in predictions}):
        path=DATASET/'Testing'/v/f'{v.lower()}.json'
        native=parse_annotation_file(path,expected_split=DatasetSplit.TESTING);sources[str(path)]=sha(path)
        for frame in native.frames:
            key=f'{v}_{frame.frame_id}'
            if key not in wanted:continue
            target=aggregate_frame_target(frame,allowed_tasks=frozenset(attrs),source=str(path))
            mask={t:getattr(target.mask,t) for t in attrs}
            gt={t:([target.phase_id] if t=='phase' else list(getattr(target,attr))) if mask[t] else None for t,attr in attrs.items()}
            truth[key]={'video_id':v,'frame_id':frame.frame_id,'gt':gt,'mask':mask}
    if set(truth)!=wanted:raise ValueError('missing Testing annotations')
    if not any(r['mask']['instrument'] for r in truth.values()):raise ValueError('no supervised frames; do not publish zero errors')
    return truth,sources

def score(out, score_filename='scores.json'):
    receipt=read(out/'receipt.json');predictions=read(out/'predictions.json')
    if receipt['state']!='PASS' or sha(out/'predictions.json')!=receipt['predictions_sha256']:raise ValueError('unsealed predictions')
    truth,sources=load_testing_truth(predictions)
    heads=('instrument','verb','target','ivt','phase');arms=('h0','without_tracker','prediction');reports={}
    for arm in arms:
        stats={}
        for task in heads:
            tp=fp=fn=n=correct=0
            for r in predictions:
                t=truth[r['key']]
                if not t['mask'][task]:continue
                a,b=set(r[arm][task]),set(t['gt'][task]);n+=1;correct+=int(a==b)
                tp+=len(a&b);fp+=len(a-b);fn+=len(b-a)
            den=2*tp+fp+fn
            stats[task]={'eligible_frames':n,'tp':tp,'fp':fp,'fn':fn,
                         'micro_precision':100*tp/(tp+fp) if n and tp+fp else None,
                         'micro_recall':100*tp/(tp+fn) if n and tp+fn else None,
                         'micro_f1':(200*tp/den if den else 100.) if n else None,
                         'exact_accuracy':100*correct/n if n else None}
        reports[arm]={'by_head':stats,'errors':sum(s['fp']+s['fn'] for s in stats.values())}
    details=[]
    for r in predictions:
        t=truth[r['key']];errors={a:sum(len(set(r[a][q])^set(t['gt'][q])) for q in heads if t['mask'][q]) for a in arms}
        details.append({'key':r['key'],'gt':t['gt'],'mask':t['mask'],'errors':errors,
                        'gate_action':r['gate_action'],'gate_score':r['gate_score'],'compact_depth':r['compact_depth'],
                        'h0':r['h0'],'without_tracker':r['without_tracker'],'pipeline':r['prediction'],
                        'output_modules':r['output_modules']})
    report={'rows':8,'arms':reports,'details':details,'receipt':receipt,'api_calls_during_scoring':0,
            'annotation_sha256':sources,'scorer_sha256':sha(Path(__file__)),
            'metric_units':'percent; undefined precision/recall denominators are null',
            'gt_join_after_sealed_inference':True,'selection':'time rank only; all eight retained',
            'phase_metric':'accuracy; report separately from four micro-F1 scores',
            'no_tracker_comparison':'same consumed answers; remove M1/M2/M3, retain v2.2 ontology/null cleanup'}
    if Path(score_filename).name!=score_filename or not score_filename.endswith('.json'):
        raise ValueError('score filename must be a local JSON basename')
    write(out/score_filename,report)
    print(json.dumps({'arms':reports,'per_frame_errors':[{k:r[k] for k in ('key','errors')} for r in details]}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','execute','score'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-paid',action='store_true')
    p.add_argument('--score-filename',default='scores.json');a=p.parse_args()
    out=a.output.resolve()
    if a.command=='prepare':prepare(out)
    elif a.command=='execute':execute(out,a.allow_paid)
    else:score(out,a.score_filename)

if __name__=='__main__':main()
