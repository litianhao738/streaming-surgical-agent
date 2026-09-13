"""Full Training preparation and bounded resumable collection. Only 'run' can POST.

Each target keeps a <=13-call ledger. A shared SQLite reservation is durable
before transport; unknown interrupted dispatches block rather than resend.
GT is read for masks at preparation, and joined only after inference sealing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

from scripts import collect_mainline_gate_training as pilot
from surgical_agent.data.schemas import InferenceSample, DatasetSplit
from surgical_agent.research.gate import collection_features_v2 as features
from surgical_agent.research.gate import terminal_utility_training as labels
from surgical_agent.research.gate.collection_budget import Budget, BudgetStop, AmbiguousDispatch

trial, old, ready = pilot.trial, pilot.old, pilot.ready
DEFAULT = ROOT/'artifacts/training/gate/full_training_20260912_v3'
PROFILE = 'full_training_gate_collection_v1'


def write(path, obj):
    # Atomic replace retry only for transient local file sharing, never API retry.
    import time
    for attempt in range(21):
        try:
            old.save(path, obj)
            return
        except PermissionError:
            if attempt == 20:
                raise
            time.sleep(.05)


def digest(path):
    return old.sha(path)


def load_base(selected):
    for image in selected['images']:
        if digest(image['path']) != image['sha256']:
            raise ValueError('input image changed: '+selected['key'])
    sample = InferenceSample(video_id=selected['video_id'], target_frame_id=selected['frame_id'],
        causal_frame_ids=tuple(selected['causal_frame_ids']), media_refs=tuple(i['path'] for i in selected['images']),
        source_split=DatasetSplit.TRAINING, alignment_version=selected['alignment_version'])
    return trial.release.make_base(sample)


def snapshot(run, selected):
    p = run/selected['tracker_snapshot']
    if digest(p) != selected['tracker_snapshot_sha256']:
        raise ValueError('Tracker snapshot changed')
    return old.read(p)


def feature_row(run, selected, h0):
    prior = old.read(run/'priors'/f"{selected['video_id']}.json")
    plan = old.read(run/'plan.json') if not hasattr(feature_row, '_plan') else feature_row._plan
    values = features.extract(h0, video_id=selected['video_id'], target_frame_id=selected['frame_id'],
        causal_frame_ids=selected['causal_frame_ids'], tracker_snapshot=snapshot(run,selected), prior=prior, gate=plan['gate'])
    return {'sample_id':selected['key'], 'video_id':selected['video_id'], 'frame_id':selected['frame_id'],
        'source_split':'Training', 'policy_id':plan['policy_id'], 'feature_version':features.FEATURE_VERSION,
        'label_version':labels.LABEL_VERSION, 'h0_labels':h0, 'features':values,
        'tracker_snapshot_sha256':selected['tracker_snapshot_sha256'],
        'prior_sha256':plan['prior_sha256'][selected['video_id']],
        'origin':'offline_reconstruction_for_reused_pilot' if selected['reused_pilot'] else 'before_repair'}


def prepare(run):
    if run.exists():
        raise ValueError('fresh preparation directory required')
    frozen = old.read(ROOT/'GATE_TRAINING_DATA_VERSION.json')
    source = Path(frozen['archive'])
    for name,h in frozen['artifact_sha256'].items():
        if digest(source/name) != h:
            raise ValueError('pilot source changed: '+name)
    for name,h in frozen['code_sha256'].items():
        if digest(ROOT/name) != h:
            raise ValueError('frozen policy code changed: '+name)
    original = old.read(source/'plan.json')
    adapter = old.common.CholecTrack20DatasetAdapter(trial.release.DATASET,causal_window_size=3)
    providers, oof_audit, oof_sources = pilot.legacy.verified_oof(pilot.OOF,trial.release.DATASET)
    reused = {s['key'] for s in original['selection']}
    run.mkdir(parents=True)
    by_video, inventory, image_hashes = {}, {}, {}
    for video in pilot.VIDEOS:
        masks = {r.inference.target_frame_id:pilot.legacy.masks_only(r) for r in adapter.iter_video(video)}
        selected = []
        for sample in adapter.iter_inference_video(video):
            if len(sample.causal_frame_ids) != 3 or not all(masks.get(sample.target_frame_id,{}).get(t,False) for t in pilot.contract.TASKS):
                continue
            if sample.source_split is not DatasetSplit.TRAINING:
                raise ValueError('Training only')
            key = f'{video}_{sample.target_frame_id}'
            images = []
            for fid,ref in zip(sample.causal_frame_ids,sample.media_refs,strict=True):
                path = str(Path(ref).resolve())
                if path not in image_hashes:
                    image_hashes[path] = digest(path)
                images.append({'frame_id':fid,'path':path,'sha256':image_hashes[path]})
            providers[video].reset(video)
            tracks = pilot.plain(providers[video].snapshot(sample))
            rel = f'tracker/{key}.json'
            write(run/rel,tracks)
            selected.append({'key':key,'video_id':video,'frame_id':sample.target_frame_id,
                'anchor_frame_id':sample.target_frame_id,'causal_frame_ids':list(sample.causal_frame_ids),
                'alignment_version':sample.alignment_version,'images':images,'tracker_snapshot':rel,
                'tracker_snapshot_sha256':digest(run/rel),'reused_pilot':key in reused})
        # Deterministic identity-only spread: each early batch spans video time.
        selected.sort(key=lambda s: hashlib.sha256(('3407:'+s['key']).encode()).hexdigest())
        by_video[video] = selected
        inventory[video] = {'complete_five_head_masks':sum(all(m.values()) for m in masks.values()),
                            'usable':len(selected),'reused':sum(s['reused_pilot'] for s in selected)}
        write(run/'priors'/f'{video}.json',old.read(source/'priors'/f'{video}.json'))
        print(json.dumps({'prepared_video':video,**inventory[video]}),flush=True)
    selection = [by_video[v][i] for i in range(max(map(len,by_video.values()))) for v in pilot.VIDEOS if i<len(by_video[v])]
    if {s['key'] for s in selection if s['reused_pilot']} != reused:
        raise ValueError('pilot identities no longer eligible')
    if len(selection) != 7376:
        raise ValueError('full-data count drift; inspect before creating a different plan')
    paths = set(trial.release.runtime_paths()) | {Path(__file__), Path(features.__file__), Path(labels.__file__),
        ROOT/'src/surgical_agent/research/gate/collection_budget.py'}
    source_hashes = {}
    for path in sorted(paths):
        rel = path.relative_to(ROOT).as_posix()
        source_hashes[rel] = digest(path)
        destination = run/'frozen_source'/rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    plan = {'profile':PROFILE,'created_utc':old.now(),'pipeline_version':ready.BASELINE_VERSION,
        'policy_id':original['policy_id'],'feature_version':features.FEATURE_VERSION,'label_version':labels.LABEL_VERSION,
        'feature_order':list(features.FEATURE_ORDER),'feature_views':list(features.VIEWS),
        'selection':selection,'inventory':inventory,'gate':original['gate'],'base_rates':original['base_rates'],
        'models':original['models'],'reviewers':original['policy']['reviewers'],'credential_root':str(ROOT),
        'limits':{'openrouter_usd':'250','aliyun_cny':'250','xai_usd':'0'},
        'maximum_new_calls':13*(len(selection)-len(reused)),'pilot_source':str(source),
        'pilot_artifact_sha256':frozen['artifact_sha256'], 'pilot_predictions_sha256':digest(source/'predictions.json'),
        'pilot_source_record_sha256':digest(ROOT/'GATE_TRAINING_DATA_VERSION.json'),
        'source_sha256':source_hashes,'oof_audit':oof_audit,'oof_source_sha256':oof_sources,
        'evaluation_source_sha256':original['evaluation_source_sha256'],
        'prior_sha256':{v:digest(run/'priors'/f'{v}.json') for v in pilot.VIDEOS},
        'root_unchanged':{name:digest(ROOT/name) for name in frozen['original_pipeline_pointers_unchanged']},
        'protocol':{'sampling':'All eligible Training targets including adjacent windows; no outcome selection. Seed3407 identity hash per video, then interleave.',
          'concurrency':'First 32 newly completed targets at 4 workers, then up to requested 8; 128-target batches; reviewers parallel within target.',
          'retries':'Zero automatic paid retries. Terminal responses including failures replay exactly. Uncertain interrupted requests block without resubmission.',
          'features':'H0 plus 5 explicit LVO prior features plus 11 basic and 18 geometry/motion Tracker statistics, saved before proposal; no future/GT/review signals.',
          'labels':'Actual terminal five-head utility. Missing H0/final remains unknown; never relabeled negative.',
          'stopping':'Budget/config errors stop dispatch. Batch transport failures >10% or valid-final fraction <90% pause the run.',
          'transport_scope':'One <=13-call ledger per target, shared durable budget. Deterministic image refusal stops its target; configuration/account failures pause the run.',
          'validation':'Training development only. Adjacent windows are not independent; no independent validation claim.',
          'phase':'Frozen joint Phase branch retained; no empty-pool or reviewer-policy changes.',
          'score':'Only after all planned new targets have terminal statuses, all inference evidence is sealed and exact raw requests replayed.'}}
    write(run/'plan.json',plan)
    # Exhaustive validation of track geometry / feature domains before any POST.
    mock_h0={'instrument':[0],'verb':[1],'target':[0],'ivt':[17],'phase':[1]}
    feature_row._plan = plan
    for s in selection:
        feature_row(run,s,mock_h0)
    del feature_row._plan
    write(run/'preparation.json',{'plan_sha256':digest(run/'plan.json'),'targets':len(selection),
        'new_targets':len(selection)-len(reused),'reused':len(reused),'geometry_checks':len(selection),'api_calls':0})
    print(json.dumps({'prepared':str(run),'targets':len(selection),'new_targets':len(selection)-len(reused),'max_new_calls':plan['maximum_new_calls']}),flush=True)


def verify(run):
    plan = old.read(run/'plan.json')
    if plan['profile'] != PROFILE or plan['feature_version'] != features.FEATURE_VERSION:
        raise ValueError('wrong collection contract')
    if old.read(run/'preparation.json')['plan_sha256'] != digest(run/'plan.json'):
        raise ValueError('plan changed after preparation')
    if 'provider_rate_envelopes_sha256' in plan and digest(run/'provider_rate_envelopes.json')!=plan['provider_rate_envelopes_sha256']:
        raise ValueError('frozen provider rate evidence changed')
    for rel,h in plan['source_sha256'].items():
        if digest(ROOT/rel) != h or digest(run/'frozen_source'/rel) != h:
            raise ValueError('source drift: '+rel)
    for video,h in plan['prior_sha256'].items():
        prior = old.read(run/'priors'/f'{video}.json')
        if digest(run/'priors'/f'{video}.json') != h or prior['excluded_video'] != video or video in prior['fit_videos']:
            raise ValueError('prior drift or query-video leakage')
    for name,h in plan['root_unchanged'].items():
        if digest(ROOT/name) != h:
            raise ValueError('mainline pointer changed: '+name)
    source = Path(plan['pilot_source'])
    for rel,h in plan['pilot_artifact_sha256'].items():
        if digest(source/rel) != h:
            raise ValueError('pilot source drift')
    if digest(source/'predictions.json') != plan['pilot_predictions_sha256']:
        raise ValueError('pilot predictions changed')
    if (run/'preflight.json').exists():
        check=old.read(run/'preflight.json')
        if check['plan_sha256']!=digest(run/'plan.json'):raise ValueError('preflight drift')
        for name,h in check['reused_features_sha256'].items():
            if digest(run/'reused_features'/name)!=h:raise ValueError('reused feature drift')
    return plan


def amend_prepare(source,run):
    """Before any paid execution, retain eligible identities and freeze rate tiers."""
    if run.exists() or (source/'budget.sqlite').exists():
        raise ValueError('fresh output and a never-executed source preparation required')
    plan=old.read(source/'plan.json');prepared=old.read(source/'preparation.json')
    if prepared['plan_sha256']!=digest(source/'plan.json'):raise ValueError('source plan drift')
    for rel,h in plan['source_sha256'].items():
        if digest(source/'frozen_source'/rel)!=h:raise ValueError('original frozen source drift')
    for s in plan['selection']:
        if digest(source/s['tracker_snapshot'])!=s['tracker_snapshot_sha256']:raise ValueError('snapshot drift')
    run.mkdir(parents=True)
    for name in ('tracker','priors'):
        shutil.copytree(source/name,run/name)
    prices=old.read(ROOT/'artifacts/research/full_gate_provider_price_check_20260912.json')
    rates={k:list(v) for k,v in old.joint.roster.RATES.items()}
    rates['base']=plan['base_rates']['gemini']
    for seat,info in prices.items():
        expected=plan['models']['gemini']['model'] if seat=='base' else plan['reviewers'][seat]
        if info['model']!=expected:raise ValueError('rate metadata model mismatch')
        rates[seat]=[str(max(Decimal(a),Decimal(b))) for a,b in zip(rates[seat],info['current_reserve_rates'])]
    plan['call_rates']=rates
    plan['created_utc']=old.now()
    plan['preparation_amendment']={'source':str(source.resolve()),'source_plan_sha256':digest(source/'plan.json'),
        'reason':'Include GPT long-context pricing tier in conservative reservations; ordinary-tier prices and all model requests unchanged.',
        'api_calls_before_amendment':0}
    write(run/'provider_rate_envelopes.json',prices)
    plan['provider_rate_envelopes_sha256']=digest(run/'provider_rate_envelopes.json')
    for rel in plan['source_sha256']:
        plan['source_sha256'][rel]=digest(ROOT/rel)
        dest=run/'frozen_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    write(run/'plan.json',plan)
    write(run/'preparation.json',{**prepared,'plan_sha256':digest(run/'plan.json'),
        'source_preparation_sha256':digest(source/'preparation.json'),'api_calls':0})
    print(json.dumps({'prepared':str(run),'new_targets':7248,'maximum_new_calls':94224,'api_calls':0}),flush=True)


def preflight(run):
    plan = verify(run)
    source = Path(plan['pilot_source'])
    saved = {r['key']:r for r in old.read(source/'predictions.json')['targets']}
    replay = trial.Replay(source/'run','gemini')
    checks = []
    feature_row._plan=plan
    with old.joint.roster.lightweight_protocol():
        for s in plan['selection']:
            if not s['reused_pilot']:
                continue
            base = load_base(s)
            raw = ready.generate_h0(replay,base,s)
            write(run/'reused_features'/f"{s['key']}.json",feature_row(run,s,old.gated.h0_from_raw(raw)))
            try:
                record = ready.repair_from_h0(replay,base,s,old.read(run/'priors'/f"{s['video_id']}.json"),plan['gate'],raw)
            except (ValueError,TypeError,KeyError,trial.ApiSchemaError):
                if saved[s['key']]['status'] != 'TARGET_FAILED':
                    raise
            else:
                if record['predictions'] != saved[s['key']]['predictions']:
                    raise ValueError('reused policy replay mismatch')
        if replay.used != set(replay.rows):
            raise ValueError('not all pilot requests replayed')
        for s in [s for s in plan['selection'] if not s['reused_pilot']][:32]:
            base, prior = load_base(s), old.read(run/'priors'/f"{s['video_id']}.json")
            a,b=trial.Mock('gemini'),trial.Mock('gemini')
            reference=trial.main.run_target(a,base,s,prior,plan['gate'])
            raw=ready.generate_h0(b,base,s)
            feature_row(run,s,old.gated.h0_from_raw(raw))
            split=ready.repair_from_h0(b,base,s,prior,plan['gate'],raw)
            if a.wires != b.wires or reference['predictions'] != split['predictions']:
                raise ValueError('feature seam changed API requests')
            checks.append(s['key'])
    del feature_row._plan
    report={'plan_sha256':digest(run/'plan.json'),'api_calls':0,'replayed_pilot_calls':len(replay.used),
        'new_target_mock_equivalence':checks,'reused_features_sha256':{p.name:digest(p) for p in (run/'reused_features').glob('*.json')}}
    write(run/'preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='reused_features_sha256'}),flush=True)


def doctor(run, online=False):
    plan=verify(run)
    checks=[]
    with old.joint.credential_context(plan),old.joint.roster.lightweight_protocol():
        for seat in old.SEATS:
            secret=old.joint.roster.transport.key_for(seat)
            if not secret.reveal():
                raise ValueError('empty credential for '+seat)
            checks.append(seat)
        if online:
            import requests
            s=plan['selection'][0]; base=load_base(s)
            mock=trial.Mock('gemini')
            trial.main.run_target(mock,base,s,old.read(run/'priors'/f"{s['video_id']}.json"),plan['gate'])
            wires={seat:wire for (stage,seat),wire in mock.wires.items() if stage in ('h0','control_graph')}
            public={}
            for seat,wire in wires.items():
                if seat=='qwen':continue
                response=requests.get('https://openrouter.ai/api/v1/models/'+wire['model']+'/endpoints',timeout=30)
                response.raise_for_status()
                data=response.json(); tag=wire['provider']['only'][0]
                endpoint=next(e for e in data['data']['endpoints'] if e['tag']==tag and e['status']==0)
                rates=plan.get('call_rates',{}).get(seat,plan['base_rates']['gemini'] if seat=='base' else old.joint.roster.RATES[seat])
                for field,limit in zip(('prompt','completion'),rates):
                    if max([Decimal(endpoint['pricing'][field]),*(Decimal(o.get(field,'0')) for o in endpoint['pricing'].get('overrides',[]))])>Decimal(limit):
                        raise ValueError('provider rate exceeds frozen reserve envelope: '+seat)
                public[seat]=data
            write(run/'provider_check.json',{'checked_utc':old.now(),'endpoints':public,'api_posts':0})
    report={'credentials_present':checks,'online_endpoints_checked':online,'paid_api_calls':0,
            'plan_sha256':digest(run/'plan.json'),'checked_utc':old.now()}
    write(run/'doctor.json',report)
    print(json.dumps(report),flush=True)


class ResumeCalls(trial.ModelCalls):
    def __init__(self, folder, plan, budget, stop):
        super().__init__(folder,'gemini',plan)
        self.limits={k:Decimal(v) for k,v in plan['limits'].items()}
        self.max_calls=13
        if 'call_rates' in plan:self.rates={k:tuple(v) for k,v in plan['call_rates'].items()}
        self.budget,self.stop=budget,stop
        self.seen=set()
        self.existing=trial.Replay(folder,'gemini') if (folder/'budget.json').exists() else None
        if self.existing:
            state=old.read(folder/'budget.json')
            self.rows=deepcopy(state['calls'])
            self.occupied={k:Decimal(v) for k,v in state['occupied'].items()}
            for r in self.rows:
                if r['status']=='DISPATCHED' or 'finished_utc' not in r:
                    raise AmbiguousDispatch('unresolved target request: '+r['target'])
                evidence=folder/'calls'/f"{r['index']:03d}_{r['target']}_{r['stage']}_{r['seat']}"
                if not (evidence/'record.json').exists() or old.read(evidence/'record.json')!=r:
                    raise AmbiguousDispatch('cached ledger/evidence disagreement: '+r['target'])
                if not (evidence/'request.json').exists() or (r['status'] in trial.VALID and not (evidence/'response.json').exists()):
                    raise AmbiguousDispatch('cached request/response incomplete: '+r['target'])

    def persist(self):
        with self.lock:
            write(self.output/'budget.json',{'limits':{k:str(v) for k,v in self.limits.items()},
                'carried_occupied':self.carried,'occupied':{k:str(v) for k,v in self.occupied.items()},
                'calls':self.rows,'stopped':self.stopped})

    def call(self,target,stage,seat,body):
        key=(target,stage,seat)
        with self.lock:
            if key in self.seen:raise ValueError('duplicate target call')
            self.seen.add(key)
        if self.existing and key in self.existing.rows:
            return self.existing.call(target,stage,seat,body)
        if self.stopped:
            return None
        if self.stop.is_set():
            raise BudgetStop('run paused before new dispatch')
        wire=trial.base_body(body,'gemini') if seat=='base' else body
        transport=old.joint.roster.transport
        reserve=transport.envelope(seat,wire,self.rates)
        identity=Budget.key(*key)
        try:
            self.budget.reserve(identity,transport.bucket(seat),reserve)
        except BudgetStop:
            self.stop.set();raise
        result=super().call(target,stage,seat,body)
        row=next((r for r in self.rows if (r['target'],r['stage'],r['seat'])==key),None)
        if row is None:
            raise AmbiguousDispatch('reserved call has no transport record')
        self.budget.settle(identity,Decimal(row['charge']))
        status=row.get('http_status')
        if status in (400,401,402,403,404):
            response=old.read(self.output/'calls'/f"{row['index']:03d}_{target}_{stage}_{seat}"/'response.json')
            error=response.get('body',{}).get('error',{})
            image_refusal=isinstance(error,dict) and error.get('code')=='data_inspection_failed'
            if not image_refusal:
                self.stop.reason=f'account_or_route_http_{status}_{seat}'
                self.stop.set()
        return result


def reconcile(run,budget):
    # Only unequivocally finished durable records may release reserved money.
    for target,stage,seat in budget.pending():
        folder=run/'targets'/target/'run'
        state=old.read(folder/'budget.json') if (folder/'budget.json').exists() else {'calls':[]}
        row=next((r for r in state['calls'] if (r['target'],r['stage'],r['seat'])==(target,stage,seat)),None)
        if row is None or row['status']=='DISPATCHED' or 'finished_utc' not in row:
            raise AmbiguousDispatch('request outcome uncertain; not resent: '+Budget.key(target,stage,seat))
        evidence=folder/'calls'/f"{row['index']:03d}_{target}_{stage}_{seat}"/'record.json'
        if old.read(evidence)!=row:
            raise AmbiguousDispatch('ledger/evidence disagreement: '+target)
        budget.settle(Budget.key(target,stage,seat),Decimal(row['charge']))


def collect_one(run,plan,s,budget,stop):
    folder=run/'targets'/s['key']
    if (folder/'terminal.json').exists():
        terminal=old.read(folder/'terminal.json')
        for rel,h in terminal['evidence_sha256'].items():
            if digest(folder/rel)!=h:raise ValueError('completed target changed')
        return old.read(folder/'prediction.json')
    base=load_base(s); calls=ResumeCalls(folder/'run',plan,budget,stop)
    result=None; row={k:s[k] for k in ('key','video_id','frame_id')}
    started=perf_counter()
    try:
        raw=ready.generate_h0(calls,base,s)
        feature=feature_row(run,s,old.gated.h0_from_raw(raw))
        if (folder/'features_before_repair.json').exists() and old.read(folder/'features_before_repair.json')!=feature:
            raise RuntimeError('pre-review feature changed on resume')
        write(folder/'features_before_repair.json',feature)
        write(folder/'h0_raw.json',raw)
        result=ready.repair_from_h0(calls,base,s,old.read(run/'priors'/f"{s['video_id']}.json"),plan['gate'],raw)
        write(folder/'result.json',result)
        row.update(status='PREDICTED',predictions=result['predictions'])
    except (BudgetStop,AmbiguousDispatch):
        raise
    except (ValueError,TypeError,KeyError,trial.ApiSchemaError) as exc:
        row.update(status='TARGET_FAILED',error_type=type(exc).__name__,predictions={a:None for a in trial.main.ARMS})
    finally:
        calls.close_ledger()
    row.update(pilot.repair_observation(result,calls.rows))
    row.update(seconds=perf_counter()-started,calls=len(calls.rows),
               failed_calls=sum(r['status'] not in trial.VALID for r in calls.rows))
    write(folder/'prediction.json',row)
    write(folder/'terminal.json',{'plan_sha256':digest(run/'plan.json'),
        'evidence_sha256':{p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*.json') if p.name!='terminal.json'}})
    return row


def execute(run,workers=8,max_new_targets=None):
    plan=verify(run)
    if old.read(run/'preflight.json')['plan_sha256']!=digest(run/'plan.json'):
        raise ValueError('preflight missing or stale')
    # OS-level lock releases on process exit, including crash; no stale PID-file deletion.
    import msvcrt
    lock_file=(run/'runner.lock').open('a+b')
    lock_file.seek(0);lock_file.write(b'0');lock_file.flush();lock_file.seek(0)
    try:msvcrt.locking(lock_file.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        lock_file.close();raise RuntimeError('another collector is already running')
    budget=Budget(run/'budget.sqlite',plan['limits'],digest(run/'plan.json'))
    stop=threading.Event(); user_stop=threading.Event()
    previous_handler=signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT,lambda *a:user_stop.set())
    feature_row._plan=plan
    try:
        reconcile(run,budget)
        pending=[s for s in plan['selection'] if not s['reused_pilot'] and not (run/'targets'/s['key']/'terminal.json').exists()]
        if max_new_targets is not None:pending=pending[:max_new_targets]
        newly=0
        with old.joint.credential_context(plan),old.joint.roster.lightweight_protocol():
            while pending and not stop.is_set() and not user_stop.is_set():
                warm=not (run/'warmup_passed.json').exists()
                batch=pending[:32 if warm else 128];pending=pending[len(batch):]
                outputs=[]
                with ThreadPoolExecutor(max_workers=4 if warm else workers) as pool:
                    def task(item):
                        if user_stop.is_set():return None
                        return collect_one(run,plan,item,budget,stop)
                    futures=[pool.submit(task,s) for s in batch]
                    for future in as_completed(futures):
                        try:row=future.result()
                        except (BudgetStop,AmbiguousDispatch) as exc:
                            stop.set();write(run/'pause.json',{'type':type(exc).__name__,'detail':str(exc),'utc':old.now()});continue
                        except Exception:
                            stop.set();raise
                        if row is None:continue
                        outputs.append(row);newly+=1
                        print(json.dumps({'new_terminal_targets':newly,'last':row['key'],'status':row['status'],**budget.summary()}),flush=True)
                if outputs:
                    fraction=sum(r['status']=='PREDICTED' for r in outputs)/len(outputs)
                    failed=sum(r['failed_calls'] for r in outputs)/max(1,sum(r['calls'] for r in outputs))
                    quality={'targets':len(outputs),'final_fraction':fraction,'failed_call_fraction':failed,'utc':old.now()}
                    write(run/'batches'/f"{old.now().replace(':','-')}.json",quality)
                    if fraction<.9 or failed>.1:
                        stop.set();write(run/'pause.json',{'type':'BATCH_QUALITY','quality':quality})
                    elif warm and len(outputs)>=32:write(run/'warmup_passed.json',quality)
        complete=sum((run/'targets'/s['key']/'terminal.json').exists() for s in plan['selection'] if not s['reused_pilot'])
        done=complete==sum(not s['reused_pilot'] for s in plan['selection'])
        write(run/'progress.json',{'new_terminal_targets':complete,'planned_new_targets':7248,
            'all_inference_complete':done,'budget':budget.summary(),'user_stopped':user_stop.is_set(),
            'paused':stop.is_set(),'halt_reason':getattr(stop,'reason',None),'utc':old.now()})
        if done:
            write(run/'inference_seal.json',{'plan_sha256':digest(run/'plan.json'),
                'targets':{s['key']:digest(run/'targets'/s['key']/'terminal.json') for s in plan['selection'] if not s['reused_pilot']}})
        print(json.dumps(old.read(run/'progress.json')),flush=True)
    finally:
        signal.signal(signal.SIGINT,previous_handler)
        if hasattr(feature_row,'_plan'):del feature_row._plan
        budget.close();lock_file.seek(0);msvcrt.locking(lock_file.fileno(),msvcrt.LK_UNLCK,1);lock_file.close()


def status(run):
    plan=old.read(run/'plan.json')
    result={'planned_total':len(plan['selection']),'reused_pilot':128,'planned_new':7248,
        'completed_new':sum((run/'targets'/s['key']/'terminal.json').exists() for s in plan['selection'] if not s['reused_pilot'])}
    if (run/'budget.sqlite').exists():
        budget=Budget(run/'budget.sqlite',plan['limits'],digest(run/'plan.json'))
        result['budget']=budget.summary();budget.close()
    if (run/'pause.json').exists():result['last_pause']=old.read(run/'pause.json')
    print(json.dumps(result,indent=2),flush=True)


def score(run):
    plan=verify(run)
    seal=old.read(run/'inference_seal.json')
    if seal['plan_sha256']!=digest(run/'plan.json') or len(seal['targets'])!=7248:
        raise ValueError('all new inference must be complete and sealed before GT scoring')
    records=[];feature_row._plan=plan
    # Exact request replay precedes GT join; only one target's raw images in memory.
    with old.joint.roster.lightweight_protocol():
        for s in plan['selection']:
            if s['reused_pilot']:continue
            folder=run/'targets'/s['key']
            if digest(folder/'terminal.json')!=seal['targets'][s['key']]:raise ValueError('terminal seal changed')
            terminal=old.read(folder/'terminal.json')
            for rel,h in terminal['evidence_sha256'].items():
                if digest(folder/rel)!=h:raise ValueError('inference evidence changed')
            row=old.read(folder/'prediction.json'); replay=trial.Replay(folder/'run','gemini');base=load_base(s)
            try:
                raw=ready.generate_h0(replay,base,s)
                feature=feature_row(run,s,old.gated.h0_from_raw(raw))
                if feature!=old.read(folder/'features_before_repair.json'):raise RuntimeError('feature replay mismatch')
                result=ready.repair_from_h0(replay,base,s,old.read(run/'priors'/f"{s['video_id']}.json"),plan['gate'],raw)
            except (ValueError,TypeError,KeyError,trial.ApiSchemaError):
                if row['status']!='TARGET_FAILED':raise
            else:
                if result['predictions']!=row['predictions']:raise RuntimeError('prediction replay mismatch')
            if replay.used!=set(replay.rows):raise ValueError('unreplayed request')
            records.append((s,row))
    source=Path(plan['pilot_source'])
    pilot_predictions={r['key']:r for r in old.read(source/'predictions.json')['targets']}
    records.extend((s,pilot_predictions[s['key']]) for s in plan['selection'] if s['reused_pilot'])
    for path,h in plan['evaluation_source_sha256'].items():
        if digest(path)!=h:raise ValueError('GT/prior source changed')
    wanted={s['key'] for s in plan['selection']};truth={}
    adapter=old.common.CholecTrack20DatasetAdapter(trial.release.DATASET,causal_window_size=3)
    for video in pilot.VIDEOS:
        for r in adapter.iter_video(video):
            key=f'{video}_{r.inference.target_frame_id}'
            if key in wanted:truth[key]=old.truth_row(r)
    training=[];missing_h0=[]
    for s,row in records:
        feature_path=run/'reused_features'/f"{s['key']}.json" if s['reused_pilot'] else run/'targets'/s['key']/'features_before_repair.json'
        if not feature_path.exists():missing_h0.append(s['key']);continue
        feature=old.read(feature_path);gt=truth[s['key']]
        final=row['predictions'][trial.main.PRIMARY]
        training.append({**feature,'final_labels':final,'gt':gt['gt'],'mask':gt['mask'],
            'labels':labels.label_outcome(feature['h0_labels'],final,gt=gt['gt'],mask=gt['mask'])})
    destination=run/'scored'
    write(destination/'training_examples.json',training)
    write(destination/'missing_h0.json',missing_h0)
    write(destination/'manifest.json',{'feature_version':features.FEATURE_VERSION,'label_version':labels.LABEL_VERSION,
        'policy_id':plan['policy_id'],'plan_sha256':digest(run/'plan.json'),'inference_seal_sha256':digest(run/'inference_seal.json'),
        'rows':len(training),'missing_h0':len(missing_h0),'known_terminal_utility':sum(r['labels']['terminal_observed'] for r in training),
        'training_examples_sha256':digest(destination/'training_examples.json'),'api_calls':0,'deployable':False})
    print(json.dumps(old.read(destination/'manifest.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','amend-prepare','preflight','doctor','run','status','score'))
    parser.add_argument('--run-dir',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,choices=(4,8),default=8)
    parser.add_argument('--max-new-targets',type=int)
    parser.add_argument('--online',action='store_true',help='Public model metadata only; never a paid POST')
    parser.add_argument('--source-preparation',type=Path)
    args=parser.parse_args();run=args.run_dir.resolve()
    if args.max_new_targets is not None and args.max_new_targets<1:parser.error('positive target limit required')
    if args.command=='run':execute(run,args.workers,args.max_new_targets)
    elif args.command=='amend-prepare':
        if args.source_preparation is None:parser.error('--source-preparation required')
        amend_prepare(args.source_preparation.resolve(),run)
    elif args.command=='doctor':doctor(run,args.online)
    else:globals()[args.command](run)
