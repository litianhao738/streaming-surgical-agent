"""Frozen Gate/Tracker paired Training pilot. Only collect --allow-paid sends requests."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys
import threading
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import collect_gate_escalation_v1 as transport
from scripts.full_official_reviewer_transport import Calls,reconcile
from scripts.run_pgp_pipeline import CachedBackend,predictor,frozen
from scripts.train_pgp_gate_no_tracker import sha,write,tracker_snapshot
from surgical_agent.research.gate.collection_budget import Budget
from surgical_agent.research.gate.pgp_runtime import run_target
from surgical_agent.research.gate.pgp_tracker_gemini38 import FrozenTracker
from surgical_agent.research.verification.tracker_evidence import make_packet,attach
from surgical_agent.research.gate.official_net_training import counts
from surgical_agent.research.gate.pgp_training import quality,COST_FIELDS
from tools.audit.pgp_gate_replay_20260913 import cost_vector
from tools.audit.gate_proposals_offline_20260913 import offline
SOURCE=ROOT/'artifacts/training/gate/full_official_reviewers_20260912_v1'
BASELINE=ROOT/'artifacts/preflight/pgp_pipeline_all_20260913_r1'
ROWS=ROOT/'artifacts/training/gate/official_behavior_net_v2_20260913/primary_rows.json'
DEFAULT=ROOT/'artifacts/training/gate/tracker_review_evidence_20260914_r2'
SEATS=('gpt','gemini','grok','deepseek')
ARMS=('control','tracker')
CODE=(Path(__file__),ROOT/'scripts/run_pgp_tracker_evidence.py',ROOT/'src/surgical_agent/research/verification/tracker_evidence.py')

def read(path):return json.loads(Path(path).read_text('utf-8'))

def prepare(out):
    baseline=[json.loads(s) for s in (BASELINE/'predictions.jsonl').read_text('utf-8').splitlines()]
    assert sha(BASELINE/'predictions.jsonl')==read(BASELINE/'receipt.json')['prediction_sha256']
    original=read(SOURCE/'plan.json');inventory={s['key']:s for s in original['selection']}
    # compact_depth > 1 is determined only by the first Qwen answer and Gate;
    # it means there is at least one genuinely needed post-Gate compact review.
    eligible=[r for r in baseline if r['gate_action']==3 and r['compact_depth']>1]
    groups=[]
    for v in ('VID103','VID23','VID31','VID96'):
        rr=sorted((r for r in eligible if inventory[r['key']]['video_id']==v),key=lambda r:inventory[r['key']]['frame_id'])
        assert len(rr)>=16
        indices=[int((j+.5)*len(rr)/16) for j in range(16)];groups.append([rr[i] for i in indices])
    chosen=[groups[v][j] for j in range(16) for v in range(4)]
    out.mkdir(exist_ok=False);(out/'evidence').mkdir();(out/'priors').mkdir()
    t=FrozenTracker(ROOT/'artifacts/training/tracker_clip_v2_oof5_20260906/oof/index.json')
    decision,manifest,model=predictor();selection=[];wires=[];extra=[]
    bindings=tracker_snapshot()
    for p in (*CODE,ROOT/'DEFAULT_PGP_GATE_VERSION.json',ROOT/'DEFAULT_PIPELINE_VERSION.json',
              BASELINE/'predictions.jsonl',ROWS,ROOT/manifest['model_artifact'],ROOT/model['estimator_artifact'],
              ROOT/'src/surgical_agent/research/gate/pgp_runtime.py',ROOT/'scripts/run_pgp_pipeline.py',
              ROOT/'docs/PGP_TRACKER_REVIEW_EVIDENCE_PROTOCOL_2026-09-14.md'):
        bindings[str(p)]=sha(p)
    with offline(),frozen.joint.roster.lightweight_protocol():
        for j,r in enumerate(chosen):
            source_s=inventory[r['key']]
            s={k:deepcopy(source_s[k]) for k in ('key','video_id','frame_id','anchor_frame_id','causal_frame_ids','images','alignment_version')}
            s['source_split']='Training';s['arm_order']=list(ARMS if j//4%2==0 else reversed(ARMS))
            rp=SOURCE/'targets'/s['key']/'result.json';s['source_result_sha256']=sha(rp);rec=read(rp)
            pp=SOURCE/'priors'/f"{s['video_id']}.json";prior=read(pp);bindings[str(pp)]=sha(pp)
            replay=run_target(CachedBackend(rec),s,prior,decision)
            # Historical receipt predates these three descriptive runtime fields.
            historical={**r,'tracker_feature_schema':False,'probe_seat':'qwen','probe_prefix':'qwen'}
            assert replay==historical,'frozen default replay mismatch'
            packet=make_packet(t.snapshot(s),s);ep=out/'evidence'/f"{s['key']}.json";write(ep,packet)
            s['evidence_sha256']=sha(ep)
            base=transport.load_base(s)
            for seat in SEATS:
                a=frozen.joint.roster.review_wire(seat,base,s,rec['pool']);b=attach(a,packet)
                frozen.check_requests(rec['h0'],[a,b])
                assert a['messages'][0]['content'][1:]==b['messages'][0]['content'][1:]
                x=read_payload(a);y=read_payload(b)
                y.pop('tracker_evidence');y.pop('tracker_evidence_instructions');assert x==y
                extra.append(len(b['messages'][0]['content'][0]['text'])-len(a['messages'][0]['content'][0]['text']))
                if j==0:wires.append({'seat':seat,'model':a['model'],'provider':a.get('provider'),'rates':original['call_rates'][seat]})
            selection.append(s)
        for v in ('VID103','VID23','VID31','VID96'):(out/'priors'/f'{v}.json').write_bytes((SOURCE/'priors'/f'{v}.json').read_bytes())
    plan=deepcopy(original);plan.update(profile='tracker-review-evidence-paired-v1',selection=selection,
        limits={'openrouter_usd':'5','glm_requests':'128','deepseek_requests':'128','aliyun_cny':'0','xai_usd':'0'},
        maximum_paid_calls=512,source_sha256=bindings,Testing_access=False,tracker_training=False,gate_training=False,
        gate_default=manifest['version'],prompt_difference='two additive tracker fields only, for remaining four compact seats',
        eligibility='frozen final Gate action 3 and causal need for further compact review',
        sampling='16 chronological rank quantiles per Training video, no GT/outcome selection',
        assessment='paired fresh control/evidence; unchanged cached Qwen probe, H0, proposal and Phase; conditional pilot, not independent evaluation')
    write(out/'plan.json',plan)
    write(out/'preflight.json',{'rows':64,'eligible_rows':len(eligible),'maximum_paid_calls':512,'api_calls':0,
        'extra_text_characters_mean':float(np.mean(extra)),'gate_decision_and_features_unchanged':True,
        'wires':wires,'plan_sha256':sha(out/'plan.json')})
    print(json.dumps(read(out/'preflight.json')),flush=True)

def read_payload(w):return json.loads(w['messages'][0]['content'][0]['text'])

def verify(out):
    p=read(out/'plan.json');assert p['profile']=='tracker-review-evidence-paired-v1'
    assert sha(out/'plan.json')==read(out/'preflight.json')['plan_sha256']
    for path,h in p['source_sha256'].items():
        if sha(path)!=h:raise ValueError('frozen source changed: '+path)
    for s in p['selection']:
        assert sha(SOURCE/'targets'/s['key']/'result.json')==s['source_result_sha256']
        assert sha(out/'evidence'/f"{s['key']}.json")==s['evidence_sha256']
        assert sha(out/'priors'/f"{s['video_id']}.json")==p['source_sha256'][str(SOURCE/'priors'/f"{s['video_id']}.json")]
    return p

def collect(out,limit,workers,allow_paid):
    if not allow_paid:raise ValueError('--allow-paid required')
    plan=verify(out)
    import msvcrt
    lock=(out/'runner.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    budget=Budget(out/'budget.sqlite',plan['limits'],sha(out/'plan.json'));stop=threading.Event();reconcile(out,budget)
    def one(s):
        if stop.is_set() or (out/'STOP').exists():stop.set();return None
        base=transport.load_base(s);rec=read(SOURCE/'targets'/s['key']/'result.json');packet=read(out/'evidence'/f"{s['key']}.json")
        for arm in s['arm_order']:
            target=s['key']+'__'+arm;ss={**s,'key':target};calls=Calls(out,plan,ss,budget,stop)
            answers={}
            try:
                # Four requests at most per arm. These are the existing remaining
                # seats, not extra rounds. Complete responses permit causal replay.
                def query(seat):
                    w=frozen.joint.roster.review_wire(seat,base,s,rec['pool'])
                    if arm=='tracker':w=attach(w,packet)
                    frozen.check_requests(rec['h0'],[w])
                    return seat,calls.call(target,'control_graph',seat,w)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for seat,raw in pool.map(query,SEATS):answers[seat]=raw
                transport.write(out/'targets'/target/'answers.json',answers)
            finally:calls.close()
            if stop.is_set():break
        return s['key']
    try:
        with frozen.joint.credential_context(plan),frozen.joint.roster.lightweight_protocol():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                done=0
                for k in pool.map(one,plan['selection'][:limit]):
                    if k:done+=1;print(json.dumps({'paired_targets':done,'budget':budget.summary()}),flush=True)
        transport.write(out/'collection_receipt.json',{'state':'STOPPED' if stop.is_set() else 'COMPLETED',
            'paired_targets':done,'budget':budget.summary(),'Testing_access':False})
    finally:budget.close();lock.close()

def score(out):
    plan=verify(out);decision,_,_=predictor();rows={r['sample_id']:r for r in read(ROWS)}
    ledger=read(ROOT/'artifacts/training/gate/official_behavior_net_v2_20260913/costs.json')['per_call_main_attempt']
    predictions={a:[] for a in ARMS};costs={a:[] for a in ARMS};joined=[];statuses=Counter();tokens={a:[0,0] for a in ARMS};seals={}
    with offline():
        for s in plan['selection']:
            results={}
            for arm in ARMS:
                target=s['key']+'__'+arm;folder=out/'targets'/target
                if not (folder/'answers.json').exists():raise ValueError('incomplete pair '+target)
                rec=read(SOURCE/'targets'/s['key']/'result.json');rec['review_raw'].update(read(folder/'answers.json'))
                result=run_target(CachedBackend(rec),s,read(out/'priors'/f"{s['video_id']}.json"),decision)
                assert result['gate_action']==3
                results[arm]=result;local=deepcopy(ledger)
                allrows=[]
                for seat in SEATS:
                    if seat in ('grok','deepseek'):r=read(folder/'changed'/f'control_graph_{seat}'/'record.json')
                    else:r=next(r for r in read(folder/'run/budget.json')['calls'] if r['seat']==seat)
                    allrows.append(r);statuses[arm+'|'+r['status']]+=1
                    if seat in ('gpt','gemini'):local['control_graph|'+seat]['usd_per_call']=float(r['charge'])
                    u=r.get('usage') or {};tokens[arm][0]+=u.get('prompt_tokens',0);tokens[arm][1]+=u.get('completion_tokens',0)
                costs[arm].append(cost_vector(result['call_keys'],local));predictions[arm].append(result)
                for p in folder.rglob('*.json'):seals[str(p)]=sha(p)
            assert results['control']['features']==results['tracker']['features']
            joined.append({**rows[s['key']],**{a:results[a]['prediction'] for a in ARMS}})
        report={'rows':len(joined),'scope':'paired conditional Training pilot, frozen final Gate; not independent validation',
            'api_calls_during_scoring':0,'Testing_access':False,'default_changed':False,'arms':{},'transport_statuses':dict(statuses),
            'paid_collection':read(out/'collection_receipt.json')['budget'],'token_totals_all_collected_reviews':tokens}
        cc=counts(joined,'cheap_labels',False);ce=cc[:,:,1:].sum(axis=(1,2));vid=np.array([r['video_id'] for r in joined])
        for arm in ARMS:
            c=counts(joined,arm,False);e=c[:,:,1:].sum(axis=(1,2));cv=np.array(costs[arm]).sum(axis=0)
            report['arms'][arm]={'original':quality(c),'merged':quality(counts(joined,arm,True)),
                **dict(zip(COST_FIELDS,cv.tolist())), 'help_frames_vs_cheap':int((e<ce).sum()),'hurt_frames_vs_cheap':int((e>ce).sum()),
                'by_video':{str(v):quality(c[vid==v]) for v in sorted(set(vid))}}
        a=report['arms']['control'];b=report['arms']['tracker'];qa=a['original'];qb=b['original']
        checks={'pooled_f1_not_lower':qb['five_head_mean_f1']+1e-12>=qa['five_head_mean_f1'],
                'pooled_errors_not_higher':qb['total_errors']<=qa['total_errors'],
                'strict_quality_gain':qb['five_head_mean_f1']>qa['five_head_mean_f1']+1e-12 or qb['total_errors']<qa['total_errors'],
                'every_video_not_worse':all(z['five_head_mean_f1']+1e-12>=a['by_video'][v]['five_head_mean_f1'] and z['total_errors']<=a['by_video'][v]['total_errors'] for v,z in b['by_video'].items()),
                'known_usd_within_10_percent':b['known_usd']<=a['known_usd']*1.1}
        report['checks']=checks;report['pilot_screen_pass']=all(checks.values())
        write(out/'report.json',report);write(out/'predictions.json',predictions);write(out/'evidence_sha256.json',seals)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','collect','score']);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--limit',type=int,default=64);p.add_argument('--workers',type=int,default=4);p.add_argument('--allow-paid',action='store_true');a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command=='collect':collect(a.output,a.limit,a.workers,a.allow_paid)
    else:score(a.output)
