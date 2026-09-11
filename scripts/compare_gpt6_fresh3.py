"""GPT-6 Astra on the sealed Sol/Gemini three-target cohort."""
import json
import shutil
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import compare_sol_gemini_fresh3 as f
t,old,main=f.t,f.old,f.main
SOURCE=f.OUT
OUT=ROOT/'artifacts/preflight/gpt6_astra_fresh3_20260911_v1'
MODEL='openai/gpt-6-astra'
original_body=t.base_body

def body(wire,name):
    result=original_body(wire,'sol' if name=='astra' else name)
    if name=='astra':
        result['model']=MODEL
    return result

def run():
    if OUT.exists():
        raise ValueError('Fresh archive required')
    t.verify(f.SOURCE)
    done=old.read(SOURCE/'completion.json')
    assert all(old.sha(SOURCE/rel)==digest for rel,digest in done['evidence_sha256'].items())
    plan=deepcopy(old.read(SOURCE/'plan.json'))
    for s in plan['selection']:
        for im in s['images']:
            assert old.sha(im['path'])==im['sha256']
    bases=t.bases_for(plan)
    response=requests.get('https://openrouter.ai/api/v1/models/'+MODEL+'/endpoints',timeout=30)
    response.raise_for_status()
    metadata=response.json()
    endpoint=next(e for e in metadata['data']['endpoints'] if e['tag']=='openai')
    assert endpoint['status']==0
    assert {'reasoning','max_tokens','response_format','structured_outputs'}<=set(endpoint['supported_parameters'])
    rates=[str(max([Decimal(endpoint['pricing'][k]),*(Decimal(o.get(k,'0')) for o in endpoint['pricing'].get('overrides',[]))])) for k in ('prompt','completion')]
    t.MODELS['astra']={'model':MODEL,'tag':'openai','provider':'OpenAI'}
    t.base_body=body
    plan['base_rates']['astra']=rates
    plan.update(profile='gpt6_astra_fresh3',model=MODEL,baseline_archive=str(SOURCE),max_calls=39,
                limits={'openrouter_usd':'8','aliyun_cny':'0.7','xai_usd':'0'},
                policy='Same frozen three targets, images, semantic prompts, low reasoning and five reviewers. Base through OpenRouter/OpenAI only; Qwen reviewer still direct Aliyun. One attempt per call; no retries or substitutes; baseline results reused without API calls.')
    old.save(OUT/'plan.json',plan)
    old.save(OUT/'provider_metadata.json',metadata)
    shutil.copyfile(__file__,OUT/'frozen_runner.py')
    shutil.copyfile(f.__file__,OUT/'frozen_target_helper.py')
    shutil.copytree(SOURCE/'priors',OUT/'priors')
    f.OUT=OUT
    checks=[]
    with old.joint.roster.lightweight_protocol():
        for s in plan['selection']:
            mocks={n:t.Mock(n) for n in ('astra','sol','gemini')}
            for n,c in mocks.items():
                row=f.run_target(c,bases[s['key']],s,plan)
                assert row['status']=='PREDICTED' and Counter(r['stage'] for r in c.rows)==Counter(main.STAGES)
            for key in mocks['astra'].wires:
                wires=[]
                for c in mocks.values():
                    w=deepcopy(c.wires[key])
                    if key[1]=='base':
                        for k in ('model','provider','temperature'):
                            w.pop(k,None)
                    wires.append(w)
                assert wires[0]==wires[1]==wires[2]
            checks.append({'key':s['key'],'paired_mock_wires_equal':True})
    old.save(OUT/'preflight.json',{'checks':checks,'plan_sha256':old.sha(OUT/'plan.json')})
    print(json.dumps({'ready':checks,'model':MODEL,'max_calls':39}),flush=True)
    rows=[]; start=perf_counter()
    with old.joint.credential_context(plan),old.joint.roster.lightweight_protocol():
        calls=t.ModelCalls(OUT/'astra','astra',plan)
        calls.max_calls=39
        calls.limits={k:Decimal(v) for k,v in plan['limits'].items()}
        try:
            for s in plan['selection']:
                row=f.run_target(calls,bases[s['key']],s,plan)
                rows.append(row)
                old.save(OUT/'astra/predictions.json',{'targets':rows})
                print(json.dumps({'key':row['key'],'status':row['status'],'seconds':row['seconds']}),flush=True)
        finally:
            calls.stopped=True
            try:calls.persist()
            finally:calls.close_ledger()
    old.save(OUT/'completion.json',{'wall_seconds':perf_counter()-start,'evidence_sha256':{str(p.relative_to(OUT)):old.sha(p) for p in OUT.rglob('*.json')}})
    replay=t.Replay(OUT/'astra','astra')
    with old.joint.roster.lightweight_protocol():
        for s,r in zip(plan['selection'],rows,strict=True):
            rr=f.run_target(replay,bases[s['key']],s,plan)
            assert rr['status']==r['status'] and rr['predictions']==r['predictions']
    assert replay.used==set(replay.rows)
    truth=old.read(SOURCE/'scored_truth.json')
    old.save(OUT/'scored_truth.json',truth)
    charges=defaultdict(Decimal); errors=[]
    for c in old.read(OUT/'astra/budget.json')['calls']:
        p=OUT/'astra/calls'/f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"/'response.json'
        raw=old.read(p) if p.exists() else {}
        message=raw.get('body',{}).get('error',{}).get('message','')
        free='No credits were charged' in message
        charges[c['account']+'|'+('provider_reported_no_charge' if free else c['charge_kind'])]+=Decimal('0') if free else Decimal(c['charge'])
        if c['status'] not in t.VALID:
            errors.append({k:c.get(k) for k in ('target','stage','seat','http_status','status')}|{'message':message})
    report=old.read(SOURCE/'comparison.json')
    report['models']['astra']={'completed':sum(r['status']=='PREDICTED' for r in rows),'metrics':t.release.metrics(rows,truth),'calls':len(replay.rows),'charges':{k:str(v) for k,v in charges.items()},'errors':errors,'target_seconds':sum(r['seconds'] for r in rows)}
    report['baseline_wall_seconds']=report.pop('wall_seconds')
    report['astra_wall_seconds']=old.read(OUT/'completion.json')['wall_seconds']
    report['baseline_reused']=True
    assert all(old.sha(ROOT/n)==h for n,h in plan['unchanged_root_hashes'].items())
    old.save(OUT/'comparison.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':run()
