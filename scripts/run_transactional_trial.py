"""New Training targets: original review, selector-only, explicit-edit review."""
import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from scripts import run_new_training_verb_guard_trial as base_trial
from scripts.check_candidate_panel_providers import ACADEMIC, redact_images
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_graph_review_trial import TimedCalls
from scripts.run_grounded_api_pipeline import score_saved
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import PROVIDERS, RATES_V2, review_wire
from scripts.run_semantic_candidate_trial import BOUNDARIES
from scripts.score_disputed_relation_trial import exact_same as same
from scripts.score_five_head_repair_trial import ReplayCalls
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import transactional_review as tx
from surgical_agent.research.verification.candidate_coordinator import make_pool

INPUTS=ROOT/'artifacts/preflight/transactional_inputs_20260909_v1'
TEMPLATE=ROOT/'src/surgical_agent/research/verification/prompts/transactional_review_v1.json'
STAGE='edit_review'
MAX_CALLS=288
LIMITS={'openrouter_usd':'6','xai_usd':'3','aliyun_cny':'3'}
VERSIONS=('h0','original','selector_only','edit_transactions')
SUCCESS=('Versus same-cohort original, explicit-edit transactions must improve Verb Precision, '
         'retain Verb Recall/F1 and IVT Precision/F1, preserve other head Precision/F1, '
         'not increase summed label errors, achieve beneficial actual changes and have no selector failures. '
         'Verb Precision must be >= H0. Report selector-only separately, never choose a winner with these labels. '
         'No post-score tuning. New target frames in old Training videos; not held-out test performance.')


def edit_wire(seat,base,selected,h0,pool):
    body=review_wire(seat,base,selected,pool)
    edits=tx.claims(h0,pool)
    schema=tx.response_schema(edits,len(base.images))
    packet={'academic_context':ACADEMIC,**read(TEMPLATE),'proposition_semantics':BOUNDARIES,
            'image_order':['t-2 seconds','t-1 second','current target'],
            'candidate_changes':edits,'response_schema':schema}
    body['messages'][0]['content'][0]['text']=json.dumps(packet,ensure_ascii=False)
    # Qwen keeps its original JSON-object mode; strict-schema providers receive the new contract.
    if body.get('response_format',{}).get('type')=='json_schema':
        body['response_format']={'type':'json_schema','json_schema':{
            'name':'explicit_edit_transactions_v1','strict':True,'schema':schema}}
    return body


def verify(output):
    plan=read(output/'plan.json')
    base_trial.verify_plan(output/'foundation')
    for k,v in {'version':tx.VERSION,'minimum_votes':tx.MIN_VOTES,'max_calls':MAX_CALLS,
                'limits':LIMITS,'success_rule':SUCCESS}.items():
        same(plan[k],v,'frozen protocol')
    for n,h in plan['extra_sources'].items():
        same(sha(ROOT/n),h,'source')
    same(sha(output/'foundation/plan.json'),plan['foundation_plan_sha256'],'foundation plan')
    return plan


def prepare(output,adapter):
    if output.exists():
        raise ValueError('fresh output directory required')
    base_trial.prepare(output/'foundation',INPUTS/'selection.json',INPUTS/'priors',INPUTS/'budget.json',adapter)
    foundation=read(output/'foundation/plan.json')
    extra=[Path(__file__).resolve(),TEMPLATE,ROOT/'tests/unit/test_transactional_review.py',
           ROOT/'tests/unit/test_transactional_wire.py',ROOT/'scripts/prepare_transactional_inputs.py']
    plan={'created_utc':now(),'version':tx.VERSION,'minimum_votes':tx.MIN_VOTES,'max_calls':MAX_CALLS,
          'limits':LIMITS,'success_rule':SUCCESS,'selection':foundation['selection'],
          'foundation_plan_sha256':sha(output/'foundation/plan.json'),
          'extra_sources':{p.relative_to(ROOT).as_posix():sha(p) for p in extra},
          'protocol':'Shared newly called H0 and original graph proposer. Original five-seat review then explicit-edit five-seat review. Same models and causal images. Selector-only uses original reviews without new calls. No retries, no extra repair rounds, Phase fixed to H0. All predictions stored before GT. Different review order is fixed, not randomized.',
          'foundation_scope':'Only reuse foundation input preparation and validation. Its old guard run profile is not this experiment. Top-level plan/call cap/arms are authoritative.'}
    save(output/'plan.json',plan)
    for p in extra:
        dst=output/'frozen_source'/p.relative_to(ROOT)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    # Check each actual seat body, preserving original image bytes/settings and academic context.
    s=plan['selection'][0];view=base_trial.InferenceOnlyAdapter(adapter);base=build_gemini_base(view,s)
    h0={'instrument':[0],'verb':[1],'target':[0],'ivt':[17],'phase':[1]}
    pool=make_pool(h0)
    save(output/'wire_smoke.json',{seat:redact_images(edit_wire(seat,base,s,h0,pool)) for seat in tx.SEATS})
    verify(output)
    print(json.dumps({'prepared':True,'targets':24,'max_calls':MAX_CALLS,'limits':LIMITS,
                      'plan_sha256':sha(output/'plan.json')}),flush=True)


class BoundCalls(TimedCalls):
    def __init__(self,output,plan):
        super().__init__(output,limits={k:Decimal(v) for k,v in LIMITS.items()},rates=RATES_V2,
                         providers=PROVIDERS,max_calls=MAX_CALLS,reasoning_seats=('grok','gemini'))
        self.known={s['key'] for s in plan['selection']};self.attempted=set()

    def call(self,target,stage,seat,body):
        if target not in self.known or (stage,seat) not in base_trial.ALLOWED_CALLS|{(STAGE,s) for s in tx.SEATS}:
            raise ValueError('undeclared call')
        with self.lock:
            key=(target,stage,seat)
            if key in self.attempted:
                raise ValueError('retry forbidden')
            self.attempted.add(key)
            save(self.output/'request_intents'/f'{target}_{stage}_{seat}.json',
                 {'fingerprint':fingerprint(body),'request':redact_images(body)})
        return super().call(target,stage,seat,body)


def run_target(calls,base,s,prior):
    original=base_trial.run_target(calls,base,s,prior)
    result={k:deepcopy(original[k]) for k in ('h0','original','status','graph','h0_raw')}
    result.update(selector_only=deepcopy(result['original']),edit_transactions=deepcopy(result['original']),
                  selector_result=None,edit_result=None,edit_raw=None,edit_seconds=0,
                  edit_status='UPSTREAM_UNAVAILABLE')
    graph=result['graph']
    if graph is None or graph['reviews'] is None:
        return result
    evidence=tx.assess_legacy(graph['reviews'],graph['pool'],len(base.images))
    result['selector_result']=tx.apply(result['h0'],graph['pool'],evidence)
    result['selector_only']=result['selector_result']['prediction']
    start=perf_counter()
    with ThreadPoolExecutor(max_workers=5) as workers:
        raw=dict(zip(tx.SEATS,workers.map(lambda seat:calls.call(s['key'],STAGE,seat,
            edit_wire(seat,base,s,result['h0'],graph['pool'])),tx.SEATS),strict=True))
    result['edit_seconds']=perf_counter()-start
    edits=tx.claims(result['h0'],graph['pool'])
    evidence=tx.assess(raw,edits,len(base.images))
    result['edit_result']=tx.apply(result['h0'],graph['pool'],evidence)
    result.update(edit_transactions=result['edit_result']['prediction'],edit_raw=raw,edit_evidence=evidence,
                  edit_status='REVIEWED')
    return result


def execute(output,adapter):
    plan=verify(output);foundation=read(output/'foundation/plan.json')
    with (output/'execution.lock').open('x') as f:
        f.write(sha(output/'plan.json'))
    calls=BoundCalls(output,plan);calls.persist()
    rows=[{'key':s['key'],'video_id':s['video_id'],'frame_id':s['frame_id'],
           **dict.fromkeys(VERSIONS),'status':'NOT_ATTEMPTED','edit_status':'NOT_ATTEMPTED'} for s in plan['selection']]
    view=base_trial.InferenceOnlyAdapter(adapter);start=perf_counter();fatal=None
    try:
        for s,row in zip(plan['selection'],rows,strict=True):
            if calls.stopped:
                row['status']='BUDGET_OR_PROVIDER_STOPPED';continue
            base=build_gemini_base(view,s)
            same(fingerprint(gemini_h0_wire(base)),foundation['h0_fingerprints'][s['key']],'frozen H0')
            prior=read(output/'foundation/priors'/f"{s['video_id']}.json")
            record=run_target(calls,base,s,prior)
            save(output/'targets'/s['key']/'result.json',record)
            row.update({k:record[k] for k in (*VERSIONS,'status','edit_status')})
            save(output/'predictions.json',rows)
            print(json.dumps({'key':s['key'],'calls':len(calls.rows),'status':row['status'],'edit_status':row['edit_status']}),flush=True)
    except Exception as exc:
        fatal=type(exc).__name__;raise
    finally:
        calls.stopped=True;calls.persist();save(output/'predictions.json',rows)
        try:
            verify(output)
        except Exception:
            fatal=fatal or 'FROZEN_SOURCE_CHANGED';raise
        finally:
            files=[output/n for n in ('plan.json','execution.lock','budget.json','predictions.json')]
            files += [p for folder in ('targets','calls','request_intents') for p in (output/folder).rglob('*.json')]
            save(output/'completion.json',{'closed_utc':now(),'fatal_error':fatal,'elapsed_seconds':perf_counter()-start,
                'post_calls':len(calls.rows),'hashes':{p.relative_to(output).as_posix():sha(p) for p in sorted(files)}})


def score(output,adapter):
    plan=verify(output);done=read(output/'completion.json');ledger=read(output/'budget.json')
    if done['fatal_error'] or not ledger['stopped']:
        raise ValueError('nonfatal closure required')
    for n,h in done['hashes'].items():
        same(sha(output/n),h,'closed inference')
    rows=read(output/'predictions.json');calls=ledger['calls']
    same(len(calls),done['post_calls'],'call count')
    if len(calls)>MAX_CALLS or len({(c['target'],c['stage'],c['seat']) for c in calls})!=len(calls):
        raise ValueError('duplicate/excess calls')
    for account,value in ledger['occupied'].items():
        costs=[Decimal(c['charge']) for c in calls if c['account']==account]
        if any(not x.is_finite() or x<0 for x in costs):
            raise ValueError('invalid charge')
        same(sum(costs),Decimal(value),'ledger arithmetic')
        if Decimal(value)>Decimal(LIMITS[account]):
            raise ValueError('budget exceeded')
    replay=ReplayCalls(output,calls);view=base_trial.InferenceOnlyAdapter(adapter);records=[]
    for s,row in zip(plan['selection'],rows,strict=True):
        same((s['key'],s['video_id'],s['frame_id']),(row['key'],row['video_id'],row['frame_id']),'identity')
        path=output/'targets'/s['key']/'result.json'
        if not path.exists():
            same(row['status'],'BUDGET_OR_PROVIDER_STOPPED','missing target');continue
        stored=read(path);base=build_gemini_base(view,s)
        rebuilt=run_target(replay,base,s,read(output/'foundation/priors'/f"{s['video_id']}.json"))
        same(base_trial.without_timing(rebuilt),base_trial.without_timing(stored),'raw replay')
        for a in VERSIONS:
            same(row[a],rebuilt[a],'prediction')
        records.append(stored)
    same(len(replay.rows),len(calls),'all raw calls replayed')
    _,truth=score_saved(adapter,[{**r,'h1':None,'final':r['edit_transactions']} for r in rows])
    truths={(t['video_id'],t['frame_id']):t for t in truth}
    metrics={a:compute_repair_comparison([{**truths[r['video_id'],r['frame_id']],
        'h0':r['h0'],'h1':None,'final':r[a]} for r in rows])['arms']['final'] for a in VERSIONS}
    details={a+'_to_'+b:[{'key':r['key'],**frame_delta(r[a],r[b],truths[r['video_id'],r['frame_id']]['gt'],
             truths[r['video_id'],r['frame_id']]['mask'])} for r in rows]
             for a,b in [('h0','original'),('h0','edit_transactions'),('original','selector_only'),('original','edit_transactions')]}
    comps={k:summarize_deltas(v) for k,v in details.items()}
    before,after=metrics['original']['tasks'],metrics['edit_transactions']['tasks']
    checks={'verb_precision_improves':after['verb']['micro_precision']>before['verb']['micro_precision'],
        'verb_precision_ge_H0':after['verb']['micro_precision']>=metrics['h0']['tasks']['verb']['micro_precision'],
        'verb_recall_not_decreased':after['verb']['micro_recall']>=before['verb']['micro_recall'],
        **{q+'_'+m+'_not_decreased':after[q][m]>=before[q][m] for q in base_trial.TASKS for m in ('micro_precision','micro_f1')},
        'net_errors_not_increased':comps['original_to_edit_transactions']['net_errors_removed']>=0,
        'actual_beneficial_changes':comps['original_to_edit_transactions']['fixed_label_errors']>0,
        'all_new_reviews_ran':all(r['edit_status']=='REVIEWED' for r in rows)}
    costs={stage:{account:{kind:str(sum(Decimal(c['charge']) for c in calls if c['stage']==stage and c['account']==account
        and c['charge_kind']==kind)) for kind in ('native','conservative_estimate','unknown_reserved')} for account in LIMITS}
        for stage in sorted({c['stage'] for c in calls})}
    timing={'h0':sum(c['elapsed_seconds'] for c in calls if c['stage']=='h0'),
        'proposal':sum(r['graph']['timing']['proposal_seconds'] for r in records if r['graph']),
        'old_review':sum(r['graph']['timing']['review_seconds'] for r in records if r['graph']),
        'edit_review':sum(r['edit_seconds'] for r in records)}
    report={'metrics':metrics,'comparisons':comps,'costs_by_stage':costs,'timing':timing,'elapsed_seconds':done['elapsed_seconds'],
            'post_calls':len(calls),'transport':dict(Counter(c['status'] for c in calls)),
            'success':all(checks.values()),'checks':checks,'raw_replayed':True}
    save(output/'metrics.json',report);save(output/'scored_truth.json',truth);save(output/'frame_deltas.json',details)
    print(json.dumps({'success':report['success'],'calls':len(calls),'checks':checks}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','execute','score'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    globals()[args.command](args.output,CholecTrack20DatasetAdapter(Path('D:/cholec_dataset'),causal_window_size=3))
