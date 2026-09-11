"""Independent closed-run metrics, provenance, vote validity and patch audit."""
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'artifacts/preflight/transactional_trial_20260909_v2'
HEADS=('instrument','verb','target','ivt','phase')


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def main():
    report=read(OUTPUT/'metrics.json');done=read(OUTPUT/'completion.json');plan=read(OUTPUT/'plan.json')
    base=read(OUTPUT/'foundation/plan.json');ledger=read(OUTPUT/'budget.json')['calls']
    assert report['raw_replayed'] and not done['fatal_error']
    files={p.relative_to(OUTPUT).as_posix() for folder in ('calls','targets','request_intents') for p in (OUTPUT/folder).rglob('*.json')}
    assert files|{'plan.json','budget.json','execution.lock','predictions.json'}==set(done['hashes'])
    for n,h in done['hashes'].items():
        assert hashlib.sha256((OUTPUT/n).read_bytes()).hexdigest()==h
    counts={};rows=read(OUTPUT/'predictions.json');truth={(t['video_id'],t['frame_id']):t for t in read(OUTPUT/'scored_truth.json')}
    assert len(rows)==len(truth)==24
    for arm in ('h0','original','selector_only','edit_transactions'):
        counts[arm]={}
        for h in HEADS:
            sets=[(set(r[arm][h]) if r[arm] is not None else set(),set(truth[r['video_id'],r['frame_id']]['gt'][h]),r[arm] is not None)
                  for r in rows if truth[r['video_id'],r['frame_id']]['mask'][h]]
            tp=sum(len(p&g) for p,g,ok in sets);fp=sum(len(p-g) for p,g,ok in sets);fn=sum(len(g-p) for p,g,ok in sets)
            result={'tp':tp,'fp':fp,'fn':fn,'valid_targets':len(sets),'failed_predictions':sum(not ok for p,g,ok in sets),
                'exact_matches':sum(ok and p==g for p,g,ok in sets),'micro_precision':tp/(tp+fp) if tp+fp else 0.0,
                'micro_recall':tp/(tp+fn) if tp+fn else 0.0,'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                'exact_set_accuracy':sum(ok and p==g for p,g,ok in sets)/len(sets)}
            assert all(math.isclose(v,report['metrics'][arm]['tasks'][h][k],abs_tol=1e-12) for k,v in result.items())
            counts[arm][h]=result
    transport=Counter();validity=Counter();states=Counter();transactions=Counter();changes=[]
    for call in ledger:
        folder=OUTPUT/'calls'/f"{call['index']:03d}_{call['target']}_{call['stage']}_{call['seat']}"
        assert read(folder/'record.json')==call
        request=read(folder/'request.json');intent=read(OUTPUT/'request_intents'/f"{call['target']}_{call['stage']}_{call['seat']}.json")
        assert request==intent['request']
        response=read(folder/'response.json');transport[str(response['http_status'])]+=1
        if call['status'] in ('JSON_PARSED','JSON_PARSED_FENCE_NORMALIZED'):
            body=response['body'];expected=base['proposer'] if call['seat']=='base' else base['models'][call['seat']]
            assert response['http_status']==200 and body['model']==expected
            if call['seat'] in base['providers']:
                assert body['provider']==base['providers'][call['seat']]
            assert body['choices'][0]['finish_reason']=='stop' and not body['choices'][0]['message'].get('refusal')
        if call['stage']=='edit_review':
            packet=json.loads(request['messages'][0]['content'][0]['text'])
            assert next(iter(packet))=='academic_context'
            assert all(e['operation'] in ('ADD','REMOVE') for e in packet['candidate_changes'])
            if call['seat']=='qwen':
                assert 'JSON' in packet['output_format']
    for row in rows:
        record=read(OUTPUT/'targets'/row['key']/'result.json')
        for e in record.get('edit_evidence',{}).values():
            states[e['state']]+=1
            for v in e['votes'].values():
                validity[v['error'] or 'VALID']+=1
        result=record.get('edit_result') or {}
        for t in result.get('transactions',[]):
            transactions[t['reason']]+=1
        gt=truth[row['video_id'],row['frame_id']]
        for h in HEADS:
            if row['original'] is not None and row['edit_transactions'] is not None and row['original'][h]!=row['edit_transactions'][h]:
                changes.append({'key':row['key'],'task':h,'original':row['original'][h],
                    'new':row['edit_transactions'][h],'gt':gt['gt'][h],'mask':gt['mask'][h],
                    'transactions':result.get('transactions',[])})
    old=Path(plan['compatibility']['aborted_source']);oldcalls=read(old/'budget.json')['calls']
    allcalls=[*oldcalls,*ledger]
    costs={a:{kind:str(sum(Decimal(c['charge']) for c in allcalls if c['account']==a and c['charge_kind']==kind))
              for kind in ('native','conservative_estimate','unknown_reserved')} for a in base['limits']}
    assert all(sum(Decimal(v) for v in kinds.values())<=Decimal(base['limits'][a]) for a,kinds in costs.items())
    for account in base['limits']:
        assert sum(Decimal(c['charge']) for c in ledger if c['account']==account)==Decimal(read(OUTPUT/'budget.json')['occupied'][account])
    out={'all_20_head_tables_verified':True,'all_closed_files_verified':True,'http_statuses':dict(transport),
         'counts':counts,'edit_item_validity':dict(validity),'evidence_states':dict(states),'transactions':dict(transactions),
         'changed_head_rows':changes,'total_costs_including_aborted':costs,'total_calls_including_aborted':len(allcalls)}
    (OUTPUT/'independent_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('counts','changed_head_rows')},indent=2))


if __name__=='__main__':
    main()
