"""Independent sealed-evidence and metric check; no API calls."""
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/preflight/sol_gemini_fresh3_20260911_v1'
def read(p):
    return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
    done = read(OUT / 'completion.json')
    assert all(sha(OUT / rel) == digest for rel,digest in done['evidence_sha256'].items())
    plan = read(OUT / 'plan.json')
    assert all(sha(ROOT / rel) == digest for rel,digest in plan['unchanged_root_hashes'].items())
    report, truth = read(OUT / 'comparison.json'), read(OUT / 'scored_truth.json')
    results, evidence = {}, {}
    for name in ('sol','gemini'):
        rows = read(OUT / name / 'predictions.json')['targets']
        evidence[name] = rows
        assert [r['key'] for r in rows] == [s['key'] for s in plan['selection']]
        for arm, expected in report['models'][name]['metrics'].items():
            fs = []
            for task in ('instrument','verb','target','ivt','phase'):
                tp=fp=fn=0
                for r in rows:
                    gt = truth[r['key']]
                    if not gt['mask'][task]:
                        continue
                    pred = r['predictions'][arm]
                    p=set(pred[task] if pred else []); g=set(gt['gt'][task])
                    tp+=len(p&g); fp+=len(p-g); fn+=len(g-p)
                f=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
                assert all(expected[task][k] == v for k,v in [('tp',tp),('fp',fp),('fn',fn),('f1',f)])
                fs.append(f)
            assert abs(sum(fs)/5-expected['mean_f1'])<1e-9
        charges, errors = defaultdict(Decimal), []
        calls=read(OUT/name/'budget.json')['calls']
        for c in calls:
            folder=OUT/name/'calls'/f"{c['index']:03d}_{c['target']}_{c['stage']}_{c['seat']}"
            raw=read(folder/'response.json') if (folder/'response.json').exists() else {}
            message=raw.get('body',{}).get('error',{}).get('message','')
            nocharge='No credits were charged' in message
            charges[c['account']+'|'+('provider_reported_no_charge' if nocharge else c['charge_kind'])]+=Decimal('0') if nocharge else Decimal(c['charge'])
            if c['status'] not in ('JSON_PARSED','JSON_PARSED_FENCE_NORMALIZED'):
                errors.append({'target':c['target'],'stage':c['stage'],'status':c['status'],'message':message})
        results[name]={'charges':{k:str(v) for k,v in charges.items()},'errors':errors,'calls':len(calls)}
    common={r['key'] for r in evidence['sol'] if r['status']=='PREDICTED'} & {r['key'] for r in evidence['gemini'] if r['status']=='PREDICTED'}
    # Per-target diagnostics use exact label symmetric differences, not averages of frame F1.
    diagnostics=[]
    for key in sorted(common):
        d={'key':key,'gt_phase':truth[key]['gt']['phase']}
        for name in evidence:
            row=next(r for r in evidence[name] if r['key']==key)
            d[name]={a:{'errors':sum(len(set(p[t])^set(truth[key]['gt'][t])) for t in truth[key]['gt']),
                          'phase':p['phase'],'ivt':p['ivt']} for a,p in row['predictions'].items()}
        diagnostics.append(d)
    audit={'sealed_hashes_passed':True,'independent_metrics_passed':True,'costs':results,
           'common_successful_targets':sorted(common),'diagnostics':diagnostics}
    (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    run()
