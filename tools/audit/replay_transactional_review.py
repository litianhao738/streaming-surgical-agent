"""Selector-only replay of old visual evidence; not a test of new edit prompts."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

from scripts.run_prior_panel_trial import save
from scripts.score_graph_review_trial import frame_delta, summarize_deltas
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification import transactional_review as tx
from tools.audit.audit_repair_failure_layers import sources


def main():
    output=ROOT/'artifacts/preflight/transactional_selector_replay_20260909_v1'
    if output.exists():
        raise ValueError('single-use exploration')
    save(output/'plan.json',{'version':tx.VERSION,'min_votes':tx.MIN_VOTES,
        'policy':'All 48 targets in two original arms; no GT-dependent policy. No threshold tuning. No new API calls.',
        'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__).resolve(),ROOT/'src/surgical_agent/research/verification/transactional_review.py']}})
    grouped={};details=[]
    for name,row,old,graph,t in sources():
        pool=graph['pool']
        reviews=graph.get('reviews') or {s:None for s in tx.SEATS}
        evidence=tx.assess_legacy(reviews,pool)
        result=tx.apply(row['h0'],pool,evidence)
        grouped.setdefault(name,[]).append({**t,'h0':old,'h1':None,'final':result['prediction']})
        details.append({'source':name,'key':row['key'],'result':result,'evidence':evidence,
                        'delta':frame_delta(old,result['prediction'],t['gt'],t['mask'])})
    report={name:{'metrics':compute_repair_comparison(rows),
                  'changes':summarize_deltas([d['delta'] for d in details if d['source']==name])}
            for name,rows in grouped.items()}
    save(output/'report.json',report);save(output/'details.json',details)
    print(json.dumps({name:{'verb_before':r['metrics']['arms']['h0']['tasks']['verb'],
                           'verb_after':r['metrics']['arms']['final']['tasks']['verb'],
                           'ivt_before':r['metrics']['arms']['h0']['tasks']['ivt']['micro_f1'],
                           'ivt_after':r['metrics']['arms']['final']['tasks']['ivt']['micro_f1'],
                           'changes':r['changes']['categories']} for name,r in report.items()},indent=2))


if __name__=='__main__':
    main()
