"""Offline decomposition; GT is used for diagnosis only, never admission."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.run_prior_panel_trial import save
from surgical_agent.research.verification.prior_panel import COMPONENTS


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def sources():
    for name, arm in [('new_training_verb_guard_20260909_v1','original'), ('verb_prompt_20260909_v1','control')]:
        folder = ROOT / 'artifacts/preflight' / name
        rows = read(folder / 'predictions.json')
        rows = rows['targets'] if isinstance(rows,dict) else rows
        truth = {(t['video_id'],t['frame_id']):t for t in read(folder / 'scored_truth.json')}
        initial = None if arm=='original' else {i['key']:i for i in read(folder / 'initial_state.json')}
        for r in rows:
            if arm=='original':
                graph = read(folder / 'targets' / r['key'] / 'result.json')['graph']
            else:
                graph = read(folder / 'targets' / r['key'] / 'control.json')
                graph['pool'] = initial[r['key']]['pool']
            yield name, r, r[arm], graph, truth[r['video_id'],r['frame_id']]


def main():
    summaries, details = {}, []
    for name,r,final,graph,t in sources():
        c = summaries.setdefault(name,Counter())
        c['targets'] += 1
        means, diags = graph.get('means') or {}, graph.get('diagnostics') or {}
        ids = {p['label_id'] for p in graph['pool']['propositions'] if p['task']=='ivt'}
        if t['mask']['ivt']:
            for i in t['gt']['ivt']:
                c['GT_IVT'] += 1
                if i not in ids:
                    reason='outside_pool'
                elif i in final['ivt']:
                    reason='selected_correct'
                elif means.get(f'ivt_{i}') is None:
                    reason='invalid_IVT_evidence'
                elif means[f'ivt_{i}']<4:
                    reason='IVT_score_below_admission'
                elif any(means.get(f'{q}_{v}') is None for q,v in COMPONENTS[i].items()):
                    reason='invalid_component_evidence'
                elif any(means[f'{q}_{v}']<4 for q,v in COMPONENTS[i].items()):
                    reason='component_score_below_admission'
                else:
                    reason='unexpected_selector_omission'
                c[reason] += 1
                details.append({'source':name,'key':r['key'],'ivt':i,'reason':reason})
            for i in set(final['ivt'])-set(t['gt']['ivt']):
                state='new_false_IVT' if i not in r['h0']['ivt'] else (
                    'old_false_IVT_invalid_review' if means.get(f'ivt_{i}') is None else
                    'old_false_IVT_supported' if means[f'ivt_{i}']>=4 else 'old_false_IVT_uncertain')
                c[state] += 1
        for p in graph['pool']['propositions']:
            pid,q,v=p['id'],p['task'],p['label_id']
            score=means.get(pid)
            if q in ('instrument','verb','target') and v in r['h0'][q] and score is not None and score<=2 and v in final[q]:
                c['component_deletion_protected_by_IVT'] += 1
                details.append({'source':name,'key':r['key'],'id':pid,'reason':'component_deletion_protected_by_IVT',
                    'valid_gt':t['mask'][q],'is_error':v not in t['gt'][q],
                    'protecting_ivt':[i for i in final['ivt'] if COMPONENTS[i][q]==v]})
            scores=diags.get(pid,{}).get('scores',[])
            if v not in r['h0'][q] and v in final[q] and any(x is not None and x<=2 for x in scores):
                c['addition_despite_valid_refutation'] += 1
                details.append({'source':name,'key':r['key'],'id':pid,'reason':'addition_despite_valid_refutation',
                    'valid_gt':t['mask'][q],'is_error':v not in t['gt'][q],'scores':scores})
    save(ROOT / 'artifacts/preflight/repair_failure_layers_20260909.json',
         {'scope':'48 previously inspected Training targets, two original arms. No API calls.',
          'summary':{k:dict(v) for k,v in summaries.items()},'details':details})
    print(json.dumps({k:dict(v) for k,v in summaries.items()},indent=2))


if __name__=='__main__':
    main()
