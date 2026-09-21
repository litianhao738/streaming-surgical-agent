"""Offline, fixed-threshold Gate sensitivity experiment; no API transport.

New heuristic thresholds are exploratory, not validation-tuned or optimal.
Selections and predictions are sealed before this run loads GT.
"""
import argparse
import csv
from copy import deepcopy
from statistics import mean, stdev
import run_mechanism_internal_ablation as r
from mechanisms import original, canonical_labels, rule_score, digest_key, HEADS, SLOTS
from metrics import evaluate


def uncertainty_score(snapshot):
    ratings = snapshot['probe_ratings_all']
    values = [ratings[p['id']] for p in snapshot['pool']['propositions']
              if not original.ambiguous(p)]
    # Ratings are ordinal evidence, not calibrated probabilities.
    candidate = mean(1. if x is None else 1.-abs(x-3.)/2. for x in values) if values else 1.
    phase = [ratings['phase_'+str(i)] for i in range(7)]
    if any(x is None for x in phase):
        phase_uncertainty = 1.
    else:
        ordered = sorted(phase, reverse=True)
        phase_uncertainty = 1.-(ordered[0]-ordered[1])/4.
    return max(candidate, phase_uncertainty)


def choose(snapshots, config):
    keys = sorted(snapshots)
    scores = {k: {'uncertainty': uncertainty_score(snapshots[k]),
                  'rule': rule_score(snapshots[k]),
                  'learned': snapshots[k]['gate_score']} for k in keys}
    selected = {
        name: [k for k in keys if scores[k][name] >= config[name+'_threshold']]
        for name in ('uncertainty','rule','learned')}
    selected['always'] = keys
    selected['skip'] = []
    # Independent deterministic Bernoulli draws, not a ranked top-K.
    cutoff = int(config['random_probability'] * 2**256)
    for seed in config['random_seeds']:
        selected['random_seed'+str(seed)] = [k for k in keys
            if int(digest_key('bernoulli_fixed_v1', seed, k),16) < cutoff]
    return selected, scores


def selection_metrics(keys, selected, cheap, full, truth):
    usable = [k for k in keys if any(truth[k]['mask'][h] for h in HEADS)]
    selected = set(selected)
    def loss(k, prediction):
        return sum(len(set(prediction[k][h]) ^ set(truth[k]['gt'][h]))
                   for h in HEADS if truth[k]['mask'][h])
    errors = {k for k in usable if loss(k,cheap)>0}
    reviewed = [k for k in usable if k in selected]
    beneficial = sum(loss(k,full)<loss(k,cheap) for k in reviewed)
    harmful = sum(loss(k,full)>loss(k,cheap) for k in reviewed)
    return {'eligible_frames':len(usable),'error_frames':len(errors),
            'reviewed_eligible_frames':len(reviewed),
            'selected_error_frames':len(errors & selected),
            'beneficial_frames':beneficial,'harmful_frames':harmful,
            'error_recall':100*len(errors & selected)/len(errors) if errors else None,
            'bvr':100*beneficial/len(reviewed) if reviewed else None}


def run(source, out):
    if out.exists(): raise ValueError('Use a new output directory')
    plan = r.load_plan(source,verify_sources=True)
    final, receipt, source_audit = (r.read(source/name) for name in
        ('final_receipt.json','prediction_receipt.json','audit.json'))
    assert final['state']=='COMPLETE' and source_audit['state']=='PASS'
    assert final['plan_sha256']==r.sha(source/'plan.json')==receipt['plan_sha256']
    assert r.sha(source/'diagnostics.json')==receipt['diagnostics_sha256']
    for name,digest in receipt['arms'].items():
        assert r.sha(source/'predictions'/(name+'.json'))==digest
    snapshots = r.read(source/'snapshots.json')
    original_selection = r.read(source/'selection.json')
    config = {'profile':'fixed_threshold_exploratory_v1','utc':r.now(),
              'source':str(source),'source_plan_sha256':r.sha(source/'plan.json'),
              'source_prediction_receipt_sha256':r.sha(source/'prediction_receipt.json'),
              'source_final_receipt_sha256':r.sha(source/'final_receipt.json'),
              'implementation_sha256':r.sha(__file__),
              'learned_threshold':original_selection['threshold'],
              'rule_threshold':.5,'uncertainty_threshold':.5,
              'random_probability':.5,'random_seeds':list(range(20)),
              'rule_definition':'Unchanged original refute + support + phase score; >=0.5',
              'uncertainty_definition':'max(mean(1-abs(rating-3)/2) over nonambiguous four-head candidates, 1-(top_phase-second_phase)/4); missing candidate rating=1; no candidates=1; any missing phase rating=1',
              'threshold_provenance':'Learned frozen original; new baselines fixed by semantic/default choice before new metrics, not calibrated on validation data',
              'bvr_definition':'Strict decrease in total masked five-head FP+FN relative to cheap; not asserted identical to learned training target',
              'evaluation_policy':'447 scoring targets, valid head masks; additionally report complete-five-head subset',
              'testing_results_previously_seen':True,'new_api_calls':0}
    out.mkdir(parents=True)
    r.write(out/'protocol.json',config)
    selected,scores = choose(snapshots,config)
    assert set(selected['learned'])==set(original_selection['learned'])
    r.write(out/'selection.json',selected)
    r.write(out/'scores.json',scores)
    r.write(out/'selection_receipt.json',{'state':'SEALED','utc':r.now(),
        'protocol_sha256':r.sha(out/'protocol.json'),'selection_sha256':r.sha(out/'selection.json'),
        'scores_sha256':r.sha(out/'scores.json'),'GT_loaded_in_this_run':False})
    cheap = {k:deepcopy(s['cheap']) for k,s in snapshots.items()}
    diagnostics = r.read(source/'diagnostics.json')
    full = {k:canonical_labels(original.repair(cheap[k],deepcopy(diagnostics[k]['full']['before_rollback'])))
            for k in snapshots}
    prior_full = r.read(source/'predictions/B4.json')
    assert all(full[k]==prior_full[k] for k in selected['learned'])
    predictions = {name:{k:deepcopy(full[k] if k in set(keys) else cheap[k]) for k in snapshots}
                   for name,keys in selected.items()}
    assert predictions['learned']==prior_full
    for name,prediction in predictions.items():r.write(out/'predictions'/(name+'.json'),prediction)
    r.write(out/'prediction_receipt.json',{'state':'SEALED','utc':r.now(),
        'selection_receipt_sha256':r.sha(out/'selection_receipt.json'),
        'arms':{n:r.sha(out/'predictions'/(n+'.json')) for n in predictions},'new_api_calls':0})
    print({'state':'PREDICTIONS_SEALED','reviewed':{n:len(v) for n,v in selected.items()}},flush=True)
    # First GT access of this run occurs only after both seals above.
    old,_ = r.imports()
    evaluation_keys = plan['evaluation_keys']
    rows = [{k:snapshots[key][k] for k in ('key','video_id','frame_id')} for key in evaluation_keys]
    truth,annotations = old.core.demo.load_testing_truth(rows)
    complete = [k for k in evaluation_keys if all(truth[k]['mask'][h] for h in HEADS)]
    prefix,historical,jobs = (r.read(source/n) for n in ('prefix_usage.json','historical_evidence.json','jobs.json'))
    refs = {k:dict(v) for k,v in historical.items()}
    for job in jobs:
        if job['panel']=='multi':
            refs[job['key']][job['seat']]=r.read(source/'completed'/(job['id']+'.json'))['evidence']
    common = [ref for frame in prefix.values() for ref in frame.values()]
    metrics, costs, table = {}, {}, []
    for name,keys in selected.items():
        details = selection_metrics(evaluation_keys,keys,cheap,full,truth)
        scores = evaluate(evaluation_keys,cheap,predictions[name],truth)
        metrics[name] = {'selection':details,'task':scores,
            'complete_five_head_selection':selection_metrics(complete,keys,cheap,full,truth),
            'per_video':{v:selection_metrics([k for k in evaluation_keys if snapshots[k]['video_id']==v],keys,cheap,full,truth)
                         for v in sorted({s['video_id'] for s in snapshots.values()})}}
        costs[name] = r.token_summary(common+[refs[k][seat] for k in keys for seat in SLOTS[1:]])
        assert costs[name]['requests']==3*len(snapshots)+4*len(keys)
        table.append({'strategy':name,'verify_percent':100*len(keys)/len(snapshots),
            'reviewed_frames':len(keys),**details,'ivt_f1':scores['by_head']['ivt']['f1'],
            'phase_accuracy':scores['by_head']['phase']['accuracy'],
            'logical_calls':costs[name]['requests'],'tokens_million_known':costs[name]['known_total_tokens']/1e6,
            'usage_missing':costs[name]['usage_missing'],'logical_rmb_known':costs[name]['known_rmb']})
    random_rows = [v for v in table if v['strategy'].startswith('random_seed')]
    numeric = [k for k in table[0] if k!='strategy']
    random_summary = {k:{'mean':mean([v[k] for v in random_rows if v[k] is not None]),
                            'sd':stdev([v[k] for v in random_rows if v[k] is not None])} for k in numeric}
    mean_row = {'strategy':'random_mean',**{k:v['mean'] for k,v in random_summary.items()}}
    main = [mean_row]+[next(v for v in table if v['strategy']==n) for n in ('uncertainty','rule','learned','always','skip')]
    r.write(out/'metrics.json',{'arms':metrics,'random_summary':random_summary,'annotation_sha256':annotations})
    r.write(out/'cost_logical.json',costs)
    for filename,values in [('table_gate_fixed_threshold',main),('all_seeds',table)]:
        with (out/(filename+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(main[0]));writer.writeheader();writer.writerows(values)
    r.write(out/'receipt.json',{'state':'COMPLETE','utc':r.now(),'new_api_calls':0,
        'protocol_sha256':r.sha(out/'protocol.json'),'table_sha256':r.sha(out/'table_gate_fixed_threshold.csv'),
        'metrics_sha256':r.sha(out/'metrics.json'),'prediction_receipt_sha256':r.sha(out/'prediction_receipt.json'),
        'checks':['original learned selection and prediction identical','source sealed prediction and diagnostic hashes',
                  'new selections and predictions sealed before new GT access','logical calls equal 3N+4K',
                  'mask-aware BVR and error recall; missing GT excluded']})
    print({'state':'COMPLETE','table':main,'new_api_calls':0},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(r.safe_output(args.source),r.safe_output(args.output))
