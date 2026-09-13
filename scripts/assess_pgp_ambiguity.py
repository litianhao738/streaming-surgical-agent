"""Frozen offline reassessment. All old results are read-only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import ast
import json
import sys
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import train_pgp_gate_no_tracker as util
from surgical_agent.research.gate import pgp_training as ref
from tools.audit import gate_proposals_offline_20260913 as legacy
from tools.audit.pgp_gate_replay_20260913 import cost_vector
OUT = ROOT / 'artifacts/training/gate/pgp_ambiguity_assessment_20260913_r2'
PROBE = ROOT / 'tools/audit/ambiguity_aware_gate_probe.py'


def quality(c):
    return ref.quality(c)


def fit(X, y, idx, kind):
    if kind == 'lr':
        return make_pipeline(StandardScaler(), LogisticRegression(solver='liblinear', C=1., class_weight='balanced', max_iter=2000, random_state=3407)).fit(X[idx], y[idx])
    pos = y[idx].mean()
    weights = np.where(y[idx] == 1, .5 / pos, .5 / (1-pos))
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15, random_state=3407).fit(X[idx], y[idx], sample_weight=weights)


def run():
    OUT.mkdir(exist_ok=False); (OUT / 'models').mkdir()
    def emit(s):
        print(s, flush=True)
        with (OUT/'run.log').open('a', encoding='utf-8') as f: f.write(s+'\n')
    with legacy.offline():
        old = util.DEFAULT
        recipe = json.loads((old/'recipe.json').read_text('utf-8'))
        rp = json.loads((old/'replay/pgp_replay_receipt.json').read_text('utf-8'))
        bindings = {**rp['source_sha256'], **recipe['tracker_frozen_sha256']}
        for base in (old, ROOT/'artifacts/training/gate/pgp_gate_net_gain_20260913_r2'):
            receipt = json.loads((base/'training_receipt.json').read_text('utf-8'))
            bindings.update({str(base/k):v for k,v in receipt['output_sha256'].items()})
        for p in (PROBE, Path(__file__), ROOT/'docs/PGP_AMBIGUITY_SELECTION_PROTOCOL_2026-09-13.md', ROOT/'DEFAULT_PGP_GATE_VERSION.json', ROOT/'DEFAULT_PIPELINE_VERSION.json'):
            bindings[str(p)] = util.sha(p)
        emit('Verifying frozen sources, prior runs and Tracker')
        for p,h in bindings.items():
            if util.sha(p) != h: raise ValueError('Changed source '+p)
        util.write(OUT/'recipe.json', {'source_sha256':bindings,'primary':'qwen_hgb','api_calls':0,'Testing_access':False,'tracker_enabled':False,'deployable':False})
        (OUT/'previous_default.json').write_bytes((ROOT/'DEFAULT_PGP_GATE_VERSION.json').read_bytes())
        emit('Replaying reviewed ambiguity policy and causal stopping; no source-script training or writes')
        tree = ast.parse(PROBE.read_text('utf-8'))
        body=[]
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='report' for t in node.targets): break
            body.append(node)
        # Capture stop depths as produced by reviewed replay, without modifying its source.
        capture=ast.parse('stop_depths.append((kc, kj))').body[0]
        for node in body:
            if isinstance(node,ast.For) and any(isinstance(t,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='kc' for a in t.targets) for t in node.body):
                node.body.append(capture)
        g={'__file__':str(PROBE),'__name__':'offline_replay_only','stop_depths':[]}
        exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])), str(PROBE),'exec'),g)
        rows=g['rows']; n=len(rows); vids=g['vid']; y=g['y']; ids=np.array([r['sample_id'] for r in rows])
        assert n==6059 and set(vids)=={'VID103','VID23','VID31','VID96'}
        assert all(r['source_split']=='Training' and r['origin']=='original_attempt_observed_behavior' for r in rows)
        names=sorted(k for k in rows[0]['features_postcheap'] if not k.startswith('tracker_'))
        XB=np.array([[r['features_postcheap'][k] for k in names] for r in rows])
        XQ=np.column_stack((XB,g['PQ'])); qnames=names+g['PROBE_NAMES']
        assert XQ.shape==(n,42) and not any('tracker' in k for k in qnames)
        cc,ca,cf=g['CC'],g['CA'],g['CF']
        assert abs(quality(ca)['five_head_mean_f1']-61.7580)<.00005
        # Independently count policy outputs using the existing primary metric implementation.
        newrows=[{**r,'amb':a} for r,a in zip(rows,g['amb_out'])]
        direct=legacy.old.counts(newrows,'amb',False)
        np.testing.assert_array_equal(direct,ca)
        cm=legacy.old.counts(rows,'cheap_labels',True); am=legacy.old.counts(newrows,'amb',True)
        ledger=json.loads((legacy.SOURCE/'costs.json').read_text('utf-8'))['per_call_main_attempt']
        costs=np.zeros((n,2,2,5)) # probe flag, route flag, cost columns
        depths=np.array(g['stop_depths']); assert depths.shape==(n,2)
        for i,(kc,kj) in enumerate(depths):
            for probe in (0,1):
                for route in (0,1):
                    keys=['h0|base']
                    if probe or route or g['rule'][i]: keys.append('proposal|base')
                    if probe: keys.append('control_graph|qwen')
                    if route:
                        keys.append('phase_recommendation|base')
                        keys += ['control_graph|'+s for s in g['ORDER'][:kc] if not(probe and s=='qwen')]
                        keys += ['joint_r1|'+s for s in g['ORDER'][:kj]]
                    costs[i,probe,route]=cost_vector(keys,ledger)
        def measure(route,probe,idx=None):
            idx=np.arange(n) if idx is None else idx; a=route[idx].astype(int)
            counts=np.where(a[:,None,None],ca[idx],cc[idx]); merged=np.where(a[:,None,None],am[idx],cm[idx])
            cv=costs[idx,int(probe),a].sum(axis=0)
            per={str(v):quality(counts[vids[idx]==v]) for v in sorted(set(vids[idx]))}
            return {'original':quality(counts),'merged':quality(merged),**dict(zip(ref.COST_FIELDS,cv.tolist())),
                    'by_video':per,'reviewed':int(a.sum())}
        def choose(s,idx,probe,criterion):
            c,f,a=(quality(z[idx]) for z in (cc,cf,ca)); candidates=[]
            for j in range(1,21):
                thr=float(np.quantile(s[idx],1-j/20)); route=np.zeros(n,bool); route[idx]=s[idx]>=thr
                m=measure(route,probe,idx); q=m['original']
                per=all(vq['five_head_mean_f1']+1e-12>=quality(cc[idx][vids[idx]==v])['five_head_mean_f1'] and vq['total_errors']<=quality(cc[idx][vids[idx]==v])['total_errors'] for v,vq in m['by_video'].items())
                target=max(c['five_head_mean_f1'],f['five_head_mean_f1']) if criterion=='original' else c['five_head_mean_f1']+.9*max(0,a['five_head_mean_f1']-c['five_head_mean_f1'])
                err=min(c['total_errors'],f['total_errors']) if criterion=='original' else c['total_errors']
                ok=per and q['five_head_mean_f1']+1e-12>=target and q['total_errors']<=err and m['reviewed']>0
                candidates.append({'grid':j,'threshold':thr,'feasible':bool(ok),'measurement':m})
            feasible=[c for c in candidates if c['feasible']]
            best=min(feasible,key=lambda c:(c['measurement']['logical_calls'],c['grid'])) if feasible else None
            return {'status':'FEASIBLE' if best else 'INFEASIBLE','threshold':best['threshold'] if best else 1.01,'candidates':candidates}
        summary={'variants':{},'baselines':{}}
        predictions={}
        for name,X,probe,kind in [('qwen_hgb',XQ,True,'hgb'),('base_hgb',XB,False,'hgb'),('qwen_lr',XQ,True,'lr')]:
            emit('Fitting independent nested evaluation: '+name)
            scores=np.zeros(n); routes={c:np.zeros(n,bool) for c in ('retention90','original')}; folds=[]; memo={}
            def fitted(train):
                key=tuple(sorted(set(vids[train])))
                if key not in memo: memo[key]=fit(X,y,train,kind)
                return memo[key]
            for v in sorted(set(vids)):
                train=np.where(vids!=v)[0]; held=np.where(vids==v)[0]; inner=np.zeros(n)
                for u in sorted(set(vids[train])):
                    tr=np.where((vids!=v)&(vids!=u))[0]; va=np.where(vids==u)[0]
                    inner[va]=fitted(tr).predict_proba(X[va])[:,1]
                model=fitted(train); scores[held]=model.predict_proba(X[held])[:,1]
                selections={c:choose(inner,train,probe,c) for c in routes}
                for c in routes: routes[c][held]=scores[held]>=selections[c]['threshold']
                folds.append({'held':str(v),'selections':selections})
                if name=='qwen_hgb': joblib.dump(model,OUT/'models'/('outer_'+str(v)+'.joblib'))
                emit(name+' held '+str(v)+': '+selections['retention90']['status'])
            res={c:measure(a,probe) for c,a in routes.items()}; res['folds']=folds
            final=choose(scores,np.arange(n),probe,'retention90'); res['final_calibration']=final
            summary['variants'][name]=res
            predictions[name+'_scores']=scores; predictions[name+'_route']=routes['retention90']; predictions[name+'_original_route']=routes['original']
            if name=='qwen_hgb':
                model=fit(X,y,np.arange(n),'hgb'); path=OUT/'models/final_estimator.joblib'; joblib.dump(model,path)
                np.testing.assert_array_equal(model.predict_proba(X),joblib.load(path).predict_proba(X))
                util.write(OUT/'models/final_research_model.json',{'schema':'pgp_ambiguity_hgb_v1','tracker_enabled':False,'deployable':False,
                    'feature_names':qnames,'threshold':final['threshold'],'selection_status':final['status'],'estimator_artifact':str(path.relative_to(ROOT)),
                    'estimator_sha256':util.sha(path),'repair_policy':'freeze_grasp_retract_verbs_and_ivts_v1','decision_rule':'whole_frame_change_threshold_v1'})
        for name,probe,route in [('cheap',False,False),('ambiguity_full',False,True),('same_probe_always',True,True)]:
            summary['baselines'][name]=measure(np.full(n,route),probe)
        for name,path in [('current_pgp',old/'semantic_benefit_original_harm/report.json'),('net_gain',ROOT/'artifacts/training/gate/pgp_gate_net_gain_20260913_r2/report.json')]:
            r=json.loads(path.read_text('utf-8')); summary['baselines'][name]={k:r[k] for k in ('original','merged',*ref.COST_FIELDS)}
        summary['baselines']['original_full']=json.loads((old/'baselines.json').read_text('utf-8'))['full']
        primary=summary['variants']['qwen_hgb']; actual=primary['retention90']; cheap=quality(cc); full=quality(cf); amb=quality(ca)
        q=actual['original']; gain=(q['five_head_mean_f1']-cheap['five_head_mean_f1'])/(amb['five_head_mean_f1']-cheap['five_head_mean_f1'])
        always=summary['baselines']['same_probe_always']; oldq=summary['baselines']['current_pgp']['original']
        checks={'all_inner_feasible':all(f['selections']['retention90']['status']=='FEASIBLE' for f in primary['folds']),
            'pooled_f1':q['five_head_mean_f1']+1e-12>=max(cheap['five_head_mean_f1'],full['five_head_mean_f1']),
            'pooled_errors':q['total_errors']<=min(cheap['total_errors'],full['total_errors']),
            'every_video':all(vq['five_head_mean_f1']+1e-12>=quality(cc[vids==v])['five_head_mean_f1'] and vq['total_errors']<=quality(cc[vids==v])['total_errors'] for v,vq in actual['by_video'].items()),
            'lower_calls':actual['logical_calls']<always['logical_calls'],'lower_known_usd':actual['known_usd']<always['known_usd'],
            'better_than_current_quality':q['five_head_mean_f1']>=oldq['five_head_mean_f1'] and q['total_errors']<=oldq['total_errors'],
            'final_calibration_feasible':primary['final_calibration']['status']=='FEASIBLE'}
        summary['selection']={'checks':checks,'research_default_eligible':all(checks.values()),'outer_gain_retention':gain,'retention90_pass':gain>=.9,'deployable':False}
        emit('Random control with identical per-video logical-call caps and Qwen prefix')
        rng=np.random.default_rng(3407); draw=[]; route=predictions['qwen_hgb_route']
        for _ in range(300):
            randomroute=np.zeros(n,bool)
            for v in sorted(set(vids)):
                ii=np.where(vids==v)[0]; extra=costs[ii,1,1,0]-costs[ii,1,0,0]; cap=extra[route[ii]].sum()
                order=rng.permutation(len(ii)); pick=order[np.cumsum(extra[order])<=cap]; randomroute[ii[pick]]=True
            m=measure(randomroute,True); draw.append([m['original']['five_head_mean_f1'],m['original']['total_errors'],m['logical_calls']])
        summary['random_control']={'draws':300,'mean':np.mean(draw,axis=0).tolist(),'scope':'same per-video call cap; prefix permutation may underspend; not a confidence interval'}
        util.save_arrays(OUT/'predictions.npz',ids=ids,videos=vids,X=XQ,targets=y,cheap=cc,ambiguity=ca,cheap_merged=cm,ambiguity_merged=am,costs=costs,depths=depths,random_draws=np.array(draw),**predictions)
        util.write(OUT/'report.json',summary)
        emit('Verifying preserved sources after fitting')
        for p,h in bindings.items():
            if util.sha(p)!=h: raise ValueError('Frozen file changed '+p)
        emit('COMPLETE '+json.dumps(summary['selection']))
        util.write(OUT/'receipt.json',{'state':'COMPLETED','frozen_files_verified':len(bindings),'tracker_files_unchanged':len(recipe['tracker_frozen_sha256']),
             'api_calls':0,'Testing_access':False,'deployable':False,'output_sha256':{str(p.relative_to(OUT)):util.sha(p) for p in OUT.rglob('*') if p.is_file()}})


if __name__=='__main__': run()
