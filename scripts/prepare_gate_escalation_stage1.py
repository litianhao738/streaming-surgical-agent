"""Identity-only balanced nonoverlapping first-stage selection; zero API calls."""
from copy import deepcopy
from pathlib import Path
import shutil
from scripts import collect_full_gate_training as source
from surgical_agent.research.gate import escalation_rule_training as escalation_training

ROOT=source.ROOT


def choose(samples, history, *, quota=125, min_anchor_gap=75):
    if quota<1 or min_anchor_gap<75:
        raise ValueError('positive quota and at least 75 raw-frame gap required')
    historical_images={f+d for f in history for d in (-50,-25,0)}
    maximal=[]
    for s in sorted(samples,key=lambda s:s['frame_id']):
        if s['reused_pilot']:continue
        if any(abs(s['frame_id']-h)<min_anchor_gap for h in history):continue
        if historical_images.intersection(s['causal_frame_ids']):continue
        if maximal and s['frame_id']-maximal[-1]['frame_id']<min_anchor_gap:continue
        maximal.append(s)
    n=min(quota,len(maximal))
    selected=[deepcopy(maximal[i*(len(maximal)-1)//(n-1)]) for i in range(n)] if n>1 else deepcopy(maximal[:n])
    for s in selected:
        s['near_history']=any(abs(f-h)<=175 for f in s['causal_frame_ids'] for h in historical_images)
    return selected, len(maximal)


def prepare(run, *, quota=125, min_anchor_gap=75):
    from scripts import collect_gate_escalation_v1 as collector
    if run.exists():raise ValueError('fresh stage directory required')
    origin=source.DEFAULT; original=source.verify(origin)
    acceptance=ROOT/'artifacts/research/joint_empty_pool_patch_20260912/acceptance.json'
    proof=source.old.read(acceptance)
    assert proof['all_arms_canonical_json_unchanged']==127 and proof['api_calls']==0
    assert source.digest(acceptance.parent/'replayed_results.json')==proof['results_sha256']
    history,hashes=source.pilot.legacy.historical_targets(ROOT/'artifacts/preflight')
    for s in original['selection']:
        if s['reused_pilot']:history.setdefault(s['video_id'],set()).add(s['frame_id'])
    groups,inventory={},{}
    for v in source.pilot.VIDEOS:
        candidates=[s for s in original['selection'] if s['video_id']==v]
        groups[v],capacity=choose(candidates,history.get(v,set()),quota=quota,min_anchor_gap=min_anchor_gap)
        inventory[v]={'requested_new':quota,'maximal_capacity':capacity,'selected_new':len(groups[v]),
            'near_history':sum(s['near_history'] for s in groups[v]),'no_quota_transfer':True}
    selected=[groups[v][i] for i in range(max(map(len,groups.values()))) for v in source.pilot.VIDEOS if i<len(groups[v])]
    reused=[s for s in original['selection'] if s['reused_pilot']]
    if not selected:raise ValueError('no eligible new windows')
    run.mkdir(parents=True)
    for name in ('priors','provider_rate_envelopes.json'):
        if (origin/name).is_dir():shutil.copytree(origin/name,run/name)
        else:shutil.copyfile(origin/name,run/name)
    plan=deepcopy(original)
    plan.update(profile=collector.PROFILE,created_utc=source.old.now(),selection=reused+selected,
        inventory=inventory,pipeline_version=collector.ready.BASELINE_VERSION,
        limits={'openrouter_usd':'20','aliyun_cny':'20','xai_usd':'0'},maximum_new_calls=13*len(selected),
        sampling={'quota_per_video':quota,'min_anchor_gap':min_anchor_gap,'history_plan_sha256':hashes,
                  'near_history_definition':'any new causal frame within 175 frames of any historical three-frame window image',
                  'rule':'earliest-finish maximal disjoint windows, uniformly thin in time; no labels/outcomes'},
        preparation_amendment={'source':str(origin),'source_plan_sha256':source.digest(origin/'plan.json'),
            'reason':'Balanced first stage and postcheap utility; empty-pool-only patch with complete replay acceptance.'},
        escalation_feature_version=escalation_training.FEATURE_VERSION,
        escalation_label_version=escalation_training.LABEL_VERSION,
        preregistration={'primary_target':'benefit','primary_features':'31 no-Tracker (11 masked) + 11 masked postcheap + proposal rule = 43 columns',
            'model':'StandardScaler + balanced LogisticRegression(liblinear,C=1,max_iter=2000,random_state=3407)',
            'threshold':0.5,'evaluation':'conditional leave-one-Training-video-out; not independent generalization',
            'data':'127 original known terminals plus all new known terminals; report all failures/unknowns',
            'pass_all':['pooled mean five-head F1 > equal-count random p97.5 (2000 draws, seed3407)',
                        'total FP+FN <= full review','estimated USD-equivalent <= 60% of full review'],
            'extra_safeguard':'also beat within-video equal-count random p97.5; report new-only cohort separately',
            'cost_definition':'frozen pilot stage-average costs; Aliyun converted at fixed 7.1 CNY/USD for this comparison; native charges separately',
            'diagnostics':['AUC/AP','per-video','safe_benefit','Tracker views','near_history','learning curves within training folds'],
            'on_pass':'prepare separate next stage; no automatic paid expansion',
            'on_fail':'stop expansion; no posthoc feature/threshold retuning to declare this stage passed'})
    plan['original_policy_id']=original['policy_id']
    import hashlib,json
    plan['policy_id']=hashlib.sha256(json.dumps({'original':original['policy_id'],'patch_version':collector.ready.BASELINE_VERSION,
        'patch_sha256':source.digest(ROOT/'src/surgical_agent/research/verification/joint_empty_pool_patch.py')},sort_keys=True).encode()).hexdigest()
    plan['compatibility']={'proof_path':str(acceptance),'proof_sha256':source.digest(acceptance),
        'original_observed_targets':127,'missing_original_target':'retained unknown; recovered prediction separately available',
        'scope':'Original 127 terminal labels only, proven identical for all arms. No general equivalence claim for changed prompts/models.'}
    plan['protocol'].update(sampling=plan['sampling']['rule'],phase='empty-four-head patch only; Phase admission unchanged',
        features='54 pre-review features retained; 43-column rule-masked postcheap views saved before any reviews',
        labels='Escalation utility is full-review minus cheap tier; original H0 utility also retained',
        validation='Training conditional LOVO; independent Testing reserved. VID110 is development calibration only.')
    extra=['scripts/collect_gate_escalation_v1.py','scripts/prepare_gate_escalation_stage1.py',
        'scripts/run_gate_escalation_mainline.py','src/surgical_agent/research/gate/escalation_training.py','src/surgical_agent/research/gate/escalation_rule_training.py',
        'src/surgical_agent/research/verification/joint_empty_pool_patch.py',
        'scripts/train_gate_escalation_v1.py','docs/GATE_ESCALATION_PROTOCOL_2026-09-12.md']
    for rel in extra:plan['source_sha256'][rel]=source.digest(ROOT/rel)
    for rel,h in plan['source_sha256'].items():
        if source.digest(ROOT/rel)!=h:raise ValueError('source drift: '+rel)
        dest=run/'frozen_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    for s in plan['selection']:
        p=origin/s['tracker_snapshot']
        if source.digest(p)!=s['tracker_snapshot_sha256']:raise ValueError('tracker drift')
        dest=run/s['tracker_snapshot'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    source.write(run/'plan.json',plan)
    source.write(run/'preparation.json',{'plan_sha256':source.digest(run/'plan.json'),'api_calls':0,
        'targets':len(plan['selection']),'new_targets':len(selected),'reused':len(reused),'inventory':inventory})
    print(json.dumps(source.old.read(run/'preparation.json')),flush=True)
