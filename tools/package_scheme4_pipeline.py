"""Package the already evaluated scheme4 model; no training or provider access."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    dest = ROOT/'artifacts/training/gate/tracker_scheme4_20260914'
    model = dest/'model.json'
    rows = read(ROOT/'artifacts/training/gate/official_behavior_net_v2_20260913/primary_rows.json')
    source = ROOT/'artifacts/training/gate/full_official_reviewers_20260912_v1'
    inventory = dest/'replay_inventory.json'
    write(inventory,{'scope':'Frozen 6059 Training inputs; no GT labels in this inventory',
        'keys':[r['sample_id'] for r in rows],
        'source_sha256':{r['sample_id']:sha(source/'targets'/r['sample_id']/'result.json') for r in rows}})
    gate = {'schema_version':'tracker_scheme4_gate_default_v1','selected_on':'2026-09-14',
        'selected_by':'explicit_user_request_integrate_scheme4', 'version':'pgp-tracker-scheme4-gate-v2-20260914',
        'decision_rule':'interaction_change_threshold_v1','model_artifact':model.relative_to(ROOT).as_posix(),
        'model_sha256':sha(model),'source_module':'src/surgical_agent/research/gate/tracker_pipeline_v2.py',
        'inference_entrypoint':'scripts/run_tracker_scheme4_pipeline.py','training_entrypoint':'scripts/assess_tracker_scheme4.py',
        'scope':'Training research full-fit model; nested outer result reported separately',
        'tracker_enabled':True,'tracker_used_in_gate_features':False,'deployable':False,'independent_validation':False,
        'previous_default_manifest':'configs/defaults/pgp-gate-before-scheme4-20260914.json'}
    write(ROOT/'DEFAULT_PGP_GATE_VERSION.json',gate)
    pipeline = {'schema_version':'default_repair_pipeline_selection_v2','selected_on':'2026-09-14',
        'selected_by':'explicit_user_request_integrate_scheme4','version':'tracker-scheme4-complete-pipeline-v1-20260914',
        'profile':'tracker_scheme4_pipeline_v1','default_entrypoint':'scripts/run_pipeline.py',
        'implementation_entrypoint':'scripts/run_tracker_scheme4_pipeline.py','scoring_entrypoint':'scripts/run_tracker_scheme4_pipeline.py',
        'scoring_requires_command':True,'accepts_dataset_root':False,'dataset_root':'D:/cholec_dataset',
        'gate_default_manifest':'DEFAULT_PGP_GATE_VERSION.json','gate_version':gate['version'],
        'previous_default_manifest':'configs/defaults/pgp-pipeline-before-scheme4-20260914.json',
        'scope':'Training-only full-fit inference; four frozen Training videos; no independent validation',
        'tracker_enabled':True,'tracker_mode':'frozen causal held-out detector/tracker predictions',
        'tracker_index':'artifacts/training/tracker_clip_v2_oof5_20260906/oof/index.json',
        'replay_inventory':inventory.relative_to(ROOT).as_posix(),'replay_inventory_sha256':sha(inventory),
        'calls_per_target_min':3,'calls_per_target_max':7,'phase_window_seconds':60,
        'paid_execution_requires_explicit_flag':True,'api_execution_validated':False,'Testing_enabled':False,
        'deployable':False,'quality_status':'TRAINING_RESEARCH_ONLY_PHASE_LATENCY_AND_IVT_DELETION_LIMITS',
        'supported_commands':['info','preflight','replay','prepare','execute','score']}
    write(ROOT/'DEFAULT_PIPELINE_VERSION.json',pipeline)
    docs = ROOT/'docs/experiments/scheme4_pipeline_20260914'; docs.mkdir(parents=True,exist_ok=True)
    write(docs/'research_report.json',read(ROOT/'artifacts/research/tracker_scheme4_20260914_r2/report.json'))
    for name in ('quality_report.json','completion_audit.json','baseline_comparison.json'):
        write(docs/name,read(ROOT/'artifacts/research/tracker_schemes56_pilot_20260914_r5'/name))


if __name__=='__main__': main()
