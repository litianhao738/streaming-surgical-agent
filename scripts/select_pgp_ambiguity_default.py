"""Select the assessed offline research default; never modifies live pipeline."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/training/gate/pgp_ambiguity_assessment_20260913_r2'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    output=RUN/'default_selection_receipt.json'
    if output.exists(): raise ValueError('selection already recorded')
    report=json.loads((RUN/'report.json').read_text('utf-8'))
    audit=ROOT/'artifacts/preflight/pgp_ambiguity_assessment_audit_20260913.json'
    assert report['selection']['research_default_eligible'] and json.loads(audit.read_text('utf-8'))['state']=='PASS'
    receipt=json.loads((RUN/'receipt.json').read_text('utf-8'))
    for p,h in receipt['output_sha256'].items(): assert sha(RUN/p)==h
    manifest=ROOT/'DEFAULT_PGP_GATE_VERSION.json'
    assert manifest.read_bytes()==(RUN/'previous_default.json').read_bytes()
    recipe=json.loads((RUN/'recipe.json').read_text('utf-8'))
    live=ROOT/'DEFAULT_PIPELINE_VERSION.json'; assert sha(live)==recipe['source_sha256'][str(live)]
    model=RUN/'models/final_research_model.json'
    m=json.loads(model.read_text('utf-8'))
    assert sha(ROOT/m['estimator_artifact'])==m['estimator_sha256'] and m['selection_status']=='FEASIBLE'
    data={'schema_version':'pgp_research_default_v1','selected_on':'2026-09-13','selected_by':'user_authorized_assessment',
      'version':'pgp-ambiguity-single-qwen-hgb-no-tracker-v1-20260913','decision_rule':'whole_frame_change_threshold_v1',
      'scope':'offline_research_default','model_artifact':str(model.relative_to(ROOT)), 'model_sha256':sha(model),
      'training_receipt':str((RUN/'receipt.json').relative_to(ROOT)),'training_receipt_sha256':sha(RUN/'receipt.json'),
      'inference_entrypoint':'scripts/predict_pgp_gate_offline.py','training_entrypoint':'scripts/assess_pgp_ambiguity.py',
      'source_module':'src/surgical_agent/research/gate/pgp_ambiguity.py','evaluation_state':'CONDITIONAL_RESEARCH_SELECTION_RETENTION90_NOT_PASSED',
      'original_quality_cost_pass':True,'retention90_pass':False,'outer_gain_retention':report['selection']['outer_gain_retention'],
      'tracker_enabled':False,'deployable':False,'independent_validation':False,'automatic_experiment_promotion':False,
      'repair_policy':m['repair_policy'],'previous_default_snapshot':str((RUN/'previous_default.json').relative_to(ROOT)),
      'selection_basis':'Frozen reassessment protocol; full-reference pooled quality and per-video cheap stability, lower cost; not a complete stability guarantee'}
    temp=ROOT/'DEFAULT_PGP_GATE_VERSION.next.json'
    with temp.open('x',encoding='utf-8') as f: json.dump(data,f,indent=2,ensure_ascii=False)
    temp.replace(manifest)
    result={'state':'SELECTED_OFFLINE_RESEARCH_ONLY','new_default_sha256':sha(manifest),
            'previous_default_sha256':sha(RUN/'previous_default.json'),'live_pipeline_unchanged_sha256':sha(live),
            'assessment_receipt_sha256':sha(RUN/'receipt.json'),'audit_sha256':sha(audit),'deleted_files':0,
            'api_calls':0,'Testing_access':False,'deployable':False}
    with output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print(json.dumps(data,indent=2))


if __name__=='__main__': run()
