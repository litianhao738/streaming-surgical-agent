"""Publish v2.2 evidence only; exclude prompt experiments and raw provider/GT data."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'docs/experiments/scheme4_v22_20260914'
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def no_truth(value):
    if isinstance(value,dict):
        assert not set(value)&{'gt','ground_truth','mask','api_key','Authorization'}
        for v in value.values():no_truth(v)
    elif isinstance(value,list):
        for v in value:no_truth(v)

def main():
    full=ROOT/'artifacts/preflight/scheme4_output_v22_all_20260914'
    audit=ROOT/'artifacts/research/scheme4_ambiguity_recall_20260914_r2'
    live=ROOT/'artifacts/experiments/scheme4_two_targets_20260914_r1'
    for folder in (full,audit,live):
        receipt=read(folder/'receipt.json')
        assert receipt['state']=='PASS'
        assert sha(folder/'predictions.jsonl')==receipt['predictions_sha256']
    assert sha(audit/'report.json')==read(audit/'receipt.json')['report_sha256']
    preds=[json.loads(s) for s in (full/'predictions.jsonl').read_text('utf-8').splitlines()]
    assert len(preds)==len({r['key'] for r in preds})==6059
    assert sum(r['logical_calls'] for r in preds)==23600
    scores=read(full/'scores.json');baseline=read(audit/'report.json')['policies']['baseline']
    assert scores['errors']==baseline['errors']==30774
    assert abs(scores['f1']-baseline['mean_f1'])<1e-10
    for t,f in scores['by_head'].items():assert abs(f-baseline['f1'][t])<1e-10
    pairs=[(full/'predictions.jsonl','pipeline_predictions.jsonl','CURRENT_V22_TRAINING_FULLFIT'),
           (full/'scores.json','pipeline_scores.json','CURRENT_V22_TRAINING_FULLFIT'),
           (full/'receipt.json','pipeline_receipt.json','HISTORICAL_EXECUTION_RECEIPT'),
           (audit/'report.json','ambiguity_recall_report.json','RESEARCH_RULES_NOT_ADOPTED'),
           (audit/'receipt.json','ambiguity_recall_receipt.json','HISTORICAL_EXECUTION_RECEIPT'),
           (audit/'predictions.jsonl','ambiguity_predictions.jsonl','RESEARCH_RULES_NOT_ADOPTED'),
           (audit/'stages.json','ambiguity_stages.json','RESEARCH_DIAGNOSTIC'),
           (live/'predictions.jsonl','two_target_v21_predictions.jsonl','HISTORICAL_V21_LIVE'),
           (live/'receipt.json','two_target_v21_receipt.json','HISTORICAL_V21_LIVE'),
           (live/'scores.json','two_target_v21_scores.json','HISTORICAL_V21_LIVE')]
    DEST.mkdir(parents=True,exist_ok=True);entries=[]
    for source,name,status in pairs:
        data=source.read_bytes()
        parsed=[json.loads(s) for s in data.decode('utf-8').splitlines()] if source.suffix=='.jsonl' else json.loads(data)
        no_truth(parsed)
        target=DEST/name
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
        entries.append({'file':name,'status':status,'source':source.relative_to(ROOT).as_posix(),
                        'bytes':len(data),'sha256':sha(target)})
    manifest={'pipeline':'scheme4','output_modules':'v2.2','proposal_prompt':'original',
              'scope':'Training research; not independent validation','files':entries,
              'historical_note':'scheme4_pipeline_20260914 pipeline outputs are v2.1; retain them as history.',
              'raw_provider_logs_and_ground_truth_included':False}
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'files':len(entries),'bytes':sum(e['bytes'] for e in entries),'rows':6059,'hash_checks':'PASS'}))

if __name__=='__main__':main()
