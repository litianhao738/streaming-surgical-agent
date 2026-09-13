"""Publish verified result data and label superseded runs; never delete evidence."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'docs/experiments/scheme4_pipeline_20260914'
def read(p): return json.loads(p.read_bytes())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))


def main():
    research=ROOT/'artifacts/research/tracker_scheme4_20260914_r2'
    pilot=ROOT/'artifacts/research/tracker_schemes56_pilot_20260914_r5'
    pipeline=ROOT/'artifacts/preflight/tracker_scheme4_default_all_20260914_r1'
    reproduction=ROOT/'artifacts/research/tracker_scheme4_reproduction_20260914_r1'
    pairs=[(research/'report.json','research_report.json'),(pilot/'quality_report.json','quality_report.json'),
           (pilot/'baseline_comparison.json','baseline_comparison.json'),(pilot/'completion_audit.json','completion_audit.json'),
           (pipeline/'receipt.json','pipeline_replay_receipt.json'),(pipeline/'scores.json','pipeline_fullfit_scores.json')]
    for source,name in pairs:
        assert read(source)==read(DEST/name), ('published summary differs',name)
    assert sha(research/'predictions.jsonl')==read(research/'report.json')['predictions_sha256']
    assert sha(pipeline/'predictions.jsonl')==read(pipeline/'receipt.json')['predictions_sha256']
    rows=[json.loads(line) for line in (research/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    current=[json.loads(line) for line in (pipeline/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    ref={r['key']:r for r in rows}
    assert len(rows)==len(ref)==len(current)==6059
    assert len({r['key'] for r in current})==6059
    assert all(r['prediction']==ref[r['key']]['predictions']['v2_fullfit_training'] for r in current)
    assert sum(r['logical_calls'] for r in current)==23600
    paired=read(pilot/'predictions.json')
    assert len(paired)==len({r['key'] for r in paired})==16
    assert all(set(r['arms'])=={'control','prior','qwen'} for r in paired)
    def no_truth(obj):
        if isinstance(obj,dict):
            assert not set(obj)&{'gt','ground_truth','mask'}
            for v in obj.values(): no_truth(v)
        elif isinstance(obj,list):
            for v in obj: no_truth(v)
    for data in (rows,current,paired): no_truth(data)
    files=[(research/'predictions.jsonl','scheme4_research_predictions.jsonl','CURRENT_RESEARCH',6059),
           (research/'predictions.npz','scheme4_gate_arrays.npz','CURRENT_RESEARCH',6059),
           (pipeline/'predictions.jsonl','scheme4_pipeline_predictions.jsonl','CURRENT_PIPELINE_FULLFIT',6059),
           (pilot/'predictions.json','schemes56_predictions.json','COMPLETED_NOT_ADOPTED',16),
           (pilot/'collection_receipt.json','schemes56_collection_receipt.json','COMPLETED_NOT_ADOPTED',16),
           (reproduction/'receipt.json','probe_reproduction_receipt.json','HISTORICAL_REPRODUCTION',None)]
    published=[]
    for source,name,status,n in files:
        target=DEST/name
        if target.exists(): assert target.read_bytes()==source.read_bytes(), ('existing publication differs',name)
        else: target.write_bytes(source.read_bytes())
        published.append({'file':name,'status':status,'rows':n,'bytes':target.stat().st_size,'sha256':sha(target),
                          'source':source.relative_to(ROOT).as_posix()})
    states=[
      ('artifacts/research/tracker_scheme4_20260914_r1','SUPERSEDED','真实 Tracker 适配修正前的中间回放；使用 r2。'),
      ('artifacts/research/tracker_scheme4_20260914_r2','CURRENT_RESEARCH','论文引用外层结果；全量拟合单独报告。'),
      ('artifacts/research/tracker_scheme4_reproduction_20260914_r1','HISTORICAL_REPRODUCTION','原探针复现证据；不是当前完整 pipeline 数字。'),
      ('artifacts/research/tracker_schemes56_pilot_20260914_r1','FAILED_PREPARATION','离线准备失败，无付费采集。'),
      ('artifacts/research/tracker_schemes56_pilot_20260914_r2','FAILED_PREPARATION','离线准备失败，无付费采集。'),
      ('artifacts/research/tracker_schemes56_pilot_20260914_r3','SUPERSEDED_PREPARATION','旧预算预览；被 r5 替代，无付费采集。'),
      ('artifacts/research/tracker_schemes56_pilot_20260914_r4','SUPERSEDED_PREPARATION','中间预览；被 r5 替代，无付费采集。'),
      ('artifacts/research/tracker_schemes56_pilot_20260914_r5','COMPLETED_NOT_ADOPTED','208 次请求配对实验的最终证据；两方案未通过，保留费用与失败记录。'),
      ('artifacts/preflight/tracker_scheme4_default_20260914_r1','LEGACY_PREFLIGHT_NOT_SCHEME4','目录名易混淆：默认切换前实际运行旧 PGP，不能引用为方案 4。'),
      ('artifacts/preflight/tracker_scheme4_default_all_20260914_r1','CURRENT_PIPELINE_FULLFIT','默认完整链路全量回放，6059 帧；不是外层验证。'),
      ('artifacts/preflight/scheme4_prepared_zero_20260914','PREFLIGHT_ONLY','零预算准备验证，无真实执行；非质量实验。'),
      ('artifacts/training/tracker','LEGACY_NOT_CURRENT','旧 Tracker 数据保留用于历史引用；当前默认使用 tracker_clip_v2_oof5_20260906。部分旧文件本地已缺失，不视为本次删除。'),
      ('artifacts/training/tracker_oof5','LEGACY_NOT_CURRENT','旧 OOF Tracker 数据保留用于历史引用；当前默认使用 clip_v2 OOF。部分旧文件本地已缺失，不视为本次删除。'),
      ('docs/experiments/tracker_pipeline_status_20260914','HISTORICAL_RESEARCH','早期探针/备选方案证据；当前数据见同级 scheme4_pipeline_20260914。')]
    for path,status,note in states:
        folder=ROOT/path; folder.mkdir(parents=True,exist_ok=True)
        (folder/'DATA_STATUS.md').write_bytes((f'# 数据状态：{status}\n\n{note}\n\n2026-09-14 按用户要求标注；原结果文件未修改，未删除复现证据。完整索引见 docs/experiments/DATA_CATALOG_2026-09-14.md。\n').encode('utf-8'))
    disposable=['tmp/scheme4_release_index_20260914.zip','tmp/scheme4_release_index_20260914',
                'tmp/scheme4_code_snapshot_20260914','tmp/scheme4_caps_zero_20260914.json','tmp/scheme4_latency_estimate.py']
    (ROOT/'tmp/SCHEME4_DISPOSABLE_DATA.md').write_bytes(('# 可清理的临时数据\n\n以下是已完成验证的导出副本、零预算配置和临时估时脚本，不是当前模型或实验原始记录。标注为 DISPOSABLE；本次未执行删除。\n\n'+'\n'.join('- '+p for p in disposable)+'\n').encode('utf-8'))
    manifest={'schema_version':'scheme4_result_publication_v1','published_on':'2026-09-14',
        'current_pipeline':read(ROOT/'DEFAULT_PIPELINE_VERSION.json')['version'],
        'new_api_calls':0,'verified_existing_summaries':6,'published_files':published,
        'data_states':[{'path':p,'status':s,'note':n} for p,s,n in states],
        'disposable_local_paths':disposable,'deleted_paths':[],
        'not_published':['raw surgical images/video','GT annotation source files','provider request/response journals and local budget database'],
        'notes':['Original source journals retained locally for audit; latest predictions and aggregate evidence are published.',
                 'NPZ labels are fused-review versus fused-cheap difference targets, not frame GT annotations.',
                 'New H0/R/G/T ablation design has no experimental results yet.']}
    write(DEST/'latest_data_manifest.json',manifest)
    print(json.dumps({'verified_existing_summaries':6,'new_result_files':len(files),
                      'published_bytes':sum(x['bytes'] for x in published),'labeled_locations':len(states),'deleted':0}))


if __name__=='__main__': main()
