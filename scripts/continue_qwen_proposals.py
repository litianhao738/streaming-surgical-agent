"""Explicitly continue a drained Qwen run with Qwen proposals; preserve mixed provenance."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import run_testing_half_complete as complete
from scripts import run_testing_half_pipeline as core
from scripts.testing_half_resume import prepare_resume

CHANGED={'scripts/run_testing_half_pipeline.py','scripts/testing_qwen_h0.py','scripts/testing_half_resume.py'}


def continue_run(source,output,aliyun_cap=None):
    if output.exists():raise ValueError('new continuation directory required')
    parent=core.read(source/'plan.json')
    old=Path(parent['core_output'])
    plan=complete.verify_core(old)
    if plan.get('base_model')!='qwen3.8-max' or plan.get('proposal_model')!='google/gemini-3.8-flash':
        raise ValueError('expected the original Qwen H0 / Gemini proposer experiment')
    if not (old/'failure.json').exists() or not (source/'failure.json').exists() or (source/'reports').exists():
        raise ValueError('source must be stopped in core stage before report generation')
    changes={}
    for field,document in [('runtime_sha256',plan),('source_sha256',parent)]:
        for name,before in document[field].items():
            after=core.sha(ROOT/name)
            if after!=before:
                if Path(name).as_posix() not in CHANGED:
                    raise ValueError('unrelated source change: '+name)
                changes[name]={'before':before,'after':after}
    with sqlite3.connect(old/'budget.sqlite') as db:
        if db.execute("SELECT count(*) FROM calls WHERE state!='TERMINAL'").fetchone()[0]:
            raise ValueError('requests still pending; wait for drain')
        # The pause changed reservation caps only, to prevent any new dispatches.
        # Restore the original contract after the process has drained.
        for account,cap in plan['limits'].items():
            db.execute('UPDATE accounts SET cap=? WHERE name=?',(cap,account))
        db.commit()
    legacy=set()
    for path in (old/'targets').rglob('record.json'):
        record=core.read(path)
        if record['status']=='DISPATCHED' or not record.get('finished_utc'):
            raise ValueError('unfinished target journal')
        if record['stage']=='proposal' and record['seat']=='base':
            legacy.add(record['target'])
    output.mkdir(parents=True)
    archive=output/'source_archive'
    archive.mkdir()
    for label,folder in [('complete',source),('core',old)]:
        dest=archive/label;dest.mkdir()
        for name in ('plan.json','prepared.json','failure.json','execution.lock','execution_started.json'):
            if (folder/name).exists():shutil.copy2(folder/name,dest/name)
    new=output/'core';new.mkdir()
    for child in old.iterdir():
        if child.name.startswith('budget.sqlite'):continue
        target=new/child.name
        if child.is_dir():shutil.copytree(child,target)
        else:shutil.copy2(child,target)
    with sqlite3.connect(old/'budget.sqlite') as src, sqlite3.connect(new/'budget.sqlite') as dst:
        src.backup(dst)
    updated=deepcopy(plan)
    updated.update(proposal_model='qwen3.8-max',legacy_proposal_targets=sorted(legacy),
        experiment_classification='Qwen H0; mixed Gemini/Qwen proposals; historical models not relabeled',
        continuation={'source':str(source),'source_core_plan_sha256':core.sha(old/'plan.json'),
                      'source_complete_plan_sha256':core.sha(source/'plan.json'),
                      'source_changes':changes,'legacy_proposal_frames':len(legacy)})
    if aliyun_cap is not None:
        from decimal import Decimal
        if not Decimal(aliyun_cap).is_finite() or Decimal(aliyun_cap)<Decimal(plan['limits']['aliyun_cny']):
            raise ValueError('continuation cap cannot be below the original cap')
        updated['limits']['aliyun_cny']=aliyun_cap
    updated['runtime_sha256']={name:core.sha(ROOT/name) for name in plan['runtime_sha256']}
    updated['runtime_sha256'][str(Path(__file__).relative_to(ROOT))]=core.sha(Path(__file__))
    core.write(new/'plan.json',updated)
    core.write(new/'prepared.json',{'plan_sha256':core.sha(new/'plan.json'),'api_calls':0,'continuation':True})
    with sqlite3.connect(new/'budget.sqlite') as db:
        db.execute("UPDATE meta SET value=? WHERE key='plan'",(core.sha(new/'plan.json'),))
        for account,cap in updated['limits'].items():
            db.execute('UPDATE accounts SET cap=? WHERE name=?',(cap,account))
    for path in (new/'results').glob('*.json'):
        result=core.read(path)
        result['proposal_model']='google/gemini-3.8-flash'
        result['continued_from']=str(old/'results'/path.name)
        core.write(path,result)
    updated_parent=deepcopy(parent)
    updated_parent.update(core_output=str(new),core_plan_sha256=core.sha(new/'plan.json'),
                          continuation=updated['continuation'],
                          experiment_classification=updated['experiment_classification'])
    updated_parent['source_sha256']={name:core.sha(ROOT/name) for name in parent['source_sha256']}
    updated_parent['source_sha256'].update(updated['runtime_sha256'])
    core.write(output/'plan.json',updated_parent)
    core.write(output/'prepared.json',{'plan_sha256':core.sha(output/'plan.json'),'api_posts':0})
    shutil.copy2(source/'failure.json',output/'failure.json')
    for name in ('execution.lock','report_provider_metadata.json'):
        if (source/name).exists():shutil.copy2(source/name,output/name)
    audit=prepare_resume(output)
    audit.update(legacy_proposal_frames=len(legacy),new_proposal_model='qwen3.8-max',
                 source=str(source),output=str(output),aliyun_cap=updated['limits']['aliyun_cny'])
    core.write(output/'continuation_receipt.json',audit)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--aliyun-cap',help='Explicitly authorized new CNY ceiling; otherwise retain original cap')
    a=p.parse_args()
    from filelock import FileLock
    with FileLock(str(a.source.resolve())+'.process.lock',timeout=0):
        continue_run(a.source.resolve(),a.output.resolve(),a.aliyun_cap)
