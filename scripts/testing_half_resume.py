"""Explicit continuation of a stopped core run, preserving plans, journals and costs."""
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CHANGES = {'scripts/run_testing_half_complete.py', 'scripts/run_testing_half_pipeline.py',
                   'scripts/testing_qwen_h0.py', 'scripts/testing_half_resume.py'}
ADDED_SOURCES = {'scripts/testing_half_transport.py', 'scripts/testing_half_progress.py',
                 'scripts/testing_half_resume.py', 'scripts/watch_testing_half.py', 'scripts/testing_half_file_io.py',
                 'scripts/pipeline_checkpoint.py'}


def reconcile_completed_reservations(core):
    """Settle only verified, successful responses already on disk; never POST."""
    from surgical_agent.research.gate.collection_budget import Budget
    from surgical_agent.artifacts.manifest import atomic_write_json
    with closing(sqlite3.connect((core/'budget.sqlite').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        pending = [json.loads(r[0]) for r in db.execute("SELECT key FROM calls WHERE state='RESERVED'")]
    verified = []
    for target, stage, seat in pending:
        paths = list((core/'targets'/target/'run/calls').glob(f'*_{target}_{stage}_{seat}/record.json'))
        changed = core/'targets'/target/'changed'/f'{stage}_{seat}'/'record.json'
        if changed.exists(): paths.append(changed)
        qwen_h0 = core/'targets'/target/seat/'record.json'
        if seat.startswith(('qwen_h0','qwen_proposal')) and qwen_h0.exists(): paths.append(qwen_h0)
        if len(paths) != 1:
            raise ValueError('unresolved paid request; no unique saved record')
        path = paths[0]
        row = read(path)
        response_path = path.with_name('response.json')
        if (row.get('status') != 'JSON_PARSED' or not row.get('finished_utc')
                or row.get('http_status') != 200 or not response_path.exists()):
            raise ValueError('unresolved paid request; no verified successful response')
        response = read(response_path)
        body = response.get('body', response)
        if body.get('model') != row['model'] or body.get('error'):
            raise ValueError('saved response identity mismatch')
        if row.get('response_sha256') and sha(response_path) != row['response_sha256']:
            raise ValueError('saved response hash mismatch')
        if row.get('charge_kind') == 'native' and Decimal(str(body['usage']['cost'])) != Decimal(row['charge']):
            raise ValueError('saved native cost mismatch')
        if path != changed and path != qwen_h0:
            ledger = read(core/'targets'/target/'run/budget.json')
            if row not in ledger['calls']:
                raise ValueError('saved target ledger differs from response record')
        verified.append(dict(identity=[target, stage, seat], charge=row['charge'],
                             record_sha256=sha(path), response_sha256=sha(response_path)))
    if not verified:
        return 0
    plan = read(core/'plan.json')
    budget = Budget(core/'budget.sqlite', plan['limits'], sha(core/'plan.json'))
    try:
        audit = core/'ledger_recovery'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        atomic_write_json(audit, {'verified': verified, 'api_calls': 0, 'before': budget.summary()})
        for item in verified:
            budget.settle(Budget.key(*item['identity']), Decimal(item['charge']))
        atomic_write_json(audit, {'verified': verified, 'api_calls': 0, 'state': 'RECONCILED', 'after': budget.summary()})
    finally:
        budget.close()
    return len(verified)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_runtime(out, plan, field):
    expected = dict(plan[field])
    policy_path = out/'resume_policy.json'
    if policy_path.exists():
        policy = read(policy_path)
        if policy['plan_sha256'] != sha(out/'plan.json'):
            raise ValueError('resume policy belongs to another plan')
        for name, change in policy['amended_sources'].items():
            if Path(name).as_posix() not in ALLOWED_CHANGES or expected.get(name) != change['before']:
                raise ValueError('unapproved resume source change')
            expected[name] = change['after']
        for name, digest in policy['added_sources'].items():
            if name not in ADDED_SOURCES or sha(ROOT/name) != digest:
                raise ValueError('resume helper changed')
    for name, digest in expected.items():
        if sha(ROOT/name) != digest:
            raise ValueError('runtime changed: '+name)


def prepare_resume(out, *, apply=True, full_check=False):
    if not full_check:
        import sys
        from scripts.pipeline_checkpoint import prepare_fast_resume
        from scripts.run_testing_half_complete import write, verify_core
        return prepare_fast_resume(out, sys.modules[__name__], verify_core, write, apply=apply)
    return prepare_resume_full(out, apply=apply)


def prepare_resume_full(out, *, apply=True):
    """Offline only. Requires every dispatched request terminal and report stage unstarted."""
    from scripts.run_testing_half_complete import write, verify_core
    plan = read(out/'plan.json')
    core = Path(plan['core_output'])
    if (out/'receipt.json').exists() or (core/'receipt.json').exists():
        raise ValueError('this resume command is for an interrupted core stage only')
    if not (out/'failure.json').exists() or not (core/'failure.json').exists():
        raise ValueError('a stopped, drained run is required')
    if (out/'reports').exists():
        raise ValueError('report stage already created; inspect before continuing')
    if plan.get('report_arms') != ['H0', 'FULL']:
        raise ValueError('paired report plan required')
    verify_core(core)
    if apply:
        reconcile_completed_reservations(core)
    with closing(sqlite3.connect((core/'budget.sqlite').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        if db.execute("SELECT COUNT(*) FROM calls WHERE state!='TERMINAL'").fetchone()[0]:
            raise ValueError('unresolved paid requests; will not resubmit')
        calls = {tuple(json.loads(key)) for key, in db.execute('SELECT key FROM calls')}
    records = {}
    evidence = {}
    for path in (core/'targets').rglob('record.json'):
        row = read(path)
        identity = row['target'], row['stage'], row['seat']
        if row['status'] == 'DISPATCHED' or not row.get('finished_utc'):
            raise ValueError('unfinished journal: '+str(path))
        if identity in records:
            raise ValueError('duplicate request evidence')
        records[identity] = row
        evidence[str(path.relative_to(core))] = sha(path)
        response = path.with_name('response.json')
        if row.get('response_sha256') and (not response.exists() or sha(response) != row['response_sha256']):
            raise ValueError('response hash mismatch')
        for p in (path.with_name('request.json'), response):
            if p.exists():
                evidence[str(p.relative_to(core))] = sha(p)
    if set(records) != calls:
        raise ValueError('shared budget and request journals differ')
    # Validate all per-target ledgers before permitting any new requests.
    for path in (core/'targets').glob('*/run/budget.json'):
        for row in read(path)['calls']:
            if records[(row['target'], row['stage'], row['seat'])] != row:
                raise ValueError('per-target ledger disagrees with request record')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = out/'attempts'/stamp
    policies = []
    for folder, field in ((out, 'source_sha256'), (core, 'runtime_sha256')):
        frozen = read(folder/'plan.json')
        amended = {}
        for name, before in frozen[field].items():
            after = sha(ROOT/name)
            if after != before:
                if Path(name).as_posix() not in ALLOWED_CHANGES:
                    raise ValueError('unexpected source drift: '+name)
                amended[name] = dict(before=before, after=after)
        policy = dict(plan_sha256=sha(folder/'plan.json'), amended_sources=amended,
                      added_sources={name: sha(ROOT/name) for name in sorted(ADDED_SOURCES)},
                      policy='deferred_frame_errors_qwen_terminal_response_retry_v1', automatic_retry=True,
                      deferred_retry_rounds=3,
                      archive=str(archive), paid_requests_preserved=len(calls))
        policies.append((folder, policy))
    for category in ('results', 'h0', 'tracker_runtime'):
        for path in (core/category).glob('*.json'):
            evidence[str(path.relative_to(core))] = sha(path)
    summary = dict(preserved_requests=len(calls), preserved_targets=len(list((core/'results').glob('*.json'))),
                   archive=str(archive), api_calls=0, applied=apply)
    if not apply:
        return summary
    archive.mkdir(parents=True, exist_ok=False)
    write(archive/'preserved_evidence.json', evidence)
    for folder, policy in policies:
        write(folder/'resume_policy.json', policy)
    for folder, names in ((out, ('failure.json', 'execution.lock')),
                          (core, ('failure.json', 'execution_started.json'))):
        for name in names:
            path = folder/name
            if path.exists():
                destination = archive/('core' if folder == core else 'complete')/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                path.unlink()
    return summary
