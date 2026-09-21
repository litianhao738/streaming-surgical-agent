"""Durable per-target checkpoints; historical result payloads stay on disk."""
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import threading


class Checkpoint:
    def __init__(self, core, selection, plan_hash):
        self.core = Path(core)
        self.selection = {s['key']: s for s in selection}
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.core/'checkpoint.sqlite', check_same_thread=False)
        try:
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
            self.db.execute('CREATE TABLE IF NOT EXISTS targets (key TEXT PRIMARY KEY, state TEXT NOT NULL)')
            binding = self.db.execute("SELECT value FROM meta WHERE key='plan'").fetchone()
            if binding and binding[0] != plan_hash:
                raise ValueError('checkpoint belongs to another plan')
            if not binding:
                # One-time migration of atomically saved legacy outputs: names only.
                results = {p.stem for p in (self.core/'results').glob('*.json')}
                h0 = {p.stem for p in (self.core/'h0').glob('*.json')}
                with self.db:
                    self.db.executemany('INSERT OR REPLACE INTO targets VALUES (?, ?)',
                        [(s['key'], 'DONE') for s in selection
                         if s['key'] in (results if s['stage'] == 'pipeline' else h0)])
                    self.db.execute("INSERT INTO meta VALUES ('plan', ?)", (plan_hash,))
            self.done = {k for k, in self.db.execute("SELECT key FROM targets WHERE state='DONE'")}
            if not self.done <= self.selection.keys():
                raise ValueError('checkpoint contains targets outside the plan')
        except BaseException:
            self.db.close()
            raise

    def recover_tail(self):
        """Only read outputs interrupted between atomic file write and DB commit."""
        recovered = 0
        for key, in self.db.execute("SELECT key FROM targets WHERE state='ACTIVE'").fetchall():
            s = self.selection[key]
            path = self.core/('results' if s['stage'] == 'pipeline' else 'h0')/(key+'.json')
            if path.exists():
                payload = json.loads(path.read_bytes())
                if not isinstance(payload, dict) or (s['stage'] == 'pipeline' and 'prediction' not in payload):
                    raise ValueError('invalid interrupted checkpoint output: '+key)
                self.finish(key)
                recovered += 1
        return recovered

    def start(self, key):
        with self.lock, self.db:
            self.db.execute('INSERT OR REPLACE INTO targets VALUES (?, ?)', (key, 'ACTIVE'))

    def finish(self, key):
        with self.lock:
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO targets VALUES (?, ?)', (key, 'DONE'))
            self.done.add(key)

    def close(self):
        self.db.close()


def prepare_fast_resume(out, module, verify_core, write, *, apply=True):
    """Caller holds the process lock. Retain budget checks without historical scans."""
    out = Path(out)
    plan = module.read(out/'plan.json')
    core = Path(plan['core_output'])
    if (out/'receipt.json').exists() or (core/'receipt.json').exists() or (out/'reports').exists():
        raise ValueError('checkpoint resume supports interrupted core stage only')
    if plan.get('report_arms') != ['H0', 'FULL']:
        raise ValueError('paired report plan required')
    if module.sha(out/'plan.json') != module.read(out/'prepared.json')['plan_sha256']:
        raise ValueError('complete plan changed')
    core_plan = verify_core(core)
    core_hash = module.sha(core/'plan.json')
    if core_hash != plan['core_plan_sha256']:
        raise ValueError('core plan changed')
    # The ledger remains authoritative for dispatched calls, including crash tails.
    if apply:
        module.reconcile_completed_reservations(core)
    with closing(sqlite3.connect((core/'budget.sqlite').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        if db.execute("SELECT COUNT(*) FROM calls WHERE state!='TERMINAL'").fetchone()[0]:
            raise ValueError('unresolved paid requests; will not resubmit')
        count = db.execute('SELECT COUNT(*) FROM calls').fetchone()[0]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = out/'attempts'/stamp
    policies = []
    for folder, field in ((out, 'source_sha256'), (core, 'runtime_sha256')):
        frozen = module.read(folder/'plan.json')
        amended = {}
        for name, before in frozen[field].items():
            after = module.sha(module.ROOT/name)
            if after != before:
                if Path(name).as_posix() not in module.ALLOWED_CHANGES:
                    raise ValueError('unexpected source drift: '+name)
                amended[name] = dict(before=before, after=after)
        policies.append((folder, dict(plan_sha256=module.sha(folder/'plan.json'),
            amended_sources=amended,
            added_sources={name: module.sha(module.ROOT/name) for name in sorted(module.ADDED_SOURCES)},
            policy='incremental_checkpoint_v1', automatic_retry=True, deferred_retry_rounds=3,
            archive=str(archive), paid_requests_preserved=count)))
    summary = dict(mode='CHECKPOINT', preserved_requests=count, api_calls=0, applied=apply)
    if not apply:
        # Dry runs must not create/migrate checkpoints or clear interruption markers.
        return summary
    cp = Checkpoint(core, core_plan['selection'], core_hash)
    try:
        summary['recovered_tail'] = cp.recover_tail()
        summary['preserved_targets'] = sum(s['stage'] == 'pipeline' and s['key'] in cp.done
                                           for s in core_plan['selection'])
    finally:
        cp.close()
    archive.mkdir(parents=True, exist_ok=False)
    write(archive/'checkpoint_resume.json', summary)
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
