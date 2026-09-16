"""Quarantine targets left uncertain by a hard interruption so a five_head_probe collection can resume; no API calls.

A soft stop (one worker raised; the others finished their in-flight request) leaves only finished
request records. `--resume` replays them for free and this tool changes nothing.

A hard kill (closed window, power loss, OOM) leaves DISPATCHED records whose outcome is unknown.
The runtime refuses to guess (AmbiguousDispatch), so for each such target this tool:
  1. moves the whole target journal (targets/<key>, its progress and error files) to quarantine/<stamp>/;
  2. deletes that target's rows from budget.sqlite while keeping every account's `occupied` total,
     so the abandoned requests stay counted as spent (conservative) and the identities can be
     dispatched again;
  3. writes recovery/<stamp>.json describing exactly what was abandoned.
The resumed collector re-runs those targets from scratch (their already-finished seats are paid
again; at most `workers` targets are affected). Dry run by default; --apply performs the changes.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]
from scripts.five_head_probe_collection_status import UNCERTAIN, classify_target, read

COLLECTOR = 'collect_five_head_probe_pilot'


def collector_processes():
    """Best-effort list of python processes running the collector; never raises."""
    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        found = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmd = ' '.join(proc.info.get('cmdline') or [])
            if COLLECTOR in cmd and 'python' in (proc.info.get('name') or '').lower():
                found.append({'pid': proc.info['pid'], 'cmdline': cmd[:200]})
        return found
    script = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*" + COLLECTOR +
              "*' -and $_.Name -like 'python*' } | ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }")
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', script],
                             capture_output=True, text=True, timeout=90).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return [{'pid': None, 'cmdline': f'process check unavailable: {type(exc).__name__}'}]
    return [{'pid': int(line.split('|', 1)[0]), 'cmdline': line.split('|', 1)[1][:200]}
            for line in out.splitlines() if line.strip() and line.split('|', 1)[0].isdigit()]


def plan_recovery(run):
    run = Path(run)
    if not (run / 'plan.json').exists():
        raise ValueError('not a collection run directory: ' + str(run))
    if (run / 'receipt.json').exists():
        return {'run': str(run), 'complete': True, 'targets': {}}
    results = {p.stem for p in (run / 'results').glob('*.json')} if (run / 'results').exists() else set()
    calls = defaultdict(list)
    if (run / 'budget.sqlite').exists():
        db = sqlite3.connect(str(run / 'budget.sqlite'))
        try:
            for key, account, charge, state in db.execute('SELECT key,account,charge,state FROM calls'):
                target, stage, seat = json.loads(key)
                calls[target].append({'key': key, 'stage': stage, 'seat': seat, 'account': account,
                                      'charge': charge, 'state': state})
        finally:
            db.close()
    pending_targets = {t for t, rows in calls.items() if any(r['state'] == 'RESERVED' for r in rows)}
    inconsistent = sorted(pending_targets & results)
    if inconsistent:
        raise RuntimeError('pending reservations for targets that already have results; inspect manually: '
                           + ', '.join(inconsistent))
    targets = {}
    if (run / 'targets').exists():
        for folder in sorted(p for p in (run / 'targets').iterdir() if p.is_dir()):
            if folder.name in results:
                continue
            kind, reasons = classify_target(folder)
            if kind == UNCERTAIN or folder.name in pending_targets:
                if folder.name in pending_targets:
                    reasons = reasons + ['budget reservation without settled outcome']
                targets[folder.name] = {'reasons': reasons, 'calls': calls.get(folder.name, [])}
    for target in sorted(pending_targets - set(targets)):
        targets[target] = {'reasons': ['budget reservation without any target journal'], 'calls': calls[target]}
    abandoned = defaultdict(Decimal)
    for info in targets.values():
        for row in info['calls']:
            abandoned[row['account']] += Decimal(row['charge'])
    return {'run': str(run), 'complete': False, 'targets': targets,
            'abandoned_charges_kept_as_occupied': {k: str(v) for k, v in sorted(abandoned.items())},
            'deleted_call_rows': sum(len(info['calls']) for info in targets.values())}


def _move(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return True
    return False


def apply_recovery(run, report):
    run = Path(run)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    quarantine = run / 'quarantine' / stamp
    quarantine.mkdir(parents=True, exist_ok=False)
    moved = {}
    for target in report['targets']:
        moved[target] = {
            'journal': _move(run / 'targets' / target, quarantine / 'targets' / target),
            'progress': _move(run / 'progress' / (target + '.json'), quarantine / 'progress' / (target + '.json')),
            'error': _move(run / 'errors' / (target + '.json'), quarantine / 'errors' / (target + '.json'))}
    # Error files from the previous attempt are archived so errors/ describes the next attempt only.
    archived_errors = 0
    if (run / 'errors').exists():
        for path in sorted((run / 'errors').glob('*.json')):
            archived_errors += _move(path, quarantine / 'errors_previous_attempt' / path.name)
    failure_moved = _move(run / 'failure.json', quarantine / 'failure.json')
    before = after = None
    if (run / 'budget.sqlite').exists():
        db = sqlite3.connect(str(run / 'budget.sqlite'))
        try:
            before = dict(db.execute('SELECT name,occupied FROM accounts').fetchall())
            with db:
                for info in report['targets'].values():
                    for row in info['calls']:
                        db.execute('DELETE FROM calls WHERE key=?', (row['key'],))
            after = dict(db.execute('SELECT name,occupied FROM accounts').fetchall())
            remaining_pending = db.execute("SELECT COUNT(*) FROM calls WHERE state='RESERVED'").fetchone()[0]
        finally:
            db.close()
        if before != after:
            raise RuntimeError('account totals changed during recovery; inspect budget.sqlite')
        if remaining_pending:
            raise RuntimeError(f'{remaining_pending} reservations still pending after recovery; inspect budget.sqlite')
    record = dict(report, stamp=stamp, quarantine=str(quarantine), moved=moved, archived_error_files=archived_errors,
                  failure_json_moved=failure_moved, occupied_before=before, occupied_after=after,
                  policy='Uncertain requests are abandoned, kept as spent, and re-dispatched fresh on resume')
    (run / 'recovery').mkdir(exist_ok=True)
    (run / 'recovery' / (stamp + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Perform the quarantine; the default is a dry run')
    parser.add_argument('--skip-process-check', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.exists():
        print(json.dumps({'state': 'NO_RUN_DIRECTORY', 'run': str(run)}), flush=True)
        return
    report = plan_recovery(run)
    if report['complete']:
        print('Collection complete; nothing to recover', flush=True)
        return
    if not report['targets']:
        print(json.dumps({'state': 'NOTHING_TO_RECOVER', 'run': str(run),
                          'note': 'No uncertain requests; resume the collector directly'}, ensure_ascii=False), flush=True)
        return
    print(json.dumps({'state': 'DRY_RUN' if not args.apply else 'APPLYING', **{k: v for k, v in report.items() if k != 'targets'},
                      'targets': {t: info['reasons'] for t, info in report['targets'].items()}}, ensure_ascii=False, indent=2), flush=True)
    if not args.apply:
        print('Dry run only. Re-run with --apply after confirming the collector is not running.', flush=True)
        return
    if not args.skip_process_check:
        running = collector_processes()
        if running:
            raise SystemExit('Refusing to recover while a collector process may be running: ' + json.dumps(running))
    record = apply_recovery(run, report)
    print(json.dumps({'state': 'RECOVERED', 'quarantine': record['quarantine'], 'targets': sorted(record['targets']),
                      'deleted_call_rows': record['deleted_call_rows'],
                      'abandoned_charges_kept_as_occupied': record['abandoned_charges_kept_as_occupied'],
                      'next': 'rerun the collector with --resume'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
