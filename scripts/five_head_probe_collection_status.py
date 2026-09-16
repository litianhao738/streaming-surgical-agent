"""Read-only progress, ETA and budget status of a five_head_probe Training collection; no API calls.

Safe to run while the collector is running: it only reads plan/receipt/results/progress/errors
and opens budget.sqlite read-only. Nothing is written.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time

UNCERTAIN = 'uncertain'      # a durable request record without a final outcome (hard interruption)
REPLAYABLE = 'replayable'    # every recorded request finished; --resume replays them for free


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def budget_status(path):
    if not path.exists():
        return None
    try:
        db = sqlite3.connect(f'file:{path.resolve().as_posix()}?mode=ro', uri=True)
        db.execute('SELECT 1 FROM accounts LIMIT 1')
    except sqlite3.OperationalError:
        db = sqlite3.connect(str(path))
    try:
        accounts = {r[0]: {'cap': r[1], 'occupied': r[2]} for r in db.execute('SELECT name,cap,occupied FROM accounts')}
        rows = db.execute('SELECT key,account,charge,state FROM calls').fetchall()
    finally:
        db.close()
    per_seat, states, pending = Counter(), Counter(), []
    for key, account, charge, state in rows:
        target, stage, seat = json.loads(key)
        per_seat[seat] += 1
        states[state] += 1
        if state == 'RESERVED':
            pending.append({'target': target, 'stage': stage, 'seat': seat, 'account': account, 'reserved': charge})
    return {'accounts': accounts, 'dispatches': len(rows), 'states': dict(states),
            'per_seat': dict(per_seat), 'pending': pending}


def record_unfinished(record):
    return record.get('status') == 'DISPATCHED' or 'finished_utc' not in record


def classify_target(folder):
    """Classify a journaled target that has no result yet: UNCERTAIN needs recovery, REPLAYABLE resumes free."""
    folder = Path(folder)
    reasons = []
    changed = folder / 'changed'
    if changed.exists():
        for seat_dir in sorted(p for p in changed.iterdir() if p.is_dir()):
            record = seat_dir / 'record.json'
            if not record.exists():
                reasons.append(f'{seat_dir.name}: request without record')
                continue
            row = read(record)
            if record_unfinished(row):
                reasons.append(f"{seat_dir.name}: {row.get('status')} without final outcome")
            elif row.get('status') == 'JSON_PARSED' and not (seat_dir / 'response.json').exists():
                reasons.append(f'{seat_dir.name}: parsed record without response')
    run = folder / 'run'
    ledger, calls_dir = run / 'budget.json', run / 'calls'
    evidence = sorted(p for p in calls_dir.iterdir() if p.is_dir()) if calls_dir.exists() else []
    if ledger.exists():
        rows = read(ledger).get('calls', [])
        for row in rows:
            if record_unfinished(row):
                reasons.append(f"run {row.get('stage')}_{row.get('seat')}: {row.get('status')} without final outcome")
        if len(rows) != len(evidence):
            reasons.append(f'run ledger has {len(rows)} rows but {len(evidence)} evidence folders')
    elif evidence:
        reasons.append(f'{len(evidence)} run evidence folders without a ledger')
    for seat_dir in evidence:
        record = seat_dir / 'record.json'
        if not record.exists():
            reasons.append(f'{seat_dir.name}: evidence without record')
        elif record_unfinished(read(record)):
            reasons.append(f'{seat_dir.name}: unfinished evidence record')
    return (UNCERTAIN if reasons else REPLAYABLE), reasons


def scan(run):
    run = Path(run)
    status = {'run': str(run), 'exists': run.exists()}
    if not run.exists():
        return status
    plan = read(run / 'plan.json') if (run / 'plan.json').exists() else None
    status['plan'] = None if plan is None else {
        'profile': plan.get('profile'), 'review_mode': plan.get('review_mode'), 'workers': plan.get('workers'),
        'targets': len(plan.get('selection', [])), 'limits': plan.get('limits'),
        'reused_pilot': (plan.get('reused_pilot') or {}).get('directory')}
    status['complete'] = (run / 'receipt.json').exists()
    if status['complete']:
        receipt = read(run / 'receipt.json')
        status['receipt'] = {k: receipt.get(k) for k in ('state', 'rows', 'reused_pilot_rows', 'joint_depths',
                                                          'missing_responses', 'collection_seconds', 'rows_sha256')}
    results = {p.stem: p for p in (run / 'results').glob('*.json')} if (run / 'results').exists() else {}
    fresh, reused, depths, missing = [], 0, Counter(), 0
    for key, path in results.items():
        try:
            row = read(path)
        except (OSError, ValueError):
            continue  # a result being written right now
        if row.get('reused_from'):
            reused += 1
        else:
            fresh.append(path.stat().st_mtime)
            depths[row.get('joint_depth')] += 1
            missing += sum(not t.get('response_present', True) for t in row.get('request_timings', []))
    total = status['plan']['targets'] if status['plan'] else None
    status['results'] = {'done': len(results), 'reused_pilot': reused, 'fresh': len(fresh),
                         'remaining': None if total is None else total - len(results),
                         'joint_depths': dict(sorted(depths.items(), key=lambda kv: str(kv[0]))),
                         'missing_responses': missing}
    now = time.time()
    if fresh:
        first, last = min(fresh), max(fresh)
        overall = (len(fresh) - 1) / max(last - first, 1.) * 3600 if len(fresh) > 1 else None
        recent = [m for m in fresh if m >= now - 1800]
        recent_rate = len(recent) / 1800 * 3600
        rate = recent_rate if len(recent) >= 8 else overall
        remaining = status['results']['remaining']
        status['throughput'] = {'frames_per_hour_recent_30min': round(recent_rate, 1),
                                'frames_per_hour_overall': None if overall is None else round(overall, 1),
                                'last_result_age_seconds': round(now - last),
                                'eta_hours': None if not rate or remaining is None else round(remaining / rate, 2)}
    errors = Counter()
    if (run / 'errors').exists():
        for path in (run / 'errors').glob('*.json'):
            try:
                e = read(path)
            except (OSError, ValueError):
                continue
            errors[f"{e.get('error_type')}: {str(e.get('error'))[:90]}"] += 1
    status['errors'] = dict(errors.most_common())
    if (run / 'failure.json').exists():
        f = read(run / 'failure.json')
        status['failure'] = {'error_type': f.get('error_type'), 'error': str(f.get('error'))[:300],
                             'written_utc': datetime.fromtimestamp((run / 'failure.json').stat().st_mtime, timezone.utc).isoformat()}
    status['budget'] = budget_status(run / 'budget.sqlite')
    uncertain, replayable, in_flight = [], [], []
    if (run / 'targets').exists():
        for folder in (run / 'targets').iterdir():
            if not folder.is_dir() or folder.name in results:
                continue
            kind, reasons = classify_target(folder)
            (uncertain if kind == UNCERTAIN else replayable).append({'target': folder.name, 'reasons': reasons})
    if (run / 'progress').exists():
        for path in (run / 'progress').glob('*.json'):
            if path.stem in results:
                continue
            try:
                p = read(path)
            except (OSError, ValueError):
                continue
            in_flight.append({'target': path.stem, 'seats': [c['seat'] for c in p.get('calls', [])],
                              'age_seconds': round(now - path.stat().st_mtime)})
    in_flight.sort(key=lambda x: x['age_seconds'])
    uncertain.sort(key=lambda x: x['target'])
    status['journaled_without_result'] = {'replayable_on_resume': len(replayable), 'uncertain_needs_recover': uncertain}
    status['in_flight_progress'] = in_flight[:16]
    status['recoveries'] = sorted(p.name for p in (run / 'recovery').glob('*.json')) if (run / 'recovery').exists() else []
    pending = bool(status['budget'] and status['budget']['pending'])
    status['needs_recover'] = pending or bool(uncertain)
    return status


def render(s):
    lines = [f"Run: {s['run']}"]
    if not s.get('exists'):
        lines.append('  (directory does not exist yet: nothing started)')
        return '\n'.join(lines)
    if s.get('plan'):
        p = s['plan']
        lines.append(f"  profile={p['profile']} mode={p['review_mode']} workers={p['workers']} targets={p['targets']}")
        lines.append(f"  limits={p['limits']}")
    r = s['results']
    lines.append(f"  results: {r['done']} done ({r['fresh']} fresh + {r['reused_pilot']} reused pilot), remaining {r['remaining']}")
    lines.append(f"  fresh joint depths {r['joint_depths']}, missing seat responses {r['missing_responses']}")
    if s.get('throughput'):
        t = s['throughput']
        lines.append(f"  throughput: {t['frames_per_hour_recent_30min']} frames/h (last 30 min), "
                     f"{t['frames_per_hour_overall']} frames/h overall, last result {t['last_result_age_seconds']} s ago, ETA {t['eta_hours']} h")
    if s.get('budget'):
        b = s['budget']
        lines.append('  budget: ' + ', '.join(f"{k} {v['occupied']}/{v['cap']}" for k, v in b['accounts'].items()))
        lines.append(f"  dispatches {b['dispatches']} states {b['states']} per seat {b['per_seat']}")
        if b['pending']:
            lines.append(f"  PENDING reservations (uncertain outcome): {len(b['pending'])}")
            for x in b['pending'][:16]:
                lines.append(f"    {x['target']} {x['stage']}/{x['seat']} {x['account']} reserved {x['reserved']}")
    j = s['journaled_without_result']
    lines.append(f"  journaled targets without result: {j['replayable_on_resume']} replayable, {len(j['uncertain_needs_recover'])} uncertain")
    for x in j['uncertain_needs_recover'][:16]:
        lines.append(f"    UNCERTAIN {x['target']}: " + '; '.join(x['reasons']))
    if s['in_flight_progress'] and not s['complete']:
        lines.append(f"  progress files without result (in flight or interrupted): {len(s['in_flight_progress'])}")
        for x in s['in_flight_progress'][:8]:
            lines.append(f"    {x['target']} seats={x['seats']} age={x['age_seconds']}s")
    if s['errors']:
        lines.append('  errors/ by type:')
        for k, v in s['errors'].items():
            lines.append(f'    {v:5d}  {k}')
    if s.get('failure'):
        f = s['failure']
        lines.append(f"  failure.json ({f['written_utc']}): {f['error_type']}: {f['error']}")
    if s['recoveries']:
        lines.append(f"  recoveries applied: {s['recoveries']}")
    if s['complete']:
        lines.append(f"  COMPLETE: receipt {s['receipt']}")
    elif s['needs_recover']:
        lines.append('  -> hard interruption detected: run recover_five_head_probe_interrupted.py before resuming')
    else:
        lines.append('  -> resumable: rerun the collector with --resume (finished journals replay for free)')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    status = scan(args.run)
    print(json.dumps(status, ensure_ascii=False, indent=2) if args.json else render(status), flush=True)


if __name__ == '__main__':
    main()
