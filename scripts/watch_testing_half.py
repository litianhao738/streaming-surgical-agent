"""Read-only tqdm monitor for an already running complete Testing pipeline.

Does not import/change the runner, dispatch requests, restart work, or write its ledger.
"""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time


def read(path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return default


def count_files(folder):
    files = list(folder.glob('*.json'))
    latest = max((p.stat().st_mtime for p in files), default=None)
    return len(files), latest


def pending_age(core, identity, now):
    target, stage, seat = identity
    if stage in ('report_generation', 'offline_evaluation'):
        path = core/'requests'/target/'status.json'
        if path.exists():
            return max(0, now-path.stat().st_mtime)
        return None
    folder = core/'targets'/target
    paths = [folder/'changed'/f'{stage}_{seat}'/'record.json']
    paths.extend((folder/'run'/'calls').glob(f'*_{target}_{stage}_{seat}/record.json'))
    for path in paths:
        record = read(path, {})
        started = record.get('started_utc')
        if started:
            try:
                return max(0, now-datetime.fromisoformat(started.replace('Z', '+00:00')).timestamp())
            except ValueError:
                pass
    return None


def ledger(path, core, now):
    if not path.exists():
        return {'dispatches': 0, 'pending': [], 'accounts': {}}
    try:
        # URI read-only: never create a DB, reserve budget, or settle pending calls.
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True, timeout=1)) as db:
            states = dict(db.execute('SELECT state,COUNT(*) FROM calls GROUP BY state').fetchall())
            pending = [json.loads(row[0]) for row in db.execute("SELECT key FROM calls WHERE state='RESERVED'")]
            accounts = {name: {'cap': cap, 'occupied_including_reservations': occupied}
                        for name, cap, occupied in db.execute('SELECT name,cap,occupied FROM accounts')}
        return {'dispatches': sum(states.values()), 'states': states,
                'pending': [{'target': key[0], 'stage': key[1], 'seat': key[2],
                             'seconds': pending_age(core, key, now)} for key in pending],
                'accounts': accounts}
    except sqlite3.Error as exc:
        return {'dispatches': None, 'pending': [], 'accounts': {}, 'read_error': type(exc).__name__}


def snapshot(out):
    now = time.time()
    plan = read(out/'plan.json', {})
    core = Path(plan.get('core_output', str(out/'core')))
    core_plan = read(core/'plan.json', {})
    pipeline_total = plan.get('pipeline_frames', core_plan.get('pipeline_frames', 8578))
    score_total = plan.get('scored_frames', core_plan.get('evaluation_target_frames', 7823))
    specs = [('H0', core/'h0', len(core_plan.get('selection', [])) or 9058),
             ('Gate+Tracker', core/'results', pipeline_total),
             ('Reports H0/FULL', out/'reports/generated', 2*pipeline_total),
             ('Report scoring', out/'reports/evaluated', 2*score_total)]
    progress = {}
    modified = []
    for name, folder, total in specs:
        count, latest = count_files(folder)
        progress[name] = {'done': count, 'total': total}
        if latest is not None:
            modified.append(latest)
    failure = read(out/'failure.json') or read(core/'failure.json') or read(out/'stopped.json')
    receipt = read(out/'receipt.json', {})
    if failure or receipt.get('state') == 'INCOMPLETE_GSR':
        state = 'STOPPED'
    elif receipt.get('state') == 'PASS':
        state = 'COMPLETE'
    elif (out/'reports/generation_seal.json').exists():
        state = 'RULE_JUDGE_SCORING'
    elif (core/'receipt.json').exists():
        state = 'REPORT_GENERATION'
    elif (out/'execution.lock').exists():
        state = 'CORE_RUNNING'
    else:
        state = 'STARTUP_CHECKS'
    budgets = {'core': ledger(core/'budget.sqlite', core, now),
               'reports': ledger(out/'reports/budget.sqlite', out/'reports', now)}
    return {'time': datetime.now(timezone.utc).isoformat(), 'state': state,
            'progress': progress, 'last_completed_seconds_ago': round(now-max(modified), 1) if modified else None,
            'failure': failure, 'budgets': budgets}


def display_status(value):
    pending = [p for b in value['budgets'].values() for p in b['pending']]
    seats = Counter(p['seat'] for p in pending)
    ages = [p['seconds'] for p in pending if p['seconds'] is not None]
    names = {'base': 'Gemini', 'qwen': 'Qwen', 'grok': 'GLM', 'deepseek': 'DeepSeek',
             'gpt': 'GPT', 'gemini': 'Gemini-review'}
    waiting = ', '.join(f'{names.get(k,k)}:{v}' for k,v in sorted(seats.items())) or 'none'
    age = f'; oldest={max(ages):.0f}s' if ages else ''
    last = value['last_completed_seconds_ago']
    quiet = f'; last completion {last:.0f}s ago' if last is not None else ''
    error = '; '+str(value['failure'].get('error_type', 'failure')) if value['failure'] else ''
    return f"{value['state']}{error} | pending={len(pending)} [{waiting}]{age}{quiet}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--once', action='store_true', help='Print one JSON snapshot, then exit')
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be at least one second')
    out = args.output.resolve()
    if not out.is_dir():
        parser.error('run output directory does not exist')
    value = snapshot(out)
    if args.once:
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return
    from tqdm import tqdm
    print('Read-only monitor. Ctrl+C closes this monitor; the pipeline keeps running.', flush=True)
    print('Report scoring includes masked rows; it is not the paid Judge request count.', flush=True)
    bars = {name: tqdm(total=p['total'], initial=p['done'], desc=name, position=i,
                       unit='frame' if i < 2 else 'report', dynamic_ncols=True, ascii=True,
                       bar_format='{desc}: {n_fmt}/{total_fmt} |{bar}| {percentage:5.1f}% [{rate_fmt}]')
            for i, (name, p) in enumerate(value['progress'].items())}
    status = tqdm(total=0, position=len(bars), bar_format='{desc}', dynamic_ncols=True)
    try:
        while True:
            for name, bar in bars.items():
                bar.update(value['progress'][name]['done']-bar.n)
                bar.refresh()
            status.set_description_str(display_status(value), refresh=True)
            if value['state'] in ('STOPPED', 'COMPLETE'):
                break
            time.sleep(args.interval)
            value = snapshot(out)
    except KeyboardInterrupt:
        pass
    finally:
        status.close()
        for bar in reversed(list(bars.values())):
            bar.close()
    print(display_status(value), flush=True)


if __name__ == '__main__':
    main()
