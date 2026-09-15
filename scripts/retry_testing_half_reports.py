"""Explicit one-pass Report/Judge recovery in a new directory; original evidence is preserved."""
import argparse
from contextlib import closing
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import run_testing_half_complete as workflow
from scripts.testing_half_resume import ALLOWED_CHANGES, ADDED_SOURCES
from surgical_agent.research.gate.collection_budget import Budget
from surgical_agent.artifacts.manifest import atomic_write_json

# These inference entrypoints/configuration are never executed during report recovery.
INFERENCE_ONLY = frozenset({
    'DEFAULT_PGP_GATE_VERSION.json', 'scripts/run_pgp_pipeline.py',
    'scripts/run_testing_half_pipeline.py', 'scripts/run_tracker_scheme4_pipeline.py',
    'src/surgical_agent/research/gate/tracker_pipeline_v2.py',
})


def validate_report_runtime(origin, original):
    expected = {Path(k).as_posix(): v for k, v in original['source_sha256'].items()}
    policy_path = origin / 'resume_policy.json'
    if policy_path.exists():
        policy = workflow.read(policy_path)
        if policy['plan_sha256'] != workflow.sha(origin / 'plan.json'):
            raise ValueError('Resume policy belongs to another plan')
        for name, change in policy['amended_sources'].items():
            name = Path(name).as_posix()
            if name not in ALLOWED_CHANGES or expected.get(name) != change['before']:
                raise ValueError('Unapproved resume source change')
            expected[name] = change['after']
        for name, digest in policy['added_sources'].items():
            if name not in ADDED_SOURCES:
                raise ValueError('Unknown resume helper')
            expected[name] = digest
    ignored = {}
    for name, digest in expected.items():
        actual = workflow.sha(ROOT / name)
        if name in INFERENCE_ONLY:
            ignored[name] = {'frozen': digest, 'current': actual, 'used_for_inference': False}
        elif actual != digest:
            raise ValueError('Report recovery runtime changed: ' + name)
    return ignored


def verify_sealed_core(core, original, origin_receipt):
    plan_hash = workflow.sha(core / 'plan.json')
    receipt = workflow.read(core / 'receipt.json')
    if (plan_hash != original['core_plan_sha256']
            or plan_hash != workflow.read(core / 'prepared.json')['plan_sha256']
            or receipt['state'] != 'PASS' or receipt['plan_sha256'] != plan_hash
            or workflow.sha(core / 'receipt.json') != origin_receipt['core_receipt_sha256']):
        raise ValueError('Core seal changed')
    for name in ('predictions', 'continuous_predictions'):
        if workflow.sha(core / (name + '.json')) != receipt[name + '_sha256']:
            raise ValueError('Core predictions changed: ' + name)
    if workflow.sha(core / 'scores_detail.json') != origin_receipt['scores_sha256']:
        raise ValueError('Core scoring export changed')


def sequential_map(function, items, workers, stop):
    if workers != 1:
        raise ValueError('Report recovery requires one worker')
    for item in items:
        if stop.is_set():
            raise RuntimeError('Report recovery stopped')
        function(item)


def write_artifact(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def reusable_rows(source):
    bad_judges = set()
    for arm in ('h0', 'full'):
        with (source / 'reports/results' / f'{arm}_reports.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('judge_status') == 'invalid_response':
                    bad_judges.add(row['judge_cache_key'])
    with closing(sqlite3.connect((source / 'reports/cache.sqlite').as_uri() + '?mode=ro', uri=True)) as db:
        rows = db.execute('SELECT * FROM calls').fetchall()
    if any(row[3] not in ('COMPLETE', 'TERMINAL_HTTP_FAILURE') for row in rows):
        raise ValueError('Unresolved dispatch exists; inspect it before recovery')
    keep, retry = [], []
    for row in rows:
        response = json.loads(row[4]) if row[4] else {}
        failed = (row[3] != 'COMPLETE' or response.get('transport_error')
                  or row[0] in bad_judges)
        if row[1] == 'report_generation':
            from surgical_agent.research.reporting.event_report_generator import parse_report
            failed = failed or not parse_report(response.get('text', ''))[0]
        (retry if failed else keep).append(row)
    return keep, retry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-usd', default='20', help='Additional budget for this recovery only')
    parser.add_argument('--interval-seconds', type=float, default=3)
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError('Use a new output directory; existing evidence cannot be overwritten')
    cap = Decimal(args.budget_usd)
    if not cap.is_finite() or cap <= 0 or not 0 <= args.interval_seconds < float('inf'):
        raise ValueError('Invalid budget or interval')
    from filelock import FileLock
    with FileLock(str(source) + '.process.lock', timeout=0):
        plan = workflow.read(source / 'plan.json')
        origin = Path(plan.get('recovery_original', str(source)))
        original = workflow.read(origin / 'plan.json')
        if workflow.sha(origin / 'plan.json') != workflow.read(origin / 'prepared.json')['plan_sha256']:
            raise ValueError('Original plan changed')
        ignored_runtime = validate_report_runtime(origin, original)
        core = Path(original['core_output'])
        source_receipt = workflow.read(origin / 'receipt.json')
        verify_sealed_core(core, original, source_receipt)
        keep, retry = reusable_rows(source)
        print(json.dumps({'reused_requests': len(keep), 'retry_requests': len(retry),
                          'additional_judge_requests': 'determined after missing reports are generated',
                          'additional_budget_usd': str(cap), 'api_posts': 0}), flush=True)
        if args.dry_run:
            return
        if not args.allow_paid:
            raise ValueError('--allow-paid is required')
        plan = {**original, 'recovery_original': str(origin), 'recovery_source': str(source),
                'recovery_source_cache_sha256': workflow.sha(source / 'reports/cache.sqlite'),
                'report_limits': {'openrouter_usd': str(cap)}, 'explicit_retry': True,
                'inference_only_runtime_audit': ignored_runtime}
        # Report generation/evaluation only need a scheduler and artifact writer.
        # Do not inherit behavior or SCOPE from the current inference runner.
        workflow.core = SimpleNamespace(bounded_map=sequential_map)
        workflow.write = write_artifact
        out.mkdir(parents=True)
        workflow.write(out / 'plan.json', plan)
        workflow.write(out / 'retry_manifest.json', {'keys': [r[0] for r in retry],
                       'reused_keys': [r[0] for r in keep], 'original_costs_retained_at': str(source)})
        reports = out / 'reports'
        reports.mkdir()
        stop = threading.Event()
        budget = Budget(reports / 'budget.sqlite', plan['report_limits'], workflow.sha(out / 'plan.json'))
        caller = workflow.ReportCaller(reports, plan, budget, stop)
        last_dispatch = [0.0]
        def paced(request):
            time.sleep(max(0, args.interval_seconds - (time.monotonic() - last_dispatch[0])))
            last_dispatch[0] = time.monotonic()
            print('Request: ' + ('Report' if 'structured_prediction_hash' in request else 'Judge'), flush=True)
            return caller(request)
        cache = workflow.ConcurrentCache(reports / 'cache.sqlite', paced, plan, stop)
        try:
            cache.db.executemany('INSERT INTO calls VALUES (?,?,?,?,?,?,?)', keep)
            cache.db.commit()
            predictions = workflow.read(core / 'continuous_predictions.json')
            workflow.generate_reports(predictions, reports, plan['report_config'], cache, 1, stop)
            scores = workflow.read(core / 'scores_detail.json')
            summary = workflow.evaluate_reports(scores, reports, plan['report_config'], cache, 1, stop)
            workflow.write(out / 'receipt.json', {'state': summary['status'],
                           'report_budget': budget.summary(), 'source': str(source),
                           'paired_complete_frames': summary['paired_complete_frames'],
                           'eligible_frames': summary['eligible_frames'],
                           'cost_note': 'Recovery budget records new calls only; cache costs include reused calls. Prior attempts remain in source directories.'})
            print(json.dumps({'status': summary['status'], 'methods': summary['methods'],
                              'summary': str(reports / 'results/summary.json')}, ensure_ascii=False), flush=True)
            if summary['status'] != 'COMPLETE':
                raise RuntimeError('Still incomplete. Inspect results; a further explicit pass may use this output as --source and a new --output.')
        except BaseException as exc:
            workflow.write(out / 'failure.json', {'error_type': type(exc).__name__,
                           'report_budget': budget.summary(), 'automatic_retry': False})
            raise
        finally:
            cache.close()
            budget.close()


if __name__ == '__main__':
    main()
