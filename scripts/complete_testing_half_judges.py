"""Bounded Judge-only completion using verified frozen requests and cached successes."""
import argparse
from collections import defaultdict
from contextlib import closing
from decimal import Decimal
import json
import math
from pathlib import Path
import sqlite3
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from filelock import FileLock
from scripts import run_testing_half_complete as w
from scripts.retry_testing_half_reports import verify_sealed_core, write_artifact
from scripts.score_pipeline_reports import load_pairs
from surgical_agent.research.gate.collection_budget import Budget
from surgical_agent.research.reporting.contracts import digest
from surgical_agent.research.reporting.evaluator import summarize, write_artifacts
from surgical_agent.research.reporting.llm_judge_calibrated import parse_calibrated
from surgical_agent.research.reporting.rule_evaluator import evaluate_rule


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_summary(rows, baseline):
    actual = summarize(rows)
    for field in ('status', 'total_frames', 'eligible_frames', 'excluded_partial_labels', 'paired_complete_frames', 'mock'):
        require(actual[field] == baseline[field], 'Source summary mismatch: ' + field)
    for arm, metrics in actual['methods'].items():
        for name, value in metrics.items():
            expected = baseline['methods'][arm][name]
            # Python 3.12+ uses improved floating-point summation. Permit only
            # roundoff in aggregates; individual scores below still match exactly.
            match = value is None and expected is None
            if value is not None and expected is not None:
                match = math.isclose(value, expected, rel_tol=0, abs_tol=1e-10)
            require(match, 'Source summary mismatch: ' + arm + '/' + name)


def apply_response(rows, response):
    """Accept solely by the original parser; never select a response by its score."""
    if response.get('transport_error'):
        return False
    results = [parse_calibrated(response['text'], row['report']) for row in rows]
    if not results or any(result is None for result in results):
        return False
    require(all(result == results[0] for result in results), 'Shared Judge key has inconsistent reports')
    result = results[0]
    for row in rows:
        row.update(llm_judge_score=result['llm_judge_score'], judge_status='scored',
                   judge_raw_response=response['text'], judge_cache_hit=False,
                   gsr_score=.8 * row['rule_score'] + .2 * result['llm_judge_score'])
        row['diagnostics'].update(result['diagnostics'])
    return True


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-paid', action='store_true')
    p.add_argument('--budget-usd', default='2')
    p.add_argument('--max-rounds', type=int, default=12)
    p.add_argument('--interval-seconds', type=float, default=3)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    source, out = a.source.resolve(), a.output.resolve()
    cap = Decimal(a.budget_usd)
    require(cap.is_finite() and cap > 0 and 1 <= a.max_rounds <= 20, 'Invalid budget or round limit')
    require(0 <= a.interval_seconds < float('inf'), 'Invalid request interval')
    require(not out.exists(), 'Use a fresh output directory')
    with FileLock(str(source) + '.process.lock', timeout=0):
        plan = w.read(source / 'plan.json')
        origin = Path(plan.get('recovery_original', str(source)))
        original = w.read(origin / 'plan.json')
        require(w.sha(origin / 'plan.json') == w.read(origin / 'prepared.json')['plan_sha256'], 'Original plan changed')
        core = Path(original['core_output'])
        verify_sealed_core(core, original, w.read(origin / 'receipt.json'))
        # This driver uses only sealed data, reporting modules and exact cached requests.
        # The changed complete-workflow entrypoint is NOT used to run inference or scoring.
        checked = {}
        for name, expected in original['source_sha256'].items():
            name = Path(name).as_posix()
            if name.startswith('src/surgical_agent/research/reporting/'):
                require(w.sha(ROOT / name) == expected, 'Frozen scoring code changed: ' + name)
                checked[name] = expected
        rows = [json.loads(line) for arm in ('h0', 'full')
                for line in (source / 'reports/results' / (arm + '_reports.jsonl')).read_text(encoding='utf-8').splitlines()]
        baseline = w.read(source / 'reports/results/summary.json')
        verify_summary(rows, baseline)
        pairs = load_pairs(w.read(core / 'scores_detail.json'))
        inputs = {(s.video_id, s.frame_id, s.source_prediction): (s, t)
                  for h, f, t in pairs for s in (h, f)}
        require(len(rows) == len(inputs), 'Source frame count mismatch')
        with closing(sqlite3.connect((source / 'reports/cache.sqlite').as_uri() + '?mode=ro', uri=True)) as db:
            cached = {r[0]: r for r in db.execute('SELECT * FROM calls')}
        require(all(r[3] in ('COMPLETE', 'TERMINAL_HTTP_FAILURE') for r in cached.values()), 'Unresolved dispatch exists')
        groups = defaultdict(list)

        class Replay:
            mock = False
            def call(self, stage, request):
                key = digest({'stage': stage, 'request': request, 'mock': False})
                saved = cached[key]
                require(saved[1] == stage and json.loads(saved[2]) == request, 'Frozen request mismatch')
                return json.loads(saved[4]), key, True

        judge = w.build_judge(plan['report_config'], Replay())
        for row in rows:
            ident = row['video_id'], row['frame_id'], row['source_prediction']
            s, truth = inputs[ident]
            require(row['canonical_appendix'] == s.appendix(), 'Report/prediction mismatch')
            rule = evaluate_rule(row['raw_response'], truth)
            require(rule['rule_score'] == row['rule_score'], 'Rule score changed')
            result = judge.evaluate(row, truth)
            require(result['judge_status'] == row['judge_status'], 'Judge status changed')
            require(result['llm_judge_score'] == row['llm_judge_score'], 'Judge score changed')
            if result['judge_status'] == 'invalid_response':
                groups[result['judge_cache_key']].append(row)
            else:
                require(result['judge_status'] in ('scored', 'masked'), 'Judge-only completion requires valid reports')
        audit = {'source': str(source), 'source_cache_sha256': w.sha(source / 'reports/cache.sqlite'),
                 'scoring_sha256': checked, 'baseline_reproduced': True,
                 'exact_cached_requests_verified': True, 'failed_unique_requests': len(groups),
                 'budget_usd': str(cap), 'max_rounds': a.max_rounds, 'interval_seconds': a.interval_seconds,
                 'driver_sha256': w.sha(Path(__file__)),
                 'transport_module_sha256': w.sha(ROOT / 'scripts/run_testing_half_complete.py'),
                 'note': 'Independent frozen-request recovery. Original workflow entrypoint has drifted; reporting modules, requests, and baseline scores verified independently. No inference, prompt, parser, or score changes.'}
        print(json.dumps({'state': 'VERIFIED', 'requests': len(groups), 'paired_complete': baseline['paired_complete_frames'], 'api_posts': 0}), flush=True)
        if a.dry_run or not a.allow_paid:
            return
        out.mkdir(parents=True)
        write_artifact(out / 'audit.json', audit)
        write_artifact(out / 'plan.json', {**plan, 'recovery_source': str(source),
                                         'recovery_original': str(origin), 'completion_audit': audit})
        w.write = write_artifact
        occupied = Decimal(0)
        total_calls = 0
        for number in range(1, a.max_rounds + 1):
            if not groups:
                break
            attempt = out / 'attempts' / ('round_%02d' % number)
            attempt.mkdir(parents=True)
            remaining = cap - occupied
            budget = Budget(attempt / 'budget.sqlite', {'openrouter_usd': str(remaining)}, w.sha(out / 'plan.json'))
            stop = threading.Event()
            caller = w.ReportCaller(attempt, plan, budget, stop)
            cache = w.ConcurrentCache(attempt / 'cache.sqlite', caller, plan, stop)
            pending = list(groups)
            try:
                for index, key in enumerate(pending, 1):
                    request = json.loads(cached[key][2])
                    require(request['model_config'] == plan['report_config']['judge'], 'Frozen Judge model changed')
                    response, returned_key, hit = cache.call('offline_evaluation', request)
                    require(returned_key == key and not hit, 'Unexpected request identity or duplicate dispatch')
                    total_calls += 1
                    valid = apply_response(groups[key], response)
                    # Keep the latest terminal failure in the merged cache as well,
                    # while every previous attempt remains in its own directory.
                    cached[key] = cache.db.execute('SELECT * FROM calls WHERE key=?', (key,)).fetchone()
                    if valid:
                        groups.pop(key)
                    print(json.dumps({'round': number, 'request': index, 'round_requests': len(pending),
                                      'valid': valid, 'remaining': len(groups), 'total_new_calls': total_calls}), flush=True)
                    if index < len(pending):
                        time.sleep(a.interval_seconds)
            finally:
                usage = budget.summary()
                occupied += Decimal(usage['accounts']['openrouter_usd']['occupied'])
                summary = write_artifacts(attempt / 'results', rows, cache.costs(), baseline['metadata'])
                write_artifact(attempt / 'receipt.json', {'state': summary['status'], 'budget': usage,
                               'cumulative_budget_occupied_usd': str(occupied), 'remaining_unique': len(groups)})
                cache.close()
                budget.close()
            if not groups:
                break
        reports = out / 'reports'
        reports.mkdir()
        final_cache = w.ConcurrentCache(reports / 'cache.sqlite', None, plan, threading.Event())
        try:
            final_cache.db.executemany('INSERT INTO calls VALUES (?,?,?,?,?,?,?)', cached.values())
            final_cache.db.commit()
            summary = write_artifacts(reports / 'results', rows, final_cache.costs(),
                                      {**baseline['metadata'], 'recovery_audit': str(out / 'audit.json')})
        finally:
            final_cache.close()
        write_artifact(out / 'receipt.json', {'state': summary['status'], 'source': str(source),
                       'paired_complete_frames': summary['paired_complete_frames'],
                       'eligible_frames': summary['eligible_frames'], 'new_calls': total_calls,
                       'new_budget_occupied_usd': str(occupied),
                       'summary_sha256': w.sha(reports / 'results/summary.json'),
                       'cost_note': 'Merged cache costs are not total experimental spend. All new attempts and budget charges are retained under attempts; historical costs remain at source.'})
        print(json.dumps({'status': summary['status'], 'methods': summary['methods'], 'summary': str(reports / 'results/summary.json')}), flush=True)
        if groups:
            raise RuntimeError('Bounded recovery exhausted; see preserved attempts')


if __name__ == '__main__':
    main()
