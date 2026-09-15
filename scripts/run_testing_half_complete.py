"""Half-Testing COMPLETE workflow: core -> Report -> structured scores -> Rule/Judge/GSR.

prepare does local work and public endpoint checks only. execute requires --allow-paid.
RAM remains off. No retries; completed core outputs can be attached with --core-output.
"""
import argparse
from concurrent.futures import Future
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import requests
from scripts import run_testing_half_pipeline as core
from scripts.score_pipeline_reports import load_pairs
from surgical_agent.api.credentials import load_api_key_file
from surgical_agent.research.gate.collection_budget import Budget, BudgetStop
from surgical_agent.research.reporting.contracts import ModelConfig, ReportInput, digest
from surgical_agent.research.reporting.event_report_generator import ReportGenerator
from surgical_agent.research.reporting.judge_selection import build_judge
from surgical_agent.research.reporting.rule_evaluator import evaluate_rule
from surgical_agent.research.reporting.evaluator import write_artifacts
from surgical_agent.research.reporting.transport import CachedCalls
from scripts.testing_half_transport import billing_failure

read, sha, write = core.read, core.sha, core.write
PROFILE = 'testing_half_complete_report_rule2_judge2low_v1'
REPORT_REFERENCE = ROOT/'configs/reporting/generation_wire.json'
REPORT_CONFIG = ROOT/'configs/reporting/gsr_v1.json'
ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'


def report_config():
    """Use the successfully exercised generator wire and the selected v2/low judge."""
    previous, selected = read(REPORT_REFERENCE), read(REPORT_CONFIG)
    if selected['rule_version'] != 'gsr_rules_2' or selected['judge_version'] != 'v2_low':
        raise ValueError('expected frozen Rule v2 and Judge v2/low')
    config = {'generator': previous['models']['report_generation'],
              'judge': selected['judge'], 'judge_version': selected['judge_version'],
              'rule_version': selected['rule_version']}
    if (config['generator']['model'] != 'google/gemini-3.8-flash'
            or config['generator']['endpoint'] != ENDPOINT):
        raise ValueError('unexpected report generator')
    return config, {
        'report_generation': {'wire_policy': previous['wire_policy']['report_generation'],
            'omit_temperature': 'report_generation' in previous['omit_temperature_for'],
            'provider_tag': previous['provider_tags']['report_generation']},
        'offline_evaluation': {'wire_policy': {'reasoning': {'effort': 'low'},
            'response_format': {'type': 'json_object'}, 'stream': False},
            'omit_temperature': True, 'provider_tag': 'openai'}}


def endpoint_rates(config, policies):
    rates, metadata = {}, {}
    for stage, name in [('report_generation', 'generator'), ('offline_evaluation', 'judge')]:
        model = config[name]['model']
        response = requests.get('https://openrouter.ai/api/v1/models/'+model+'/endpoints', timeout=30)
        response.raise_for_status()
        data = response.json()
        endpoints = [e for e in data['data']['endpoints']
                     if e['tag'] == policies[stage]['provider_tag'] and e['status'] == 0]
        if len(endpoints) != 1:
            raise ValueError('pinned report route unavailable: '+model)
        endpoint = endpoints[0]
        if not {'reasoning', 'response_format'} <= set(endpoint['supported_parameters']):
            raise ValueError('report route does not support frozen parameters')
        rates[stage] = {key: str(max(Decimal(endpoint['pricing'][key]),
            *(Decimal(o.get(key, '0')) for o in endpoint['pricing'].get('overrides', []))))
            if endpoint['pricing'].get('overrides') else str(Decimal(endpoint['pricing'][key]))
            for key in ('prompt', 'completion')}
        if any(not Decimal(v).is_finite() or Decimal(v) < 0 for v in rates[stage].values()):
            raise ValueError('invalid endpoint pricing')
        metadata[stage] = data
    return rates, metadata


class ConcurrentCache(CachedCalls):
    """Same cache namespace/schema as CachedCalls; coalesce concurrent identical requests."""
    def __init__(self, path, caller, plan, stop, *, mock=False):
        super().__init__(path, caller, max_calls=plan['max_report_dispatches'], mock=mock)
        self.db.close()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.plan, self.stop = plan, stop
        self.lock, self.inflight = threading.RLock(), {}

    def call(self, stage, request):
        if stage not in self.hits:
            raise ValueError('invalid report stage')
        policy = self.plan['policies'][stage]
        for field in ('wire_policy', 'omit_temperature', 'provider_tag'):
            if field in request and request[field] != policy[field]:
                raise ValueError('report wire policy drift')
        request = {**request, **policy}
        key = digest({'stage': stage, 'request': request, 'mock': self.mock})
        with self.lock:
            future = self.inflight.get(key)
            if future is None:
                row = self.db.execute('SELECT status,response FROM calls WHERE key=?', (key,)).fetchone()
                if row:
                    if row[0] not in ('COMPLETE', 'TERMINAL_HTTP_FAILURE'):
                        raise RuntimeError('previous report dispatch unresolved/failed; no automatic retry')
                    self.hits[stage] += 1
                    return json.loads(row[1]), key, True
                if self.stop.is_set() or self.dispatched >= self.max_calls:
                    raise BudgetStop('report stopped or request cap reached')
                self.db.execute("INSERT INTO calls VALUES (?,?,?,'PENDING',NULL,NULL,?)",
                                (key, stage, json.dumps(request), int(self.mock)))
                self.db.commit()
                self.dispatched += 1
                self.session_dispatches[stage] += 1
                future = self.inflight[key] = Future()
                owner = True
            else:
                owner = False
        if not owner:
            response = future.result()
            with self.lock:
                self.hits[stage] += 1
            return deepcopy(response), key, True
        started = time.perf_counter()
        try:
            response = self.caller(request)
            if not isinstance(response, dict) or not isinstance(response.get('text'), str):
                raise ValueError('invalid response envelope')
            encoded = json.dumps(response, allow_nan=False)
            with self.lock:
                terminal = 'TERMINAL_HTTP_FAILURE' if response.get('transport_error') else 'COMPLETE'
                self.db.execute("UPDATE calls SET status=?,response=?,seconds=? WHERE key=?",
                                (terminal, encoded, time.perf_counter()-started, key))
                self.db.commit()
                future.set_result(deepcopy(response))
            return response, key, False
        except BaseException as exc:
            self.stop.set()
            with self.lock:
                self.db.execute("UPDATE calls SET status='FAILED',seconds=? WHERE key=?",
                                (time.perf_counter()-started, key))
                self.db.commit()
                future.set_exception(exc)
            raise


class ReportCaller:
    def __init__(self, out, plan, budget, stop):
        self.out, self.plan, self.budget, self.stop = out, plan, budget, stop
        self.secret = load_api_key_file(ROOT/'docs/API.txt')

    def __call__(self, request):
        stage = 'report_generation' if 'structured_prediction_hash' in request else 'offline_evaluation'
        config = request['model_config']
        expected = self.plan['report_config']['generator' if stage == 'report_generation' else 'judge']
        if config != expected or self.stop.is_set():
            raise BudgetStop('report config drift or stopped before dispatch')
        key = digest({'stage': stage, 'request': request, 'mock': False})
        wire = {'model': config['model'], 'temperature': config['temperature'],
                'max_tokens': config['max_tokens'], 'messages': [{'role': 'user', 'content': request['prompt']}],
                **request['wire_policy'], 'provider': {'only': [request['provider_tag']],
                    'order': [request['provider_tag']], 'allow_fallbacks': False, 'require_parameters': True}}
        if request['omit_temperature']:
            wire.pop('temperature')
        rates = self.plan['rates'][stage]
        reserve = 2*(Decimal(len(request['prompt'].encode())+512)*Decimal(rates['prompt'])
                     + Decimal(config['max_tokens'])*Decimal(rates['completion']))
        identity = Budget.key(key, stage, config['model'])
        folder = self.out/'requests'/key
        self.budget.reserve(identity, 'openrouter_usd', reserve)
        write(folder/'request.json', wire)
        write(folder/'status.json', {'state': 'DISPATCHED', 'reserve_usd': str(reserve)})
        started = time.perf_counter()
        try:
            response = requests.post(ENDPOINT, json=wire,
                headers={'Authorization': 'Bearer '+self.secret.reveal()},
                timeout=(15, 180), allow_redirects=False)
            try:
                body = response.json()
            except ValueError:
                body = {'non_json': response.text[:2000]}
            body = json.loads(json.dumps(body).replace(self.secret.reveal(), '[REDACTED]'))
            write(folder/'response.json', body)
            usage = body.get('usage') or {}
            cost = usage.get('cost')
            charge = Decimal(str(cost)) if cost is not None else reserve
            self.budget.settle(identity, charge)
            write(folder/'status.json', {'state': 'RECEIVED', 'http_status': response.status_code,
                'seconds': time.perf_counter()-started, 'budget_charge_usd': str(charge), 'native_cost_usd': cost})
            if billing_failure(response.status_code, body.get('error')):
                raise BudgetStop('report provider balance/credits exhausted')
            if response.status_code >= 400 or body.get('error'):
                # Preserve an explicit terminal failure; the same request is not resent.
                return {'text': '', 'usage': usage, 'cost': {'USD': float(cost)} if cost is not None else None,
                        'transport_error': {'http_status': response.status_code, 'error': body.get('error')}}
            if response.status_code != 200 or body.get('model') != config['model']:
                raise RuntimeError('report provider failure/model mismatch; inspect saved response')
            choice = body['choices'][0]
            if choice.get('finish_reason') != 'stop' or choice['message'].get('refusal'):
                raise RuntimeError('report response incomplete/refused')
            return {'text': choice['message']['content'], 'usage': usage,
                    'cost': {'USD': float(cost)} if cost is not None else None,
                    'provider_request_id': body.get('id'), 'returned_model': body.get('model')}
        except requests.RequestException as exc:
            # Keep the uncertain charge and a terminal failure, not a fake report.
            self.budget.settle(identity, reserve)
            failure = {'error_type': type(exc).__name__, 'http_status': None,
                       'automatic_retry': False, 'continued': True}
            write(folder/'failure.json', failure)
            write(folder/'status.json', {'state': 'TERMINAL_NETWORK_FAILURE',
                'seconds': time.perf_counter()-started, 'budget_charge_usd': str(reserve),
                **failure})
            return {'text': '', 'usage': {}, 'cost': None, 'transport_error': failure}
        except BaseException as exc:
            self.stop.set()
            write(folder/'failure.json', {'error_type': type(exc).__name__, 'automatic_retry': False})
            raise


def report_path(out, source):
    return out/'generated'/f'{source.video_id}_{source.frame_id}_{source.source_prediction}.json'


def generate_reports(predictions, out, config, cache, workers, stop):
    generator = ReportGenerator(ModelConfig(**config['generator']), cache)
    sources = [ReportInput.from_prediction(r['video_id'], r['frame_id'], arm, r[head])
               for r in predictions for arm, head in [('H0', 'h0'), ('FULL', 'prediction')]]
    counter, lock = [0], threading.Lock()
    def task(source):
        report = generator.generate(source)
        write(report_path(out, source), report)
        with lock:
            counter[0] += 1
    core.bounded_map(task, sources, workers, stop)
    seal = {str(report_path(out, s).relative_to(out)): sha(report_path(out, s)) for s in sources}
    for arm in ('H0', 'FULL'):
        with (out/f'{arm.lower()}_reports.jsonl').open('x', encoding='utf-8') as stream:
            for s in sources:
                if s.source_prediction == arm:
                    stream.write(json.dumps(read(report_path(out, s)), ensure_ascii=False)+'\n')
    write(out/'generation_seal.json', {'files': seal, 'reports': len(sources), 'ground_truth_used': False})


def evaluate_reports(scores, out, config, cache, workers, stop):
    seal = read(out/'generation_seal.json')
    for path, expected in seal['files'].items():
        if sha(out/path) != expected:
            raise ValueError('generated report changed before evaluation')
    # First access to GT in this module occurs after the entire generation seal.
    pairs = load_pairs(scores)
    judge = build_judge(config, cache)
    def task(pair):
        h0, full, truth = pair
        for source in (h0, full):
            path = report_path(out, source)
            if str(path.relative_to(out)) not in seal['files']:
                raise ValueError('unsealed report')
            report = read(path)
            if report['canonical_appendix'] != source.appendix():
                raise ValueError('report/prediction mismatch')
            rule = evaluate_rule(report['raw_response'], truth)
            if not report['raw_response']:
                rule.update(rule_score=None, diagnostics={**rule['diagnostics'], 'status': 'missing_provider_response'})
            assessed = judge.evaluate(report, truth)
            value = assessed['llm_judge_score']
            row = {**report, **rule, **{k: v for k, v in assessed.items() if k != 'diagnostics'},
                   'diagnostics': {**rule['diagnostics'], **assessed.get('diagnostics', {})},
                   'gsr_score': .8*rule['rule_score']+.2*value
                       if rule['rule_score'] is not None and value is not None else None}
            write(out/'evaluated'/path.name, row)
    core.bounded_map(task, pairs, workers, stop)
    evaluated = [read(out/'evaluated'/report_path(out, s).name) for h, f, _ in pairs for s in (h, f)]
    return write_artifacts(out/'results', evaluated, cache.costs(),
        {'generator_seal_sha256': sha(out/'generation_seal.json'),
         'judge_version': config['judge_version'], 'rule_version': config['rule_version'],
         'scored_frames': len(pairs), 'generation_frames': seal['reports']//2})


def verify_core(core_out, *, completed=False):
    plan = read(core_out/'plan.json')
    if (sha(core_out/'plan.json') != read(core_out/'prepared.json')['plan_sha256']
            or plan['profile'] != core.PROFILE or plan['evaluation_target_frames'] != 7823
            or plan['pipeline_frames'] != 8578 or plan['ram'] is not False
            or plan['scope_sha256'] != sha(core.SCOPE/'run_scope.json')):
        raise ValueError('core is not the frozen half-Testing plan')
    if completed:
        receipt = read(core_out/'receipt.json')
        if receipt['state'] != 'PASS' or receipt['plan_sha256'] != sha(core_out/'plan.json'):
            raise ValueError('core inference not sealed')
        for name in ('predictions', 'continuous_predictions'):
            if sha(core_out/(name+'.json')) != receipt[name+'_sha256']:
                raise ValueError('core predictions changed')
        actual = read(core_out/'continuous_predictions.json')
        if {r['key'] for r in actual} != {s['key'] for s in plan['selection'] if s['stage'] == 'pipeline'}:
            raise ValueError('core prediction scope mismatch')
    return plan


def prepare(out, core_out, report_cap, core_limits, *, resume=False):
    core.check_prepare_destination(out, resume, {'core'})
    cap = Decimal(report_cap)
    if not cap.is_finite() or cap <= 0:
        raise ValueError('report budget must be positive USD')
    config, policies = report_config()
    rates, metadata = endpoint_rates(config, policies)
    if not load_api_key_file(ROOT/'docs/API.txt').reveal():
        raise ValueError('missing OpenRouter report credential')
    if core_out is not None:
        core_out = core_out.resolve()
        verify_core(core_out)
    else:
        core_out = out/'core'
        if resume and (core_out/'prepared.json').exists():
            existing = verify_core(core_out)
            for path, expected in existing['runtime_sha256'].items():
                if sha(ROOT/path) != expected:
                    raise ValueError('prepared core runtime changed; cannot silently reseal: '+path)
        else:
            core.prepare(core_out, core.SCOPE, core_limits.resolve(), resume=resume, streaming=True)
    core_plan = verify_core(core_out)
    bound = {Path(__file__), Path(core.__file__), REPORT_REFERENCE, REPORT_CONFIG}
    bound.update((ROOT/'src/surgical_agent/research/reporting').glob('*.py'))
    bound.update(ROOT/p for p in core_plan['runtime_sha256'])
    plan = {'profile': PROFILE, 'core_output': str(core_out),
            'core_plan_sha256': sha(core_out/'plan.json'), 'report_config': config, 'policies': policies,
            'rates': rates, 'report_limits': {'openrouter_usd': str(cap)},
            'max_report_dispatches': 2*8578+2*7823, 'workers_max': 8, 'ram': False,
            'scored_frames': 7823, 'pipeline_frames': 8578, 'report_arms': ['H0', 'FULL'],
            'automatic_retry': False,
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(bound)}}
    write(out/'report_provider_metadata.json', metadata)
    write(out/'plan.json', plan)
    write(out/'prepared.json', {'plan_sha256': sha(out/'plan.json'), 'api_posts': 0})
    print(json.dumps({'state': 'PREPARED_COMPLETE_PIPELINE', 'core': str(core_out),
        'report_and_judge_cap_usd': str(cap), 'report_model': config['generator']['model'],
        'judge_model': config['judge']['model'], 'ram': False}), flush=True)


def execute(out, workers, allow_paid):
    if not allow_paid:
        raise ValueError('--allow-paid required for core + Report + Judge')
    if not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    plan = read(out/'plan.json')
    if plan['profile'] != PROFILE or sha(out/'plan.json') != read(out/'prepared.json')['plan_sha256']:
        raise ValueError('complete workflow plan changed')
    from scripts.testing_half_resume import validate_runtime
    validate_runtime(out, plan, 'source_sha256')
    core_out = Path(plan['core_output'])
    if sha(core_out/'plan.json') != plan['core_plan_sha256']:
        raise ValueError('core plan changed')
    verify_core(core_out)
    with (out/'execution.lock').open('x') as stream:
        stream.write(sha(out/'plan.json'))
    started = time.perf_counter()
    budget = cache = None
    try:
        if not (core_out/'receipt.json').exists():
            core.execute(core_out, workers, True)
        verify_core(core_out, completed=True)
        report_out = out/'reports'
        report_out.mkdir()
        stop = threading.Event()
        budget = Budget(report_out/'budget.sqlite', plan['report_limits'], sha(out/'plan.json'))
        caller = ReportCaller(report_out, plan, budget, stop)
        cache = ConcurrentCache(report_out/'cache.sqlite', caller, plan, stop)
        predictions = read(core_out/'continuous_predictions.json')
        generate_reports(predictions, report_out, plan['report_config'], cache, workers, stop)
        # Generate all text before the offline scoring step reads annotations.
        if not (core_out/'scores_detail.json').exists():
            core.score(core_out)
        scores = read(core_out/'scores_detail.json')
        predicted = {r['key']: r for r in read(core_out/'predictions.json')}
        if len(scores['details']) != len(predicted) or {r['key'] for r in scores['details']} != set(predicted):
            raise ValueError('score export scope mismatch')
        for row in scores['details']:
            if row['pipeline'] != predicted[row['key']]['prediction'] or row['h0'] != predicted[row['key']]['h0']:
                raise ValueError('score export predictions changed')
        summary = evaluate_reports(scores, report_out, plan['report_config'], cache, workers, stop)
        write(out/'receipt.json', {'state': 'PASS' if summary['status'] == 'COMPLETE' else 'INCOMPLETE_GSR',
            'plan_sha256': sha(out/'plan.json'), 'core_receipt_sha256': sha(core_out/'receipt.json'),
            'scores_sha256': sha(core_out/'scores_detail.json'),
            'report_summary_sha256': sha(report_out/'results/summary.json'),
            'wall_seconds': time.perf_counter()-started, 'report_budget': budget.summary(),
            'scored_frames': len(predicted), 'report_generated_frames': len(predictions),
            'eligible_gsr_frames': summary['eligible_frames'], 'ram': False})
        if summary['status'] != 'COMPLETE':
            raise RuntimeError('GSR incomplete; inspect saved missing/invalid responses; no automatic retry')
        print(json.dumps({'state': 'COMPLETE_PIPELINE_PASS', 'gsr': summary['methods'],
                          'report_summary': str(report_out/'results/summary.json')}), flush=True)
    except BaseException as exc:
        write(out/'failure.json', {'error_type': type(exc).__name__, 'automatic_retry': False,
                                  'report_budget': budget.summary() if budget else None})
        raise
    finally:
        if cache is not None:
            cache.close()
        if budget is not None:
            budget.close()


def run(out, core_out, report_cap, core_limits, workers, allow_paid):
    """One user command, with actual on-demand Tracker inference (no batch precomputation)."""
    if not allow_paid:
        raise ValueError('--allow-paid required before starting the complete workflow')
    if not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    if not (out/'prepared.json').exists():
        prepare(out, core_out, report_cap, core_limits, resume=out.exists())
    else:
        existing = read(out/'plan.json')
        prepared_core = read(Path(existing['core_output'])/'plan.json')
        if prepared_core.get('tracker_mode') != 'on_demand':
            raise ValueError('existing plan uses batch Tracker; use its execute command or a new output directory')
    execute(out, workers, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'prepare', 'execute', 'resume', 'status'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--core-output', type=Path, help='Attach an existing prepared/completed half run')
    parser.add_argument('--report-budget-usd', default='30')
    parser.add_argument('--core-budget-limits', type=Path, default=core.SCOPE/'suggested_budget_limits.json')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--resume-prepare', action='store_true', help='Retry interrupted local preparation in the same directory')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.command in ('run', 'execute', 'resume'):
        from filelock import FileLock
        from scripts.testing_half_progress import PipelineProgress
        from scripts.testing_half_resume import prepare_resume
        from scripts.testing_half_file_io import retry_local_writes
        # OS-backed lock is released on process exit, including unexpected termination.
        out.parent.mkdir(parents=True, exist_ok=True)
        with (FileLock(str(out)+'.process.lock', timeout=0), retry_local_writes(),
              PipelineProgress(out, stage='RESUME_CHECKS: verifying existing requests and results'
                               if args.command == 'resume' else None) as progress):
            if args.command == 'resume':
                if not args.allow_paid:
                    raise ValueError('--allow-paid required for resume')
                if not 1 <= args.workers <= 8:
                    raise ValueError('workers must be between 1 and 8')
                print(json.dumps(prepare_resume(out)), flush=True)
                progress.stage = None
                execute(out, args.workers, True)
            elif args.command == 'run':
                run(out, args.core_output, args.report_budget_usd, args.core_budget_limits, args.workers, args.allow_paid)
            else:
                execute(out, args.workers, args.allow_paid)
        return
    if args.command == 'run':
        run(out, args.core_output, args.report_budget_usd, args.core_budget_limits, args.workers, args.allow_paid)
    elif args.command == 'prepare':
        prepare(out, args.core_output, args.report_budget_usd, args.core_budget_limits, resume=args.resume_prepare)
    elif args.command == 'execute':
        execute(out, args.workers, args.allow_paid)
    else:
        plan = read(out/'plan.json'); core_out = Path(plan['core_output'])
        print(json.dumps({'core_frames_done': len(list((core_out/'results').glob('*.json'))),
            'reports_done': len(list((out/'reports/generated').glob('*.json'))),
            'evaluated_reports': len(list((out/'reports/evaluated').glob('*.json'))),
            'core_sealed': (core_out/'receipt.json').exists(),
            'complete_receipt': read(out/'receipt.json') if (out/'receipt.json').exists() else None,
            'stopped': (out/'failure.json').exists()}))


if __name__ == '__main__':
    main()
