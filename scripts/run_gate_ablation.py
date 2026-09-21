"""Four-group Gate x Tracker ablation on a frozen contiguous one-third Testing subset.

One paid all-review collection (Gate action forced to 1 on every frame) journals every
response; the Tracker-off arm is the existing same-response counterfactual. Gate-on groups
are exact offline replays of a prefix of the same responses, because the Gate consumes only
H0, proposal and the pre-Gate five-head probe, and the post-Gate panel is consumed in a fixed
order with a deterministic early stop. Report/Rule/Judge/GSR run afterwards for H0 and all
four groups with one shared response cache. Credentials come from an explicit ablation
directory, never from the project docs/API.txt.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import requests
from scripts import run_testing_half_pipeline as core
from scripts import run_testing_half_complete as complete
from scripts import check_candidate_panel_providers as cp
from scripts import reviewer_routes_official as routes_official
from scripts import reviewer_routes_v3 as routes_v3
from surgical_agent.api.credentials import load_api_key_file
from surgical_agent.research.gate.collection_budget import Budget
from surgical_agent.research.gate.tracker_pipeline_v2 import CausalPhaseFilter

read, sha, write = core.read, core.sha, core.write
ORIGINAL_LOAD_GATE = core.app.load_gate
ORIGINAL_RUN_TARGET = core.run_target
DEFAULT_CREDENTIALS = ROOT / '消融实验API'
DEFAULT_SCOPE = ROOT / 'artifacts/evaluation/testing_third_ablation_20260916'
CRED_FILES = {'openrouter': 'docs/openrouter-backup-API.txt', 'aliyun': 'docs/aliyun_API KEY',
              'glm': 'docs/GLM-API.txt', 'deepseek': 'docs/DS-API.txt'}
MAIN_FILES = {'openrouter': 'docs/API.txt', 'aliyun': 'docs/aliyun_API KEY',
              'glm': 'docs/GLM-API.txt', 'deepseek': 'docs/DS-API.txt'}
GATE_MODE = 'force_review_all_frames'
GROUPS = {'g1_gate_on_tracker_on': ('gate_on', True), 'g2_gate_on_tracker_off': ('gate_on', False),
          'g3_gate_off_tracker_on': ('gate_off', True), 'g4_gate_off_tracker_off': ('gate_off', False)}
HEADS = ('instrument', 'verb', 'target', 'ivt', 'phase')


# ----------------------------------------------------------------------------- scope / gate
def scope_rows(scope):
    """Generalised copy of core.scope_rows: frame counts come from run_scope.json."""
    spec = read(scope / 'run_scope.json')
    if spec.get('base_model', 'google/gemini-3.8-flash') != 'google/gemini-3.8-flash':
        raise ValueError('ablation scope must reuse the Gemini H0 base model')
    if sha(scope / 'frame_inventory.jsonl') != spec['frame_inventory_sha256']:
        raise ValueError('frame inventory changed')
    for path, digest in spec['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('scope source changed: ' + path)
    selected = core.rows(scope / 'frame_inventory.jsonl')
    if (len(selected) != spec['pipeline_frames'] + spec['warmup_h0_frames']
            or sum(s['evaluation_target'] for s in selected) != spec['evaluation_target_frames']
            or sum(s['stage'] == 'pipeline' for s in selected) != spec['pipeline_frames']):
        raise ValueError('frame inventory does not match the declared scope counts')
    for v in spec['videos']:
        actual = [s['frame_id'] for s in selected if s['video_id'] == v['video_id']]
        if actual != list(range(v['warmup_start_frame'], v['end_frame_exclusive'], 25)):
            raise ValueError('timeline is not contiguous: ' + v['video_id'])
    return spec, selected


class ForcedReview:
    """Gate off = the same runtime with action fixed to 1; the model score is kept for logging."""
    def __init__(self, real):
        self.real = real
        self.phase_review_enabled = real.phase_review_enabled
        self.review_mode = real.review_mode

    def __call__(self, features):
        score, _ = self.real(features)
        return score, 1


def forced_load_gate(root=ROOT):
    decide, manifest, model = ORIGINAL_LOAD_GATE(root)
    return ForcedReview(decide), manifest, model


def recording_run_target(backend, *args, **kwargs):
    """Persist the proposal response and the un-forced Gate action alongside the live result."""
    result = ORIGINAL_RUN_TARGET(backend, *args, **kwargs)
    if hasattr(backend, 'proposal_raw'):  # live WireBackend only; the in-run replay object has none
        result['proposal_raw'] = deepcopy(backend.proposal_raw)
        result['gate_action_forced'] = True
        result['gate_mode'] = GATE_MODE
    return result


def apply_runtime_patches():
    core.scope_rows = scope_rows
    core.app.load_gate = forced_load_gate
    core.run_target = recording_run_target


# ----------------------------------------------------------------------------- credentials
def apply_credential_patches(cred_root):
    cred_root = Path(cred_root)
    for name, relative in CRED_FILES.items():
        if not (cred_root / relative).is_file():
            raise FileNotFoundError('ablation credential missing: ' + name)
    cp.MODELS = {seat: tuple(CRED_FILES['openrouter'] if x == MAIN_FILES['openrouter'] else x for x in entry)
                 for seat, entry in cp.MODELS.items()}
    routes_official.ROOT = cred_root


def _digest(text):
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()


def verify_credentials(plan, cred_root, *, online=False):
    """Every runtime credential path must resolve to the ablation files and differ from docs/."""
    cred_root = Path(cred_root)
    expected = {k: _digest((cred_root / f).read_text(encoding='utf-8-sig')) for k, f in CRED_FILES.items()}
    main = {k: _digest((ROOT / f).read_text(encoding='utf-8-sig')) for k, f in MAIN_FILES.items() if (ROOT / f).exists()}
    config = plan['reviewer_config']
    with core.app.frozen.joint.credential_context(plan):
        got = {'openrouter': _digest(cp.key_for('gpt').reveal()),
               'openrouter_gemini_seat': _digest(cp.key_for('gemini').reveal()),
               'aliyun': _digest(cp.key_for('qwen').reveal()),
               'glm': _digest(routes_official.credentials('grok', config)[1]),
               'deepseek': _digest(routes_official.credentials('deepseek', config)[1])}
        qwen_url, qwen_key = routes_v3.credentials('qwen', config)
        got['aliyun_review_route'] = _digest(qwen_key)
        endpoint = None
        if online:
            response = requests.get(qwen_url.rstrip('/') + '/models', headers={'Authorization': 'Bearer ' + qwen_key}, timeout=30)
            listed = [m.get('id') for m in response.json().get('data', [])] if response.status_code == 200 else []
            endpoint = {'qwen_models_http_status': response.status_code, 'qwen37_flash_listed': 'qwen3.7-flash' in listed}
            if response.status_code in (401, 403):
                raise ValueError('ablation Aliyun credential rejected by the configured Qwen endpoint')
    matches = {'openrouter': got['openrouter'] == expected['openrouter'],
               'openrouter_gemini_seat': got['openrouter_gemini_seat'] == expected['openrouter'],
               'aliyun': got['aliyun'] == expected['aliyun'],
               'aliyun_review_route': got['aliyun_review_route'] == expected['aliyun'],
               'glm': got['glm'] == expected['glm'], 'deepseek': got['deepseek'] == expected['deepseek']}
    distinct = {k: got[k] != main[k] for k in ('openrouter', 'aliyun', 'glm', 'deepseek') if k in main}
    if not all(matches.values()):
        raise ValueError('a runtime credential path does not resolve to the ablation files: ' + json.dumps(matches))
    if not all(distinct.values()):
        raise ValueError('an ablation credential equals the main project credential: ' + json.dumps(distinct))
    return {'credential_root': str(cred_root), 'files': {k: str(cred_root / f) for k, f in CRED_FILES.items()},
            'file_sha256_prefix': {k: v[:8] for k, v in expected.items()},
            'runtime_resolves_to_ablation_files': matches, 'differs_from_main_docs': distinct,
            'report_judge_credential': str(cred_root / CRED_FILES['openrouter']), 'endpoint_check': endpoint}


class AblationReportCaller(complete.ReportCaller):
    def __init__(self, out, plan, budget, stop, secret_file):
        self.out, self.plan, self.budget, self.stop = out, plan, budget, stop
        self.secret = load_api_key_file(secret_file)


# ----------------------------------------------------------------------------- prepare / collect
def prepare(out, scope, limits, cred_root):
    if out.exists():
        raise ValueError('output exists; use a new directory')
    apply_runtime_patches()
    apply_credential_patches(cred_root)
    spec, _ = scope_rows(scope)
    core_out = out / 'core'
    core.prepare(core_out, scope, limits.resolve(), resume=False, streaming=True)
    plan = read(core_out / 'plan.json')
    plan.update(evaluation_target_frames=spec['evaluation_target_frames'], pipeline_frames=spec['pipeline_frames'],
                warmup_h0_frames=spec['warmup_h0_frames'],
                maximum_paid_calls=plan['new_h0_calls'] + spec['pipeline_frames'] * 6,
                credential_root=str(Path(cred_root).resolve()), gate_mode=GATE_MODE,
                ablation={**spec['ablation'], 'scope': str(scope), 'runner': 'scripts/run_gate_ablation.py'})
    write(core_out / 'plan.json', plan)
    write(core_out / 'prepared.json', {'plan_sha256': sha(core_out / 'plan.json'), 'api_calls': 0})
    credentials = verify_credentials(plan, cred_root, online=True)
    _, manifest, model = ORIGINAL_LOAD_GATE()
    write(out / 'ablation.json', {
        'profile': 'gate_tracker_four_group_ablation_v1', 'scope': str(scope), 'scope_sha256': sha(scope / 'run_scope.json'),
        'groups': GROUPS, 'gate': {'version': manifest['version'], 'threshold': model['threshold'], 'feature_count': len(model['feature_names'])},
        'gate_off_definition': spec['ablation']['gate_off_definition'], 'collection_gate_mode': GATE_MODE,
        'pipeline_frames': spec['pipeline_frames'], 'evaluation_target_frames': spec['evaluation_target_frames'],
        'new_h0_calls': plan['new_h0_calls'], 'limits': plan['limits'], 'credentials': credentials,
        'paid_api_calls_so_far': 0, 'ram': False})
    print(json.dumps({'state': 'PREPARED_ABLATION', 'output': str(out), 'pipeline_frames': spec['pipeline_frames'],
                      'scored_frames': spec['evaluation_target_frames'], 'new_h0_calls': plan['new_h0_calls'],
                      'limits': plan['limits'], 'credentials_ok': True, 'api_calls': 0}, ensure_ascii=False), flush=True)


def progress_thread(core_out, plan, stop):
    def loop():
        started = time.time()
        while not stop.wait(120):
            done = len(list((core_out / 'results').glob('*.json')))
            h0 = len(list((core_out / 'h0').glob('*.json')))
            print(json.dumps({'elapsed_min': round((time.time() - started) / 60, 1), 'pipeline_done': done,
                              'pipeline_total': plan['pipeline_frames'], 'h0_done': h0,
                              'budget': budget_summary(core_out)}), flush=True)
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread


def budget_summary(core_out):
    path = core_out / 'budget.sqlite'
    if not path.exists():
        return None
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        return {r[0]: {'cap': r[1], 'occupied': r[2]} for r in db.execute('SELECT * FROM accounts')}


def collect(out, workers, allow_paid):
    if not allow_paid:
        raise ValueError('--allow-paid required; the collection sends paid requests')
    core_out = out / 'core'
    plan = read(core_out / 'plan.json')
    if plan.get('gate_mode') != GATE_MODE or sha(core_out / 'plan.json') != read(core_out / 'prepared.json')['plan_sha256']:
        raise ValueError('not a prepared ablation collection plan')
    apply_runtime_patches()
    apply_credential_patches(plan['credential_root'])
    verify_credentials(plan, plan['credential_root'])
    stop = threading.Event()
    progress_thread(core_out, plan, stop)
    try:
        core.execute(core_out, workers, True)
    finally:
        stop.set()
    replay(out, keep_m3_without_tracker=False)


def resume(out, workers, allow_paid, assume_stopped):
    core_out = out / 'core'
    if (core_out / 'receipt.json').exists():
        raise ValueError('core collection already sealed; run replay/report')
    if not (core_out / 'failure.json').exists() and not assume_stopped:
        raise ValueError('failure.json missing: confirm the previous process is dead, then pass --assume-stopped')
    from scripts.testing_half_resume import reconcile_completed_reservations
    reconciled = reconcile_completed_reservations(core_out)
    with sqlite3.connect((core_out / 'budget.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
        unresolved = db.execute("SELECT COUNT(*) FROM calls WHERE state!='TERMINAL'").fetchone()[0]
    if unresolved:
        raise ValueError(f'{unresolved} paid requests have no settled outcome; inspect before resuming')
    for path in (core_out / 'targets').rglob('record.json'):
        row = read(path)
        if row.get('status') == 'DISPATCHED' or not row.get('finished_utc'):
            raise ValueError('unfinished request journal: ' + str(path))
    archive = core_out / 'attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive.mkdir(parents=True)
    for name in ('execution_started.json', 'failure.json'):
        if (core_out / name).exists():
            shutil.move(str(core_out / name), str(archive / name))
    write(core_out / 'resume_policy.json', {'policy': 'gate_ablation_resume_v1', 'plan_sha256': sha(core_out / 'plan.json'),
                                            'amended_sources': {}, 'added_sources': {}, 'reconciled_reservations': reconciled,
                                            'archive': str(archive), 'automatic_retry': False})
    print(json.dumps({'state': 'RESUMING', 'reconciled': reconciled, 'archive': str(archive)}), flush=True)
    collect(out, workers, allow_paid)


# ----------------------------------------------------------------------------- replay
class ReplayBackend:
    """Sealed responses of one target; refuses anything that was not collected."""
    parse_h0 = staticmethod(deepcopy)
    normalize_compact = staticmethod(core.app.frozen.common.normalize_five)

    def __init__(self, result):
        if 'proposal_raw' not in result or 'five_head_raw' not in result:
            raise ValueError('result lacks journaled proposal/five-head responses: ' + result['key'])
        self.r = result

    def h0(self):
        return deepcopy(self.r['h0'])

    def proposal(self, *args):
        return deepcopy(self.r['proposal_raw'])

    def five_head(self, seat, *args):
        if seat not in self.r['five_head_raw']:
            raise ValueError('replay requested an uncollected five-head seat: ' + seat)
        return deepcopy(self.r['five_head_raw'][seat])

    def compact(self, *args):
        raise ValueError('compact control_graph review is not part of the five_head_probe runtime')

    def phase_recommendation(self, *args):
        raise ValueError('phase recommendation is not requested in five_head_probe mode')

    def joint(self, *args):
        raise ValueError('joint_r1 is not requested in five_head_probe mode')


def replay(out, *, keep_m3_without_tracker=False):
    core_out = out / 'core'
    plan, receipt = read(core_out / 'plan.json'), read(core_out / 'receipt.json')
    if receipt['state'] != 'PASS' or receipt['plan_sha256'] != sha(core_out / 'plan.json'):
        raise ValueError('collection not sealed')
    if plan.get('gate_mode') != GATE_MODE:
        raise ValueError('collection was not an all-review collection')
    decide_real, manifest, model = ORIGINAL_LOAD_GATE()
    modes = {'gate_on': decide_real, 'gate_off': ForcedReview(decide_real)}
    filters = {g: CausalPhaseFilter(60 if (tracker or keep_m3_without_tracker) else 0) for g, (_, tracker) in GROUPS.items()}
    rows = {g: [] for g in GROUPS}
    priors = {v: read(core_out / 'priors' / (v + '.json')) for v in {s['video_id'] for s in plan['selection']}}
    stats = {'frames': 0, 'gate_on_opened': 0, 'gate_on_model_action_agrees_with_logged': 0,
             'depth_hist': {m: {} for m in modes}, 'logical_calls': {m: 0 for m in modes},
             'gate_off_replay_equals_live': 0, 'gate_on_prefix_verified': 0}
    for s in sorted(plan['selection'], key=lambda s: (s['video_id'], s['frame_id'])):
        video, frame = s['video_id'], s['frame_id']
        if s['stage'] != 'pipeline':
            h0 = read(core_out / 'h0' / (s['key'] + '.json'))
            for f in filters.values():
                f.apply(video, frame, h0['phase'][0])
            continue
        result = read(core_out / 'results' / (s['key'] + '.json'))
        selected = {k: s[k] for k in ('key', 'video_id', 'frame_id', 'causal_frame_ids', 'stage', 'evaluation_target')}
        selected['source_split'] = 'Testing'
        snapshot_path = core_out / 'tracker_runtime' / (s['key'] + '.json')
        if not snapshot_path.exists():
            snapshot_path = core_out / 'tracker' / (s['key'] + '.json')
        snapshot = read(snapshot_path)
        stats['frames'] += 1
        outputs = {}
        for mode, decide in modes.items():
            with_t = ORIGINAL_RUN_TARGET(ReplayBackend(result), selected, priors[video], decide, tracker_snapshot=snapshot,
                                         phase_filter=CausalPhaseFilter(0), output='v2.2', inference_split='Testing')
            without = ORIGINAL_RUN_TARGET(ReplayBackend(result), selected, priors[video], decide, tracker_snapshot=None,
                                          phase_filter=CausalPhaseFilter(0), output='v2.2', inference_split='Testing')
            if with_t['call_keys'] != without['call_keys'] or with_t['features'] != result['features']:
                raise ValueError('replay routing/features diverged from the collection: ' + s['key'])
            if mode == 'gate_off':
                if (with_t['prediction'] != result['prediction'] or without['prediction'] != result['without_tracker']
                        or with_t['call_keys'] != result['call_keys']):
                    raise ValueError('all-review replay differs from the live result: ' + s['key'])
                stats['gate_off_replay_equals_live'] += 1
            else:
                n = len(with_t['call_keys'])
                if result['call_keys'][:n] != with_t['call_keys']:
                    raise ValueError('Gate-on replay is not a prefix of the collected responses: ' + s['key'])
                stats['gate_on_prefix_verified'] += 1
                stats['gate_on_opened'] += with_t['gate_action']
                stats['gate_on_model_action_agrees_with_logged'] += int(with_t['gate_action'] == int(result['gate_score'] >= model['threshold']))
            depth = with_t['joint_depth']
            stats['depth_hist'][mode][str(depth)] = stats['depth_hist'][mode].get(str(depth), 0) + 1
            stats['logical_calls'][mode] += with_t['logical_calls']
            outputs[(mode, True)], outputs[(mode, False)] = with_t, without
        for g, (mode, tracker) in GROUPS.items():
            res = outputs[(mode, tracker)]
            pred = deepcopy(res['prediction'])
            pred['phase'] = [filters[g].apply(video, frame, res['phase_before_smoothing'])]
            counterpart = outputs[(mode, False)]['prediction']
            rows[g].append({'key': s['key'], 'video_id': video, 'frame_id': frame, 'evaluation_target': s['evaluation_target'],
                            'h0': res['h0'], 'prediction': pred, 'without_tracker': deepcopy(counterpart),
                            'prediction_before_smoothing': res['prediction'], 'phase_before_smoothing': res['phase_before_smoothing'],
                            'gate_action': res['gate_action'], 'gate_score': res['gate_score'],
                            'gate_forced': mode == 'gate_off', 'compact_depth': res['compact_depth'], 'joint_depth': res['joint_depth'],
                            'logical_calls': res['logical_calls'], 'call_keys': res['call_keys'],
                            'output_modules': res['output_modules'], 'phase_decision': res['phase_decision'],
                            'group': g, 'tracker': tracker, 'gate_mode': mode})
    groups_dir = out / 'groups'
    if groups_dir.exists():
        shutil.rmtree(groups_dir)
    selection_text = (f"Frozen contiguous leading one-third of each Testing video's half interval; "
                      f"{plan['evaluation_target_frames']} annotated frames; group ")
    summary = {}
    for g, (mode, tracker) in GROUPS.items():
        folder = groups_dir / g
        scored = [r for r in rows[g] if r['evaluation_target']]
        write(folder / 'continuous_predictions.json', rows[g])
        write(folder / 'predictions.json', scored)
        write(folder / 'receipt.json', {'state': 'PASS', 'rows': len(scored), 'group': g, 'gate_mode': mode, 'tracker': tracker,
                                         'source_receipt_sha256': sha(core_out / 'receipt.json'),
                                         'predictions_sha256': sha(folder / 'predictions.json'),
                                         'continuous_predictions_sha256': sha(folder / 'continuous_predictions.json'),
                                         'offline_replay': True, 'api_calls': 0})
        summary[g] = score_group(folder, selection_text + g)
    overview = {'profile': 'gate_tracker_four_group_ablation_v1', 'groups': GROUPS,
                'scored_frames': plan['evaluation_target_frames'], 'pipeline_frames': plan['pipeline_frames'],
                'gate': {'version': manifest['version'], 'threshold': model['threshold']},
                'tracker_off_keeps_m3': keep_m3_without_tracker,
                'equivalence': stats, 'gate_on_open_rate': stats['gate_on_opened'] / max(stats['frames'], 1),
                'metrics': summary, 'budget': budget_summary(core_out), 'api_calls_during_replay': 0}
    write(out / 'ablation_summary.json', overview)
    print(json.dumps({'state': 'REPLAYED', 'frames': stats['frames'], 'gate_on_open_rate': overview['gate_on_open_rate'],
                      'logical_calls': stats['logical_calls'], 'metrics': summary}, ensure_ascii=False), flush=True)
    return overview


def score_group(folder, selection_text):
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        core.demo.score(folder, 'scores_detail.json')
    report = read(folder / 'scores_detail.json')
    report.update(rows=len(read(folder / 'predictions.json')), selection=selection_text, scorer_sha256=sha(Path(__file__)))
    write(folder / 'scores_detail.json', report)
    write(folder / 'scores.json', {k: v for k, v in report.items() if k != 'details'})
    arm = report['arms']['prediction']['by_head']
    return {h: {'f1': arm[h]['micro_f1'], 'precision': arm[h]['micro_precision'], 'recall': arm[h]['micro_recall'],
                'accuracy': arm[h]['exact_accuracy'], 'n': arm[h]['eligible_frames']} for h in HEADS} | {
        'errors': report['arms']['prediction']['errors'],
        'h0_f1': {h: report['arms']['h0']['by_head'][h]['micro_f1'] for h in HEADS},
        'h0_phase_accuracy': report['arms']['h0']['by_head']['phase']['exact_accuracy']}


# ----------------------------------------------------------------------------- report
def report(out, workers, cap, allow_paid, cred_root):
    if not allow_paid:
        raise ValueError('--allow-paid required for Report/Judge')
    if not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    cap = Decimal(cap)
    if not cap.is_finite() or cap <= 0:
        raise ValueError('report budget must be positive USD')
    secret_file = Path(cred_root) / CRED_FILES['openrouter']
    if _digest(secret_file.read_text(encoding='utf-8-sig')) == _digest((ROOT / MAIN_FILES['openrouter']).read_text(encoding='utf-8-sig')):
        raise ValueError('ablation Report credential equals the main OpenRouter credential')
    overview = read(out / 'ablation_summary.json')
    reports = out / 'reports'
    reports.mkdir(exist_ok=True)
    if (reports / 'plan.json').exists():
        plan = read(reports / 'plan.json')
    else:
        config, policies = complete.report_config()
        rates, metadata = complete.endpoint_rates(config, policies)
        plan = {'profile': 'gate_ablation_report_rule2_judge2low_v1', 'report_config': config, 'policies': policies, 'rates': rates,
                'report_limits': {'openrouter_usd': str(cap)}, 'max_report_dispatches': 2 * overview['scored_frames'] * (len(GROUPS) + 1),
                'workers_max': 8, 'ram': False, 'scored_frames': overview['scored_frames'], 'report_arms': ['H0', 'FULL'],
                'groups': list(GROUPS), 'automatic_retry': False, 'shared_cache': True,
                'generation_scope': 'evaluation-target frames only; H0 reports/judgements are shared across groups via the cache',
                'credential': str(secret_file), 'ablation_summary_sha256': sha(out / 'ablation_summary.json')}
        write(reports / 'report_provider_metadata.json', metadata)
        write(reports / 'plan.json', plan)
    if plan['ablation_summary_sha256'] != sha(out / 'ablation_summary.json'):
        raise ValueError('ablation replay changed after the report plan was frozen')
    stop = threading.Event()
    budget = Budget(reports / 'budget.sqlite', plan['report_limits'], sha(reports / 'plan.json'))
    caller = AblationReportCaller(reports, plan, budget, stop, secret_file)
    cache = complete.ConcurrentCache(reports / 'cache.sqlite', caller, plan, stop)
    summaries = {}
    try:
        for g in GROUPS:
            folder = reports / g
            if (folder / 'results' / 'summary.json').exists():
                summaries[g] = read(folder / 'results' / 'summary.json')
                continue
            if folder.exists():
                raise ValueError('incomplete report directory; inspect before rerunning: ' + str(folder))
            predictions = read(out / 'groups' / g / 'predictions.json')
            complete.generate_reports(predictions, folder, plan['report_config'], cache, workers, stop)
            scores = read(out / 'groups' / g / 'scores_detail.json')
            summaries[g] = complete.evaluate_reports(scores, folder, plan['report_config'], cache, workers, stop)
            print(json.dumps({'group': g, 'status': summaries[g]['status'], 'methods': summaries[g].get('methods')}, ensure_ascii=False), flush=True)
        write(reports / 'summary.json', {'groups': {g: {'status': s['status'], 'eligible_frames': s.get('eligible_frames'),
                                                        'methods': s.get('methods')} for g, s in summaries.items()},
                                         'budget': budget.summary(), 'shared_cache_costs': cache.costs(),
                                         'note': 'H0 arm is identical across groups (shared cache); costs are pooled'})
        print(json.dumps({'state': 'REPORTED', 'summary': str(reports / 'summary.json'), 'budget': budget.summary()}, ensure_ascii=False), flush=True)
    finally:
        cache.close()
        budget.close()


def status(out):
    core_out = out / 'core'
    info = {'core_prepared': (core_out / 'prepared.json').exists(), 'core_sealed': (core_out / 'receipt.json').exists(),
            'core_stopped': (core_out / 'failure.json').exists(), 'core_running_marker': (core_out / 'execution_started.json').exists(),
            'h0_done': len(list((core_out / 'h0').glob('*.json'))), 'pipeline_done': len(list((core_out / 'results').glob('*.json'))),
            'pipeline_total': read(core_out / 'plan.json')['pipeline_frames'] if (core_out / 'plan.json').exists() else None,
            'budget': budget_summary(core_out), 'replayed': (out / 'ablation_summary.json').exists(),
            'reports_done': [g for g in GROUPS if (out / 'reports' / g / 'results' / 'summary.json').exists()]}
    print(json.dumps(info, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'collect', 'resume', 'replay', 'report', 'status'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', type=Path, default=DEFAULT_SCOPE)
    parser.add_argument('--budget-limits', type=Path)
    parser.add_argument('--credentials', type=Path, default=DEFAULT_CREDENTIALS, help='directory holding docs/<ablation key files>')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--report-budget-usd', default='40')
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--assume-stopped', action='store_true')
    parser.add_argument('--keep-m3-without-tracker', action='store_true', help='replay only: keep the 60 s phase filter in the Tracker-off groups')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.command == 'prepare':
        limits = args.budget_limits or args.scope / 'suggested_budget_limits.json'
        prepare(out, args.scope.resolve(), limits, args.credentials.resolve())
    elif args.command == 'collect':
        collect(out, args.workers, args.allow_paid)
    elif args.command == 'resume':
        resume(out, args.workers, args.allow_paid, args.assume_stopped)
    elif args.command == 'replay':
        replay(out, keep_m3_without_tracker=args.keep_m3_without_tracker)
    elif args.command == 'report':
        report(out, args.workers, args.report_budget_usd, args.allow_paid, args.credentials.resolve())
    else:
        status(out)


if __name__ == '__main__':
    main()
