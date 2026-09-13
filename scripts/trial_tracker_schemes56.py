"""Prepare, collect and score separate small Training pilots for schemes 5 and 6.

Only collect --allow-paid can send requests. Scheme 5 runs first; scheme 6 changes
only Qwen and explicitly shares the fresh control proposal/four remaining replies.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys
import threading
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import assess_tracker_scheme4 as study
from scripts.assess_tracker_review_evidence_r3 import RefusalTolerantCalls
from scripts import full_official_reviewer_transport as T
from scripts.run_pgp_pipeline import CachedBackend, frozen, predictor
from surgical_agent.research.gate.collection_budget import Budget, BudgetStop, AmbiguousDispatch
from surgical_agent.research.gate.pgp_tracker_gemini38 import FrozenTracker
from surgical_agent.research.gate.tracker_pipeline_v2 import run_interaction, output_modules, current_classes
from surgical_agent.research.retrieval.tracker_prior_candidates import retrieve_tracker_hints
from surgical_agent.research.verification.tracker_evidence import attach, make_packet
from surgical_agent.research.verification.candidate_coordinator import make_pool
from tools.audit.gate_proposals_offline_20260913 import offline

DEFAULT = ROOT / 'artifacts/research/tracker_schemes56_pilot_20260914_r5'
SEATS = ('qwen', 'gpt', 'gemini', 'grok', 'deepseek')
LIMITS = {'openrouter_usd': '5', 'aliyun_cny': '6', 'glm_requests': '32', 'deepseek_requests': '32', 'xai_usd': '0'}


def proposal_wire(base, selected, h0, pool, prior, classes, variant):
    if variant == 'prior':
        hints = retrieve_tracker_hints(h0, prior, video_id=selected['video_id'], tracker_classes=classes)
    else:
        hints = frozen.retrieve_candidate_hints(h0, prior, video_id=selected['video_id'])
    return frozen.proposal_wire(base, selected, h0, pool, hints['packet'])


def compact_wire(base, selected, h0, pool, packet, variant, seat):
    wire = frozen.joint.roster.review_wire(seat, base, selected, pool)
    if variant == 'qwen' and seat == 'qwen':
        wire = attach(wire, packet)
    frozen.check_requests(h0, [wire])
    return wire


def failure_requires_stop(row, seat):
    """Narrow refusal exemption; unknown dispatch outcomes cannot silently continue."""
    if row.get('status') == 'DISPATCHED' or 'finished_utc' not in row:
        return True
    if row.get('http_status') is None:
        return True
    status = row['http_status']
    err = row.get('provider_error') or {}
    allowed = status == 400 and isinstance(err, dict) and (
        seat == 'grok' and str(err.get('code')) == '1301' or
        seat == 'qwen' and err.get('code') == 'data_inspection_failed')
    if status in (400, 401, 402, 403, 404) and not allowed:
        return True
    kind = row.get('error_type', row.get('exception_type', ''))
    if kind in ('ReadTimeout', 'ConnectTimeout', 'Timeout', 'ConnectionError'):
        return True
    return bool(row.get('global_stop'))


class GuardedCalls(RefusalTolerantCalls):
    def call(self, target, stage, seat, body):
        answer = super().call(target, stage, seat, body)
        rows = self.rows if seat in T.CHANGED else self.delegate.rows
        row = next((r for r in reversed(rows) if (r['target'], r['stage'], r['seat']) == (target, stage, seat)), None)
        if row is None or failure_requires_stop(row, seat):
            self.stop.set()
            raise AmbiguousDispatch('stopped transport; inspect durable record before any continuation')
        return answer


def prepare(out):
    if not (study.DEFAULT / 'report.json').exists():
        raise ValueError('scheme 4 must finish first')
    out.mkdir(exist_ok=False)
    (out / 'evidence').mkdir(); (out / 'previews').mkdir()
    original = study.read(study.SOURCE / 'plan.json')
    inventory = {s['key']: s for s in original['selection']}
    rows = study.read(study.ROWS)
    groups = []
    for v in ('VID103', 'VID23', 'VID31', 'VID96'):
        group = sorted([r for r in rows if r['video_id'] == v], key=lambda r: r['frame_id'])
        groups.append([group[int((j + .5) * len(group) / 4)] for j in range(4)])
    chosen = [groups[v][j] for j in range(4) for v in range(4)]
    tracker = FrozenTracker(study.TRACKER / 'index.json')
    selection, changed, preview_caps = [], [], Counter()
    with offline(), frozen.joint.roster.lightweight_protocol():
        for row in chosen:
            source = inventory[row['sample_id']]
            selected = {k: deepcopy(source[k]) for k in ('key', 'video_id', 'frame_id', 'anchor_frame_id', 'causal_frame_ids', 'images', 'alignment_version')}
            selected['source_split'] = 'Training'
            record_path = study.SOURCE / 'targets' / selected['key'] / 'result.json'
            record = study.read(record_path); h0 = record['h0']; prior = study.read(study.SOURCE / 'priors' / (selected['video_id'] + '.json'))
            snapshot = tracker.snapshot(selected); classes = current_classes(snapshot, selected)
            if classes is None:
                raise ValueError('selected frame has no valid Tracker')
            packet = make_packet(snapshot, selected)
            study.write(out / 'evidence' / (selected['key'] + '.json'), packet)
            selected['tracker_classes'] = sorted(classes)
            selected['source_result_sha256'] = study.sha(record_path)
            selected['evidence_sha256'] = study.sha(out / 'evidence' / (selected['key'] + '.json'))
            base = T.old.load_base(selected); pool = make_pool(h0)
            control = proposal_wire(base, selected, h0, pool, prior, classes, 'control')
            prior_wire = proposal_wire(base, selected, h0, pool, prior, classes, 'prior')
            # Proposal alone already permits prior hints in the frozen pipeline.
            # Retain the H0/GT checks, exempting only this intended hint field.
            from surgical_agent.research.verification.prior_gated_joint import assert_ungated_request
            for wire in (control, prior_wire):
                packet_check = deepcopy(frozen.joint.packet_of(wire))
                packet_check.pop('candidate_relation_hints', None)
                assert_ungated_request(packet_check, h0)
            if control != prior_wire:
                changed.append(selected['key'])
            previews = {'control_proposal': control, 'tracker_prior_proposal': prior_wire}
            # Reviewer previews use the old pool only; live pools derive from fresh proposals.
            for seat in SEATS:
                wire = compact_wire(base, selected, h0, record['pool'], packet, 'control', seat)
                tweaked = compact_wire(base, selected, h0, record['pool'], packet, 'qwen', seat)
                if seat != 'qwen':
                    assert wire == tweaked
                previews['control_' + seat] = wire
                if seat == 'qwen':
                    previews['tracker_qwen'] = tweaked
                for candidate in ((wire, wire, tweaked) if seat == 'qwen' else (wire, wire)):
                    if seat in ('grok', 'deepseek'):
                        preview_caps[seat + '_requests'] += 1
                    else:
                        routed = T.routes.route_body(seat, candidate, original['reviewer_config']) if seat == 'qwen' else candidate
                        account = frozen.joint.roster.transport.bucket(seat)
                        preview_caps[account] += frozen.joint.roster.transport.envelope(seat, routed, original['call_rates'])
            for wire in (control, prior_wire):
                preview_caps['openrouter_usd'] += frozen.joint.roster.transport.envelope('base', wire, original['call_rates'])
            safe = {k: frozen.joint.roster.transport.redact_images(w) for k, w in previews.items()}
            study.write(out / 'previews' / (selected['key'] + '.json'), safe)
            selection.append(selected)
    plan = deepcopy(original)
    plan.update(profile='tracker-schemes56-paired-v1', selection=selection, limits=LIMITS,
                maximum_paid_calls=208, Testing_access=False, VID110_access=False,
                sampling='4 chronological quantiles per video from all 6059 rows; no Gate/GT filtering',
                sequence=['scheme5_control_and_prior', 'scheme6_qwen_only_on_fresh_control_pool'],
                reuse='H0 and Phase fixed; scheme6 explicitly reuses fresh control proposal and four non-Qwen reviews',
                gate='old frozen Qwen Gate as migration control; no accuracy/calibration guarantee after input change',
                postprocessing='scheme4 corrected M1/M2 and same cached causal Phase history in all arms',
                stopping='No retries; unknown outcomes, identity/route mismatch, non-exempt 400-404 or budget stop the run',
                primary='paired F1 nondecrease, errors nonincrease, one strict quality gain, every video nonworse, known USD <=110%; report CNY and request accounts separately')
    bound = [Path(__file__), study.PROTOCOL, ROOT / 'docs/TRACKER_SCHEMES56_PREFLIGHT_2026-09-14.md', ROOT / 'scripts/assess_tracker_review_evidence_r3.py',
             ROOT / 'scripts/full_official_reviewer_transport.py', ROOT / 'scripts/collect_gate_escalation_v1.py',
             ROOT / 'scripts/run_pgp_pipeline.py', ROOT / 'scripts/reviewer_routes_official.py',
             ROOT / 'src/surgical_agent/research/gate/tracker_pipeline_v2.py',
             ROOT / 'src/surgical_agent/research/retrieval/prior_candidates.py',
             ROOT / 'src/surgical_agent/research/retrieval/tracker_prior_candidates.py',
             ROOT / 'src/surgical_agent/research/verification/tracker_evidence.py',
             ROOT / 'src/surgical_agent/research/gate/collection_budget.py',
             study.DEFAULT / 'predictions.npz', study.ROWS,
             ROOT / 'DEFAULT_PGP_GATE_VERSION.json', ROOT / 'DEFAULT_PIPELINE_VERSION.json']
    _, default, model = predictor()
    bound.extend([ROOT / default['model_artifact'], ROOT / model['estimator_artifact']])
    bound.extend(study.SOURCE / 'priors' / (v + '.json') for v in ('VID103', 'VID23', 'VID31', 'VID96'))
    plan['source_sha256'] = {str(p): study.sha(p) for p in bound}
    study.write(out / 'plan.json', plan)
    summary = {'state': 'PREPARED_NOT_COLLECTED', 'rows': 16, 'api_calls': 0, 'maximum_paid_calls': 208,
               'maximum_requests_by_seat': {'base': 32, 'qwen': 48, 'gpt': 32, 'gemini': 32, 'grok': 32, 'deepseek': 32},
               'limits': LIMITS, 'preview_reservation_estimates': {k: str(v) for k, v in preview_caps.items()},
               'prior_prompt_changed_frames': changed, 'plan_sha256': study.sha(out / 'plan.json'),
               'preview_limit': 'Review previews use historical pools; fresh proposals can change pool size. Per-dispatch reservations enforce caps.'}
    study.write(out / 'preflight.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def verify(out):
    plan = study.read(out / 'plan.json')
    if plan['profile'] != 'tracker-schemes56-paired-v1' or study.sha(out / 'plan.json') != study.read(out / 'preflight.json')['plan_sha256']:
        raise ValueError('plan mismatch')
    for p, digest in plan['source_sha256'].items():
        if study.sha(p) != digest:
            raise ValueError('frozen source changed: ' + p)
    for s in plan['selection']:
        assert study.sha(study.SOURCE / 'targets' / s['key'] / 'result.json') == s['source_result_sha256']
        assert study.sha(out / 'evidence' / (s['key'] + '.json')) == s['evidence_sha256']
    return plan


def collect(out, allow_paid):
    if not allow_paid:
        raise ValueError('explicit paid approval and --allow-paid required')
    plan = verify(out)
    if (out / 'STOP').exists() or (out / 'collection_receipt.json').exists():
        raise ValueError('existing completed/stopped run; no implicit continuation')
    import msvcrt
    lock = (out / 'runner.lock').open('a+b'); lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    budget = Budget(out / 'budget.sqlite', plan['limits'], study.sha(out / 'plan.json'))
    stop = threading.Event()
    T.reconcile(out, budget)
    try:
        with frozen.joint.credential_context(plan), frozen.joint.roster.lightweight_protocol():
            for phase in ('scheme5', 'scheme6'):
                for i, s in enumerate(plan['selection']):
                    arms = ('control', 'prior') if i % 2 == 0 else ('prior', 'control')
                    if phase == 'scheme6':
                        arms = ('qwen',)
                    source = study.read(study.SOURCE / 'targets' / s['key'] / 'result.json')
                    base = T.old.load_base(s); h0 = source['h0']
                    prior = study.read(study.SOURCE / 'priors' / (s['video_id'] + '.json'))
                    packet = study.read(out / 'evidence' / (s['key'] + '.json'))
                    for arm in arms:
                        target = s['key'] + '__' + arm; folder = out / 'targets' / target
                        if (folder / 'responses.json').exists():
                            continue
                        calls = GuardedCalls(out, plan, {**s, 'key': target}, budget, stop)
                        try:
                            if arm == 'qwen':
                                data = study.read(out / 'targets' / (s['key'] + '__control') / 'responses.json')
                                pool = data['pool']; data['reused_from'] = s['key'] + '__control'
                                wire = compact_wire(base, s, h0, pool, packet, arm, 'qwen')
                                data['reviews']['qwen'] = calls.call(target, 'control_graph', 'qwen', wire)
                            else:
                                pool = make_pool(h0)
                                wire = proposal_wire(base, s, h0, pool, prior, set(s['tracker_classes']), arm)
                                proposal = calls.call(target, 'proposal', 'base', wire)
                                if proposal is not None:
                                    try:
                                        pool = make_pool(h0, proposal, pool)
                                    except ValueError:
                                        pass
                                reviews = {}
                                def query(seat):
                                    w = compact_wire(base, s, h0, pool, packet, arm, seat)
                                    return seat, calls.call(target, 'control_graph', seat, w)
                                seat, value = query('qwen'); reviews[seat] = value
                                with ThreadPoolExecutor(max_workers=4) as workers:
                                    for seat, value in workers.map(query, SEATS[1:]):
                                        reviews[seat] = value
                                data = {'proposal': proposal, 'pool': pool, 'reviews': reviews}
                            study.write(folder / 'responses.json', data)
                        finally:
                            calls.close()
                        if stop.is_set():
                            raise BudgetStop('global stop')
                    print(json.dumps({'phase': phase, 'targets_completed': i + 1, 'budget': budget.summary()}), flush=True)
        study.write(out / 'collection_receipt.json', {'state': 'COMPLETED', 'rows': len(plan['selection']), 'budget': budget.summary()})
    except Exception as exc:
        (out / 'STOP').write_text(type(exc).__name__ + ': inspect durable journal before continuation', encoding='utf-8')
        raise
    finally:
        budget.close(); lock.close()


def score(out):
    plan = verify(out)
    if study.read(out / 'collection_receipt.json')['state'] != 'COMPLETED':
        raise ValueError('complete collection required')
    rows = {r['sample_id']: r for r in study.read(study.ROWS)}
    with np.load(study.DEFAULT / 'predictions.npz') as z:
        phase = dict(zip(z['ids'].tolist(), z['phase_nested'].tolist()))
    gate, _, _ = predictor(); allcounts = {a: [] for a in ('control', 'prior', 'qwen')}; predictions = []
    ledger = study.read(study.BASE / 'official_behavior_net_v2_20260913/costs.json')['per_call_main_attempt']
    account_costs = {a: Counter() for a in allcounts}
    rawcounts = {a: [] for a in allcounts}; coverage = {a: Counter() for a in allcounts}
    def decision(features):
        score, action = gate(features)
        return score, int(bool(action))
    def record_for(key, arm, stage, seat):
        if arm == 'qwen' and seat != 'qwen':
            arm = 'control'
        folder = out / 'targets' / (key + '__' + arm)
        if seat in T.CHANGED:
            return study.read(folder / 'changed' / (stage + '_' + seat) / 'record.json')
        return next(r for r in study.read(folder / 'run/budget.json')['calls'] if r['stage'] == stage and r['seat'] == seat)
    with offline():
        for s in plan['selection']:
            source = study.read(study.SOURCE / 'targets' / s['key'] / 'result.json')
            prior = study.read(study.SOURCE / 'priors' / (s['video_id'] + '.json'))
            record = {'key': s['key'], 'arms': {}}
            for arm in allcounts:
                data = study.read(out / 'targets' / (s['key'] + '__' + arm) / 'responses.json')
                updated = {**source, 'proposal_raw': data['proposal'], 'review_raw': data['reviews']}
                result = run_interaction(CachedBackend(updated), s, prior, decision)
                prediction, trace = output_modules(result['prediction'], set(s['tracker_classes']))
                prediction['phase'] = [phase[s['key']]]
                allcounts[arm].append(study.counts(prediction, rows[s['key']]['gt']))
                rawcounts[arm].append(study.counts(result['prediction'], rows[s['key']]['gt']))
                pool_ivts = {p['label_id'] for p in data['pool']['propositions'] if p['task'] == 'ivt'}
                truth_ivts = set(rows[s['key']]['gt']['ivt'])
                coverage[arm].update(true_ivts_covered=len(pool_ivts & truth_ivts), true_ivts_total=len(truth_ivts),
                                     candidate_ivts=len(pool_ivts), false_candidate_ivts=len(pool_ivts - truth_ivts))
                account_costs[arm]['logical_calls'] += result['logical_calls']
                for call in result['call_keys']:
                    stage, seat = call.split('|')
                    if stage == 'h0':
                        account_costs[arm]['openrouter_usd'] += ledger[call].get('usd_per_call', 0.)
                        continue
                    transport_record = record_for(s['key'], arm, stage, seat)
                    account_costs[arm][transport_record['account']] += float(transport_record['charge'])
                    if transport_record.get('charge_kind') == 'unknown_reserved':
                        account_costs[arm]['unknown_charge_calls'] += 1
                record['arms'][arm] = {'prediction': prediction, 'before_output_modules': result['prediction'],
                                       'call_keys': result['call_keys'], 'output_modules': trace,
                                       'gate_score': result['gate_score'], 'gate_action': result['gate_action']}
            predictions.append(record)
    report = {'rows': len(predictions), 'api_calls': 0, 'default_changed': False,
              'scope': 'Small reused-Training migration pilot; not an independently calibrated Gate or validation cohort',
              'arms': {},
              'collection': study.read(out / 'collection_receipt.json')}
    videos = np.array([s['video_id'] for s in plan['selection']])
    for arm, c in allcounts.items():
        c = np.array(c)
        report['arms'][arm] = {**study.quality(c), 'before_output_modules': study.quality(np.array(rawcounts[arm])),
                              'per_video': {v: study.quality(c[videos == v]) for v in sorted(set(videos))},
                              'cost_accounts_per_inference_path': dict(account_costs[arm]),
                              'candidate_coverage': dict(coverage[arm])}
    control = report['arms']['control']; checks = {}
    for arm in ('prior', 'qwen'):
        trial = report['arms'][arm]
        tests = {'f1_not_lower': trial['f1'] + 1e-12 >= control['f1'],
                 'errors_not_higher': trial['errors'] <= control['errors'],
                 'strict_quality_gain': trial['f1'] > control['f1'] + 1e-12 or trial['errors'] < control['errors'],
                 'every_video_not_worse': all(q['f1'] + 1e-12 >= control['per_video'][v]['f1'] and q['errors'] <= control['per_video'][v]['errors'] for v, q in trial['per_video'].items()),
                 'known_usd_within_10_percent': account_costs[arm]['openrouter_usd'] <= 1.1 * account_costs['control']['openrouter_usd'],
                 'no_unknown_charge_calls': account_costs[arm]['unknown_charge_calls'] == account_costs['control']['unknown_charge_calls'] == 0}
        checks[arm] = {'checks': tests, 'pilot_screen_pass': all(tests.values())}
    report['paired_checks'] = checks
    changed_keys = set(study.read(out / 'preflight.json')['prior_prompt_changed_frames'])
    changed_mask = np.array([s['key'] in changed_keys for s in plan['selection']])
    report['scheme5_prompt_changed_subset'] = {
        'rows': int(changed_mask.sum()), 'keys': sorted(changed_keys),
        'arms': {a: study.quality(np.array(allcounts[a])[changed_mask]) for a in ('control', 'prior')} if changed_mask.any() else {},
        'interpretation': 'Only this subset changes the scheme5 input; identical-request stochastic differences are not Tracker evidence.'}
    report['cost_note'] = 'Virtual inference costs include shared cached replies in each arm and a historical H0 USD estimate. CNY and GLM/DeepSeek request counts are separate; USD is not all-provider total cost.'
    study.write(out / 'predictions.json', predictions)
    study.write(out / 'quality_report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'collect', 'score'))
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--allow-paid', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    elif args.command == 'collect':
        collect(args.output, args.allow_paid)
    else:
        score(args.output)
