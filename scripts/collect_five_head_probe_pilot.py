"""Fresh five-head Training collection: 32-frame pilot or all 6,059 frames."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import run_testing_half_pipeline as core
from scripts.assess_tracker_scheme4 import ROWS, SOURCE, load_trackers
from scripts.collect_gate_escalation_v1 import load_base
from scripts.run_pgp_pipeline import WireBackend
from surgical_agent.research.gate.collection_budget import Budget
from surgical_agent.research.gate.tracker_pipeline_v2 import output_modules, null_label_cleanup
from tqdm import tqdm


def choose(rows):
    chosen = []
    videos = sorted({r['video_id'] for r in rows})
    assert len(videos) == 4 and all(r['source_split'] == 'Training' for r in rows)
    for video in videos:
        available = sorted((r for r in rows if r['video_id'] == video), key=lambda r: r['frame_id'])
        chosen.extend(available[((2*i+1)*len(available))//16] for i in range(8))
    assert len({r['sample_id'] for r in chosen}) == 32
    return chosen


def yes(features):
    assert len(features) == 54
    return 1., 1


yes.phase_review_enabled = True
yes.review_mode = 'five_head_probe'


def reusable_results(folder):
    """Reuse only sealed results from the identical inference protocol."""
    if folder is None:
        return {}, None
    folder = folder.resolve()
    receipt, plan = core.read(folder/'receipt.json'), core.read(folder/'plan.json')
    assert receipt['state'] == 'PASS' and receipt['review_mode'] == 'five_head_probe'
    assert core.sha(folder/'rows.jsonl') == receipt['rows_sha256']
    for rel, digest in plan['runtime_sha256'].items():
        # Collector scheduling may change; model inputs and inference must not.
        if rel.replace('\\', '/') != 'scripts/collect_five_head_probe_pilot.py':
            assert core.sha(ROOT/rel) == digest, 'Cached inference protocol changed: '+rel
    rows = [json.loads(line) for line in (folder/'rows.jsonl').read_text('utf-8').splitlines()]
    assert len(rows) == receipt['rows'] and len({r['sample_id'] for r in rows}) == len(rows)
    for row in rows:
        assert row['source_split'] == 'Training' and len(row['features']) == 54
        assert row['source_sha256'] == plan['source_sha256'][row['sample_id']]
    return {r['sample_id']: r for r in rows}, {'directory':str(folder),
        'rows_sha256':receipt['rows_sha256'], 'plan_sha256':core.sha(folder/'plan.json')}


class PilotBackend(WireBackend):
    def __init__(self, calls, base, selected, record):
        super().__init__(calls, base, selected)
        self.record = record

    def h0(self):
        return deepcopy(self.record['h0_raw'])

    def proposal(self, *args):
        return deepcopy(self.record['proposal_raw'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    p.add_argument('--allow-paid', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--all-training', action='store_true', help='Collect all 6,059 Training frames')
    p.add_argument('--reuse-pilot', type=Path, help='Reuse a completed matching five-head pilot')
    p.add_argument('--check-only', action='store_true', help='Validate inventory and reuse without API calls')
    args = p.parse_args()
    if not args.allow_paid and not args.check_only:
        p.error('--allow-paid required to collect fresh responses')
    out = args.output.resolve()
    started = time.perf_counter()
    inventory = core.read(ROWS)
    assert len(inventory) == 6059 and all(r['source_split'] == 'Training' for r in inventory)
    rows = inventory if args.all_training else choose(inventory)
    reused, reuse_binding = reusable_results(args.reuse_pilot)
    assert set(reused) <= {r['sample_id'] for r in rows}
    if args.check_only:
        # Exercise loading a real input as well as validating sealed cached responses.
        source_selection = {s['key']:s for s in core.read(SOURCE/'plan.json')['selection']}
        load_base(source_selection[rows[0]['sample_id']])
        print(json.dumps({'state':'PASS','rows':len(rows),'reused':len(reused),
            'new_frames':len(rows)-len(reused),'api_calls':0}), flush=True)
        return
    out.mkdir(parents=True, exist_ok=args.resume)
    if (out/'receipt.json').exists():
        receipt = core.read(out/'receipt.json')
        assert receipt['state'] == 'PASS' and receipt['rows'] == len(rows)
        assert receipt['rows_sha256'] == core.sha(out/'rows.jsonl')
        print('Collection already complete; no new API calls', flush=True)
        return
    videos = sorted({r['video_id'] for r in rows})
    trackers, tracker_hashes = load_trackers(videos)
    source_plan = core.read(SOURCE/'plan.json')
    selection = {s['key']: s for s in source_plan['selection']}
    sealed = core.read(ROOT/'artifacts/training/gate/pgp_gate_no_tracker_20260913_r1/replay/pgp_reviewer_source_sha256.json')
    inputs, priors = {}, {}
    for row in rows:
        key = row['sample_id']
        selected = deepcopy(selection[key])
        for field in ('reuse_cache', 'reuse_cache_sha256'):
            selected.pop(field, None)
        selected['source_split'] = 'Training'
        inputs[key] = selected
    for video in videos:
        prior_path = SOURCE/'priors'/(video+'.json')
        assert core.sha(prior_path) == source_plan['prior_sha256'][video]
        priors[video] = core.read(prior_path)
        assert priors[video]['excluded_video'] == video and video not in priors[video]['fit_videos']
    plan = {k: deepcopy(source_plan[k]) for k in ('base_rates', 'call_rates', 'credential_root',
        'endpoints', 'models', 'reviewer_config', 'reviewer_config_sha256')}
    plan.update(profile='five_head_probe_training_collection_v2', review_mode='five_head_probe',
        selection=list(inputs.values()), workers=args.workers, maximum_paid_calls=5*(len(rows)-len(reused)),
        limits={'openrouter_usd':'60' if args.all_training else '10',
                'aliyun_cny':'30' if args.all_training else '5', 'glm_requests':str(len(rows)),
                'deepseek_requests':str(len(rows)), 'xai_usd':'0'},
        sampling_rule='All 6059 Training rows in original order' if args.all_training else 'Eight temporal quantile midpoints per Training video; no truth-based selection',
        reused_pilot=reuse_binding,
        forced_review=True, adaptive_exact_early_stop=True, reports=False,
        cached=['h0', 'proposal', 'tracker'], tracker_sha256=tracker_hashes,
        source_sha256={k: sealed[str((SOURCE/'targets'/k/'result.json').resolve())] for k in inputs},
        runtime_sha256={str(path.relative_to(ROOT)):core.sha(path) for path in (
            Path(__file__), ROOT/'scripts/run_pgp_pipeline.py', ROOT/'scripts/run_testing_half_pipeline.py',
            ROOT/'src/surgical_agent/research/gate/tracker_pipeline_v2.py',
            ROOT/'src/surgical_agent/research/gate/unified_review.py',
            ROOT/'src/surgical_agent/research/verification/prompts/five_head_probe_v1.txt')})
    if (out/'plan.json').exists():
        assert core.read(out/'plan.json') == plan, 'Resume plan drift'
    else:
        core.write(out/'plan.json', plan)
    stop = threading.Event()
    budget = Budget(out/'budget.sqlite', plan['limits'], core.sha(out/'plan.json'))

    def run(row):
        key, video = row['sample_id'], row['video_id']
        destination = out/'results'/(key+'.json')
        if destination.exists():
            saved = core.read(destination)
            assert saved['sample_id'] == key and saved['source_sha256'] == plan['source_sha256'][key]
            return saved
        if key in reused:
            saved = deepcopy(reused[key])
            assert saved['source_sha256'] == plan['source_sha256'][key]
            saved['reused_from'] = reuse_binding['directory']
            core.write(destination, saved)
            return saved
        if stop.is_set():
            raise RuntimeError('Collection stopped before target dispatch')
        tick = time.perf_counter()
        selected = inputs[key]
        delegate = core.GuardedCalls(out, plan, selected, budget, stop)
        timings = []
        class Calls:
            def call(self, target, stage, seat, body):
                assert target == key and stage == 'five_head_v1'
                assert seat in core.tracker_pipeline.original.SEATS
                assert seat not in [r['seat'] for r in timings] and len(timings) < 5
                start = time.perf_counter()
                answer = delegate.call(target, stage, seat, body)
                timings.append({'seat':seat, 'seconds':time.perf_counter()-start,
                                'response_present':answer is not None})
                core.write(out/'progress'/(key+'.json'), {'sample_id':key, 'calls':timings})
                return answer
        try:
            path = SOURCE/'targets'/key/'result.json'
            assert core.sha(path) == plan['source_sha256'][key], key
            # Bound image memory to the active workers rather than all Training frames.
            backend = PilotBackend(Calls(), load_base(selected), selected, core.read(path))
            ts = trackers[video].get(row['frame_id'])
            snapshot = None if ts is None else {'status':'AVAILABLE','video_id':video,
                'source_max_frame_id':row['frame_id'], 'frames':[{'frame_id':row['frame_id'],
                'tracks':[{'instrument_id':c,'score':1.0} for c in sorted(ts)]}]}
            result = core.run_target(backend, selected, priors[video], yes,
                tracker_snapshot=snapshot, phase_filter=core.CausalPhaseFilter(0), output='v2.2')
            cheap, _ = output_modules(result['cheap'], ts, ontology_filter=True)
            cheap, _ = null_label_cleanup(cheap, result['probe_ratings'])
            assert result['compact_depth'] == 0 and result['joint_depth'] == len(timings)
            assert [t['seat'] for t in timings].count('qwen') == 1
            result.update(sample_id=key, source_split='Training', video_id=video, frame_id=row['frame_id'],
                cheap=cheap, reviewed=result['prediction'], source_sha256=plan['source_sha256'][key],
                five_head_raw=backend.five_head_raw, request_timings=timings,
                elapsed_seconds=time.perf_counter()-tick, tracker_reused=True)
            core.write(destination, result)
            return result
        except BaseException as exc:
            stop.set()
            core.write(out/'errors'/(key+'.json'), {'error_type':type(exc).__name__, 'error':str(exc)})
            raise
        finally:
            delegate.close()

    try:
        results = []
        collection_start = time.perf_counter()
        print(f'{len(rows)} Training targets; {len(reused)} reused pilot results; cached H0/proposal/Tracker', flush=True)
        with core.app.frozen.joint.credential_context(plan), core.app.frozen.joint.roster.lightweight_protocol():
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run, row) for row in rows]
                with tqdm(total=len(rows), desc='Five-head collection', unit='frame', mininterval=1, dynamic_ncols=True) as bar:
                    for future in as_completed(futures):
                        results.append(future.result())
                        bar.update(1)
        order = {row['sample_id']:i for i,row in enumerate(rows)}
        results.sort(key=lambda r: order[r['sample_id']])
        with (out/'rows.jsonl').open('w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False)+'\n')
        receipt = {'state':'PASS', 'rows':len(rows), 'reused_pilot_rows':len(reused), 'review_mode':'five_head_probe', 'output_modules':'v2.2',
            'workers':args.workers, 'rows_sha256':core.sha(out/'rows.jsonl'), 'budget':budget.summary(),
            'elapsed_seconds':time.perf_counter()-started, 'collection_seconds':time.perf_counter()-collection_start,
            'joint_depths':dict(Counter(r['joint_depth'] for r in results)),
            'missing_responses':sum(not t['response_present'] for r in results for t in r['request_timings']),
            'scope':'Training collection; no production Gate fitted or activated',
            'budget_scope':'New dispatches only; reused pilot charges excluded'}
        core.write(out/'receipt.json', receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    except BaseException as exc:
        stop.set()
        core.write(out/'failure.json', {'error_type':type(exc).__name__, 'error':str(exc), 'budget':budget.summary()})
        raise
    finally:
        budget.close()


if __name__ == '__main__':
    main()
