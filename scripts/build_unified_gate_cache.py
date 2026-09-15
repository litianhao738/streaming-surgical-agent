"""Replay joint-panel Training responses through the shared runtime; no network."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import run_testing_half_pipeline as core
from scripts.assess_tracker_scheme4 import ROWS, SOURCE, load_trackers
from scripts.run_pgp_pipeline import CachedBackend
from surgical_agent.research.gate.tracker_pipeline_v2 import output_modules, null_label_cleanup
from tools.audit.gate_proposals_offline_20260913 import offline


def replay_item(item):
    row, selected, prior, ts, expected = item
    key = row['sample_id']
    path = SOURCE/'targets'/key/'result.json'
    digest = core.sha(path)
    assert digest == expected, key
    record = core.read(path)
    assert set(record['joint_raw']) == set(core.tracker_pipeline.original.SEATS), key
    def yes(_): return 1., 1
    yes.phase_review_enabled = True
    yes.review_mode = 'unified'
    snapshot = None if ts is None else {'status':'AVAILABLE','video_id':row['video_id'],
        'source_max_frame_id':row['frame_id'], 'frames':[{'frame_id':row['frame_id'],
        'tracks':[{'instrument_id':c,'score':1.0} for c in sorted(ts)]}]}
    with offline():
        tracked = core.run_target(CachedBackend(record), selected, prior, yes,
            tracker_snapshot=snapshot, phase_filter=core.CausalPhaseFilter(0), output='v2.2')
    cheap, _ = output_modules(tracked['cheap'], ts, ontology_filter=True)
    cheap, _ = null_label_cleanup(cheap, tracked['probe_ratings'])
    assert tracked['compact_depth'] == 1 and tracked['joint_depth'] >= 1
    assert not any(k.startswith('control_graph|') and k != 'control_graph|qwen' for k in tracked['call_keys'])
    return {'sample_id':key,'source_split':'Training','features':tracked['features'],
        'cheap':cheap,'reviewed':tracked['prediction'],'joint_depth':tracked['joint_depth'],
        'logical_calls':tracked['logical_calls'],'source_sha256':digest,
        'schema_fallbacks':tracked.get('schema_fallbacks',[])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4: parser.error('--workers must be 1..4')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if (args.output/'receipt.json').exists(): raise ValueError('Replay already complete')
    start = time.perf_counter()
    rows = core.read(ROWS)
    assert len(rows) == 6059 and all(r['source_split'] == 'Training' for r in rows)
    videos = sorted({r['video_id'] for r in rows})
    trackers, tracker_hashes = load_trackers(videos)
    priors = {v: core.read(SOURCE/'priors'/(v+'.json')) for v in videos}
    selection = {r['key']: r for r in core.read(SOURCE/'plan.json')['selection']}
    sealed = core.read(ROOT/'artifacts/training/gate/pgp_gate_no_tracker_20260913_r1/replay/pgp_reviewer_source_sha256.json')
    output = args.output/'rows.jsonl'
    counts = {}; source_hashes = {}
    completed = []
    if output.exists():
        if not args.resume: raise ValueError('Use --resume for interrupted offline replay')
        raw = output.read_bytes()
        end = raw.rfind(b'\n') + 1
        completed = [json.loads(line) for line in raw[:end].splitlines()]
        assert [r['sample_id'] for r in completed] == [r['sample_id'] for r in rows[:len(completed)]]
        # Discard only an unfinished trailing line from this interrupted local output.
        if end != len(raw):
            with output.open('r+b') as f: f.truncate(end)
        for r in completed:
            key = r['sample_id']; path = SOURCE/'targets'/key/'result.json'
            assert r['source_sha256'] == sealed[str(path.resolve())]
            source_hashes[key] = r['source_sha256']
            counts[r['joint_depth']] = counts.get(r['joint_depth'],0)+1
    def items():
        for row in rows[len(completed):]:
            selected = deepcopy(selection[row['sample_id']]); selected['source_split'] = 'Training'
            path = SOURCE/'targets'/row['sample_id']/'result.json'
            yield (row, selected, priors[row['video_id']], trackers[row['video_id']].get(row['frame_id']), sealed[str(path.resolve())])
    print(f'Resuming at {len(completed)}/{len(rows)} with {args.workers} local workers', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool, output.open('a', encoding='utf-8') as stream:
        for i, result in enumerate(pool.map(replay_item, items(), chunksize=8), len(completed)):
            stream.write(json.dumps(result)+'\n'); stream.flush()
            counts[result['joint_depth']] = counts.get(result['joint_depth'],0)+1
            source_hashes[result['sample_id']] = result['source_sha256']
            if (i+1) % 500 == 0: print(f'Unified cached replay {i+1}/{len(rows)}',flush=True)
    core.write(args.output/'receipt.json', {'state':'PASS','rows':len(rows),'api_calls':0,
        'review_mode':'unified','output_modules':'v2.2','rows_sha256':core.sha(output),'joint_depths':counts,
        'tracker_sha256':tracker_hashes,'source_sha256':source_hashes,'elapsed_seconds':time.perf_counter()-start,
        'runtime_sha256':{str(p.relative_to(ROOT)):core.sha(p) for p in [Path(__file__),
            ROOT/'scripts/run_testing_half_pipeline.py', ROOT/'src/surgical_agent/research/gate/tracker_pipeline_v2.py',
            ROOT/'src/surgical_agent/research/gate/unified_review.py']}})
    print(json.dumps({'state':'PASS','rows':len(rows),'joint_depths':counts,'seconds':time.perf_counter()-start}),flush=True)


if __name__ == '__main__': main()
