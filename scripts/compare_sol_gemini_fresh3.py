"""Fixed prospective three-target paired follow-up; no retries or Gate collection."""
from __future__ import annotations
import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import compare_mainline_backbones as t
old, main = t.old, t.main
SOURCE = ROOT / 'artifacts/preflight/mainline_backbone_training6_20260911_v2'
OUT = ROOT / 'artifacts/preflight/sol_gemini_fresh3_20260911_v1'
NAMES = ('sol', 'gemini')


def run_target(calls, base, s, plan):
    row = {k: s[k] for k in ('key', 'video_id', 'frame_id')}
    start = perf_counter()
    try:
        record = main.run_target(calls, base, s, old.read(OUT / 'priors' / f"{s['video_id']}.json"), plan['gate'])
    except (ValueError, TypeError, KeyError, t.ApiSchemaError) as exc:
        row.update(status='TARGET_FAILED', error_type=type(exc).__name__, predictions={a: None for a in main.ARMS})
    else:
        row.update(status='PREDICTED', predictions=record['predictions'], timing=record['timing_seconds'])
    row['seconds'] = perf_counter() - start
    return row


def run():
    if OUT.exists():
        raise ValueError('Fresh archive required; no automatic reruns')
    original = t.verify(SOURCE)
    adapter = old.common.CholecTrack20DatasetAdapter(t.release.DATASET, causal_window_size=3)
    # Exclude known selected frames in archived preflight plans, without inspecting outcomes.
    used = defaultdict(set)
    for path in (ROOT / 'artifacts/preflight').rglob('plan.json'):
        try:
            data = old.read(path)
        except (ValueError, OSError):
            continue
        for s in data.get('selection', []):
            if isinstance(s, dict) and 'video_id' in s and 'frame_id' in s:
                used[s['video_id']].update(s.get('causal_frame_ids', [s['frame_id']]))
    selection, bases = [], {}
    for video in ('VID103', 'VID23', 'VID31'):
        samples = list(adapter.iter_inference_video(video))
        masks = {r.inference.target_frame_id: old.truth_row(r)['mask'] for r in adapter.iter_video(video)}
        anchor = samples[len(samples) // 4].target_frame_id
        eligible = [s for s in samples if len(s.causal_frame_ids) == 3
                    and s.source_split is t.DatasetSplit.TRAINING
                    and all(masks.get(s.target_frame_id, {}).get(task, False) for task in old.TASKS)
                    and all(abs(f - x) > 175 for f in s.causal_frame_ids for x in used[video])]
        sample = min(eligible, key=lambda s: (abs(s.target_frame_id - anchor), s.target_frame_id))
        base = t.release.make_base(sample)
        key = f'{video}_{sample.target_frame_id}'
        images = []
        for fid, im in zip(sample.causal_frame_ids, base.images, strict=True):
            p = OUT / 'images' / video / f'{fid}.png'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(im.content)
            images.append({'path': str(p), 'sha256': old.sha(p), 'frame_id': fid})
        selection.append({'key': key, 'video_id': video, 'frame_id': sample.target_frame_id,
                          'causal_frame_ids': list(sample.causal_frame_ids), 'images': images})
        bases[key] = base
        dest = OUT / 'priors' / f'{video}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / 'priors' / dest.name, dest)
    unchanged = {n: old.sha(ROOT / n) for n in ('DEFAULT_PIPELINE_VERSION.json', 'BEST_PIPELINE_VERSION.json', 'TRAINING_BASE_MODEL_SELECTION.json')}
    plan = deepcopy(original)
    plan.update(selection=selection, profile='sol_gemini_fresh3', max_calls=78,
                prior_source=str(SOURCE), unchanged_root_hashes=unchanged,
                selection_rule='One quarter-time Training target per VID103/23/31; all masks valid; >175 frame gap from selected frames in archived preflight plans; no selection on label values or predictions.',
                comparison='Sol vs Gemini, low reasoning, unchanged five reviewers and mechanism; two models parallel per target; no retries; failure predictions empty; rank by final mean F1 then IVT F1 among complete models. Report prior six separately.',
                gate_data_collected=False)
    old.save(OUT / 'plan.json', plan)
    shutil.copyfile(__file__, OUT / 'frozen_runner.py')
    checks = []
    with old.joint.roster.lightweight_protocol():
        for s in selection:
            wires = {}
            for n in NAMES:
                mock = t.Mock(n)
                row = run_target(mock, bases[s['key']], s, plan)
                assert row['status'] == 'PREDICTED'
                assert Counter(r['stage'] for r in mock.rows) == Counter(main.STAGES)
                wires[n] = deepcopy(mock.wires)
            for key in wires['sol']:
                for n in NAMES:
                    if key[1] == 'base':
                        for field in ('model', 'provider', 'temperature'):
                            wires[n][key].pop(field, None)
                assert wires['sol'][key] == wires['gemini'][key]
            checks.append({'target': s['key'], 'paired_mock_wire_equal': True, 'mock_calls': 26})
    old.save(OUT / 'preflight.json', {'checks': checks, 'plan_sha256': old.sha(OUT / 'plan.json')})
    print(json.dumps({'selected': checks, 'max_paid_calls': 78}), flush=True)
    rows = {n: [] for n in NAMES}
    start = perf_counter()
    with old.joint.credential_context(plan), old.joint.roster.lightweight_protocol():
        calls = {n: t.ModelCalls(OUT / n, n, plan) for n in NAMES}
        for c in calls.values():
            c.max_calls = 39
        try:
            for s in selection:
                def one(n):
                    row = run_target(calls[n], bases[s['key']], s, plan)
                    print(json.dumps({'model': n, 'target': s['key'], 'status': row['status'], 'seconds': row['seconds']}), flush=True)
                    return n, row
                with ThreadPoolExecutor(max_workers=2) as pool:
                    for n, row in pool.map(one, NAMES):
                        rows[n].append(row)
                        old.save(OUT / n / 'predictions.json', {'targets': rows[n]})
        finally:
            for c in calls.values():
                c.stopped = True
                try:
                    c.persist()
                finally:
                    c.close_ledger()
    old.save(OUT / 'completion.json', {'wall_seconds': perf_counter()-start,
        'evidence_sha256': {str(p.relative_to(OUT)): old.sha(p) for p in OUT.rglob('*.json')}})
    with old.joint.roster.lightweight_protocol():
        for n in NAMES:
            replay = t.Replay(OUT / n, n)
            for s, row in zip(selection, rows[n], strict=True):
                rr = run_target(replay, bases[s['key']], s, plan)
                assert rr['status'] == row['status'] and rr['predictions'] == row['predictions']
            assert replay.used == set(replay.rows)
    truth = {}
    for s in selection:
        r = next(r for r in adapter.iter_video(s['video_id']) if r.inference.target_frame_id == s['frame_id'])
        truth[s['key']] = old.truth_row(r)
    old.save(OUT / 'scored_truth.json', truth)
    results = {}
    for n in NAMES:
        charges, errors = defaultdict(Decimal), []
        budget = old.read(OUT / n / 'budget.json')
        for c in budget['calls']:
            charges[c['account']+'|'+c['charge_kind']] += Decimal(c['charge'])
            if c['status'] not in t.VALID:
                errors.append({k: c.get(k) for k in ('target','stage','seat','status','http_status')})
        results[n] = {'completed': sum(r['status']=='PREDICTED' for r in rows[n]),
                      'metrics': t.release.metrics(rows[n], truth), 'calls': len(budget['calls']),
                      'charges': {k: str(v) for k,v in charges.items()}, 'errors': errors,
                      'target_seconds': sum(r['seconds'] for r in rows[n])}
    assert all(old.sha(ROOT / n) == h for n,h in unchanged.items())
    report = {'models': results, 'wall_seconds': old.read(OUT / 'completion.json')['wall_seconds'],
              'gate_data_collected': False, 'root_pointers_unchanged': True, 'raw_replay_passed': True}
    old.save(OUT / 'comparison.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    run()
