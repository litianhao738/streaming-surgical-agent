"""Bounded paired Training collection for the five-head mainline Gate.

Prepare and preflight are offline except public provider metadata. Execute
collects one H0 and at most twelve frozen repair calls per target. Score joins
GT only after inference has been sealed and all requests replayed locally.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from time import perf_counter

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import compare_mainline_backbones as trial
from scripts import prepare_final_only_gate_training as legacy
from scripts import run_gate_ready_mainline as ready
from surgical_agent.research.gate import mainline_training as contract

old, main = trial.old, trial.main
SOURCE = ROOT / 'artifacts/preflight/mainline_backbone_training6_20260911_v2'
PROFILE = 'mainline_five_head_gate_collection_v1'
VIDEOS = ('VID103', 'VID23', 'VID31', 'VID96')
LIMITS = {'openrouter_usd': '15', 'aliyun_cny': '12', 'xai_usd': '0'}
OOF = ROOT / 'artifacts/training/tracker_clip_v2_oof5_20260906/oof/index.json'


def plain(value):
    """Snapshot providers freeze mappings; materialize JSON without changing values."""
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def feature_row(base, raw, s, plan, snapshot):
    h0 = old.gated.h0_from_raw(raw)
    features = contract.extract_features(h0, target_frame_id=s['frame_id'],
        causal_frame_ids=s['causal_frame_ids'], tracker_snapshot=snapshot)
    return {'sample_id': s['key'], 'video_id': s['video_id'], 'frame_id': s['frame_id'],
            'source_split': 'Training', 'policy_id': plan['policy_id'],
            'feature_version': contract.FEATURE_VERSION, 'label_version': contract.LABEL_VERSION,
            'features_no_tracker': contract.mask_tracker(features), 'features_with_tracker': features,
            'oof_artifact_sha256': snapshot['artifact_sha256'], 'h0_labels': h0,
            'tracker_snapshot_sha256': s['tracker_snapshot_sha256']}


def repair_observation(record, call_rows):
    """Distinguish completed HTTP/JSON calls from usable repair supervision."""
    expected = {('h0', 'base'), ('proposal', 'base'), ('phase_recommendation', 'base')}
    expected |= {(stage, seat) for stage in ('control_graph', 'joint_r1') for seat in old.SEATS}
    transport = (len(call_rows) == 13
                 and {(r.get('stage'), r.get('seat')) for r in call_rows} == expected
                 and all(r.get('status') in trial.VALID for r in call_rows))
    reasons = [] if transport else ['transport_incomplete']
    if not isinstance(record, dict):
        reasons.append('result_missing')
    else:
        proposal = record.get('proposal_raw')
        if not isinstance(proposal, dict) or 'invalid' in proposal:
            reasons.append('proposal_invalid')
        rec = record.get('phase_recommendation')
        if (not isinstance(rec, dict) or 'error' not in rec or rec['error'] is not None
                or old.joint.phase_choice_error(rec.get('raw'), 3) is not None):
            reasons.append('phase_recommendation_invalid')
        for prefix, pool_name in (('', 'pool'), ('joint_', 'joint_pool')):
            formats = record.get(prefix+'format_diagnostics')
            if not isinstance(formats, dict) or set(formats) != set(old.SEATS):
                reasons.append(prefix+'format_missing')
            else:
                for seat, diag in formats.items():
                    extra = 'unbound_rows' if prefix else 'envelope_errors'
                    if (not isinstance(diag, dict) or not isinstance(diag.get('errors'), dict)
                            or not isinstance(diag.get(extra), list)
                            or any(diag.get(k) for k in ('errors','envelope_errors','unbound_rows'))):
                        reasons.append(prefix+'format_invalid:'+seat)
            raw_pool = record.get(pool_name)
            pool = raw_pool.get('propositions') if isinstance(raw_pool, dict) else None
            if not isinstance(pool, list) or any(not isinstance(p,dict) or 'id' not in p for p in pool):
                reasons.append(prefix+'pool_invalid')
                pool = []
            ids = {p['id'] for p in pool}
            means = record.get(prefix+'means')
            diagnostics = record.get(prefix+'diagnostics')
            if (not isinstance(means, dict) or set(means) != ids
                    or any(type(v) not in (int,float) or not math.isfinite(v) or not 1 <= v <= 5 for v in means.values())):
                reasons.append(prefix+'ratings_incomplete')
            if (not isinstance(diagnostics, dict) or set(diagnostics) != ids
                    or any(not isinstance(d,dict) or 'invalid' not in d or d['invalid'] for d in diagnostics.values())):
                reasons.append(prefix+'ratings_invalid')
        phase_decision = record.get('phase_decision')
        if not isinstance(phase_decision, dict) or phase_decision.get('valid_panel') is not True:
            reasons.append('phase_panel_incomplete')
    return {'transport_complete': transport, 'repair_fully_observed': not reasons,
            'repair_unknown_reasons': reasons}


def prepare(output, per_video=32):
    if output.exists():
        raise ValueError('Fresh output required')
    if per_video != 32:
        raise ValueError('This authorized protocol fixes 32 targets per video')
    original = trial.verify(SOURCE)
    adapter = old.common.CholecTrack20DatasetAdapter(trial.release.DATASET, causal_window_size=3)
    providers, oof_audit, oof_sources = legacy.verified_oof(OOF, trial.release.DATASET)
    excluded, history_hashes = legacy.historical_targets(ROOT / 'artifacts/preflight')
    selection_by_video, inventory = {}, {}
    for video in VIDEOS:
        samples = list(adapter.iter_inference_video(video))
        masks = {r.inference.target_frame_id: legacy.masks_only(r) for r in adapter.iter_video(video)}
        eligible = [s for s in samples if s.source_split is trial.DatasetSplit.TRAINING
                    and len(s.causal_frame_ids) == 3
                    and all(masks.get(s.target_frame_id, {}).get(t, False) for t in old.TASKS)
                    and all(abs(fid - historical) > 175 for fid in s.causal_frame_ids
                            for historical in excluded.get(video, set()))]
        # Equal-width causal windows: earliest-finish selection has maximal
        # cardinality. Uniform thinning preserves spacing and time coverage.
        maximal, used = [], set()
        for candidate in sorted(eligible, key=lambda s: s.target_frame_id):
            if all(abs(f-x) > 175 for f in candidate.causal_frame_ids for x in used):
                maximal.append(candidate)
                used.update(candidate.causal_frame_ids)
        if len(maximal) < per_video:
            raise ValueError('Insufficient new spaced targets for ' + video)
        chosen = []
        for number in range(per_video):
            sample = maximal[number * (len(maximal)-1) // (per_video-1)]
            anchor = sample.target_frame_id
            base = trial.release.make_base(sample)
            key = f'{video}_{sample.target_frame_id}'
            images = []
            for fid, image in zip(sample.causal_frame_ids, base.images, strict=True):
                path = output / 'images' / video / f'{fid}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(image.content)
                images.append({'path': str(path.resolve()), 'frame_id': fid, 'sha256': old.sha(path)})
            providers[video].reset(video)
            snapshot_path = output / 'tracker' / f'{key}.json'
            old.save(snapshot_path, plain(providers[video].snapshot(sample)))
            chosen.append({'key': key, 'video_id': video, 'frame_id': sample.target_frame_id,
                'anchor_frame_id': anchor, 'causal_frame_ids': list(sample.causal_frame_ids),
                'images': images, 'tracker_snapshot': str(snapshot_path.resolve()),
                'tracker_snapshot_sha256': old.sha(snapshot_path)})
        selection_by_video[video] = chosen
        inventory[video] = {'all_targets': len(samples), 'complete_five_head_targets': sum(all(x.values()) for x in masks.values()),
                            'eligible_after_history': len(eligible), 'maximal_spaced_capacity': len(maximal), 'selected': len(chosen)}
        prior_path = output / 'priors' / f'{video}.json'
        prior_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / 'priors' / prior_path.name, prior_path)
    # Interleave videos so the four-target warm-up covers all four distributions.
    selection = [selection_by_video[v][i] for i in range(per_video) for v in VIDEOS]
    response = requests.get('https://openrouter.ai/api/v1/models/google/gemini-3.8-flash/endpoints', timeout=30)
    response.raise_for_status()
    metadata = response.json()
    endpoint = next(e for e in metadata['data']['endpoints'] if e['tag'] == 'google-ai-studio')
    assert endpoint['status'] == 0
    assert {'reasoning','max_tokens','response_format','structured_outputs','temperature'} <= set(endpoint['supported_parameters'])
    rates = [str(max([Decimal(endpoint['pricing'][k]), *(Decimal(x.get(k, '0')) for x in endpoint['pricing'].get('overrides', []))])) for k in ('prompt', 'completion')]
    old.save(output / 'provider_metadata.json', metadata)
    sources = {p.relative_to(ROOT).as_posix(): old.sha(p) for p in trial.release.runtime_paths()}
    for rel in sources:
        dest = output / 'frozen_source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, dest)
    from surgical_agent.artifacts.manifest import sha256_mapping
    policy = {'mainline': main.VERSION, 'base': trial.MODELS['gemini'], 'reviewers': old.joint.roster.MODELS,
              'gate': original['gate'], 'phase_threshold': 4, 'source_sha256': sources,
              'feature_version': contract.FEATURE_VERSION, 'label_version': contract.LABEL_VERSION}
    plan = {**original, 'profile': PROFILE, 'created_utc': old.now(), 'selection': selection,
        'source_sha256': sources, 'policy_id': sha256_mapping(policy), 'policy': policy,
        'base_rates': {**original['base_rates'], 'gemini': rates}, 'max_calls': 13 * len(selection),
        'limits': LIMITS, 'oof_source_sha256': oof_sources, 'oof_audit': oof_audit,
        'source_prior_archive': str(SOURCE), 'inventory': inventory, 'history_plan_sha256': history_hashes,
        'feature_version': contract.FEATURE_VERSION, 'label_version': contract.LABEL_VERSION,
        'provider_metadata_sha256': old.sha(output/'provider_metadata.json'),
        'root_unchanged': {n: old.sha(ROOT/n) for n in ('DEFAULT_PIPELINE_VERSION.json','BEST_PIPELINE_VERSION.json','TRAINING_BASE_MODEL_SELECTION.json')},
        'protocol': {'selection': '32 uniformly spaced indices from earliest-finish maximal window set per Training video; full five-head masks, three causal images; >175 raw-frame gap across new windows and from historical planned targets; no label-value/outcome selection',
            'pre_api_amendment': 'v1 stopped on snapshot serialization; v2 could not select 32 VID96 targets with >225 historical-frame clearance (maximum 24). Before any paid calls or GT label inspection, reduce historical clearance to >175 and use maximal-set uniform thinning; new-window clearance unchanged.',
            'concurrency': '4 warm-up targets; up to 8 independent targets after no warm-up transport failures; panels parallel within target',
            'retries': 'No automatic API retry, resume, or target replacement. Interrupted archives remain sealed for explicit recovery.',
            'labels': 'Five-head utility; pre-verification features sealed before repair and GT. Incomplete transport or semantically invalid proposal/recommendation/panel yields unknown benefit, never a fabricated negative. Fallback outputs remain in pipeline metrics.',
            'training': 'Collection and readiness only; no model fitting or deployment.',
            'tracker': 'OOF predicted tracks, prepared sequentially; immutable snapshots used by workers; tracker affects Gate features only.'}}
    for stale in ('metadata_hashes', 'protocol_sha256', 'predeclared_standard'):
        plan.pop(stale, None)
    plan['models'] = {'gemini': trial.MODELS['gemini']}
    old.save(output / 'plan.json', plan)
    print(json.dumps({'prepared': str(output), 'targets': len(selection), 'max_calls': plan['max_calls'], 'inventory': inventory}), flush=True)


def verify(output):
    p = old.read(output / 'plan.json')
    if p['profile'] != PROFILE or p['feature_version'] != contract.FEATURE_VERSION or p['label_version'] != contract.LABEL_VERSION:
        raise ValueError('Protocol version changed')
    if old.sha(output/'provider_metadata.json') != p['provider_metadata_sha256']:
        raise ValueError('Provider metadata changed')
    for rel, h in p['source_sha256'].items():
        if old.sha(ROOT/rel) != h or old.sha(output/'frozen_source'/rel) != h:
            raise ValueError('Runtime drift: '+rel)
    for name, h in p['prior_sha256'].items():
        if name not in VIDEOS:
            continue
        prior = old.read(output/'priors'/f'{name}.json')
        if old.sha(output/'priors'/f'{name}.json') != h or name in prior['fit_videos']:
            raise ValueError('Prior drift/leakage')
    for paths in (p['oof_source_sha256'], p['evaluation_source_sha256']):
        for path, h in paths.items():
            if old.sha(path) != h:
                raise ValueError('Data/Tracker source changed: '+path)
    for s in p['selection']:
        if old.sha(s['tracker_snapshot']) != s['tracker_snapshot_sha256']:
            raise ValueError('Tracker snapshot changed')
        for im in s['images']:
            if old.sha(im['path']) != im['sha256']:
                raise ValueError('Image changed')
    return p


def preflight(output):
    plan = verify(output)
    bases = trial.bases_for(plan)
    reports = []
    with old.joint.roster.lightweight_protocol():
        for s in plan['selection']:
            prior = old.read(output/'priors'/f"{s['video_id']}.json")
            a, b, skip = trial.Mock('gemini'), trial.Mock('gemini'), trial.Mock('gemini')
            reference = main.run_target(a, bases[s['key']], s, prior, plan['gate'])
            raw = ready.generate_h0(b, bases[s['key']], s)
            features = feature_row(bases[s['key']], raw, s, plan, old.read(s['tracker_snapshot']))
            result = ready.repair_from_h0(b, bases[s['key']], s, prior, plan['gate'], raw)
            skipped = ready.run_target(skip, bases[s['key']], s, prior, plan['gate'], verify=False)
            assert a.wires == b.wires and reference['predictions'] == result['predictions']
            assert Counter(r['stage'] for r in b.rows) == Counter(main.STAGES)
            assert len(skip.rows) == 1 and all(v == reference['h0'] for v in skipped['predictions'].values())
            assert features['features_no_tracker'] == contract.mask_tracker(features['features_with_tracker'])
            reports.append({'key': s['key'], 'all_on_wire_and_prediction_identical': True, 'all_off_calls': 1, 'on_calls': 13, 'tracker_features_valid': True})
    old.save(output/'preflight.json', {'plan_sha256': old.sha(output/'plan.json'), 'api_calls': 0, 'targets': reports})
    print(json.dumps({'preflight_passed': len(reports), 'paid_calls': 0}), flush=True)


def execute(output, workers=8):
    plan = verify(output)
    check = old.read(output/'preflight.json')
    if check['plan_sha256'] != old.sha(output/'plan.json') or len(check['targets']) != 128:
        raise ValueError('Preflight incomplete')
    if workers not in (4, 8):
        raise ValueError('Only bounded 4/8 target workers allowed')
    bases = trial.bases_for(plan)
    with (output/'execution.lock').open('x', encoding='utf-8') as f:
        f.write(old.sha(output/'plan.json'))
    rows, started, fatal = {}, perf_counter(), None
    with old.joint.credential_context(plan), old.joint.roster.lightweight_protocol():
        calls = trial.ModelCalls(output/'run', 'gemini', plan)
        calls.max_calls = plan['max_calls']
        calls.limits = {k: Decimal(v) for k,v in plan['limits'].items()}
        def one(s):
            key = s['key']; start = perf_counter()
            result_record = None
            row = {k: s[k] for k in ('key','video_id','frame_id')}
            if calls.stopped:
                return {**row, 'status': 'NOT_DISPATCHED', 'seconds': 0, 'predictions': {a: None for a in main.ARMS}}
            try:
                h0_started = perf_counter()
                raw = ready.generate_h0(calls, bases[key], s)
                h0_seconds = perf_counter() - h0_started
                features = feature_row(bases[key], raw, s, plan, old.read(s['tracker_snapshot']))
                # Each worker owns a different target path. No shared prediction writes.
                old.save(output/'targets'/key/'features_before_repair.json', features)
                old.save(output/'targets'/key/'h0_raw.json', raw)
                record = ready.repair_from_h0(calls, bases[key], s,
                    old.read(output/'priors'/f"{s['video_id']}.json"), plan['gate'], raw,
                    h0_seconds=h0_seconds)
                result_record = record
            except (ValueError, TypeError, KeyError, trial.ApiSchemaError) as exc:
                row.update(status='TARGET_FAILED', error_type=type(exc).__name__, predictions={a: None for a in main.ARMS})
            else:
                old.save(output/'targets'/key/'result.json', record)
                row.update(status='PREDICTED', predictions=record['predictions'], timing=record['timing_seconds'])
            row['seconds'] = perf_counter()-start
            with calls.lock:
                actual = [r for r in calls.rows if r['target'] == key]
                row.update(repair_observation(result_record, actual))
                row['call_statuses'] = dict(Counter(r['status'] for r in actual))
            old.save(output/'targets'/key/'prediction.json', row)
            return row
        def record(row):
            rows[row['key']] = row
            ordered = [rows[s['key']] for s in plan['selection'] if s['key'] in rows]
            old.save(output/'predictions.json', {'targets': ordered})
            with calls.lock:
                summary = {'completed_targets': len(rows), 'total_targets': 128,
                    'predicted': sum(r['status']=='PREDICTED' for r in rows.values()),
                    'fully_observed': sum(r.get('repair_fully_observed',False) for r in rows.values()),
                    'calls': len(calls.rows), 'seconds': perf_counter()-started, 'stopped': calls.stopped}
            old.save(output/'progress.json', summary)
            print(json.dumps({**summary,'last_target':row['key'],'status':row['status']}),flush=True)
        def batch(items, count):
            with ThreadPoolExecutor(max_workers=count) as pool:
                futures = [pool.submit(one,s) for s in items]
                try:
                    for future in as_completed(futures):
                        record(future.result())
                except Exception:
                    with calls.lock:
                        calls.stopped = True
                    for pending in futures:
                        pending.cancel()
                    raise
        try:
            batch(plan['selection'][:4], 4)
            with calls.lock:
                warmup_ok = all(r['status'] in trial.VALID for r in calls.rows)
            actual_workers = workers if warmup_ok else 4
            old.save(output/'concurrency.json', {'warmup_passed':warmup_ok,'workers_after_warmup':actual_workers})
            batch(plan['selection'][4:], actual_workers)
        except Exception as exc:
            fatal = type(exc).__name__
            raise
        finally:
            calls.stopped = True
            try:
                calls.persist()
            finally:
                calls.close_ledger()
            old.save(output/'completion.json', {'fatal_error':fatal,'wall_seconds':perf_counter()-started,
                'targets':len(rows),'calls':len(calls.rows),
                'evidence_sha256':{str(p.relative_to(output)):old.sha(p) for p in output.rglob('*.json') if 'frozen_source' not in p.parts}})


def score(output):
    plan, done = verify(output), old.read(output/'completion.json')
    if done['fatal_error'] or done['targets'] != 128:
        raise ValueError('Incomplete execution; preserve archive for explicit resume')
    for rel,h in done['evidence_sha256'].items():
        if old.sha(output/rel) != h:
            raise ValueError('Sealed evidence changed: '+rel)
    bases, saved = trial.bases_for(plan), old.read(output/'predictions.json')['targets']
    replay = trial.Replay(output/'run','gemini')
    feature_rows = []
    with old.joint.roster.lightweight_protocol():
        for s,row in zip(plan['selection'],saved,strict=True):
            assert s['key'] == row['key']
            try:
                raw = ready.generate_h0(replay,bases[s['key']],s)
                feature = feature_row(bases[s['key']],raw,s,plan,old.read(s['tracker_snapshot']))
                assert feature == old.read(output/'targets'/s['key']/'features_before_repair.json')
                feature_rows.append(feature)
                result = ready.repair_from_h0(replay,bases[s['key']],s,
                    old.read(output/'priors'/f"{s['video_id']}.json"),plan['gate'],raw)
            except (ValueError,TypeError,KeyError,trial.ApiSchemaError):
                if row['status'] not in ('TARGET_FAILED','NOT_DISPATCHED'):
                    raise
            else:
                assert result['predictions'] == row['predictions']
    assert replay.used == set(replay.rows)
    old.save(output/'features_before_gt.json',feature_rows)
    # GT is joined only after the complete inference and replay have been sealed.
    adapter = old.common.CholecTrack20DatasetAdapter(trial.release.DATASET,causal_window_size=3)
    truth = {}
    wanted={s['key'] for s in plan['selection']}
    for video in VIDEOS:
        for r in adapter.iter_video(video):
            key=f'{video}_{r.inference.target_frame_id}'
            if key in wanted:truth[key]=old.truth_row(r)
    old.save(output/'scored_truth.json',truth)
    by_key={r['key']:r for r in saved}; training=[]
    for feature in feature_rows:
        row=by_key[feature['sample_id']]; gt=truth[row['key']]
        labels=contract.label_outcome(feature['h0_labels'],row['predictions'][main.PRIMARY],
            gt=gt['gt'],mask=gt['mask'],repair_observed=row.get('repair_fully_observed',False))
        training.append({**feature,'labels':labels,'gt':gt['gt'],'mask':gt['mask'],
                         'final_labels':row['predictions'][main.PRIMARY]})
    old.save(output/'training_examples.json',training)
    readiness={name:contract.readiness(training,target=name) for name in contract.TARGETS}
    budget=old.read(output/'run/budget.json'); charges=defaultdict(Decimal)
    failures=[]
    for c in budget['calls']:
        charges[c['account']+'|'+c['charge_kind']]+=Decimal(c['charge'])
        if c['status'] not in trial.VALID:
            failures.append({k:c.get(k) for k in ('target','stage','seat','http_status','status','exception_type')})
    assert all(old.sha(ROOT/n)==h for n,h in plan['root_unchanged'].items())
    report={'profile':PROFILE,'policy_id':plan['policy_id'],'planned_targets':128,
        'statuses':dict(Counter(r['status'] for r in saved)),'fully_observed':sum(r.get('repair_fully_observed',False) for r in saved),
        'training_rows':len(training),'calls':len(budget['calls']),'replayed_calls':len(replay.used),
        'wall_seconds':done['wall_seconds'],'charges':{k:str(v) for k,v in charges.items()},
        'call_failures':failures,'metrics':trial.release.metrics(saved,truth),'readiness':readiness,
        'training_examples_sha256':old.sha(output/'training_examples.json'),
        'features_before_gt_sha256':old.sha(output/'features_before_gt.json'),
        'feature_version':contract.FEATURE_VERSION,'label_version':contract.LABEL_VERSION,
        'root_pointers_unchanged':True,'gate_fit_started':False,'deployable':False,
        'interpretation':'Training-only first collection. Readiness is an engineering class-support check, not independent validation or a deployment certificate.'}
    old.save(output/'collection_report.json',report)
    print(json.dumps({k:report[k] for k in ('planned_targets','statuses','fully_observed','training_rows','calls','wall_seconds','charges','readiness')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','preflight','execute','score'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args(); output=args.output.resolve()
    if args.command=='execute':execute(output,workers=args.workers)
    else:globals()[args.command](output)
