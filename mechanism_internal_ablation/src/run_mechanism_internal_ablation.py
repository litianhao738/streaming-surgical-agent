"""Frozen small-batch Gate and repair ablations with shared five-seat evidence.

prepare/replay/score are offline. Only collect --allow-paid sends model requests.
All newly written artifacts remain inside mechanism_internal_ablation/outputs.
"""
import os
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_variable] = '1'
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time
import uuid

HOME = Path(__file__).resolve().parents[1]
ROOT = HOME.parent
sys.path[:0] = [str(HOME/'src'), str(ROOT), str(ROOT/'src')]
sys.dont_write_bytecode = True

from mechanisms import HEADS, SLOTS, aggregate, decide, difference, probe_ratings, selections
from metrics import evaluate, random_summary, summary_values

SOURCE = ROOT/'artifacts/experiments/gate_ablation_third_20260916_r1/core'
INVENTORY = ROOT/'artifacts/evaluation/testing_third_ablation_20260916/frame_inventory.jsonl'
DEFAULT_OUT = HOME/'outputs/testing480_20260920_r1'
PROFILE = 'mechanism_internal_small_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_output(path):
    out = Path(path).resolve()
    if not out.is_relative_to((HOME/'outputs').resolve()) or out == (HOME/'outputs').resolve():
        raise ValueError('Output must be a run directory under mechanism_internal_ablation/outputs')
    return out


def imports():
    from scripts import run_gate_ablation as old
    from surgical_agent.research.gate import tracker_pipeline_v2 as runtime
    return old, runtime


@contextmanager
def protocol_context(plan, *, credentials=False):
    old, _ = imports()
    # Existing context managers are process-local; project source files are untouched.
    if credentials:
        old.apply_credential_patches(plan['credential_root'])
        old.verify_credentials(plan, plan['credential_root'], online=False)
        with old.core.app.frozen.joint.credential_context(plan), old.core.app.frozen.joint.roster.lightweight_protocol():
            yield
    else:
        with old.core.app.frozen.joint.roster.lightweight_protocol():
            yield


class Capture:
    def call(self, target, stage, seat, body):
        self.body = deepcopy(body)
        return None


def wire_for(snapshot, selected, seat):
    old, _ = imports()
    capture = Capture()
    base = old.core.demo.make_base(selected)
    backend = old.core.app.WireBackend(capture, base, selected)
    from surgical_agent.research.verification.prior_gated_joint import joint_pool
    backend.five_head(seat, snapshot['h0'], joint_pool(snapshot['pool']))
    return capture.body


def routed_safe(wire, seat, plan):
    from scripts import full_official_reviewer_transport as transport
    if seat in transport.CHANGED:
        wire = transport.routes.route_body(seat, wire, plan['reviewer_config'])
    return transport.old.old.joint.roster.transport.redact_images(wire)


def request_semantics(safe):
    """Ignore JSON packet serialization only; keep text, image hashes and parameters exact."""
    value = deepcopy(safe)
    for message in value.get('messages', []):
        if not isinstance(message.get('content'), list): continue
        for part in message['content']:
            if part.get('type') == 'text':
                try: part['text'] = json.loads(part['text'])
                except (ValueError, TypeError): pass
    return value


def restore_frozen_packet_text(wire, frozen):
    """Repeated Qwen calls use the original packet bytes, including key order."""
    wire = deepcopy(wire)
    for message, old_message in zip(wire['messages'], frozen['messages'], strict=True):
        if not isinstance(message.get('content'), list): continue
        for part, old_part in zip(message['content'], old_message['content'], strict=True):
            if part.get('type') == 'text': part['text'] = old_part['text']
    return wire


def journal_index(source, key):
    records = {}
    for path in (source/'targets'/key).rglob('record.json'):
        r = read(path)
        identity = r['stage']+'|'+r['seat']
        if identity in records:
            raise ValueError('Duplicate historical request: '+key+'/'+identity)
        if not r.get('finished_utc') or r['status'] == 'DISPATCHED':
            raise ValueError('Unresolved historical request: '+str(path))
        response_path = path.parent/'response.json'
        if r.get('response_sha256') and sha(response_path) != r['response_sha256']:
            raise ValueError('Historical response hash changed')
        response = read(response_path) if response_path.exists() else {}
        body = response.get('body', response)
        records[identity] = {'record_path': str(path), 'record_sha256': sha(path),
                             'request_path': str(path.parent/'request.json'),
                             'request_sha256': sha(path.parent/'request.json'),
                             'response_path': str(response_path) if response_path.exists() else None,
                             'response_sha256': sha(response_path) if response_path.exists() else None,
                             'record': r, 'usage': body.get('usage') or r.get('usage') or {}}
    return records


def parsed_cached(ref, seat, plan):
    from scripts import full_official_reviewer_transport as transport
    r = ref['record']
    valid = {'JSON_PARSED'} if seat in transport.CHANGED else set(transport.old.trial.VALID)
    if r['status'] not in valid:
        return None
    response = read(ref['response_path'])
    body = response.get('body', response)
    if seat in transport.CHANGED:
        transport.routes.check_response(seat, body, plan['reviewer_config'])
        return transport.parse_changed(body, {'model': r['model']})
    return transport.old.trial.parse_review_json(body['choices'][0]['message']['content'])[0]


def rebuild(row, result, prior, real_gate):
    old, runtime = imports()
    class CachedPrefix:
        parse_h0 = staticmethod(deepcopy)
        def h0(self): return deepcopy(result['h0'])
        def proposal(self, *args): return deepcopy(result['proposal_raw'])
        def five_head(self, seat, *args):
            if seat != 'qwen': raise AssertionError('Post-Gate data accessed during prefix reconstruction')
            return deepcopy(result['five_head_raw']['qwen'])
    class Skip:
        review_mode = 'five_head_probe'
        phase_review_enabled = True
        def __call__(self, features): return real_gate(features)[0], 0
    selected = {k: row[k] for k in ('key', 'video_id', 'frame_id', 'causal_frame_ids')}
    selected['source_split'] = 'Testing'
    rebuilt = runtime.run_interaction(CachedPrefix(), selected, prior, Skip(), inference_split='Testing')
    if rebuilt['features'] != result['features'] or rebuilt['cheap'] != result['cheap']:
        raise ValueError('Prefix differs from frozen cache: '+row['key'])
    if abs(rebuilt['gate_score']-result['gate_score']) > 1e-12:
        raise ValueError('Gate inference differs from frozen score')
    original = runtime.original
    pool = original.make_pool(result['h0'])
    if result['proposal_raw'] is not None:
        try: pool = original.make_pool(result['h0'], result['proposal_raw'], pool)
        except ValueError: pass
    return {**selected, 'h0': result['h0'], 'cheap': result['cheap'], 'pool': pool,
            'features': rebuilt['features'], 'gate_score': rebuilt['gate_score'],
            'probe_ratings_all': probe_ratings(pool, result['five_head_raw']['qwen'])}


def prepare(out, frames_per_video):
    if out.exists():
        raise ValueError('Use a new run directory; preparation never overwrites a run')
    if not 1 <= frames_per_video <= 120:
        raise ValueError('Small-batch preparation supports 1..120 frames per video')
    old, _ = imports()
    spec = read(HOME/'protocol.json')
    model_path = ROOT/spec['gate']['model_path']
    if sha(INVENTORY) != spec['scope']['inventory_sha256'] or sha(model_path) != spec['gate']['model_sha256']:
        raise ValueError('Frozen source inventory/model changed')
    real_gate, gate_manifest, model = old.ORIGINAL_LOAD_GATE()
    if gate_manifest['model_sha256'] != spec['gate']['model_sha256']:
        raise ValueError('Default Gate differs from protocol')
    source_plan = read(SOURCE/'plan.json')
    source_receipt = read(SOURCE/'receipt.json')
    if source_receipt['state'] != 'PASS' or source_receipt['plan_sha256'] != sha(SOURCE/'plan.json'):
        raise ValueError('Source collection is not sealed')
    inventory = [json.loads(line) for line in INVENTORY.read_text(encoding='utf-8-sig').splitlines()]
    chosen = []
    for video in spec['scope']['videos']:
        group = sorted((s for s in inventory if s['video_id'] == video and s['stage'] == 'pipeline'),
                       key=lambda s: s['frame_id'])
        chosen += group[:frames_per_video]
    out.mkdir(parents=True)
    snapshots, evidence, input_rows, prefix_usage = {}, {}, {}, {}
    source_refs = {str(INVENTORY): sha(INVENTORY), str(SOURCE/'plan.json'): sha(SOURCE/'plan.json'),
                   str(SOURCE/'receipt.json'): sha(SOURCE/'receipt.json'), str(model_path): sha(model_path)}
    prior_hashes = {}
    with protocol_context(source_plan):
        for index, row in enumerate(chosen):
            key, video = row['key'], row['video_id']
            source_result = SOURCE/'results'/(key+'.json')
            result = read(source_result)
            source_refs[str(source_result)] = sha(source_result)
            prior_path = SOURCE/'priors'/(video+'.json')
            prior = read(prior_path)
            if set(prior['fit_videos']) & set(spec['scope']['videos']) or prior['excluded_video'] != video:
                raise ValueError('Testing video in prior fitting source')
            prior_hashes[video] = sha(prior_path)
            snapshot = rebuild(row, result, prior, real_gate)
            selected_path = SOURCE/'resolved_inputs'/(key+'.json')
            selected = read(selected_path)
            source_refs[str(selected_path)] = sha(selected_path)
            # Keep evaluation metadata separate from snapshots/model requests.
            input_rows[key] = {k: v for k, v in selected.items() if k != 'evaluation_target'}
            refs = journal_index(SOURCE, key)
            evidence[key] = {}
            for seat in SLOTS:
                wire = wire_for(snapshot, selected, seat)
                safe = routed_safe(wire, seat, source_plan)
                name = 'five_head_v1|'+seat
                if seat in result['five_head_raw']:
                    ref = refs[name]
                    frozen_request = read(ref['request_path'])
                    if request_semantics(frozen_request) != request_semantics(safe):
                        raise ValueError('Cached/new wire mismatch: '+key+'/'+seat)
                    safe = frozen_request
                    parsed = parsed_cached(ref, seat, source_plan)
                    if parsed != result['five_head_raw'][seat]:
                        raise ValueError('Parsed answer not bound to source response: '+key+'/'+seat)
                    evidence[key][seat] = {**ref, 'raw': parsed}
                write(out/'requests'/key/(seat+'.json'), safe)
            if 'qwen' not in evidence[key]:
                raise ValueError('Shared initial probe missing')
            pref = {'proposal': refs['proposal|base'], 'probe': evidence[key]['qwen']}
            if 'h0|base' in refs:
                pref['h0'] = refs['h0|base']
            else:
                hpath = Path(selected['cached_record_path'])
                hrecord = read(hpath)
                if sha(hpath) != selected['cached_record_sha256']:
                    raise ValueError('Cached H0 changed')
                pref['h0'] = {'record_path': str(hpath), 'record_sha256': sha(hpath),
                              'record': {'account': 'openrouter_usd'}, 'usage': hrecord.get('usage') or {}}
            prefix_usage[key] = pref
            snapshots[key] = snapshot
            if (index+1) % 40 == 0:
                print(json.dumps({'stage': 'prepare', 'frames': index+1, 'total': len(chosen)}), flush=True)
    selection = selections(snapshots, model['threshold'])
    jobs = []
    for key in snapshots:
        for seat in SLOTS[1:]:
            if seat not in evidence[key]:
                jobs.append({'id': key+'__multi__'+seat, 'key': key, 'panel': 'multi',
                             'slot': seat, 'seat': seat, 'replicate_id': 0})
        if key in set(selection['learned']):
            for rep, slot in enumerate(SLOTS[1:], 1):
                jobs.append({'id': key+'__qwen5__r'+str(rep), 'key': key, 'panel': 'qwen5',
                             'slot': slot, 'seat': 'qwen', 'replicate_id': rep})
    # Interleave models to obtain early actual latency evidence without changing the scope.
    jobs.sort(key=lambda j: (j['key'], j['replicate_id'], j['seat']))
    counts = Counter(j['seat'] for j in jobs)
    expected_seconds = sum(counts[s]*{'qwen':12.6,'gpt':8.9,'gemini':5.3,'grok':18.1,'deepseek':4.8}[s] for s in counts)
    write(out/'snapshots.json', snapshots)
    write(out/'selection.json', selection)
    write(out/'inputs.json', input_rows)
    write(out/'historical_evidence.json', evidence)
    write(out/'prefix_usage.json', prefix_usage)
    write(out/'jobs.json', jobs)
    plan = deepcopy(source_plan)
    plan['selection'] = list(input_rows.values())
    plan.update(profile=PROFILE, created_utc=now(), source_core=str(SOURCE),
                frames_per_video=frames_per_video, pipeline_frames=len(chosen),
                evaluation_keys=[s['key'] for s in chosen if s['evaluation_target']],
                evaluation_target_frames=sum(s['evaluation_target'] for s in chosen),
                prior_sha256=prior_hashes, gate_threshold=model['threshold'],
                expected_K=selection['K'], new_jobs=len(jobs), new_jobs_by_seat=dict(counts),
                maximum_workers=8, fixed_panel_depth=5, primary_output='before_all_output_modules',
                source_hashes=source_refs, estimated_ideal_8_worker_seconds=expected_seconds/8,
                limits={'openrouter_usd':'3','aliyun_cny':'6',
                        'glm_requests':str(counts['grok']), 'deepseek_requests':str(counts['deepseek']), 'xai_usd':'0'})
    plan['files_sha256'] = {name: sha(out/name) for name in ('snapshots.json','selection.json','inputs.json',
                                                           'historical_evidence.json','prefix_usage.json','jobs.json')}
    plan['implementation_sha256'] = {str(p): sha(p) for p in (HOME/'src').glob('*.py')}
    plan['request_hashes'] = {str(p.relative_to(out)): sha(p) for p in (out/'requests').rglob('*.json')}
    write(out/'plan.json', plan)
    write(out/'prepared.json', {'plan_sha256': sha(out/'plan.json'), 'api_calls': 0, 'state': 'PREPARED'})
    print(json.dumps({'state':'PREPARED','frames':len(chosen),'scored':plan['evaluation_target_frames'],
                      'selected':selection['K'],'new_calls':len(jobs),'by_seat':dict(counts),
                      'ideal_8_worker_minutes':round(expected_seconds/480,1),'output':str(out)}), flush=True)


def load_plan(out, *, verify_sources=False):
    plan = read(out/'plan.json')
    if plan['profile'] != PROFILE or sha(out/'plan.json') != read(out/'prepared.json')['plan_sha256']:
        raise ValueError('Prepared plan changed')
    for name, digest in plan['files_sha256'].items():
        if sha(out/name) != digest: raise ValueError('Frozen input changed: '+name)
    for name, digest in plan['implementation_sha256'].items():
        if sha(name) != digest: raise ValueError('Frozen implementation changed: '+name)
    if verify_sources:
        for name, digest in plan['source_hashes'].items():
            if sha(name) != digest: raise ValueError('Historical source changed: '+name)
        for video, digest in plan['prior_sha256'].items():
            if sha(Path(plan['source_core'])/'priors'/(video+'.json')) != digest:
                raise ValueError('Prior changed')
    return plan


def collect(out, *, workers, allow_paid, max_minutes, limit_jobs=None):
    if not allow_paid: raise ValueError('--allow-paid required')
    if not 1 <= workers <= 8: raise ValueError('Use 1..8 global concurrent requests')
    plan = load_plan(out, verify_sources=True)
    from scripts.testing_half_transport import GuardedCalls
    from surgical_agent.research.gate.collection_budget import Budget, BudgetStop
    jobs, snapshots, inputs = (read(out/name) for name in ('jobs.json','snapshots.json','inputs.json'))
    lock_path = out/'collection.lock'
    with lock_path.open('x') as stream:
        json.dump({'pid':os.getpid(),'started_utc':now()}, stream)
    stop, lock, tick = threading.Event(), threading.Lock(), time.perf_counter()
    budget = Budget(out/'budget.sqlite', plan['limits'], sha(out/'plan.json'))
    finished = sum((out/'completed'/(j['id']+'.json')).exists() for j in jobs)
    counters = {'done': finished, 'new': 0}

    def task(job):
        done_path = out/'completed'/(job['id']+'.json')
        if done_path.exists():
            return
        if stop.is_set() or time.perf_counter()-tick > max_minutes*60:
            stop.set(); raise BudgetStop('Collection time budget reached; saved work is resumable')
        wire = wire_for(snapshots[job['key']], inputs[job['key']], job['seat'])
        request_path = out/'requests'/job['key']/(job['seat']+'.json')
        if sha(request_path) != plan['request_hashes'][str(request_path.relative_to(out))]:
            raise ValueError('Frozen request changed')
        frozen_request = read(request_path)
        if request_semantics(routed_safe(wire, job['seat'], plan)) != request_semantics(frozen_request):
            raise ValueError('Generated request differs from frozen request')
        wire = restore_frozen_packet_text(wire, frozen_request)
        if routed_safe(wire, job['seat'], plan) != frozen_request:
            raise ValueError('Unable to restore exact frozen request packet')
        # A separate target identity for each repetition prevents wire-hash deduplication.
        selected = {'key':job['id']}
        delegate = GuardedCalls(out, plan, selected, budget, stop)
        try:
            answer = delegate.call(job['id'], 'five_head_v1', job['seat'], wire)
        finally:
            delegate.close()
        refs = journal_index(out, job['id'])
        ref = refs['five_head_v1|'+job['seat']]
        write(done_path, {'job':job, 'raw':answer, 'evidence':ref, 'finished_utc':now()})
        with lock:
            counters['done'] += 1
            counters['new'] += 1
            if counters['done'] % 20 == 0:
                elapsed = time.perf_counter()-tick
                remaining = len(jobs)-counters['done']
                progress = {'state':'COLLECTING', **counters, 'total':len(jobs),
                            'elapsed_minutes':round(elapsed/60,1),
                            'estimated_remaining_minutes':round(remaining*elapsed/max(1,counters['new'])/60,1),
                            'budget':budget.summary()}
                write(out/'progress.json', progress)
                print(json.dumps(progress), flush=True)

    try:
        pending = [j for j in jobs if not (out/'completed'/(j['id']+'.json')).exists()]
        if limit_jobs is not None: pending = pending[:limit_jobs]
        with protocol_context(plan, credentials=True), ThreadPoolExecutor(max_workers=workers) as pool:
            iterator = iter(pending)
            running = set()
            for _ in range(workers):
                job = next(iterator, None)
                if job is not None: running.add(pool.submit(task, job))
            while running:
                done, running = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    try: future.result()
                    except BaseException:
                        stop.set()
                        raise
                    job = next(iterator, None)
                    if job is not None and not stop.is_set(): running.add(pool.submit(task, job))
        complete = all((out/'completed'/(j['id']+'.json')).exists() for j in jobs)
        state = {'state':'COLLECTED' if complete else 'PILOT_COMPLETE', 'done':counters['done'],
                 'total':len(jobs),'seconds':time.perf_counter()-tick,'budget':budget.summary(),
                 'plan_sha256':sha(out/'plan.json')}
        if complete:
            state['completed_sha256'] = {j['id']:sha(out/'completed'/(j['id']+'.json')) for j in jobs}
            write(out/'collection_receipt.json',state)
        write(out/'progress.json',state)
        print(json.dumps({k:v for k,v in state.items() if k != 'completed_sha256'}),flush=True)
    except BaseException as exc:
        write(out/'failure.json', {'state':'STOPPED','error_type':type(exc).__name__,
                                 'message':str(exc),'utc':now(),'done':counters['done']})
        raise
    finally:
        budget.close()
        lock_path.unlink()


def replay(out):
    plan = load_plan(out, verify_sources=True)
    receipt = read(out/'collection_receipt.json')
    if receipt['state'] != 'COLLECTED' or receipt['plan_sha256'] != sha(out/'plan.json'):
        raise ValueError('Collection not sealed')
    snapshots, selection, historical, jobs = (read(out/name) for name in
                                              ('snapshots.json','selection.json','historical_evidence.json','jobs.json'))
    multi = {key:{seat:ref['raw'] for seat,ref in refs.items()} for key,refs in historical.items()}
    single = {key:{'qwen':historical[key]['qwen']['raw']} for key in selection['learned']}
    for job in jobs:
        path = out/'completed'/(job['id']+'.json')
        if sha(path) != receipt['completed_sha256'][job['id']]: raise ValueError('Collected answer changed')
        result = read(path)
        for kind in ('record', 'request', 'response'):
            ref = result['evidence']
            if ref.get(kind+'_sha256') and sha(ref[kind+'_path']) != ref[kind+'_sha256']:
                raise ValueError('New response journal changed')
        (multi if job['panel']=='multi' else single)[job['key']][job['slot']] = result['raw']
    priors = {video:read(Path(plan['source_core'])/'priors'/(video+'.json')) for video in plan['prior_sha256']}
    baseline = {key:deepcopy(s['cheap']) for key,s in snapshots.items()}
    full, variants, diagnostics = {}, {'B1':{},'B2':{},'B3':{}}, {}
    for index,(key,snapshot) in enumerate(snapshots.items()):
        panel = aggregate(snapshot['pool'],multi[key])
        prior = priors[snapshot['video_id']]
        full[key], detail = decide(snapshot,panel,prior)
        diagnostics[key] = {'full':detail,'multi_evidence':panel['diagnostics'],
                             'multi_formats':panel['formats']}
        if key in single:
            qpanel = aggregate(snapshot['pool'],single[key])
            variants['B1'][key],diagnostics[key]['B1'] = decide(snapshot,qpanel,prior)
            variants['B2'][key],diagnostics[key]['B2'] = decide(snapshot,panel,prior,prior_after=False)
            variants['B3'][key],diagnostics[key]['B3'] = decide(snapshot,panel,prior,rollback=False)
            diagnostics[key]['qwen_evidence'] = qpanel['diagnostics']
            diagnostics[key]['qwen_raw_unique'] = len({object_sha(v) for v in single[key].values()})
            assert variants['B2'][key]['phase'] == full[key]['phase']
            assert all(variants['B3'][key][h] == full[key][h] for h in ('instrument','target','phase'))
        if (index+1)%80 == 0: print(json.dumps({'stage':'replay','frames':index+1}),flush=True)
    def routed(keys,reviewed):
        selected=set(keys)
        return {key:deepcopy(reviewed[key] if key in selected else baseline[key]) for key in snapshots}
    arms={'A0':baseline,'B0':baseline,'A2':routed(selection['rule'],full),
          'A3':routed(selection['learned'],full)}
    arms['B4']=arms['A3']
    for seed,keys in selection['random'].items(): arms['A1_seed'+seed]=routed(keys,full)
    for name,reviewed in variants.items(): arms[name]=routed(selection['learned'],reviewed)
    for name,prediction in arms.items(): write(out/'predictions'/(name+'.json'),prediction)
    write(out/'diagnostics.json',diagnostics)
    write(out/'prediction_receipt.json',{'state':'SEALED','plan_sha256':sha(out/'plan.json'),
          'arms':{name:sha(out/'predictions'/(name+'.json')) for name in arms},
          'diagnostics_sha256':sha(out/'diagnostics.json'),'api_calls':0,'utc':now()})
    print(json.dumps({'state':'REPLAYED','arms':len(arms),'frames':len(snapshots),'api_calls':0}),flush=True)


def token_summary(refs):
    from scripts.summarize_ablation_metrics_cost import price_rmb
    prompt=completion=unknown=0
    money=Decimal(0); missing_money=0
    for ref in refs:
        usage=ref.get('usage') or {}
        if usage.get('prompt_tokens') is None or usage.get('completion_tokens') is None:
            unknown+=1
        else:
            prompt+=int(usage['prompt_tokens'])
            completion+=max(int(usage['completion_tokens']),int(usage.get('total_tokens',0))-int(usage['prompt_tokens']))
        cost,_=price_rmb(ref['record'],usage)
        if cost is None: missing_money+=1
        else: money+=cost
    return {'requests':len(refs),'prompt_tokens':prompt,'completion_tokens':completion,
            'known_total_tokens':prompt+completion,'usage_missing':unknown,
            'known_rmb':float(money),'cost_missing':missing_money,
            'pricing_basis':'historical rates and USD/CNY=6.7065; estimates except provider usage.cost'}


def costs(out,plan,selection):
    prefix,historical,jobs=(read(out/name) for name in ('prefix_usage.json','historical_evidence.json','jobs.json'))
    multi={k:{s:r for s,r in refs.items()} for k,refs in historical.items()}
    single={k:{} for k in selection['learned']}
    actual=[]
    for job in jobs:
        ref=read(out/'completed'/(job['id']+'.json'))['evidence']
        actual.append(ref)
        (multi if job['panel']=='multi' else single)[job['key']][job['slot']]=ref
    common=[ref for value in prefix.values() for ref in value.values()]
    selected_by_arm={'A0':[],'B0':[],'A2':selection['rule'],'A3':selection['learned'],'B4':selection['learned'],
                     **{name:selection['learned'] for name in ('B1','B2','B3')},
                     **{'A1_seed'+seed:keys for seed,keys in selection['random'].items()}}
    logical={}
    for name,keys in selected_by_arm.items():
        pool=single if name=='B1' else multi
        logical[name]=token_summary(common+[pool[k][s] for k in keys for s in SLOTS[1:]])
    write(out/'cost_actual.json',token_summary(actual))
    write(out/'cost_logical_by_arm.json',logical)
    return logical


def score(out):
    plan=load_plan(out)
    receipt=read(out/'prediction_receipt.json')
    if receipt['state']!='SEALED' or receipt['plan_sha256']!=sha(out/'plan.json'):
        raise ValueError('Predictions must be sealed before GT access')
    snapshots,selection=read(out/'snapshots.json'),read(out/'selection.json')
    arms={}
    for name,digest in receipt['arms'].items():
        path=out/'predictions'/(name+'.json')
        if sha(path)!=digest: raise ValueError('Predictions changed')
        arms[name]=read(path)
    old,_=imports()
    rows=[{k:snapshots[key][k] for k in ('key','video_id','frame_id')} for key in plan['evaluation_keys']]
    truth,sources=old.core.demo.load_testing_truth(rows)
    keys=plan['evaluation_keys']; selected=set(selection['learned'])
    baseline={k:s['cheap'] for k,s in snapshots.items()}
    metrics={}
    for name,prediction in arms.items():
        metrics[name]={'all':evaluate(keys,baseline,prediction,truth),
                       'learned_selected':evaluate([k for k in keys if k in selected],baseline,prediction,truth),
                       'per_video':{v:evaluate([k for k in keys if snapshots[k]['video_id']==v],baseline,prediction,truth)
                                    for v in sorted({s['video_id'] for s in snapshots.values()})}}
    summary=random_summary([metrics['A1_seed'+str(seed)]['all'] for seed in range(20)])
    logical=costs(out,plan,selection)
    write(out/'metrics.json',{'arms':metrics,'random_summary':summary,'annotation_sha256':sources,
                            'evaluation_targets':len(keys),'baseline':'cheap','boundary':'before_output_modules'})
    columns=['arm','instrument_f1','verb_f1','target_f1','ivt_f1','phase_accuracy','harm_percent','net_gain_pp',
             'reviewed_frames','review_percent','tokens_million_known','usage_missing','cost_rmb_known']
    def row(name):
        n=0 if name in ('A0','B0') else selection['K']
        return {'arm':name,**summary_values(metrics[name]['all']), 'reviewed_frames':n,
                'review_percent':100*n/len(snapshots),'tokens_million_known':logical[name]['known_total_tokens']/1e6,
                'usage_missing':logical[name]['usage_missing'],'cost_rmb_known':logical[name]['known_rmb']}
    random_row={'arm':'A1_Random_mean','reviewed_frames':selection['K'],'review_percent':100*selection['K']/len(snapshots),
                **{field:value['mean'] for field,value in summary.items()}}
    for field,key in [('tokens_million_known','known_total_tokens'),('usage_missing','usage_missing'),('cost_rmb_known','known_rmb')]:
        random_row[field]=statistics.mean(logical['A1_seed'+str(i)][key] for i in range(20))/(1e6 if field=='tokens_million_known' else 1)
    tables={'table_gate':[row('A0'),random_row,row('A2'),row('A3')],
            'table_repair':[row(name) for name in ('B0','B2','B3','B4')]}
    for filename,rows in tables.items():
        with (out/(filename+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    diagnostic=read(out/'diagnostics.json')
    activation={name:Counter() for name in ('full','B1','B2','B3')}
    for key,detail in diagnostic.items():
        if key not in selected: continue
        for name,counter in activation.items():
            d=detail[name]
            log=d['prior'] or {}
            counter['frames']+=1
            counter['prior_added_labels']+=len(log.get('prior_added',[]))
            counter['prior_vetoed_labels']+=len(log.get('vetoed',[]))
            counter['schema_fallback_frames']+=d['schema_fallback'] is not None
            counter['rollback_changed_frames']+=any(v['added'] or v['removed'] for v in d['rollback_difference'].values())
    write(out/'activation_summary.json',{k:dict(v) for k,v in activation.items()})
    def fmt(value): return 'NA' if value is None else f'{value:.3f}' if isinstance(value,(int,float)) else str(value)
    lines=['# 机制内部消融结果','',f"范围：{len(snapshots)} 个推理点、{len(keys)} 个评分目标，Learned 送审 {selection['K']} 帧。",
           '', '主输出位于 Tracker、v2.2 清理和阶段平滑之前；修复参照为共同 cheap。',
           'I/V/T/IVT 为 micro-F1，Phase 为 Accuracy；净修收益单位为百分点，其余比例单位为 %。','']
    for title,rows in [('Gate 策略消融',tables['table_gate']),('多模型修复机制消融',tables['table_repair'])]:
        lines += ['## '+title,'','| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
        lines += ['| '+' | '.join(fmt(r.get(c)) for c in columns)+' |' for r in rows]
        lines += ['']
    lines += ['Random 20 次的样本标准差：'+', '.join(k+'='+fmt(v['sd']) for k,v in summary.items()),'',
              '费用：cost_actual.json 为本轮新增；表中为每组包含 H0、候选提案和初审的逻辑成本。未知 usage 未计为零，已列缺失数。',
              '本表是已有 Testing 子集的探索性机制分析。单因素删减是条件贡献；少量连续片段不代表全量 Testing。',
              '本轮未计算 bootstrap 区间；逐视频、共同送审范围指标见 metrics.json，触发数见 activation_summary.json。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'final_receipt.json',{'state':'COMPLETE','utc':now(),'plan_sha256':sha(out/'plan.json'),
          'tables':{n:sha(out/(n+'.csv')) for n in tables},'metrics_sha256':sha(out/'metrics.json'),
          'report_sha256':sha(out/'report.md'),'api_calls_during_scoring':0})
    print(json.dumps({'state':'COMPLETE','output':str(out),'tables':tables},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','collect','replay','score','finish','status'))
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--frames-per-video',type=int,default=60)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--max-minutes',type=float,default=120)
    parser.add_argument('--limit-jobs',type=int)
    parser.add_argument('--allow-paid',action='store_true')
    args=parser.parse_args();out=safe_output(args.output)
    if args.command=='prepare': prepare(out,args.frames_per_video)
    elif args.command=='collect': collect(out,workers=args.workers,allow_paid=args.allow_paid,max_minutes=args.max_minutes,limit_jobs=args.limit_jobs)
    elif args.command=='replay': replay(out)
    elif args.command=='score': score(out)
    elif args.command=='finish': replay(out);score(out)
    else:
        plan=read(out/'plan.json')
        print(json.dumps({'frames':plan['pipeline_frames'],'selected':plan['expected_K'],
                          'jobs':plan['new_jobs'],'done':len(list((out/'completed').glob('*.json'))),
                          'complete':(out/'final_receipt.json').exists(),
                          'progress':read(out/'progress.json') if (out/'progress.json').exists() else None},ensure_ascii=False))


if __name__=='__main__':
    main()
