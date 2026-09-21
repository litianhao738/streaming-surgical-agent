"""Frozen half-Testing run with five-head Gate, phase review and Tracker v2.2; RAM off."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import run_scheme4_testing_demo as demo
from scripts import run_tracker_scheme4_pipeline as app
from scripts.testing_half_transport import GuardedCalls
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.research.gate.collection_budget import Budget, BudgetStop
from surgical_agent.research.gate import tracker_pipeline_v2 as tracker_pipeline
from surgical_agent.research.gate.tracker_pipeline_v2 import CausalPhaseFilter

PROFILE = 'testing_half_gate_tracker_v1'
SCOPE = ROOT / 'artifacts/evaluation/testing_half_probe_gate_tracker_plan_20260916'
REFERENCE = ROOT / 'artifacts/experiments/scheme4_testing_demo_20260914_r3'
read, sha = demo.read, demo.sha


def run_target(*args, **kwargs):
    """Keep valid initial labels if panel expansion exceeds the output contract."""
    from types import FunctionType, SimpleNamespace
    from surgical_agent.api.errors import ApiSchemaError
    original = tracker_pipeline.original
    fallbacks = []

    def select(current, *selection_args, **selection_kwargs):
        try:
            return original.select_prior_gated(current, *selection_args, **selection_kwargs)
        except ApiSchemaError as exc:
            # Validate the fallback; malformed inputs must not become valid results.
            original.make_pool(current)
            note = {'stage': 'review_selection', 'error_type': type(exc).__name__,
                    'reason': str(exc), 'fallback': 'validated_initial_labels'}
            fallbacks.append(note)
            return deepcopy(current), note

    interaction = tracker_pipeline.run_interaction
    isolated = FunctionType(interaction.__code__,
        {**interaction.__globals__, 'original': SimpleNamespace(
            **{**vars(original), 'select_prior_gated': select})},
        interaction.__name__, interaction.__defaults__, interaction.__closure__)
    isolated.__kwdefaults__ = interaction.__kwdefaults__
    target = tracker_pipeline.run_target
    invoke = FunctionType(target.__code__, {**target.__globals__, 'run_interaction': isolated},
                          target.__name__, target.__defaults__, target.__closure__)
    invoke.__kwdefaults__ = target.__kwdefaults__
    result = invoke(*args, **kwargs)
    if fallbacks:
        result['schema_fallbacks'] = fallbacks
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, value)


def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def scope_rows(scope):
    spec = read(scope / 'run_scope.json')
    if spec.get('base_model', 'google/gemini-3.8-flash') not in ('google/gemini-3.8-flash', 'qwen3.8-max'):
        raise ValueError('Unsupported H0 model; import and validate the other base model first')
    if spec.get('base_model') == 'qwen3.8-max':
        parent = Path(spec['parent_scope_path'])
        if sha(parent/'run_scope.json') != spec['parent_scope_sha256']:
            raise ValueError('Gemini comparison scope changed')
        old = rows(parent/'frame_inventory.jsonl')
        new = rows(scope/'frame_inventory.jsonl')
        fields = ('key','video_id','frame_id','stage','evaluation_target','causal_frame_ids')
        if [[s[k] for k in fields] for s in old] != [[s[k] for k in fields] for s in new]:
            raise ValueError('Qwen and Gemini comparison timelines differ')
    if sha(scope / 'frame_inventory.jsonl') != spec['frame_inventory_sha256']:
        raise ValueError('frame inventory changed')
    for path, digest in spec['source_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('scope source changed: ' + path)
    selected = rows(scope / 'frame_inventory.jsonl')
    if (len(selected) != 9058 or sum(s['evaluation_target'] for s in selected) != 7823
            or sum(s['stage'] == 'pipeline' for s in selected) != 8578):
        raise ValueError('unexpected half-Testing scope')
    for v in spec['videos']:
        actual = [s['frame_id'] for s in selected if s['video_id'] == v['video_id']]
        if actual != list(range(v['warmup_start_frame'], v['end_frame_exclusive'], 25)):
            raise ValueError('timeline is not contiguous')
    return spec, selected


def store_image(out, content, frame, detail):
    digest = hashlib.sha256(content).hexdigest()
    path = out / 'images' / digest[:2] / (digest + '.jpg')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if sha(path) != digest:
        raise ValueError('image cache corrupted')
    return dict(frame_id=frame, path=str(path), sha256=digest,
                mime_type='image/jpeg', detail=detail)


def extract_cached(selected, out):
    needed = [s for s in selected if 'cached_h0' in s and s['stage'] == 'pipeline']
    for directory in sorted({s['prepared_dir'] for s in needed}):
        folder = Path(directory)
        pending = {s['custom_id']: s for s in needed if s['prepared_dir'] == directory}
        for chunk in read(folder / 'manifest.json')['chunks']:
            if not pending:
                break
            path = folder / chunk['path'].replace('\\', '/')
            if demo.stream_sha(path) != chunk['sha256']:
                raise ValueError('delivered chunk changed: ' + str(path))
            with path.open(encoding='utf-8-sig') as stream:
                for line in stream:
                    item = json.loads(line)
                    s = pending.pop(item['custom_id'], None)
                    if s is None:
                        continue
                    blocks = [b for m in item['body']['messages'] if isinstance(m['content'], list)
                              for b in m['content'] if b.get('type') == 'image_url']
                    if len(blocks) != 3:
                        raise ValueError('cached H0 must have exactly three delivered images')
                    s['images'] = []
                    for frame, block in zip(s['causal_frame_ids'], blocks, strict=True):
                        prefix, encoded = block['image_url']['url'].split(',', 1)
                        if prefix != 'data:image/jpeg;base64':
                            raise ValueError('expected JPEG')
                        # An omitted detail uses the provider's default auto mode.
                        # Do not invent low/high settings for a reused H0 image.
                        detail = block['image_url'].get('detail', 'auto')
                        if detail not in ('auto', 'low', 'high'):
                            raise ValueError('invalid delivered image detail')
                        s['images'].append(store_image(out, base64.b64decode(encoded, validate=True),
                                                       frame, detail))
                    s['input_chunk_sha256'] = chunk['sha256']
            print(f'Images: {folder.name}/{path.name}; {len(pending)} remaining', flush=True)
        if pending:
            raise ValueError('missing cached images')


def decode_missing(selected, out):
    import cv2
    import numpy as np
    for video in sorted({s['video_id'] for s in selected}):
        group = [s for s in selected if s['video_id'] == video]
        missing = [s for s in group if 'cached_h0' not in s]
        # Verify actual decode alignment against delivered JPEGs near both boundaries.
        cached = [s for s in group if s.get('images')]
        probes = [cached[0], cached[-1]]
        probe_map = {s['frame_id']: s['images'][-1] for s in probes}
        wanted = {f for s in missing for f in s['causal_frame_ids']} | set(probe_map)
        cap = cv2.VideoCapture(str(demo.DATASET / 'Testing' / video / (video.lower() + '.mp4')))
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - 25) > .01:
            raise ValueError('video missing or FPS mismatch: ' + video)
        images, audits = {}, []
        first, last = min(wanted), max(wanted)
        reference = cv2.imread(probes[0]['images'][-1]['path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, first - 1)
        try:
            for frame in range(first, last + 1):
                if frame not in wanted:
                    if not cap.grab():
                        raise ValueError('early end of video')
                    continue
                ok, image = cap.read()
                if not ok:
                    raise ValueError('decode failed')
                image = cv2.resize(image, (reference.shape[1], reference.shape[0]))
                if frame in probe_map:
                    expected = cv2.imread(probe_map[frame]['path'])
                    mae = float(np.abs(image.astype(float) - expected.astype(float)).mean())
                    if mae > 5:
                        raise ValueError(f'decoder alignment mismatch: {video}/{frame}, MAE={mae}')
                    audits.append(dict(frame_id=frame, decoder_index=frame-1, mae=mae))
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise ValueError('JPEG encode failed')
                images[frame] = store_image(out, encoded.tobytes(), frame, 'high')
        finally:
            cap.release()
        for s in missing:
            s['images'] = [{**images[f], 'detail': ('high' if f == s['frame_id'] else 'low')}
                           for f in s['causal_frame_ids']]
        write(out / 'alignment' / (video + '.json'), audits)
        print(f'Decoded {video}: {len(missing)} new H0 windows; alignment passed', flush=True)


def make_h0_base(s):
    from surgical_agent.api.contracts import ApiRequest, ApiImageInput
    from surgical_agent.perception.main_h0 import (load_main_h0_prompt, main_h0_input,
                                                   MAIN_H0_PROMPT_VERSION, MAIN_H0_SCHEMA_VERSION)
    checked = demo.make_base(s)
    return ApiRequest(provider='openrouter', model_identifier='google/gemini-3.8-flash',
        endpoint_identifier='https://openrouter.ai/api/v1/chat/completions',
        prompt_version=MAIN_H0_PROMPT_VERSION, response_schema_version=MAIN_H0_SCHEMA_VERSION,
        payload={'system_text': load_main_h0_prompt(), 'image_details': ['low', 'low', 'high'],
                 'openrouter_routing_profile': 'strict_google_ai_studio',
                 'openrouter_image_detail_mode': 'explicit_v1',
                 'input_text': json.dumps(main_h0_input(video_id=s['video_id'],
                    target_frame_id=s['frame_id'], frame_ids=tuple(s['causal_frame_ids'])),
                    sort_keys=True, separators=(',', ':'))},
        images=tuple(ApiImageInput(str(f), im.mime_type, im.content)
                     for f, im in zip(s['causal_frame_ids'], checked.images, strict=True)),
        generation_parameters={'temperature': 0, 'max_output_tokens': 4096, 'reasoning': {'effort': 'low'}})


def prepare_tracker(selected, out):
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    from surgical_agent.tracking.config import TrackerTrainingConfig
    from surgical_agent.tracking.detector import build_instrument_detector, load_tracker_checkpoint, decode_detections
    from surgical_agent.tracking.associator import CausalHungarianAssociator
    manifest = read(demo.TRACKER / 'training_manifest.json')
    if sha(demo.TRACKER / 'checkpoint.pt') != manifest['checkpoint_sha256']:
        raise ValueError('tracker checkpoint changed')
    config = TrackerTrainingConfig(**manifest['config'])
    torch.set_num_threads(4)
    model = build_instrument_detector(config, use_pretrained=False)
    load_tracker_checkpoint(demo.TRACKER / 'checkpoint.pt', model=model, map_location='cpu')
    model.eval()
    cache = {}
    for n, s in enumerate((s for s in selected if s['stage'] == 'pipeline'), 1):
        association = CausalHungarianAssociator(iou_threshold=config.association_iou_threshold,
                                               max_age=config.max_age, max_frame_id_gap=25)
        association.reset(s['video_id'])
        frames = []
        for im in s['images']:
            digest = im['sha256']
            if digest not in cache:
                with Image.open(im['path']) as image:
                    rgb = image.convert('RGB'); width, height = rgb.size; tensor = to_tensor(rgb)
                with torch.inference_mode():
                    detected = model([tensor])[0]
                cache[digest] = decode_detections(detected, width=width, height=height,
                                                  score_threshold=config.score_threshold)
            tracks = association.update(im['frame_id'], cache[digest])
            frames.append({'frame_id': im['frame_id'], 'tracks': [t.as_mapping() for t in tracks]})
        path = out / 'tracker' / (s['key'] + '.json')
        write(path, dict(status='AVAILABLE', video_id=s['video_id'], source_split='Testing',
                         source_max_frame_id=s['frame_id'], frames=frames,
                         checkpoint_sha256=manifest['checkpoint_sha256']))
        s['tracker_sha256'] = sha(path)
        if n % 25 == 0:
            print(f'Tracker {n}/8578', flush=True)


def check_prepare_destination(out, resume, allowed_directories):
    """Retry local preparation only; never overwrite a plan or paid execution evidence."""
    if not out.exists():
        return
    if not resume:
        raise ValueError('output exists; use --resume-prepare only for interrupted local preparation')
    if not out.is_dir():
        raise ValueError('prepare output must be a directory')
    for child in out.iterdir():
        if child.name == 'preparation_scope.json' and child.is_file():
            continue
        if child.name not in allowed_directories or not child.is_dir():
            raise ValueError('cannot resume preparation over sealed/unknown artifacts: '+child.name)


def prepare(out, scope, limits, *, resume=False, streaming=False):
    check_prepare_destination(out, resume, {'priors', 'images', 'alignment', 'tracker', 'tracker_runtime'})
    spec, selected = scope_rows(scope)
    reference_plan = read(REFERENCE / 'plan.json')
    if sha(REFERENCE / 'plan.json') != read(REFERENCE / 'prepared.json')['plan_sha256']:
        raise ValueError('reference plan changed')
    caps = read(limits)
    from decimal import Decimal
    if set(caps) != set(read(scope / 'suggested_budget_limits.json')) or any(Decimal(x) < 0 for x in caps.values()):
        raise ValueError('invalid account caps')
    source_path = ROOT/spec['h0_index'] if spec.get('h0_index') else demo.INDEX
    if spec.get('base_model') == 'qwen3.8-max' and not spec.get('h0_index'):
        raise ValueError('Qwen H0 index required; Gemini fallback is forbidden')
    source = {s['key']: s for s in rows(source_path)}
    for s in selected:
        s['source_split'] = 'Testing'
        if s['h0_action'] == 'reuse_exact_three_frame_h0':
            original = source[s['key']]
            if spec.get('base_model') == 'qwen3.8-max' and original.get('model') != 'qwen3.8-max':
                raise ValueError('H0 model identity mismatch')
            if (sha(original['record_path']) != original['record_sha256']
                    or original['causal_frame_ids'] != s['causal_frame_ids']):
                raise ValueError('cached H0 evidence changed')
            s['cached_h0'] = deepcopy(original['h0'])
    _, gate_manifest, _ = app.load_gate()
    if (gate_manifest['model_sha256'] != spec['pipeline']['gate_model_sha256']
            or sha(demo.TRACKER / 'checkpoint.pt') != spec['pipeline']['tracker_checkpoint_sha256']):
        raise ValueError('model differs from frozen scope')
    scope_binding = {'profile': PROFILE, 'scope_sha256': sha(scope/'run_scope.json'), 'limits': caps}
    binding_path = out/'preparation_scope.json'
    if binding_path.exists() and read(binding_path) != scope_binding:
        raise ValueError('interrupted preparation belongs to a different scope/budget')
    legacy_preparation = binding_path.exists()
    out.mkdir(parents=True, exist_ok=resume)
    write(binding_path, scope_binding)
    for video in sorted({s['video_id'] for s in selected}):
        prior_path = REFERENCE / 'priors' / (video + '.json')
        evidence = next(s for s in reference_plan['selection'] if s['video_id'] == video)
        prior = read(prior_path)
        if (sha(prior_path) != evidence['prior_sha256'] or prior['excluded_video'] != video
                or set(prior['fit_videos']) & {s['video_id'] for s in selected}):
            raise ValueError('prior provenance mismatch')
        prior_out = out / 'priors' / (video + '.json')
        if prior_out.exists() and read(prior_out) != prior:
            raise ValueError('existing preparation prior differs: '+video)
        write(prior_out, prior)
    if not streaming:
        extract_cached(selected, out)
        decode_missing(selected, out)
        prepare_tracker(selected, out)
    else:
        # Import before binding runtime sources. No image extraction or detector sweep.
        from scripts import testing_half_streaming
        from scripts import testing_tracker_cache
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('GPU Tracker requires .venv-tracker-gpu/Scripts/python.exe; CUDA is unavailable')
        print('Streaming mode: images and trained Tracker are evaluated per target; no batch Tracker preparation', flush=True)
    plan = deepcopy(reference_plan)
    plan.update(profile=PROFILE, selection=selected, limits=caps, ram=False,
                scope_path=str(scope.resolve()), scope_sha256=sha(scope / 'run_scope.json'), automatic_retry=False,
                base_model=spec.get('base_model', 'google/gemini-3.8-flash'),
                h0_provenance=spec.get('data_provenance', 'reused Gemini H0; current Gate downstream'),
                maximum_paid_calls=52562, maximum_workers=8, evaluation_target_frames=7823,
                pipeline_frames=8578, warmup_h0_frames=480, h0_reuse='exact_three_frame_only',
                new_h0_calls=sum('cached_h0' not in s for s in selected), phase_window_seconds=60,
                phase_review_enabled=bool(gate_manifest.get('phase_review_enabled')),
                review_mode=gate_manifest.get('review_mode', 'separate'),
                gate_version=gate_manifest['version'], gate_model_sha256=gate_manifest['model_sha256'],
                model_sha256=gate_manifest['model_sha256'])
    if plan['base_model'] == 'qwen3.8-max':
        from scripts.testing_qwen_h0 import prepare_config
        plan['qwen_h0'] = prepare_config(plan)
        plan['proposal_model'] = 'qwen3.8-max'
    if plan['phase_review_enabled']:
        plan['maximum_paid_calls'] = plan['new_h0_calls'] + plan['pipeline_frames'] * (
            6 if plan['review_mode'] == 'five_head_probe' else 8 if plan['review_mode'] == 'unified' else 12)
    if streaming:
        plan.update(input_mode='on_demand', tracker_mode='on_demand',
                    tracker_device='cuda:0', tracker_cache=str(ROOT/'artifacts/cache/tracker_testing'),
                    tracker_environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                                             gpu=torch.cuda.get_device_name(0)),
                    tracker_checkpoint_sha256=spec['pipeline']['tracker_checkpoint_sha256'])
        wanted = {s['key'] for s in selected if s['stage'] == 'pipeline'}
        plan['legacy_tracker_sha256'] = {p.stem: sha(p) for p in (out/'tracker').glob('*.json')
                                         if legacy_preparation and p.stem in wanted}
    for field in ('preparation_seconds', 'tracker_seconds', 'protocol_sha256'):
        plan.pop(field, None)
    plan.pop('demo_runtime_sha256', None)
    plan['prior_sha256'] = {v: sha(out / 'priors' / (v + '.json')) for v in {s['video_id'] for s in selected}}
    model_path = ROOT / gate_manifest['model_artifact']
    bound = {Path(__file__), ROOT / 'DEFAULT_PGP_GATE_VERSION.json', model_path,
             ROOT/'src/surgical_agent/research/gate/unified_review.py',
             ROOT/'src/surgical_agent/research/verification/prompts/five_head_probe_v1.txt',
             model_path.parent / read(model_path)['estimator_file']}
    if streaming:
        bound.update((ROOT/'src/surgical_agent/tracking').rglob('*.py'))
        bound.add(demo.TRACKER/'training_manifest.json')
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            p = Path(path).resolve()
            if p.is_relative_to(ROOT) and p.suffix == '.py' and p.is_file():
                bound.add(p)
    plan['runtime_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(bound)}
    write(out / 'plan.json', plan)
    write(out / 'prepared.json', {'plan_sha256': sha(out / 'plan.json'), 'api_calls': 0})
    print('PREPARED: 8578 pipeline frames; 7823 scored; RAM off; API calls=0', flush=True)


def bounded_map(function, items, workers, stop):
    """At most workers outstanding jobs; stop scheduling immediately on failure."""
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(function, s) for s in list_next(iterator, workers)}
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                if stop.is_set():
                    raise BudgetStop('execution stopped')
                pending.update(pool.submit(function, s) for s in list_next(iterator, len(done)))
        except BaseException:
            stop.set()
            for future in pending:
                future.cancel()
            raise


def list_next(iterator, n):
    from itertools import islice
    return list(islice(iterator, n))


def deferred_map(function, items, workers, stop, out, *, retry_rounds=3):
    """Drain healthy frames first, then retry frame failures in bounded later passes."""
    from surgical_agent.research.gate.collection_budget import AmbiguousDispatch
    pending = list(items)
    for attempt in range(retry_rounds + 1):
        failed = []
        lock = threading.Lock()
        def guarded(item):
            try:
                function(item)
            except (BudgetStop, AmbiguousDispatch):
                stop.set()
                raise
            except Exception as exc:
                if stop.is_set():
                    raise
                write(out/'deferred_errors'/(item['key']+'.json'),
                      dict(target=item['key'], error_type=type(exc).__name__,
                           attempt=attempt, state='PENDING_RETRY'))
                with lock:
                    failed.append(item)
                print(f"DEFERRED {item['key']}: {type(exc).__name__}; pass={attempt}", flush=True)
            else:
                (out/'deferred_errors'/(item['key']+'.json')).unlink(missing_ok=True)
        bounded_map(guarded, pending, workers, stop)
        if not failed:
            return
        pending = sorted(failed, key=lambda item: item['key'])
        print(f'Deferred frames: {len(pending)}; completed pass {attempt}', flush=True)
    raise RuntimeError(f'{len(pending)} frames remain after {retry_rounds} deferred retry rounds; resume to retry')


def finalize(out, selection):
    phase = CausalPhaseFilter(60)
    predictions, all_results = [], []
    for s in sorted(selection, key=lambda s: (s['video_id'], s['frame_id'])):
        h0 = read(out / 'h0' / (s['key'] + '.json'))
        if s['stage'] != 'pipeline':
            phase.apply(s['video_id'], s['frame_id'], h0['phase'][0])
            continue
        result = read(out / 'results' / (s['key'] + '.json'))
        if result['cheap']['phase'] != h0['phase']:
            raise ValueError('cheap phase differs from H0')
        raw_phase = result['phase_before_smoothing'] if result.get('phase_review_enabled') else h0['phase'][0]
        smoothed = phase.apply(s['video_id'], s['frame_id'], raw_phase)
        result['prediction']['phase'] = [smoothed]
        all_results.append(result)
        if s['evaluation_target']:
            predictions.append(result)
    write(out / 'continuous_predictions.json', all_results)
    write(out / 'predictions.json', predictions)
    return len(predictions)


def execute(out, workers, allow_paid):
    if not allow_paid:
        raise ValueError('--allow-paid required; execute incurs API charges')
    if not 1 <= workers <= 8:
        raise ValueError('workers must be between 1 and 8')
    plan = read(out / 'plan.json')
    if plan['profile'] != PROFILE or sha(out / 'plan.json') != read(out / 'prepared.json')['plan_sha256']:
        raise ValueError('unsealed plan')
    from scripts.testing_half_resume import validate_runtime
    validate_runtime(out, plan, 'runtime_sha256')
    from scripts.pipeline_checkpoint import Checkpoint
    checkpoint = Checkpoint(out, plan['selection'], sha(out/'plan.json'))
    checkpoint.recover_tail()
    for video, digest in plan['prior_sha256'].items():
        if sha(out / 'priors' / (video + '.json')) != digest:
            raise ValueError('prior changed')
    if plan.get('tracker_environment'):
        import torch
        actual = dict(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        if actual != plan['tracker_environment']:
            raise ValueError('Tracker GPU environment differs from prepared plan')
    for s in plan['selection']:
        if s['key'] in checkpoint.done:
            continue
        if s.get('images'):
            demo.make_base(s)
        if (plan.get('tracker_mode') != 'on_demand' and s['stage'] == 'pipeline'
                and sha(out / 'tracker' / (s['key'] + '.json')) != s['tracker_sha256']):
            raise ValueError('tracker snapshot changed')
    # Exclusive marker prevents overlapping runs and automatic uncertain-outcome retries.
    with (out / 'execution_started.json').open('x') as stream:
        json.dump({'workers': workers, 'automatic_retry': True, 'deferred_retry_rounds': 3}, stream)
    budget = Budget(out / 'budget.sqlite', plan['limits'], sha(out / 'plan.json'))
    stop = threading.Event()
    decide, _, _ = app.load_gate()
    started = time.perf_counter()
    counter, lock = [len(checkpoint.done)], threading.Lock()
    input_runtime = tracker_runtime = None

    def task(s):
        if stop.is_set():
            raise BudgetStop('stopped before target')
        if s['key'] in checkpoint.done:
            return
        checkpoint.start(s['key'])
        if input_runtime is not None and (s['stage'] == 'pipeline' or 'cached_h0' not in s):
            s = input_runtime.resolve(s)
        delegate = GuardedCalls(out, plan, s, budget, stop)
        proposal_model = ('google/gemini-3.8-flash' if s['key'] in plan.get('legacy_proposal_targets', [])
                          else plan.get('proposal_model','google/gemini-3.8-flash'))
        if proposal_model == 'qwen3.8-max':
            from scripts.testing_qwen_h0 import ProposalCalls
            delegate = ProposalCalls(delegate,out,plan,s,budget,stop)
        try:
            # Persist local evidence even if H0 or a later remote call fails.
            snapshot = (tracker_runtime.snapshot(s) if tracker_runtime is not None and s['stage'] == 'pipeline'
                        else None)
            if 'cached_h0' in s:
                h0 = s['cached_h0']
            else:
                wire = app.frozen.gemini_h0_wire(make_h0_base(s))
                if plan.get('base_model') == 'qwen3.8-max':
                    from scripts.testing_qwen_h0 import call_h0
                    raw = call_h0(out, plan, s, budget, stop, wire)
                else:
                    raw = delegate.call(s['key'], 'h0', 'base', wire)
                if raw is None and getattr(delegate, 'last_http_failure', None):
                    write(out/'unavailable_h0'/(s['key']+'.json'),
                          {**delegate.last_http_failure, 'prediction': None, 'automatic_retry': False})
                    with lock:
                        counter[0] += 1
                    return  # Finish the rest of the batch without inventing a prediction.
                h0 = app.frozen.gated.h0_from_raw(raw)
            write(out / 'h0' / (s['key'] + '.json'), h0)
            if s['stage'] == 'pipeline':
                selected = {**s, 'cached_h0': h0}
                backend = demo.Backend(delegate, demo.make_base(s), selected)
                prior = read(out / 'priors' / (s['video_id'] + '.json'))
                if snapshot is None:
                    snapshot = read(out / 'tracker' / (s['key'] + '.json'))
                result = run_target(backend, selected, prior, decide,
                    tracker_snapshot=snapshot,
                    phase_filter=CausalPhaseFilter(0), output='v2.2', inference_split='Testing')
                class Replay:
                    parse_h0 = staticmethod(deepcopy)
                    normalize_compact = staticmethod(app.frozen.common.normalize_five)
                    def h0(self): return deepcopy(h0)
                    def proposal(self, *args): return deepcopy(backend.proposal_raw)
                    def compact(self, seat, *args): return deepcopy(backend.review_raw[seat])
                    def phase_recommendation(self, *args): return deepcopy(backend.phase_recommendation_raw)
                    def joint(self, seat, *args): return deepcopy(backend.joint_raw[seat])
                    def five_head(self, seat, *args): return deepcopy(backend.five_head_raw[seat])
                without = run_target(Replay(), selected, prior, decide, tracker_snapshot=None,
                    phase_filter=CausalPhaseFilter(0), output='v2.2', inference_split='Testing')
                if result['call_keys'] != without['call_keys']:
                    raise ValueError('tracker changed Gate/review routing')
                result.update(video_id=s['video_id'], frame_id=s['frame_id'], source_split='Testing',
                              without_tracker=without['prediction'], cached_h0='cached_h0' in s,
                              base_model=plan.get('base_model','google/gemini-3.8-flash'),
                              proposal_model=proposal_model)
                if result['phase_review_enabled']:
                    result.update(phase_recommendation_raw=getattr(backend, 'phase_recommendation_raw', None),
                                  joint_raw=getattr(backend, 'joint_raw', {}))
                if result['review_mode'] == 'five_head_probe':
                    result['five_head_raw'] = deepcopy(backend.five_head_raw)
                write(out / 'results' / (s['key'] + '.json'), result)
            checkpoint.finish(s['key'])
            with lock:
                counter[0] += 1
        finally:
            delegate.close()

    try:
        if plan.get('input_mode') == 'on_demand':
            from scripts.testing_half_streaming import Inputs, Tracker
            input_runtime = Inputs(plan['selection'], out)
            tracker_runtime = Tracker(out, plan['tracker_checkpoint_sha256'], plan['legacy_tracker_sha256'])
        with app.frozen.joint.credential_context(plan), app.frozen.joint.roster.lightweight_protocol():
            remaining = [s for s in plan['selection'] if s['key'] not in checkpoint.done]
            print(f"Checkpoint: {len(checkpoint.done)} saved targets; {len(remaining)} remaining", flush=True)
            deferred_map(task, remaining, workers, stop, out)
        unavailable = list((out/'unavailable_h0').glob('*.json'))
        if unavailable:
            raise ValueError(f'Batch drained but {len(unavailable)} H0 predictions unavailable; outputs remain incomplete')
        count = finalize(out, plan['selection'])
        if count != plan['evaluation_target_frames']:
            raise ValueError('incomplete scoring scope')
        write(out / 'receipt.json', dict(state='PASS', rows=count, workers=workers,
            budget=budget.summary(), downstream_seconds=time.perf_counter()-started,
            predictions_sha256=sha(out / 'predictions.json'), plan_sha256=sha(out / 'plan.json'),
            continuous_predictions_sha256=sha(out / 'continuous_predictions.json'), ram=False))
        print('PASS: predictions sealed; run score next', flush=True)
    except BaseException as exc:
        write(out / 'failure.json', dict(state='STOPPED', error_type=type(exc).__name__, message=str(exc),
              completed_targets=counter[0], budget=budget.summary()))
        raise
    finally:
        if input_runtime is not None:
            input_runtime.close()
        budget.close()
        checkpoint.close()


def score(out):
    # Reuse the established strict task-mask scorer, then replace demo-only metadata.
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        demo.score(out, 'scores_detail.json')
    report = read(out / 'scores_detail.json')
    report.update(rows=len(read(out / 'predictions.json')),
                  selection='Frozen contiguous half-duration interval per Testing video; 7823 annotated frames',
                  scorer_sha256=sha(Path(__file__)))
    write(out / 'scores_detail.json', report)
    write(out / 'scores.json', {k: v for k, v in report.items() if k != 'details'})
    print(json.dumps({'rows': report['rows'], 'arms': report['arms'], 'scores': str(out/'scores.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'execute', 'score', 'status'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', type=Path, default=SCOPE)
    parser.add_argument('--budget-limits', type=Path, default=SCOPE / 'suggested_budget_limits.json')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--resume-prepare', action='store_true', help='Retry interrupted local preparation in place')
    args = parser.parse_args()
    out = args.output.resolve()
    if args.command == 'prepare': prepare(out, args.scope.resolve(), args.budget_limits.resolve(), resume=args.resume_prepare)
    elif args.command == 'execute': execute(out, args.workers, args.allow_paid)
    elif args.command == 'score': score(out)
    else:
        print(json.dumps({'h0_completed': len(list((out/'h0').glob('*.json'))),
                          'pipeline_completed': len(list((out/'results').glob('*.json'))),
                          'sealed': (out/'receipt.json').exists(), 'stopped': (out/'failure.json').exists()}))


if __name__ == '__main__':
    main()
