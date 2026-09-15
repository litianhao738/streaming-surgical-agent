from contextlib import nullcontext
from copy import deepcopy
import json
import threading
import time

import pytest
from scripts import run_testing_half_pipeline as runner
from tests.unit.test_gate_ready_mainline import inputs, RecordingMock


@pytest.mark.parametrize('action', [0, 1])
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('phase_enabled,review_mode', [(False, 'separate'), (True, 'separate'), (True, 'unified')])
def test_execute_cached_new_warmup_and_replay_offline(tmp_path, monkeypatch, inputs, action, streaming, phase_enabled, review_mode):
    base, selected, prior, _ = inputs
    mock = RecordingMock()
    raw = mock.call('seed', 'h0', 'base', runner.app.frozen.gemini_h0_wire(base))
    h0 = runner.app.frozen.gated.h0_from_raw(raw)
    selection = []
    for i, (stage, cached) in enumerate([('warmup_h0_only', True), ('warmup_h0_only', False),
                                        ('pipeline', True), ('pipeline', False)]):
        frame = 100 + i*25
        s = {**selected, 'frame_id': frame, 'key': f'query_{frame}',
             'causal_frame_ids': [frame-50, frame-25, frame], 'source_split': 'Testing',
             'stage': stage, 'evaluation_target': stage == 'pipeline'}
        if cached:
            s['cached_h0'] = deepcopy(h0)
        if stage == 'pipeline':
            p = tmp_path/'tracker'/f'{s["key"]}.json'
            runner.write(p, {'status': 'AVAILABLE', 'video_id': 'query', 'source_max_frame_id': frame,
                             'frames': [{'frame_id': frame, 'tracks': []}]})
            s['tracker_sha256'] = runner.sha(p)
        selection.append(s)
    runner.write(tmp_path/'priors/query.json', prior)
    plan = {'profile': runner.PROFILE, 'selection': selection,
            'limits': {'openrouter_usd': '1'}, 'runtime_sha256': {},
            'prior_sha256': {'query': runner.sha(tmp_path/'priors/query.json')},
            'evaluation_target_frames': 2}
    resolved, tracked = [], []
    if streaming:
        from scripts import testing_half_streaming as live
        class Inputs:
            def __init__(self, *args): pass
            def resolve(self, s): resolved.append(s['key']); return s
            def close(self): pass
        class Tracker:
            def __init__(self, *args): pass
            def snapshot(self, s):
                tracked.append(s['key'])
                return runner.read(tmp_path/'tracker'/f'{s["key"]}.json')
        monkeypatch.setattr(live, 'Inputs', Inputs)
        monkeypatch.setattr(live, 'Tracker', Tracker)
        plan.update(input_mode='on_demand', tracker_mode='on_demand',
                    tracker_checkpoint_sha256='test', legacy_tracker_sha256={})
    runner.write(tmp_path/'plan.json', plan)
    runner.write(tmp_path/'prepared.json', {'plan_sha256': runner.sha(tmp_path/'plan.json')})
    calls = {}
    class Calls(RecordingMock):
        def __init__(self, out, plan, s, budget, stop):
            super().__init__(); calls[s['key']] = self
        def close(self): pass
    monkeypatch.setattr(runner, 'GuardedCalls', Calls)
    monkeypatch.setattr(runner.demo, 'make_base', lambda s: base)
    monkeypatch.setattr(runner, 'make_h0_base', lambda s: base)
    def decide(_): return .5, action
    decide.phase_review_enabled = phase_enabled
    decide.review_mode = review_mode
    monkeypatch.setattr(runner.app, 'load_gate', lambda: (decide, {}, {}))
    monkeypatch.setattr(runner.app.frozen.joint, 'credential_context', lambda _: nullcontext())
    runner.execute(tmp_path, 8, True)
    assert runner.read(tmp_path/'receipt.json')['rows'] == 2
    assert not calls['query_100'].wires
    assert {k[1] for k in calls['query_125'].wires} == {'h0'}
    assert all(k[1] != 'h0' for k in calls['query_150'].wires)
    assert sum(k[1] == 'h0' for k in calls['query_175'].wires) == 1
    predictions = runner.read(tmp_path/'predictions.json')
    assert [r['frame_id'] for r in predictions] == [150, 175]
    assert all(r['output_modules']['tracker_available'] for r in predictions)
    for r in predictions:
        assert r['phase_review_enabled'] == phase_enabled
        stages = {key.split('|')[0] for key in r['call_keys']}
        assert ('joint_r1' in stages) == (phase_enabled and bool(action))
        if phase_enabled and action:
            assert r['joint_raw'] and 'phase_recommendation_raw' in r
            assert r['without_tracker']['phase'] == [r['phase_before_smoothing']]
        if review_mode == 'unified':
            assert r['logical_calls'] <= 9
            assert {k for k in r['call_keys'] if k.startswith('control_graph|')} == {'control_graph|qwen'}
    if streaming:
        assert set(resolved) == {'query_125', 'query_150', 'query_175'}
        assert set(tracked) == {'query_150', 'query_175'}
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path, 8, True)
    runner.write(tmp_path/'resume_policy.json', {'plan_sha256': runner.sha(tmp_path/'plan.json'),
                                              'amended_sources': {}, 'added_sources': {}})
    (tmp_path/'execution_started.json').unlink()
    monkeypatch.setattr(runner, 'GuardedCalls', lambda *args: pytest.fail('completed targets must not dispatch'))
    runner.execute(tmp_path, 8, True)
    assert runner.read(tmp_path/'predictions.json') == predictions


def test_global_queue_bounded_and_failure_stops_submission():
    active = [0, 0]
    lock = threading.Lock()
    def task(n):
        with lock:
            active[0] += 1; active[1] = max(active)
        time.sleep(.005)
        with lock: active[0] -= 1
    runner.bounded_map(task, range(32), 8, threading.Event())
    assert 1 < active[1] <= 8
    stop = threading.Event()
    seen = []
    def fail(n):
        seen.append(n)
        raise ValueError('stop')
    with pytest.raises(ValueError):
        runner.bounded_map(fail, range(100), 8, stop)
    assert stop.is_set() and len(seen) <= 8


def test_m3_finalize_sorts_and_excludes_warmup_and_annotation_gaps(tmp_path):
    selection = []
    for frame, phase, stage, evaluate in [(100, 2, 'warmup_h0_only', False),
            (125, 2, 'pipeline', False), (150, 3, 'pipeline', True)]:
        key = f'VID01_{frame}'
        selection.append(dict(key=key, video_id='VID01', frame_id=frame, stage=stage,
                              evaluation_target=evaluate))
        runner.write(tmp_path/'h0'/f'{key}.json', {'phase': [phase]})
        if stage == 'pipeline':
            runner.write(tmp_path/'results'/f'{key}.json',
                         {'key': key, 'cheap': {'phase': [phase]}, 'prediction': {'phase': [phase]}})
    assert runner.finalize(tmp_path, selection[::-1]) == 1
    assert runner.read(tmp_path/'predictions.json')[0]['prediction']['phase'] == [2]
    assert len(runner.read(tmp_path/'continuous_predictions.json')) == 2


def test_h0_wire_exact_protocol_and_three_images(tmp_path):
    s = dict(video_id='VID01', frame_id=100, causal_frame_ids=[50, 75, 100])
    s['images'] = [runner.store_image(tmp_path, f'jpeg{f}'.encode(), f,
                                     'high' if f == 100 else 'low') for f in s['causal_frame_ids']]
    wire = runner.app.frozen.gemini_h0_wire(runner.make_h0_base(s))
    content = wire['messages'][1]['content']
    text = json.loads(content[0]['text'])
    assert text['causal_frame_ids'] == [50, 75, 100]
    assert text['relative_seconds'] == [-2, -1, 0]
    assert [b['image_url']['detail'] for b in content[1:]] == ['low', 'low', 'high']
    assert wire['provider']['only'] == ['google-ai-studio']
    assert wire['max_tokens'] == 4096


def test_explicit_paid_flag_and_worker_bounds(tmp_path):
    with pytest.raises(ValueError, match='allow-paid'):
        runner.execute(tmp_path, 8, False)
    with pytest.raises(ValueError, match='workers'):
        runner.execute(tmp_path, 9, True)


@pytest.mark.parametrize('details', [(None, None, None), ('low', None, 'high')])
def test_delivered_images_without_detail_default_auto_and_retry(tmp_path, details):
    import base64
    folder = tmp_path/'delivery'
    folder.mkdir()
    content = []
    for i, detail in enumerate(details):
        image = {'url': 'data:image/jpeg;base64,'+base64.b64encode(f'original-jpeg-{i}'.encode()).decode()}
        if detail is not None:
            image['detail'] = detail
        content.append({'type': 'image_url', 'image_url': image})
    chunk = folder/'requests.jsonl'
    chunk.write_text(json.dumps({'custom_id': 'test_100', 'body': {'messages': [
        {'role': 'user', 'content': content}]}})+'\n', encoding='utf-8')
    runner.write(folder/'manifest.json', {'chunks': [{'path': chunk.name, 'sha256': runner.sha(chunk)}]})
    s = dict(custom_id='test_100', key='VID01_100', prepared_dir=str(folder), cached_h0={},
             stage='pipeline', causal_frame_ids=[50, 75, 100])
    out = tmp_path/'prepared'
    runner.extract_cached([s], out)
    assert [im['detail'] for im in s['images']] == [d or 'auto' for d in details]
    before = deepcopy(s['images'])
    for i, im in enumerate(s['images']):
        assert runner.Path(im['path']).read_bytes() == f'original-jpeg-{i}'.encode()
    runner.extract_cached([s], out)
    assert s['images'] == before


@pytest.mark.parametrize('marker', ['plan.json', 'prepared.json', 'execution_started.json', 'budget.sqlite', 'unknown.txt'])
def test_resume_prepare_cannot_overwrite_sealed_or_unknown_artifacts(tmp_path, marker):
    (tmp_path/marker).write_text('preserve')
    with pytest.raises(ValueError, match='sealed/unknown'):
        runner.check_prepare_destination(tmp_path, True, {'priors', 'images'})
    assert (tmp_path/marker).read_text() == 'preserve'


def test_prepare_requires_explicit_resume_for_local_leftovers(tmp_path):
    (tmp_path/'priors').mkdir()
    with pytest.raises(ValueError, match='resume-prepare'):
        runner.check_prepare_destination(tmp_path, False, {'priors', 'images'})
    runner.check_prepare_destination(tmp_path, True, {'priors', 'images'})
