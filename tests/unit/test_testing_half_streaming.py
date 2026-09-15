import base64
from copy import deepcopy
import json

import pytest
from scripts import testing_half_streaming as live


def test_delivery_is_incremental_and_preserves_missing_detail(tmp_path, monkeypatch):
    folder = tmp_path/'delivery'; folder.mkdir()
    selected, chunks = [], []
    for i in range(2):
        frame = 100+25*i
        s = dict(key=f'VID01_{frame}', video_id='VID01', frame_id=frame,
                 causal_frame_ids=[frame-50, frame-25, frame], cached_h0={},
                 prepared_dir=str(folder), custom_id=str(i), stage='pipeline')
        selected.append(s)
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+
                    base64.b64encode(f'image-{f}'.encode()).decode()}} for f in s['causal_frame_ids']]
        path = folder/f'{i}.jsonl'
        path.write_text(json.dumps({'custom_id': str(i), 'body': {'messages': [{'content': content}]}})+'\n')
        chunks.append({'path': path.name, 'sha256': live.core.sha(path)})
    live.core.write(folder/'manifest.json', {'chunks': chunks})
    hashes = []
    original = live.core.demo.stream_sha
    def sha(path): hashes.append(path.name); return original(path)
    monkeypatch.setattr(live.core.demo, 'stream_sha', sha)
    out = tmp_path/'run'
    inputs = live.Inputs(selected, out)
    try:
        assert not out.exists() and not hashes
        first = inputs.resolve(selected[0])
        assert hashes == ['0.jsonl']  # no scan/materialization of the next chunk yet
        assert [im['detail'] for im in first['images']] == ['auto']*3
        assert live.Path(first['images'][0]['path']).read_bytes() == b'image-50'
        inputs.resolve(selected[1])
        assert hashes == ['0.jsonl', '1.jsonl']
        assert inputs.resolve(selected[0]) == first  # out-of-order reuse, no rewind
    finally:
        inputs.close()


def test_legacy_tracker_reuse_requires_bound_hash_and_never_loads_model(tmp_path, monkeypatch):
    training = tmp_path/'training'; training.mkdir()
    (training/'checkpoint.pt').write_bytes(b'checkpoint')
    checkpoint = live.core.sha(training/'checkpoint.pt')
    live.core.write(training/'training_manifest.json', {'checkpoint_sha256': checkpoint})
    monkeypatch.setattr(live.core.demo, 'TRACKER', training)
    s = dict(key='VID01_100', video_id='VID01', frame_id=100, causal_frame_ids=[50,75,100],
             images=[{'frame_id': f, 'sha256': str(f)} for f in [50,75,100]])
    original = dict(status='AVAILABLE', video_id='VID01', source_max_frame_id=100,
                    frames=[{'frame_id': f, 'tracks': []} for f in [50,75,100]],
                    checkpoint_sha256=checkpoint)
    path = tmp_path/'tracker/VID01_100.json'
    live.core.write(path, original)
    tracker = live.Tracker(tmp_path, checkpoint, {'VID01_100': live.core.sha(path)}, cache_root=tmp_path/'shared')
    monkeypatch.setattr(tracker, '_load', lambda: pytest.fail('cached target must not run the model'))
    result = tracker.snapshot(s)
    assert result['frames'] == original['frames'] and result['reused_local_preparation']
    assert live.core.read(path) == original  # original cache preserved
    assert tracker.snapshot(s) == result
    other = live.Tracker(tmp_path/'other_run', checkpoint, cache_root=tmp_path/'shared')
    monkeypatch.setattr(other, '_load', lambda: pytest.fail('another experiment must reuse cached results'))
    assert other.snapshot(s) == result
    assert (tmp_path/'other_run/tracker_runtime/VID01_100.json').exists()
    changed = deepcopy(s); changed['images'][0]['sha256'] = 'changed'
    with pytest.raises(ValueError, match='binding mismatch'):
        tracker.snapshot(changed)


def test_one_command_requires_paid_flag_and_uses_existing_local_directory(tmp_path, monkeypatch):
    from scripts import run_testing_half_complete as complete
    calls = []
    monkeypatch.setattr(complete, 'prepare', lambda *args, **kwargs: calls.append(('prepare', kwargs)))
    monkeypatch.setattr(complete, 'execute', lambda *args: calls.append(('execute', args[1:])))
    with pytest.raises(ValueError, match='allow-paid'):
        complete.run(tmp_path, None, '30', tmp_path/'limits', 8, False)
    assert not calls
    complete.run(tmp_path, None, '30', tmp_path/'limits', 8, True)
    assert calls == [('prepare', {'resume': True}), ('execute', (8, True))]
