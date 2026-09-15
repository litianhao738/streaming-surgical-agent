import json

import pytest

from scripts.testing_tracker_cache import SnapshotCache


def test_cache_separates_inputs_weights_and_video_and_detects_corruption(tmp_path):
    sample = dict(video_id='VID01', frame_id=100, images=[dict(frame_id=100, sha256='image')])
    cache = SnapshotCache(tmp_path, {'checkpoint_sha256': 'weights'})
    snapshot = {'frames': [], 'inference_runtime': {'device': 'cpu'}}
    with cache.entry(sample) as (path, identity):
        cache.write(path, identity, snapshot)
        assert cache.read(path, identity) == snapshot
    for different, contract in [
        ({**sample, 'video_id': 'VID02'}, cache.contract),
        ({**sample, 'images': [dict(frame_id=100, sha256='different')]}, cache.contract),
        (sample, {'checkpoint_sha256': 'new_weights'}),
    ]:
        other = SnapshotCache(tmp_path, contract)
        with other.entry(different) as (other_path, other_identity):
            assert other_path != path and other.read(other_path, other_identity) is None
    value = json.loads(path.read_text())
    value['snapshot']['frames'] = ['corrupted']
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='integrity mismatch'):
        cache.read(path, identity)
