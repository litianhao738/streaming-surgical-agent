"""Content-bound Testing Tracker snapshots shared by independent experiments."""
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager

from filelock import FileLock


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class SnapshotCache:
    def __init__(self, root, contract):
        self.root = Path(root)
        self.contract = contract

    def identity(self, s):
        return dict(protocol='testing-three-frame-reset-v1', source_split='Testing',
                    video_id=s['video_id'], frame_id=s['frame_id'], contract=self.contract,
                    images=[{'frame_id': i['frame_id'], 'sha256': i['sha256']} for i in s['images']])

    @contextmanager
    def entry(self, s):
        identity = self.identity(s)
        key = digest(identity)
        path = self.root/digest(self.contract)/s['video_id']/(key+'.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        # Independent processes cannot infer and publish the same target concurrently.
        with FileLock(str(path)+'.lock', timeout=600):
            yield path, identity

    def read(self, path, identity):
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding='utf-8'))
        if value['identity'] != identity or digest(value['snapshot']) != value['snapshot_sha256']:
            raise ValueError('shared Tracker cache integrity mismatch')
        return value['snapshot']

    def write(self, path, identity, snapshot):
        # Use the project atomic writer through the caller's serialization convention.
        from scripts.run_testing_half_pipeline import write
        write(path, dict(identity=identity, snapshot=snapshot, snapshot_sha256=digest(snapshot)))
