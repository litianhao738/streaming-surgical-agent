"""On-demand image loading and trained Tracker inference for the half-Testing runner."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import threading

from scripts import run_testing_half_pipeline as core


class Inputs:
    """Read delivery chunks incrementally; verify each opened chunk once, preserve JPEGs."""
    def __init__(self, selection, out):
        self.out = out
        self.lock = threading.RLock()
        self.ready, self.streams, self.decoded = {}, {}, {}
        self.wanted = {(s['prepared_dir'], s['custom_id']): s
                       for s in selection if 'cached_h0' in s}
        self.probes = {}
        for s in selection:
            if 'cached_h0' in s:
                self.probes.setdefault(s['video_id'], s)

    def _cached(self, s):
        if s['key'] in self.ready:
            return deepcopy(self.ready[s['key']])
        directory = s['prepared_dir']
        if directory not in self.streams:
            chunks = iter(core.read(Path(directory)/'manifest.json')['chunks'])
            self.streams[directory] = {'chunks': chunks, 'stream': None}
        state = self.streams[directory]
        while s['key'] not in self.ready:
            if state['stream'] is None:
                chunk = next(state['chunks'], None)
                if chunk is None:
                    raise ValueError('missing original request: '+s['key'])
                path = Path(directory)/chunk['path'].replace('\\', '/')
                print('Reading source chunk on demand: '+path.name, flush=True)
                if core.demo.stream_sha(path) != chunk['sha256']:
                    raise ValueError('delivered chunk changed')
                state.update(stream=path.open(encoding='utf-8-sig'), chunk=chunk)
            line = state['stream'].readline()
            if not line:
                state['stream'].close(); state['stream'] = None
                continue
            item = json.loads(line)
            wanted = self.wanted.get((directory, item['custom_id']))
            if wanted is None:
                continue
            blocks = [b for m in item['body']['messages'] if isinstance(m['content'], list)
                      for b in m['content'] if b.get('type') == 'image_url']
            if len(blocks) != 3:
                raise ValueError('three original JPEGs required')
            images = []
            for frame, block in zip(wanted['causal_frame_ids'], blocks, strict=True):
                prefix, encoded = block['image_url']['url'].split(',', 1)
                detail = block['image_url'].get('detail', 'auto')
                if prefix != 'data:image/jpeg;base64' or detail not in ('auto', 'low', 'high'):
                    raise ValueError('invalid delivered image')
                images.append(core.store_image(self.out, base64.b64decode(encoded, validate=True), frame, detail))
            self.ready[wanted['key']] = {**wanted, 'images': images,
                                         'input_chunk_sha256': state['chunk']['sha256']}
        return deepcopy(self.ready[s['key']])

    def _new(self, s):
        import cv2
        import numpy as np
        video = s['video_id']
        if video not in self.decoded:
            probe = self._cached(self.probes[video])
            expected = cv2.imread(probe['images'][-1]['path'])
            path = core.demo.DATASET/'Testing'/video/(video.lower()+'.mp4')
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-25) > .01:
                cap.release()
                raise ValueError('video unavailable or FPS mismatch')
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, probe['frame_id']-1)
                ok, image = cap.read()
                if not ok:
                    raise ValueError('alignment probe decode failed')
                image = cv2.resize(image, (expected.shape[1], expected.shape[0]))
                mae = float(np.abs(image.astype(float)-expected.astype(float)).mean())
                if mae > 5:
                    raise ValueError('decoder alignment mismatch: '+video)
                core.write(self.out/'alignment_live'/(video+'.json'),
                           {'frame_id': probe['frame_id'], 'mae': mae, 'decoder_offset': -1})
            except BaseException:
                cap.release()
                raise
            self.decoded[video] = {'cap': cap, 'size': (expected.shape[1], expected.shape[0]), 'frames': {}}
        state = self.decoded[video]
        images = []
        for frame in s['causal_frame_ids']:
            if frame not in state['frames']:
                state['cap'].set(cv2.CAP_PROP_POS_FRAMES, frame-1)
                ok, image = state['cap'].read()
                if not ok:
                    raise ValueError('frame decode failed')
                image = cv2.resize(image, state['size'])
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise ValueError('JPEG encoding failed')
                state['frames'][frame] = core.store_image(self.out, encoded.tobytes(), frame, 'high')
            images.append({**state['frames'][frame], 'detail': 'high' if frame == s['frame_id'] else 'low'})
        return {**s, 'images': images}

    def resolve(self, s):
        with self.lock:
            resolved = self._cached(s) if 'cached_h0' in s else self._new(s)
            core.demo.make_base(resolved)  # Verify input bytes before any paid request.
            core.write(self.out/'resolved_inputs'/(s['key']+'.json'), resolved)
            return resolved

    def close(self):
        for state in self.streams.values():
            if state['stream'] is not None:
                state['stream'].close()
        for state in self.decoded.values():
            state['cap'].release()


class Tracker:
    """One GPU model per process, with durable, cross-experiment target reuse."""
    def __init__(self, out, checkpoint_sha256, legacy_hashes=None, *, device='cuda:0', cache_root=None):
        from scripts.testing_tracker_cache import SnapshotCache
        self.out, self.expected = out, checkpoint_sha256
        self.device = device
        self.legacy_hashes = legacy_hashes or {}
        self.manifest = core.read(core.demo.TRACKER/'training_manifest.json')
        if self.manifest['checkpoint_sha256'] != self.expected or core.sha(core.demo.TRACKER/'checkpoint.pt') != self.expected:
            raise ValueError('Tracker checkpoint drift')
        self.model = None
        self.lock, self.detections = threading.RLock(), {}
        self.cache = SnapshotCache(cache_root or core.ROOT/'artifacts/cache/tracker_testing',
            dict(checkpoint_sha256=self.expected, config=self.manifest.get('config', {}),
                 association_max_frame_id_gap=25,
                 implementation_sha256={p.name: core.sha(p) for p in
                     (core.ROOT/'src/surgical_agent/tracking').glob('*.py')}))

    def _load(self):
        import torch
        from surgical_agent.tracking.config import TrackerTrainingConfig
        from surgical_agent.tracking.detector import build_instrument_detector, load_tracker_checkpoint
        self.config = TrackerTrainingConfig(**self.manifest['config'])
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: launch with .venv-tracker-gpu/Scripts/python.exe')
        torch.set_num_threads(4)
        self.model = build_instrument_detector(self.config, use_pretrained=False)
        load_tracker_checkpoint(core.demo.TRACKER/'checkpoint.pt', model=self.model, map_location='cpu')
        self.model.to(self.device).eval()
        print('Trained Tracker loaded once on '+self.device+'; shared cache enabled', flush=True)

    def snapshot(self, s):
        from surgical_agent.research.gate.tracker_pipeline_v2 import current_classes
        binding = [{'frame_id': im['frame_id'], 'sha256': im['sha256']} for im in s['images']]
        path = self.out/'tracker_runtime'/(s['key']+'.json')
        with self.lock, self.cache.entry(s) as (shared_path, identity):
            shared = self.cache.read(shared_path, identity)
            if path.exists():
                snapshot = core.read(path)
                if (snapshot.get('input_images') != binding or snapshot.get('checkpoint_sha256') != self.expected
                        or current_classes(snapshot, s) is None):
                    raise ValueError('runtime Tracker cache binding mismatch')
                snapshot.setdefault('inference_runtime', {'device': 'cpu', 'origin': 'legacy_runtime'})
                if shared is None:
                    self.cache.write(shared_path, identity, snapshot)
                return snapshot
            if shared is not None:
                if (shared.get('input_images') != binding or shared.get('checkpoint_sha256') != self.expected
                        or current_classes(shared, s) is None):
                    raise ValueError('shared Tracker snapshot invalid')
                core.write(path, shared)
                return shared
            # Legacy files belong to this same interrupted preparation and were SHA-bound
            # in the new plan. Keep the originals, adopt into the byte-bound runtime cache.
            legacy = self.out/'tracker'/(s['key']+'.json')
            if s['key'] in self.legacy_hashes:
                if core.sha(legacy) != self.legacy_hashes[s['key']]:
                    raise ValueError('legacy Tracker cache changed')
                snapshot = core.read(legacy)
                if (snapshot.get('checkpoint_sha256') != self.expected or current_classes(snapshot, s) is None
                        or [f['frame_id'] for f in snapshot['frames']] != s['causal_frame_ids']):
                    raise ValueError('legacy Tracker provenance mismatch')
                snapshot = {**snapshot, 'input_images': binding, 'reused_local_preparation': True}
                snapshot['inference_runtime'] = {'device': 'cpu', 'origin': 'legacy_preparation'}
                self.cache.write(shared_path, identity, snapshot)
                core.write(path, snapshot)
                return snapshot
            if self.model is None:
                self._load()
            import torch
            from PIL import Image
            from torchvision.transforms.functional import to_tensor
            from surgical_agent.tracking.detector import decode_detections
            from surgical_agent.tracking.associator import CausalHungarianAssociator
            association = CausalHungarianAssociator(iou_threshold=self.config.association_iou_threshold,
                max_age=self.config.max_age, max_frame_id_gap=25)
            association.reset(s['video_id'])
            frames = []
            for im in s['images']:
                if core.sha(im['path']) != im['sha256']:
                    raise ValueError('Tracker image changed')
                if im['sha256'] not in self.detections:
                    with Image.open(im['path']) as image:
                        rgb = image.convert('RGB'); width, height = rgb.size; tensor = to_tensor(rgb).to(self.device)
                    with torch.inference_mode():
                        detected = self.model([tensor])[0]
                    self.detections[im['sha256']] = decode_detections(detected, width=width, height=height,
                                                                     score_threshold=self.config.score_threshold)
                tracks = association.update(im['frame_id'], self.detections[im['sha256']])
                frames.append({'frame_id': im['frame_id'], 'tracks': [t.as_mapping() for t in tracks]})
            snapshot = dict(status='AVAILABLE', video_id=s['video_id'], source_split='Testing',
                source_max_frame_id=s['frame_id'], frames=frames, checkpoint_sha256=self.expected,
                input_images=binding, reused_local_preparation=False)
            snapshot['inference_runtime'] = dict(device=str(next(self.model.parameters()).device),
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(self.device) if self.device.startswith('cuda') else None)
            self.cache.write(shared_path, identity, snapshot)
            core.write(path, snapshot)
            return snapshot
