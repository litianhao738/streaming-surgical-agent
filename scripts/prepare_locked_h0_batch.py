"""Prepare the selected Batch H0 protocol offline; never submit paid API jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.perception.locked_baseline import (
    build_locked_batch_request,
    canonical_bytes,
    load_baseline_lock,
    verify_locked_batch_request,
)


def prepare(args):
    lock = load_baseline_lock()
    if args.check_lock:
        print(json.dumps({"status": "LOCK_VERIFIED", "baseline_id": lock["baseline_id"],
                          "protocol_sha256": lock["protocol_sha256"], "provider_calls": 0}))
        return
    if (args.dataset_root is None or args.video_id is None or args.start is None
            or args.output is None or args.count < 1):
        raise ValueError("require dataset-root, video-id, start, output and positive count")
    dataset = args.dataset_root.resolve()
    output = args.output.resolve()
    if output == dataset or output.is_relative_to(dataset):
        raise ValueError("output must be outside the read-only dataset")
    if output.exists():
        raise ValueError("use a new output directory; existing results are preserved")
    import torch

    torch.set_num_threads(2)
    adapter = CholecTrack20DatasetAdapter(dataset, causal_window_size=3)
    samples = []
    for sample in adapter.iter_inference_video(args.video_id):
        if sample.target_frame_id >= args.start:
            samples.append(sample)
            if len(samples) == args.count:
                break
    if len(samples) != args.count or samples[0].target_frame_id != args.start:
        raise ValueError("exact start or requested target count unavailable")
    loader = CausalApiMediaLoader()
    contexts = CausalPerceptionContextBuilder(
        max_frames=3, max_images=3, selection_strategy="fixed_all",
        history_image_detail="low", target_image_detail="high",
    )
    output.mkdir(parents=True)
    rows, shards, shard, count, size = [], [], None, 0, 0
    try:
        for sample in samples:
            loaded = loader.load(sample)
            context = contexts.build(loaded.runtime_sample, loaded.frames, workflow_snapshot={},
                                     memory_snapshot={}, prior_finalized_prediction=None)
            request = build_locked_batch_request(video_id=sample.video_id,
                target_frame_id=sample.target_frame_id, images=context.images)
            verify_locked_batch_request(request)
            line = canonical_bytes(request)+b"\n"
            if (shard is None or size+len(line) > lock["max_jsonl_file_bytes"]
                    or count >= lock["max_jsonl_requests"]):
                if shard is not None:
                    shard.close()
                path = output / f"requests_{len(shards):03d}.jsonl"
                shard = path.open("xb")
                shards.append(path)
                size, count = 0, 0
            shard.write(line)
            size += len(line)
            count += 1
            rows.append({"custom_id": request["custom_id"], "source_split": sample.source_split.value,
                         "causal_frame_ids": list(sample.causal_frame_ids),
                         "request_sha256": hashlib.sha256(line[:-1]).hexdigest(),
                         "images": [{"id": image.identifier, "sha256": image.sha256}
                                    for image in context.images]})
    finally:
        if shard is not None:
            shard.close()
    manifest = {"status": "PREPARED_NOT_SUBMITTED", "baseline_id": lock["baseline_id"],
                "protocol_sha256": lock["protocol_sha256"], "baseline_lock": lock,
                "provider_calls": 0, "target_count": len(rows), "rows": rows,
                "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                           "size_bytes": p.stat().st_size} for p in shards]}
    atomic_write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "baseline_id": lock["baseline_id"],
                      "targets": len(rows), "files": len(shards), "provider_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-lock", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--video-id")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path)
    prepare(parser.parse_args())
