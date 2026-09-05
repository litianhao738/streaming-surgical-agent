"""Offline exact-image and raw-GT audit of one completed pure H0 smoke."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from surgical_agent.artifacts.manifest import atomic_write_json, sha256_file
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.api_media import CausalApiMediaLoader
from surgical_agent.config.loader import load_api_config
from surgical_agent.perception.context_builder import CausalPerceptionContextBuilder
from surgical_agent.api.request_hash import canonical_request_metadata
from scripts.run_pure_h0_smoke import SchemaExplicitRequestBuilder, FinalOnlyRequestBuilder, evaluate_offline


def run(args):
    predictions = json.loads((args.run / "predictions.json").read_text(encoding="utf-8"))
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    ledger = [json.loads(line) for line in (args.run / "api_usage.jsonl").read_text().splitlines()]
    usage = {row["request_hash"]: row for row in ledger}
    config = load_api_config(args.config)
    adapter = CholecTrack20DatasetAdapter(args.dataset, causal_window_size=3)
    selected = {row["frame_id"]: row for row in predictions}
    annotation_path = args.dataset / "Validation" / args.video / (args.video.lower() + ".json")
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    context_builder = CausalPerceptionContextBuilder(max_frames=3, max_images=3,
        selection_strategy="fixed_all", history_image_detail="low", target_image_detail="auto")
    checks = []
    for sample in adapter.iter_inference_video(args.video):
        if sample.target_frame_id not in selected:
            continue
        loaded = CausalApiMediaLoader().load(sample)
        context = context_builder.build(loaded.runtime_sample, loaded.frames,
            workflow_snapshot={}, memory_snapshot={}, prior_finalized_prediction=None)
        request = SchemaExplicitRequestBuilder(config=config).build(context)
        final = FinalOnlyRequestBuilder(config=config).build(context)
        metadata = canonical_request_metadata(request)
        recorded = usage[metadata.request_hash]["request"]
        assert [dict(identifier=x.identifier, sha256=x.sha256) for x in metadata.images] == [
            dict(identifier=x["identifier"], sha256=x["sha256"]) for x in recorded["images"]]
        assert final.images == request.images
        assert final.payload["input_text"] == request.payload["input_text"]
        assert final.generation_parameters == request.generation_parameters
        assert all(int(Path(path).stem) == fid for path, fid in zip(sample.media_refs, sample.causal_frame_ids))
        instances = raw["annotations"][str(sample.target_frame_id)]
        gt = {task: sorted({item[key] for item in instances}) for task, key in
              (("instrument", "instrument"), ("verb", "verb"), ("target", "target"), ("ivt", "triplet"), ("phase", "phase"))}
        expected = next(row for row in summary["evaluation"]["frames"] if row["frame_id"] == sample.target_frame_id)
        assert gt == expected["gt"]
        checks.append({"frame_id": sample.target_frame_id,
                       "causal_frame_ids": list(sample.causal_frame_ids),
                       "media_paths": list(sample.media_refs),
                       "uploaded_image_hashes_reproduced": True,
                       "final_only_has_identical_visual_input": True,
                       "raw_gt_equals_scored_gt": True,
                       "raw_instance_count": len(instances), "gt": gt,
                       "raw_tool_boxes": [item["tool_bbox"] for item in instances]})
    assert len(checks) == len(predictions)
    assert evaluate_offline(adapter, args.video, predictions) == summary["evaluation"]
    result = {"annotation_path": str(annotation_path), "annotation_sha256": sha256_file(annotation_path),
              "reference_run": str(args.run), "scoring_reproduced": True, "frames": checks}
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", default="VID110")
    run(parser.parse_args())
