"""Compare GT box policies on all effective annotations, without inference."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.data.parser import parse_annotation_file
from surgical_agent.data.schemas import DatasetSplit
from surgical_agent.tracking.box_policy import (
    BBoxPolicy,
    apply_bbox_policy,
    new_box_audit,
    record_box_policy_result,
)
from surgical_agent.tracking.training_data import (
    build_detection_training_records,
    supervision_qualified_training_video_ids,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = ROOT / "artifacts/preflight/tracker_remediation_20260905/box_policy_full_annotation_audit.json"
    if output.exists():
        raise FileExistsError(output)
    adapter = CholecTrack20DatasetAdapter("D:/cholec_dataset")
    report = {"schema_version": "tracker_box_policy_comparison_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha(__file__), "repair_manifest_sha256": sha(adapter.dataset_root / "repair_manifest.json"),
              "no_training_or_inference": True, "testing_scope": "GT box availability counts only, no test scores or class distributions",
              "box_audit_scope": "instrument-valid instances on effective routes; VID30 repaired; VID31 instance supervision disabled",
              "videos": {}, "training_records": {}, "splits": {}}
    for entry in adapter.entries.values():
        print(f"Count boxes: {entry.video_id}", flush=True)
        audits = {p.value: new_box_audit(p) for p in BBoxPolicy}
        frame_counts = Counter()
        supervised_frames = 0
        if entry.split is DatasetSplit.TESTING:
            annotation = parse_annotation_file(entry.annotation_file, expected_split=entry.split)
            frames = (f.instances for f in annotation.frames)
        else:
            frames = (s.evaluation.instances for s in adapter.iter_video(entry.video_id)
                      if s.evaluation is not None and s.evaluation.instance_supervision_available)
        for instances in frames:
            supervised_frames += 1
            for policy in BBoxPolicy:
                kept = 0
                for instance in instances:
                    if not instance.mask.instrument:
                        continue
                    result = apply_bbox_policy(instance.bbox, policy=policy)
                    record_box_policy_result(audits[policy.value], result)
                    kept += not result.dropped
                frame_counts[policy.value] += bool(kept)
        report["videos"][entry.video_id] = {"split": entry.split.value, "instance_supervised_frames": supervised_frames,
                                            "frames_with_kept_boxes": dict(frame_counts), "policies": audits}
    qualified = supervision_qualified_training_video_ids(adapter)
    for policy in BBoxPolicy:
        audit = new_box_audit(policy)
        records = build_detection_training_records(adapter, qualified, bbox_policy=policy, box_audit=audit)
        report["training_records"][policy.value] = {
            "records": len(records), "gt_boxes": sum(len(r.targets) for r in records),
            "per_video_records": dict(Counter(r.video_id for r in records)), "box_audit": audit,
        }
        assert len(records) == sum(report["videos"][v]["frames_with_kept_boxes"][policy.value] for v in qualified)
    for split in DatasetSplit:
        report["splits"][split.value] = {}
        rows = [v for v in report["videos"].values() if v["split"] == split.value]
        for policy in BBoxPolicy:
            total = new_box_audit(policy)
            for row in rows:
                for key,value in row["policies"][policy.value].items():
                    if key != "bbox_policy":
                        total[key] += int(value)
            total["kept_gt_boxes"] = total["kept_unchanged"] + total["clipped"]
            assert total["input_boxes"] == total["kept_gt_boxes"] + total["dropped_total"]
            report["splits"][split.value][policy.value] = total
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha(output), "training_records": report["training_records"], "splits": report["splits"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
