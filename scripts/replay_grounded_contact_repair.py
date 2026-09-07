"""Replay saved grounded proposals/reviews with checked final-only admission.

No images are uploaded and no provider client is created. Saved GT/masks are
joined only after admission, for diagnosis rather than policy selection.
"""

import argparse
import hashlib
import json
import re
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.artifacts.manifest import atomic_write_json
from surgical_agent.evaluation.repair_comparison import compute_repair_comparison
from surgical_agent.research.verification.final_only_grounded import (
    CHECKED_REPAIR_VERSION,
    finalize_grounded_repair,
)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index(rows):
    result = {}
    for row in rows:
        frame_id = row["frame_id"]
        if type(frame_id) is not int or frame_id < 1 or frame_id in result:
            raise ValueError("invalid or duplicate frame identity")
        result[frame_id] = row
    return result


def _historical_cost(paths):
    stages = {}
    for path in paths:
        stage = path.parent.name.split("_", 1)[1]
        if stage not in {"h0", "h0_format_retry", "locator", "proposal", "review"}:
            raise ValueError("unrecognized historical call stage")
        totals = stages.setdefault(stage, {"provider_calls": 0, "priced_cost_usd": Decimal(0),
                                          "unpriced_calls": 0})
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            usage = json.loads(line)
            if usage["provider"] != "openrouter":
                raise ValueError("this historical ledger adapter expects OpenRouter USD")
            count = usage["provider_call_count"]
            if type(count) is not int or count < 0:
                raise ValueError("invalid provider call count")
            totals["provider_calls"] += count
            cost = usage.get("provider_cost")
            if cost is None:
                totals["unpriced_calls"] += count
            else:
                amount = Decimal(str(cost))
                if not amount.is_finite() or amount < 0:
                    raise ValueError("invalid historical provider cost")
                totals["priced_cost_usd"] += amount
    repair_stages = [v for k, v in stages.items() if not k.startswith("h0")]
    result = {
        "note": "Historical billed usage; replay makes zero new calls. Missing prices are not zero cost.",
        "provider_calls": sum(v["provider_calls"] for v in stages.values()),
        "priced_cost_usd": float(sum(v["priced_cost_usd"] for v in stages.values())),
        "unpriced_calls": sum(v["unpriced_calls"] for v in stages.values()),
        "repair_added_provider_calls": sum(v["provider_calls"] for v in repair_stages),
        "repair_added_priced_cost_usd": float(sum(v["priced_cost_usd"] for v in repair_stages)),
        "repair_unpriced_calls": sum(v["unpriced_calls"] for v in repair_stages),
        "by_stage": {k: {**v, "priced_cost_usd": float(v["priced_cost_usd"])}
                     for k, v in stages.items()},
    }
    return result


def replay(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(run):
        raise ValueError("use a fresh output directory outside the historical run")
    paths = [run / name for name in ("predictions.json", "summary.json", "plan.json")]
    usage_paths = sorted((run / "calls").glob("*/api_usage.jsonl"))
    if not usage_paths:
        raise ValueError("historical call ledger is required")
    paths.extend(usage_paths)
    before = {str(p.relative_to(run)): _hash(p) for p in paths}
    predictions = _index(_read(run / "predictions.json"))
    summary, plan = _read(run / "summary.json"), _read(run / "plan.json")
    scored = _index(summary["evaluation"]["h0"]["frames"])
    expected = plan["targets"]
    if (len(expected) != len(set(expected)) or set(predictions) != set(expected)
            or set(scored) != set(expected)):
        raise ValueError("predictions, GT masks and planned targets must match exactly")
    metadata = plan["input_metadata"]
    video_ids = set()
    for fid in expected:
        identifier = metadata[str(fid)]["images"][-1]["identifier"]
        match = re.fullmatch(r"cholectrack20:(VID[0-9]+):frame:([0-9]+)", identifier)
        if match is None or int(match[2]) != fid:
            raise ValueError("target frame is not bound to the source request")
        video_ids.add(match[1])
    if len(video_ids) != 1:
        raise ValueError("historical smoke adapter requires one identified video")
    video_id = video_ids.pop()

    # Admission sees only stored model outputs. It never receives scored GT.
    replayed, decisions = {}, []
    for fid, row in sorted(predictions.items()):
        if row["status"] != "OK":
            replayed[fid] = {"h0": None, "h1": None, "final": None}
            decisions.append({"frame_id": fid, "decision": "H0_FAILURE"})
            continue
        if row["h0"] != scored[fid]["h0"]:
            raise ValueError("saved scoring H0 differs from prediction H0")
        try:
            result = finalize_grounded_repair(
                row["h0"], row.get("locator"), row.get("proposal"), row.get("review"),
                proposal_slot=row.get("proposal_slot", "FIRST"),
            )
        except ApiSchemaError as exc:
            raise ValueError(f"invalid saved H0 at {video_id}/{fid}") from exc
        replayed[fid] = {k: result[k] for k in ("h0", "h1", "final")}
        decisions.append({"frame_id": fid, "legacy_decision": row["admission"]["decision"],
                          **{k: result[k] for k in ("decision", "reason", "review_required")}})

    legacy_rows, checked_rows = [], []
    for fid, row in sorted(predictions.items()):
        labels = scored[fid]
        identity_and_gt = {"video_id": video_id, "frame_id": fid,
                           "gt": labels["gt"], "mask": labels["mask"]}
        legacy = {k: row.get(k) if row["status"] == "OK" else None
                  for k in ("h0", "h1", "final")}
        legacy_rows.append({**identity_and_gt, **legacy})
        checked_rows.append({**identity_and_gt, **replayed[fid]})
    cost = _historical_cost(usage_paths)
    if cost["provider_calls"] != summary["provider_calls"]:
        raise ValueError("historical ledger call count differs from run summary")
    if {str(p.relative_to(run)): _hash(p) for p in paths} != before:
        raise ValueError("source evidence changed during replay")
    report = {
        "schema_version": "grounded_contact_replay_v1",
        "admission_version": CHECKED_REPAIR_VERSION,
        "source_run": str(run), "source_sha256": before, "source_unchanged": True,
        "source_h0_prompt_versions": sorted({m["prompt_version"] for m in metadata.values()}),
        "gt_source": "saved summary GT and task masks, previously audited; not reloaded here",
        "scope": "historical diagnostic replay; not a new model or current Batch accuracy experiment",
        "new_provider_calls": 0, "new_cost_usd": 0,
        "historical_cost": cost,
        "legacy": compute_repair_comparison(legacy_rows),
        "checked": compute_repair_comparison(checked_rows),
        "decisions": decisions,
        "implementation_sha256": {
            str(p.relative_to(ROOT)): _hash(p) for p in (
                Path(__file__),
                ROOT / "src/surgical_agent/research/verification/final_only_grounded.py",
                ROOT / "src/surgical_agent/evaluation/repair_comparison.py",
            )
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output / "report.json", report)
    print(json.dumps({"report": str(output / "report.json"), "targets": len(predictions),
                      "new_provider_calls": 0, "new_cost_usd": 0,
                      "historical_repair_cost_usd": cost["repair_added_priced_cost_usd"]}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.run, args.output)
