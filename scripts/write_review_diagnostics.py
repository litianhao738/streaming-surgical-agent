"""Write precise review rejection reasons from closed logs, without changing results."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_candidate_panel_trial import sha
from scripts.run_prior_panel_trial import read, save
from surgical_agent.research.verification.review_diagnostic_details import (
    explain_invalid_items,
)


def write_report(output):
    output = Path(output)
    completion = read(output / "completion.json")
    if completion.get("fatal_error"):
        raise ValueError("nonfatal closed inference required")
    expected = completion.get("inference_artifact_sha256", completion.get("hashes", {}))
    result, reasons, original_hashes = {}, Counter(), {}
    for path in sorted((output / "targets").rglob("*.json")):
        if path.name not in ("pipeline.json", "result.json"):
            continue
        rel = path.relative_to(output).as_posix()
        digest = sha(path)
        if expected.get(rel) != digest:
            raise ValueError("unbound or changed source record: " + rel)
        record = read(path)
        graphs = record.get("arms", {"graph": record["graph"]} if "graph" in record else {})
        for arm, graph in graphs.items():
            graph = graph or {}
            detail = explain_invalid_items(graph.get("diagnostics") or {}, graph.get("format_diagnostics") or {})
            result[path.parent.name + "/" + arm] = detail
            for item in detail["invalid_item_details"].values():
                for seat in item.values():
                    reasons.update(seat["recorded_reasons"])
        original_hashes[rel] = digest
    if not original_hashes:
        raise ValueError("no supported closed review records")
    report = {"schema_version": "review_rejection_details_v1", "records": result,
        "reason_counts": dict(reasons), "input_sha256": original_hashes,
        "predictions_changed": False, "votes_changed": False, "api_calls": 0}
    path = output / "review_diagnostics.json"
    if path.exists() and read(path) != report:
        raise ValueError("preserve previous diagnostic report")
    save(path, report)
    for rel, digest in original_hashes.items():
        assert sha(output / rel) == digest
    return path, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    path, report = write_report(parser.parse_args().output)
    print({"path": str(path), "reason_counts": report["reason_counts"], "predictions_changed": False})
