"""Optional append-only integration over already finalized predictions."""
import json
from pathlib import Path

from . import VERSION
from .contracts import digest
from .rule_evaluator import evaluate_rule


def evaluate_pair(h0, full, truth, generator, judge):
    if (h0.video_id, h0.frame_id) != (full.video_id, full.frame_id) or (h0.source_prediction, full.source_prediction) != ("H0", "FULL"):
        raise ValueError("paired H0/FULL identities required")
    rows = []
    for source in (h0, full):
        identity = {"video_id": source.video_id, "frame_id": source.frame_id, "source_prediction": source.source_prediction}
        if not truth.report_valid:
            rows.append({**identity, "report_valid": False, "rule_score": None, "llm_judge_score": None, "gsr_score": None, "status": "masked"})
            continue
        report = generator.generate(source)
        if report["canonical_appendix"] != source.appendix():
            raise ValueError("canonical appendix mismatch")
        rule = evaluate_rule(report["raw_response"], truth)
        assessed = judge.evaluate(report, truth)
        value = assessed["llm_judge_score"]
        rows.append({**report, **rule, **{k: v for k, v in assessed.items() if k != "diagnostics"},
                     "diagnostics": {**rule["diagnostics"], **assessed.get("diagnostics", {})},
                     "gsr_score": .8*rule["rule_score"]+.2*value if value is not None else None})
    return rows


def summarize(rows):
    indexed = {}
    for row in rows:
        key = row["video_id"], row["frame_id"]
        arm = row["source_prediction"]
        if arm not in ("H0", "FULL") or arm in indexed.setdefault(key, {}):
            raise ValueError("duplicate or unknown arm")
        indexed[key][arm] = row
    if any(set(pair) != {"H0", "FULL"} for pair in indexed.values()):
        raise ValueError("unpaired frame")
    if any(pair["H0"]["report_valid"] != pair["FULL"]["report_valid"] for pair in indexed.values()):
        raise ValueError("mismatched supervision")
    eligible = [p for p in indexed.values() if p["H0"]["report_valid"]]
    scored = [p for p in eligible if all(p[a].get("gsr_score") is not None for a in ("H0", "FULL"))]
    models = {(r.get("generator_provider"), r.get("generator_model"), r.get("prompt_version"), digest(r.get("generation_parameters")), r.get("ontology_hash"), r.get("judge_provider"), r.get("judge_model"), r.get("judge_prompt_version"), digest(r.get("judge_parameters"))) for pair in scored for r in pair.values()}
    if len(models) > 1:
        raise ValueError("model/prompt/parameter mismatch in paired evaluation")
    means = {}
    for arm in ("H0", "FULL"):
        means[arm] = {}
        for metric in ("rule_score", "llm_judge_score", "gsr_score"):
            values = [p for p in eligible if all(p[a].get(metric) is not None for a in ("H0", "FULL"))]
            means[arm][metric] = sum(p[arm][metric] for p in values)/len(values) if values and len(values)==len(eligible) else None
    return {"protocol": VERSION, "total_frames": len(indexed), "eligible_frames": len(eligible),
            "excluded_partial_labels": len(indexed)-len(eligible), "paired_complete_frames": len(scored),
            "status": "COMPLETE" if eligible and len(scored)==len(eligible) else "INCOMPLETE",
            "mock": any(r.get("mock", False) for r in rows), "methods": means,
            "aggregation": "equal frame means; missing responses are not zero or silently excluded"}


def write_artifacts(output, rows, costs, metadata=None):
    summary = summarize(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for filename, selected in (("h0_reports.jsonl", [r for r in rows if r["source_prediction"]=="H0"]),
                               ("full_reports.jsonl", [r for r in rows if r["source_prediction"]=="FULL"]),
                               ("rule_scores.jsonl", [{k: r.get(k) for k in ("video_id", "frame_id", "source_prediction", "report_valid", "rule_score", "diagnostics")} for r in rows]),
                               ("judge_scores.jsonl", [{k: r.get(k) for k in ("video_id", "frame_id", "source_prediction", "report_valid", "llm_judge_score", "gsr_score", "judge_model", "judge_status", "diagnostics")} for r in rows])):
        (output/filename).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in selected), encoding="utf-8")
    summary.update(costs=costs, metadata=metadata or {})
    (output/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return summary
