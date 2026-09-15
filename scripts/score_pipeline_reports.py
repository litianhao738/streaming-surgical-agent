"""Append GSR report evaluation: prepare (zero-call), cached, or synthetic mock QA.

No paid-execution CLI is enabled in this first integration pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from surgical_agent.research.reporting.contracts import GroundTruth, ModelConfig, ReportInput, digest
from surgical_agent.research.reporting.event_report_generator import ReportGenerator, PROMPT, PROMPT_VERSION
from surgical_agent.research.reporting.llm_judge import LLMJudge
from surgical_agent.research.reporting.judge_selection import build_judge
from surgical_agent.research.reporting.transport import CachedCalls
from surgical_agent.research.reporting.evaluator import evaluate_pair, write_artifacts


def load_pairs(source):
    pairs, seen = [], set()
    for row in source["details"]:
        video, frame = row["key"].rsplit("_", 1)
        key = (video, int(frame))
        if key in seen:
            raise ValueError("duplicate source frame")
        seen.add(key)
        h0 = ReportInput.from_prediction(*key, "H0", row["h0"])
        full = ReportInput.from_prediction(*key, "FULL", row["pipeline"])
        mask, gt = row["mask"], row["gt"]
        phase = gt["phase"]
        if mask["phase"] and (not isinstance(phase, list) or len(phase) != 1):
            raise ValueError("invalid supervised phase")
        truth = GroundTruth(tuple(gt["ivt"]) if gt["ivt"] is not None else None,
                            phase[0] if phase else None, mask["ivt"], mask["phase"])
        pairs.append((h0, full, truth))
    return pairs


def mock_caller(request):
    """Synthetic responses for wiring tests, never experimental data."""
    if "structured_prediction_hash" in request:
        state = json.loads(request["prompt"].split("Canonical surgical state:\n", 1)[1])
        events = []
        for triple in state["ivt"]:
            i, v, t = triple.split(",")
            events.append(f"The {i} has no identified action or target" if v == "null_verb" else f"The {i} {v} the {t}")
        text = "; ".join(events) if events else "No interaction is identified"
        raw = {"report": text + " during " + state["phase"] + "."}
    else:
        raw = {"factual_consistency": 80, "clarity": 80, "coherence": 80, "overall": 80, "major_error": False}
        if request.get("prompt_version") == "gsr_report_v1_judge_2":
            raw.update(audit=dict.fromkeys(("instrument","action","target","phase","coverage","unsupported_claims"), "match"), errors=[])
    return {"text": json.dumps(raw), "usage": {}, "cost": {}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT/"configs/reporting/gsr_v1.json")
    p.add_argument("--mode", choices=("prepare", "cached", "mock"), default="prepare")
    p.add_argument("--cache", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("output already exists; historical runs are immutable")
    content = a.scores.read_bytes()
    pairs = load_pairs(json.loads(content))
    config = json.loads(a.config.read_bytes())
    generator_config, judge_config = (ModelConfig(**config[k]) for k in ("generator", "judge"))
    meta = {"source_scores_sha256": hashlib.sha256(content).hexdigest(), "config": config,
            "source_kind": "sealed prediction offline score export", "config_hash": digest(config)}
    if a.mode == "prepare":
        a.output.mkdir(parents=True)
        requests = [{"video_id": s.video_id, "frame_id": s.frame_id, "source_prediction": s.source_prediction,
                     "canonical_appendix": s.appendix(), "prompt_version": PROMPT_VERSION,
                     "model_config": generator_config.to_dict(), "prompt": PROMPT+json.dumps(s.state(), sort_keys=True)}
                    for h0, full, _ in pairs for s in (h0, full)]
        # All predictions exported without GT-based generation flags. The mask and
        # GT are isolated in a separate file; never send metadata/index rows to LLM.
        (a.output/"generation_plan.json").write_text(json.dumps({"requests": requests, "metadata": meta}, indent=2), encoding="utf-8")
        evaluation = [{"video_id": h.video_id, "frame_id": h.frame_id, "report_valid": t.report_valid,
                       "ground_truth": t.state() if t.report_valid else None} for h, _, t in pairs]
        (a.output/"offline_evaluation_inputs.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        print(json.dumps({"status": "PREPARED", "frames": len(pairs), "eligible": sum(t.report_valid for _, _, t in pairs), "paid_calls": 0}))
        return
    if a.cache is None:
        raise ValueError("--cache is required for cached/mock mode")
    if a.mode == "cached" and not a.cache.is_file():
        raise ValueError("cached mode requires an existing cache")
    cache = CachedCalls(a.cache, mock_caller if a.mode=="mock" else None,
                        max_calls=4*len(pairs) if a.mode=="mock" else 0, mock=a.mode=="mock")
    try:
        generator, judge = ReportGenerator(generator_config, cache), build_judge(config, cache)
        rows = [r for h, f, t in pairs for r in evaluate_pair(h, f, t, generator, judge)]
        summary = write_artifacts(a.output, rows, cache.costs(), meta)
        print(json.dumps(summary, indent=2))
    finally:
        cache.close()


if __name__ == "__main__":
    main()
