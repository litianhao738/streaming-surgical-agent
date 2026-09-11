"""Bounded two-target GLM transport/contract smoke; no H0, proposer or GT calls."""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_glm_parallel_repair as glm
from scripts.check_candidate_panel_providers import redact_images
from scripts.run_repair_revision_trial import normalize_review


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "execution.lock").open("x") as handle:
        handle.write("Single execution: at most four calls, USD 0.20")
    source = ROOT / "artifacts/preflight/default_lightweight_prepare_20260910_v1"
    plan = glm.read(source / "plan.json")
    rows = plan["selection"][:2]
    adapter = glm.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    jobs = []
    frozen = {}
    for row in rows:
        archived = ROOT / "artifacts/preflight/parallel_phase_eight_20260909_v1/targets" / row["key"] / "pipeline.json"
        pool = glm.read(archived)["graph"]["pool"]
        frozen[str(archived)] = glm.sha(archived)
        base = glm.compact.original.build_gemini_base(adapter, row)
        phase = plan["phase_inputs"][row["key"]]
        for stage, body in (("graph", glm.review_wire("grok", base, row, pool)),
                            ("phase", glm.phase_wire("grok", phase))):
            assert body["reasoning"] == {"effort": "low", "exclude": True}
            assert body["model"] == "z-ai/glm-5.3-flash"
            glm.save(output / "requests" / f"{row['key']}_{stage}.json", redact_images(body))
            jobs.append((row, stage, body, pool))
    import shutil
    for path in [*glm.RUNTIME_FILES, Path(__file__).resolve()]:
        frozen[str(path)] = glm.sha(path)
        target = output / "frozen_source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    glm.save(output / "plan.json", {"selection": rows, "models": glm.MODELS,
        "reasoning": {"effort": "low", "exclude": True, "mandatory": True},
        "provider": glm.PROVIDERS["grok"], "max_calls": 4, "usd_limit": "0.20", "frozen_hashes": frozen,
        "scope": "Interface/contract smoke only, fixed first two archived targets. No repair accuracy evaluation."})
    results = []
    start = perf_counter()
    with glm.lightweight_protocol():
        calls = glm.GLMCalls(output, limits={"openrouter_usd": Decimal("0.20"),
            "xai_usd": Decimal(0), "aliyun_cny": Decimal(0)}, rates=glm.RATES,
            providers=glm.PROVIDERS, max_calls=4)

        def one(job):
            row, stage, body, pool = job
            begin = perf_counter()
            raw = calls.call(row["key"], stage, "grok", body)
            seconds = perf_counter()-begin
            if stage == "phase":
                validation = glm.compact.original.phase_choice_error(raw, image_count=3)
            else:
                normalized, formatting = normalize_review(raw, pool, seat="grok", image_count=3)
                validation = {"formatting": formatting, "normalized": normalized}
            return {"key": row["key"], "stage": stage, "seconds": seconds,
                "parsed": raw is not None, "validation": validation}

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(one, jobs))
        calls.stopped = True
        calls.persist()
    summary = {"results": results, "wall_seconds": perf_counter()-start,
        "calls": len(calls.rows), "native_usd": str(sum(Decimal(r['charge']) for r in calls.rows if r['charge_kind']=='native')),
        "unknown_cost_calls": sum(r['charge_kind']=='unknown_reserved' for r in calls.rows),
        "reasoning_tokens": [r.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens') for r in calls.rows],
        "status": [r['status'] for r in calls.rows]}
    glm.save(output / "summary.json", summary)
    print({k:v for k,v in summary.items() if k != "results"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
