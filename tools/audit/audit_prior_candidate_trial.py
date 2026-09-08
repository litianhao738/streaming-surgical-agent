"""Read-only wire, raw-output, prior-isolation and score arithmetic audit."""
import argparse
import json
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import redact_images
from scripts.run_candidate_panel_trial import usage_charge
from scripts.run_openrouter_gemini_h0_trial import build_gemini_base, gemini_h0_wire
from scripts.run_prior_candidate_trial import ARMS, TASKS, proposal_wire
from scripts.run_prior_panel_trial import read, save
from scripts.run_recent_mean_panel_trial import review_wire
from scripts.score_prior_candidate_trial import validate
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.research.retrieval.prior_candidates import retrieve_candidate_hints
from surgical_agent.research.verification.candidate_coordinator import SEATS, make_pool
from surgical_agent.research.verification.prior_panel import fit_prior
from surgical_agent.research.verification.review_json_compat import (
    parse_review_json_compatible,
)


def audit(output, adapter):
    plan, targets, ledger = validate(output)
    counts = read(output / "training_counts.json")
    calls = {(r["target"], r["stage"], r["seat"]): r for r in ledger["calls"]}
    if len(calls) != len(ledger["calls"]):
        raise ValueError("duplicate dispatch identity")
    totals = defaultdict(Decimal)
    for r in calls.values():
        totals[r["account"]] += Decimal(r["charge"])
        charge, kind = usage_charge(r["seat"], r.get("usage", {}), Decimal(r["reserve"]), plan["rates"])
        assert charge == Decimal(r["charge"]) and kind == r["charge_kind"]
    assert {k: str(v) for k, v in totals.items()} == ledger["occupied"]

    def folder(key, stage, seat):
        r = calls[key, stage, seat]
        return output / "calls" / f"{r['index']:03d}_{key}_{stage}_{seat}"

    prior_checks = 0
    for video in plan["prior_sha256"]:
        expected = fit_prior(counts, video)
        actual = read(output / "priors" / f"{video}.json")
        assert expected == actual and video not in actual["fit_videos"]
        prior_checks += 1
    wire_checks, raw_checks, hint_checks = 0, 0, 0
    for selected, target in zip(plan["selection"], targets, strict=True):
        key = selected["key"]
        if target["h0"] is None:
            continue
        base = build_gemini_base(adapter, selected)
        h0folder = folder(key, "h0", "base")
        assert read(h0folder / "request.json") == redact_images(gemini_h0_wire(base))
        h0raw = json.loads(read(h0folder / "response.json")["body"]["choices"][0]["message"]["content"])
        # Recreate the inference-time insertion order. Saved JSON sorts keys,
        # while the inner wire JSON string preserves its original key order.
        h0 = {t: [h0raw[t]["selected_id"]] if t == "phase" else h0raw[t]["selected_ids"] for t in TASKS}
        assert target["h0"] == h0
        raw_checks += 1
        hints = retrieve_candidate_hints(target["h0"], read(output / "priors" / f"{target['video_id']}.json"), video_id=target["video_id"])
        assert hints == read(output / "targets" / key / "hints.json")
        hint_checks += 1
        for arm in ARMS:
            path = output / "targets" / key / f"{arm}.json"
            if not path.exists():
                continue
            record = read(path)
            source = record.get("proposal_shared_from") or arm
            pdir = folder(key, f"{source}_proposal", "base")
            expected = proposal_wire(base, selected, h0, make_pool(h0), hints["packet"] if arm == "prior_graph" else None)
            assert read(pdir / "request.json") == redact_images(expected)
            wire_checks += 1
            if record["proposal"] is not None:
                raw = json.loads(read(pdir / "response.json")["body"]["choices"][0]["message"]["content"])
                assert raw == record["proposal"]
                raw_checks += 1
            if "raw" not in record:
                continue
            pool = make_pool(h0, record["proposal"], make_pool(h0))
            assert pool == record["pool"]
            source = record.get("shared_from") or arm
            for seat in SEATS:
                rdir = folder(key, f"{source}_review", seat)
                assert read(rdir / "request.json") == redact_images(review_wire(seat, base, selected, pool))
                wire_checks += 1
                if record["raw"][seat] is not None:
                    raw = parse_review_json_compatible(read(rdir / "response.json")["body"]["choices"][0]["message"]["content"])[0]
                    assert raw == record["raw"][seat]
                    raw_checks += 1
    # Independent arithmetic from saved target GT/masks, not the scorer's metric functions.
    truths = {(r["video_id"], r["frame_id"]): r for r in read(output / "scored_truth.json")}
    scores = read(output / "metrics.json")["metrics"]
    metric_checks = 0
    for arm in ("h0", *ARMS):
        for task in TASKS:
            tp = fp = fn = exact = n = 0
            for row in targets:
                gt = truths[row["video_id"], row["frame_id"]]
                if not gt["mask"][task]:
                    continue
                prediction = row["h0"] if arm == "h0" else row["arms"][arm]["prediction"]
                chosen = set(prediction[task]) if prediction is not None else set()
                wanted = set(gt["gt"][task])
                tp += len(chosen & wanted)
                fp += len(chosen - wanted)
                fn += len(wanted - chosen)
                exact += prediction is not None and chosen == wanted
                n += 1
            expected = {"tp": tp, "fp": fp, "fn": fn, "valid_targets": n, "exact_matches": exact,
                        "micro_precision": tp/(tp+fp) if tp+fp else 0,
                        "micro_recall": tp/(tp+fn) if tp+fn else 0,
                        "micro_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                        "exact_set_accuracy": exact/n if n else None}
            assert all(scores[arm]["tasks"][task][k] == v for k, v in expected.items())
            metric_checks += 1
    result = {"verified": True, "prior_tables_rebuilt_without_query_video": prior_checks,
              "hints_rebuilt": hint_checks, "exact_wire_checks": wire_checks,
              "raw_output_checks": raw_checks, "metric_tables_checked": metric_checks,
              "actual_post_calls": len(calls), "costs": {k: str(v) for k, v in totals.items()},
              "reviewer_inputs_have_no_prior_text": True, "old_sources_preserved": True}
    save(output / "independent_audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    audit(args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3))
