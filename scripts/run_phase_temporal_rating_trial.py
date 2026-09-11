"""Same-32 paired Phase ratings: short versus up-to-30-second causal context."""
import argparse
import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_phase_rating_trial as ratings
from surgical_agent.research.verification.phase_rating import (
    apply_ratings,
    rating_error,
)

trial = ratings.trial
ORIGINAL_BODY = ratings.body
BASE_VERIFY = trial.verify


def context_input(row, samples):
    frames = sorted(samples)
    end = frames.index(row["frame_id"])
    start = end
    while start and frames[start] - frames[start-1] == 25:
        start -= 1
    segment = frames[start:end+1]
    ids = trial.phase.choose_history(frames, row["frame_id"], (30, 10, 0))
    if len(ids) < 3:
        ids = sorted({segment[0], *ids})
    if len(ids) < 3 and len(segment) >= 3:
        midpoint = (segment[0] + row["frame_id"]) / 2
        middle = min(segment[1:-1], key=lambda f: (abs(f-midpoint), f))
        ids = sorted({*ids, middle})
    if len(ids) != 3:
        raise ValueError("three distinct real images unavailable; stop before inference")
    if any(f not in segment for f in ids) or ids[-1] != row["frame_id"]:
        raise ValueError("context crossed a gap or target")
    selected = {k: row[k] for k in ("key", "video_id", "frame_id")}
    selected.update(causal_frame_ids=ids, images=[{"frame_id": f, "path": str(samples[f].media_refs[-1]),
        "sha256": trial.old.sha(samples[f].media_refs[-1])} for f in ids])
    assert selected["images"][-1]["sha256"] == row["images"][-1]["sha256"]
    return selected


def body(arm, seat, selected, initial):
    images = selected if arm == "compact_repeat" else selected["phase_context"]
    return ORIGINAL_BODY("revised_phase", seat, images, initial)


def verify(output):
    plan = BASE_VERIFY(output)
    for row in plan["selection"]:
        context = row["phase_context"]
        assert all(context[k] == row[k] for k in ("key", "video_id", "frame_id"))
        ids = context["causal_frame_ids"]
        assert len(ids) == len(set(ids)) == 3 and ids == sorted(ids) and ids[-1] == row["frame_id"]
        assert row["frame_id"] - ids[0] <= 750
        assert context["images"][-1] == row["images"][-1]
        for f, im in zip(ids, context["images"], strict=True):
            assert f == im["frame_id"]
            assert trial.old.sha(im["path"]) == im["sha256"]
    return plan


def configure():
    ratings.configure()
    trial.PROFILE = "phase_mean4_short_vs_causal30_same32_v1"
    trial.RULE = ("Same 32 inspected Training target identities, H0 and cached four-head predictions. "
        "compact_repeat=FRESH SHORT RATING [-2,-1,0]s; revised_phase=CONTEXT RATING [-30,-10,0]s, "
        "or earliest available same-contiguous-segment frame instead of unavailable -30s. "
        "If the earliest frame duplicates the -10s anchor, use the nearest real internal midpoint as the middle image. "
        "Exactly three real distinct causal images in both arms, identical current image and low/low/high detail. "
        "No crossing gaps, padding, future images, previous-target memory or reference GT. "
        "Same simple score prompt, H0 hypothesis, seven definitions, five models/routes and original mean4 selector. "
        "All five whole seven-phase responses valid; unique highest mean >=4 and above H0 stage to switch. "
        "No four-head inference, no graph modifications, no extra round, no retries. 320 calls maximum. "
        "OR $2/xAI $2/Ali CNY1 caps. Alternate arm order; five concurrent reviewers. "
        "Close inference and replay raw responses before task-masked GT scoring. "
        "Report full32, exact30s subset and pair-valid sensitivity, changes/cost/time/all heads. "
        "Context Phase must beat same-run short AND H0 for success; KEEP-only not success. "
        "Development diagnosis only, Testing unused, no automatic default promotion or disabling Phase.")
    ratings.CHOICE_APPLY = apply_ratings
    ratings.body = body
    trial.phase_body = body
    trial.verify = verify


def prepare(output, adapter):
    if output.exists():
        raise ValueError("new output directory required")
    previous = trial.source_runner.verify(trial.SOURCE)
    done = trial.old.read(trial.SOURCE / "completion.json")
    assert done["fatal_error"] is None and trial.old.read(trial.SOURCE / "budget.json")["stopped"]
    for name, digest in done["hashes"].items():
        assert trial.old.sha(trial.SOURCE / name) == digest
    selected = deepcopy(previous["selection"])
    samples = {v: {s.target_frame_id: s for s in adapter.iter_inference_video(v)} for v in {r["video_id"] for r in selected}}
    initials, source_paths = {}, {trial.SOURCE / n for n in ("plan.json", "completion.json", "predictions.json")}
    for row in selected:
        assert adapter.entries[row["video_id"]].split is trial.old.DatasetSplit.TRAINING
        row["phase_context"] = context_input(row, samples[row["video_id"]])
        prior_path = trial.SOURCE / "priors" / f"{row['video_id']}.json"
        record_path = trial.SOURCE / "targets" / row["key"] / "result.json"
        prior = trial.old.read(prior_path)
        trial.source_runner.validate_prior(prior, row["video_id"], trial.old.InferenceOnlyAdapter(adapter))
        initial = trial.adapt_initial(trial.old.read(record_path), prior)
        initials[row["key"]] = initial
        source_paths.update((prior_path, record_path))
        for arm in trial.ARMS:
            for seat in trial.old.SEATS:
                trial.old.save(output / "preflight_requests" / row["key"] / f"{arm}_{seat}.json",
                    trial.old.redact_images(body(arm, seat, row, initial)))
    trial.old.save(output / "initial_state.json", initials)
    trial.old.save(output / "metadata.json", trial.old.metadata_preflight())
    deps = {*ROOT.glob("scripts/*.py"), *ROOT.glob("src/**/*.py"), *ROOT.glob("src/**/*.txt"),
        *ROOT.glob("src/**/*.json"), *ROOT.glob("src/**/*.csv"), ROOT / "DEFAULT_PIPELINE_VERSION.json",
        ROOT / "configs/perception/joint_openrouter_h0.yaml", ROOT / "tests/unit/test_phase_rating.py",
        ROOT / "tests/unit/test_phase_temporal_rating.py", ROOT / "tools/audit/audit_phase_mechanism_comparison.py"}
    deps = sorted(p for p in deps if p.is_file())
    for path in deps:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    local = [p for p in output.rglob("*.json") if "frozen_source" not in p.parts]
    trial.old.save(output / "plan.json", {"profile": trial.PROFILE, "rule": trial.RULE, "created_utc": trial.old.now(),
        "selection": selected, "models": trial.old.MODELS, "proposer": trial.legacy.PROPOSER, "providers": trial.old.PROVIDERS,
        "max_calls": trial.MAX_CALLS, "limits": trial.LIMITS,
        "sources": {p.relative_to(ROOT).as_posix(): trial.old.sha(p) for p in deps},
        "inputs": {p.relative_to(output).as_posix(): trial.old.sha(p) for p in local},
        "source_archive": {str(p.resolve()): trial.old.sha(p) for p in source_paths}})
    verify(output)
    print(json.dumps({"prepared": True, "max_calls": trial.MAX_CALLS,
        "offsets": dict(Counter(str([(f-r['frame_id'])/25 for f in r['phase_context']['causal_frame_ids']]) for r in selected)),
        "plan_sha256": trial.old.sha(output / "plan.json")}), flush=True)


def preflight(output, adapter):
    from jsonschema import Draft202012Validator
    plan, initial = verify(output), trial.old.read(output / "initial_state.json")
    identities = []

    class Mock:
        stopped = False

        def call(self, target, stage, seat, request):
            identities.append((target, stage, seat))
            assert trial.old.redact_images(request) == trial.old.read(output / "preflight_requests" / target / f"{stage}_{seat}.json")
            packet = json.loads(request["messages"][0]["content"][0]["text"])
            assert next(iter(packet)) == "academic_context"
            pid = initial[target]["h0"]["phase"][0]
            value = {"ratings": {str(p): 5 if p == pid else 1 for p in range(7)}, "image_indices": [2],
                "observation": "Offline fixture."}
            Draft202012Validator(packet["response_schema"]).validate(value)
            return value

    for index, row in enumerate(plan["selection"]):
        for seat in trial.old.SEATS:
            a, b = body("compact_repeat", seat, row, initial[row["key"]]), body("revised_phase", seat, row, initial[row["key"]])
            pa = json.loads(a["messages"][0]["content"][0]["text"])
            pb = json.loads(b["messages"][0]["content"][0]["text"])
            assert {k for k in pa if pa[k] != pb[k]} == {"images"}
            assert a["messages"][0]["content"][-1] == b["messages"][0]["content"][-1]
            a["messages"] = b["messages"] = None
            assert a == b
        base = trial.old.build_gemini_base(trial.old.InferenceOnlyAdapter(adapter), row)
        record = trial.run_case(Mock(), base, row, initial[row["key"]], index)
        assert all(record["predictions"][arm] == initial[row["key"]]["graph_r1"]["prediction"] for arm in trial.ARMS)
    assert len(identities) == len(set(identities)) == 320
    trial.old.save(output / "offline_preflight.json", {"calls": 320, "api_calls": 0, "query_gt_reads": 0,
        "same_prompt_schema_model_target_only_historical_images_changed": True})
    print("320 mock calls passed; only history images and timestamps changed; zero API/query GT", flush=True)


def audit(output):
    from tools.audit.audit_phase_mechanism_comparison import main
    main(output)
    plan = verify(output)
    details = {}
    for arm in trial.ARMS:
        valid, errors, reasons = [], Counter(), Counter()
        for row in plan["selection"]:
            record = trial.old.read(output / "targets" / row["key"] / "result.json")
            branch = record["branches"][arm]
            raw = branch["raw"]
            failures = {s: e for s in trial.old.SEATS if (e := rating_error(raw[s], 3))}
            errors.update(f"{s}:{e}" for s, e in failures.items())
            reasons[branch["decision"]["reason"]] += 1
            expected = deepcopy(record["predictions"]["graph_before_phase"])
            if not failures:
                valid.append(row["key"])
                sums = {p: sum(raw[s]["ratings"][str(p)] for s in trial.old.SEATS) for p in range(7)}
                maximum = max(sums.values())
                top = [p for p, v in sums.items() if v == maximum]
                if len(top) == 1 and maximum >= 20 and maximum > sums[expected["phase"][0]]:
                    expected["phase"] = top
                assert branch["decision"]["means"] == {f"phase_{p}": v/5 for p, v in sums.items()}
            assert expected == branch["prediction"]
        details[arm] = {"valid_targets": valid, "valid_panels": len(valid), "errors": dict(errors), "decisions": dict(reasons)}
    trial.old.save(output / "temporal_rating_audit.json", details)
    print(json.dumps(details), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "score", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    adapter = trial.old.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        prepare(args.output, adapter)
    elif args.command == "preflight":
        preflight(args.output, adapter)
    elif args.command == "audit":
        audit(args.output)
    else:
        getattr(trial, args.command)(args.output, adapter)
