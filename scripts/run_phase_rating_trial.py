"""Same-32 isolated Phase: simple choice versus simple seven-phase mean ratings."""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_phase_mechanism_comparison as trial
from surgical_agent.research.verification.phase_rating import (
    apply_ratings,
    rating_error,
    rating_schema,
)

CHOICE_APPLY = trial.phase.apply_phase_choices
PROMPTS = ROOT / "src/surgical_agent/research/verification/prompts"


def body(arm, seat, selected, initial):
    out = trial.phase.phase_wire(seat, selected)
    packet = json.loads(out["messages"][0]["content"][0]["text"])
    suffix = "choice" if arm == "compact_repeat" else "ratings"
    packet["instructions"] = ((PROMPTS / "phase_simple_common_v1.txt").read_text(encoding="utf-8").strip() + "\n\n"
        + (PROMPTS / f"phase_simple_{suffix}_v1.txt").read_text(encoding="utf-8").strip()).replace(
            "{current_image_index}", str(packet["current_image_index"]))
    pid = initial["h0"]["phase"][0]
    packet["reference_phase_hypothesis"] = {"phase_id": pid, "phase_name": trial.phase.PHASE_GUIDE[pid]["name"],
        "status": "Untrusted model hypothesis; not visual evidence."}
    if arm == "revised_phase":
        packet["task"] = "Rate visual support for all seven current surgical workflow phases at the final target image."
        packet["response_schema"] = rating_schema()
        if out["response_format"]["type"] == "json_schema":
            out["response_format"]["json_schema"] = {"name": "phase_evidence_ratings_v1", "strict": True,
                "schema": packet["response_schema"]}
    out["messages"][0]["content"][0]["text"] = json.dumps(packet, ensure_ascii=False)
    return out


def run_case(calls, base, selected, initial, index, skipped_arms=()):
    before = initial["graph_r1"]["prediction"]
    predictions = {"h0": deepcopy(initial["h0"]), "cached_default": deepcopy(initial["cached_default"]),
        "graph_before_phase": deepcopy(before), **{a: deepcopy(before) for a in trial.ARMS}}
    order = trial.ARMS[index % 2:] + trial.ARMS[:index % 2]
    branches, skipped = {}, []
    for arm in order:
        start = perf_counter()
        raw = {s: None for s in trial.old.SEATS}
        if calls.stopped or arm in skipped_arms:
            skipped.append(arm)
        else:
            with ThreadPoolExecutor(max_workers=5) as pool:
                raw = dict(zip(trial.old.SEATS, pool.map(lambda s, arm=arm: calls.call(selected["key"], arm, s,
                    body(arm, s, selected, initial)), trial.old.SEATS), strict=True))
        selector = CHOICE_APPLY if arm == "compact_repeat" else apply_ratings
        prediction, decision = selector(before, raw, 3)
        branches[arm] = {"raw": raw, "prediction": prediction, "decision": decision, "seconds": perf_counter() - start}
        predictions[arm] = prediction
    return {"predictions": predictions, "branches": branches, "order": list(order), "skipped_arms": skipped}


def configure():
    trial.PROFILE = "phase_simple_choice_vs_mean4_same32_v1"
    # Reuse the generic runner's arm names; their meaning is frozen explicitly.
    trial.ARMS = trial.PHASE_ARMS = ("compact_repeat", "revised_phase")
    trial.VERSIONS = ("h0", "cached_default", "graph_before_phase", *trial.ARMS)
    trial.MAX_CALLS = 320
    trial.LIMITS = {"openrouter_usd": "2", "xai_usd": "2", "aliyun_cny": "1"}
    trial.RULE = ("Same 32 inspected Training targets; cached H0/graph four-head outputs unchanged. "
        "compact_repeat means NEW SIMPLE CHOICE, revised_phase means NEW SIMPLE SEVEN-PHASE MEAN RATINGS. "
        "Both use identical simple common instructions, untrusted H0 Phase, original three images, definitions/models/routes. "
        "Choice uses original five-valid three-vote selection; ratings use all-five valid seven-integer answers, "
        "arithmetic mean >=4, unique highest and strictly above H0 phase via existing phase_apply. "
        "Only task/output/rubric differ; this is scoring-contract plus aggregation, not aggregation-only. "
        "No prior, repair LLM, four-head calls, future frames, memory, retries or extra rounds. "
        "Rotate arm order per target; five concurrent seats. 320 request cap, OR $2/xAI $2/Ali CNY1. "
        "Close all inference then raw replay before GT scoring. Missing GT task-masked. "
        "Report all attempts/fallbacks, changes, F1/precision/set accuracy and cost/time. "
        "Success requires rated Phase F1 above same-run simple choice AND H0, no automatic promotion. "
        "Comparison with archived longer v2 is exploratory and confounded by run variation. Development only, Testing unused.")
    trial.phase_body = body
    trial.run_case = run_case


def preflight(output, adapter):
    from jsonschema import Draft202012Validator
    plan, initial = trial.verify(output), trial.old.read(output / "initial_state.json")

    class Mock:
        stopped = False

        def __init__(self):
            self.rows = []

        def call(self, target, stage, seat, request):
            self.rows.append((target, stage, seat))
            assert trial.old.redact_images(request) == trial.old.read(
                output / "preflight_requests" / target / f"{stage}_{seat}.json")
            packet = json.loads(request["messages"][0]["content"][0]["text"])
            assert next(iter(packet)) == "academic_context"
            pid = initial[target]["h0"]["phase"][0]
            assert packet["reference_phase_hypothesis"]["phase_id"] == pid
            value = {"image_indices": [2], "observation": "Offline structural fixture."}
            if stage == "compact_repeat":
                value["phase_id"] = pid
            else:
                value["ratings"] = {str(p): 5 if p == pid else 1 for p in range(7)}
            Draft202012Validator(packet["response_schema"]).validate(value)
            return value

    calls = Mock()
    for index, selected in enumerate(plan["selection"]):
        base = trial.old.build_gemini_base(trial.old.InferenceOnlyAdapter(adapter), selected)
        record = trial.run_case(calls, base, selected, initial[selected["key"]], index)
        assert all(record["predictions"][a] == initial[selected["key"]]["graph_r1"]["prediction"] for a in trial.ARMS)
    assert len(calls.rows) == len(set(calls.rows)) == 320
    trial.old.save(output / "offline_preflight.json", {"mock_calls": 320, "api_calls": 0, "query_gt_reads": 0,
        "plan_sha256": trial.old.sha(output / "plan.json"), "frozen_requests_and_schemas_verified": True})
    print("320 frozen-request/schema mock checks passed; 0 API and query GT reads", flush=True)


def audit(output):
    from tools.audit.audit_phase_mechanism_comparison import main
    main(output)
    plan = trial.verify(output)
    details = {}
    for arm in trial.ARMS:
        counts, errors, valid = Counter(), Counter(), []
        for selected in plan["selection"]:
            record = trial.old.read(output / "targets" / selected["key"] / "result.json")
            branch = record["branches"][arm]
            counts[branch["decision"]["reason"]] += 1
            raw = branch["raw"]
            check = trial.phase.phase_choice_error if arm == "compact_repeat" else rating_error
            failures = {s: e for s in trial.old.SEATS if (e := check(raw[s], 3))}
            errors.update(f"{s}:{e}" for s, e in failures.items())
            before = record["predictions"]["graph_before_phase"]
            expected = deepcopy(before)
            if not failures:
                valid.append(selected["key"])
                if arm == "revised_phase":
                    sums = {p: sum(raw[s]["ratings"][str(p)] for s in trial.old.SEATS) for p in range(7)}
                    top = max(sums.values())
                    winners = [p for p, v in sums.items() if v == top]
                    if len(winners) == 1 and top >= 20 and top > sums[before["phase"][0]]:
                        expected["phase"] = winners
                    assert branch["decision"]["means"] == {f"phase_{p}": v/5 for p, v in sums.items()}
                else:
                    votes = Counter(r["phase_id"] for r in raw.values() if r["phase_id"] is not None)
                    winners = [p for p, n in votes.items() if n >= 3]
                    if winners:
                        expected["phase"] = winners
            assert expected == branch["prediction"]
        details[arm] = {"decisions": dict(counts), "errors": dict(errors), "valid_panels": len(valid), "valid_targets": valid}
    trial.old.save(output / "phase_rating_audit.json", details)
    print(json.dumps(details), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "score", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    adapter = trial.old.CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "preflight":
        preflight(args.output, adapter)
    elif args.command == "audit":
        audit(args.output)
    else:
        getattr(trial, args.command)(args.output, adapter)
