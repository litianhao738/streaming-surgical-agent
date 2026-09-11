"""Same 32 targets, format-only clarification of the revised Phase template."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_phase_mechanism_comparison as trial


def configure():
    trial.PROFILE = "phase_hypothesis_instance_format_confirmation_v2"
    trial.ARMS = trial.PHASE_ARMS = ("revised_phase",)
    trial.VERSIONS = ("h0", "cached_default", "graph_before_phase", "revised_phase")
    trial.MAX_CALLS = 160
    trial.LIMITS = {"openrouter_usd": "1", "xai_usd": "1", "aliyun_cny": "1"}
    trial.TEMPLATE = trial.ROOT / "src/surgical_agent/research/verification/prompts/phase_hypothesis_review_v2_instance.txt"
    trial.RULE = ("Same 32 already inspected Training targets and cached H0/graph predecessor. Revised Phase v1 "
        "unchanged except explicit JSON data-instance-only wording, three exact keys, no schema metadata/wrapper. "
        "Same images, hypothesis, seven phase definitions, schema, models and original three-of-five selection. "
        "No GT rules, changed thresholds, retries, H0/proposal/four-head inference or legacy repair calls. "
        "160 requests maximum, $1 OpenRouter + $1 xAI + CNY1 incremental caps; report alongside all prior attempts. "
        "Format correction must not be called semantic repair success unless Phase beats H0. "
        "Compare with archived fresh compact/original/revised-v1/legacy outputs with their own failures disclosed. "
        "No independent generalization or automatic default promotion; no GT scoring until inference closes and raw replay passes.")


def preflight(output, adapter):
    from tools.audit.preflight_phase_mechanism_comparison import MockCalls
    plan = trial.verify(output)
    initials = trial.old.read(output / "initial_state.json")
    calls = MockCalls(initials)
    for index, selected in enumerate(plan["selection"]):
        base = trial.old.build_gemini_base(trial.old.InferenceOnlyAdapter(adapter), selected)
        record = trial.run_case(calls, base, selected, initials[selected["key"]], index)
        assert record["predictions"]["revised_phase"] == initials[selected["key"]]["graph_r1"]["prediction"]
    assert len(calls.rows) == 160
    trial.old.save(output / "offline_preflight.json", {"mock_calls": 160, "api_calls": 0, "query_gt_reads": 0,
        "plan_sha256": trial.old.sha(output / "plan.json")})
    print("160 request/schema mock checks passed; no API or query GT reads", flush=True)


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
        from tools.audit.audit_phase_mechanism_comparison import main
        main(args.output)
    else:
        getattr(trial, args.command)(args.output, adapter)
