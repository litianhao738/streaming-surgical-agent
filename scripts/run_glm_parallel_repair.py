"""User-selected GLM low-reasoning reviewer roster for both default repair branches.

Keep historical protocols immutable. Scope model, transport and accounting
bindings to this CLI process and restore them after all workers finish.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_candidate_panel_trial as transport
from scripts import run_compact_parallel_repair as compact
from scripts import run_lightweight_parallel_repair as previous
from scripts.run_candidate_panel_trial import sha
from scripts.run_evidence_feedback_trial import fingerprint
from scripts.run_prior_panel_trial import read, save
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter

VERSION = "parallel-phase-repair-v1.3.0-glm-low"
PROFILE = "fixed_h0_parallel_graph_and_phase_compact_glm_low_v1"
MODELS = {**compact.original.MODELS, "grok": "z-ai/glm-5.3-flash", "qwen": "qwen3.5-35b-a3b"}
PROVIDERS = {**compact.original.PROVIDERS, "grok": "Together"}
RATES = {**compact.original.RATES_V2, "grok": ("0.00000015", "0.0000005")}
FAMILIES = {"grok": "GLM", "qwen": "Qwen", "gpt": "GPT", "gemini": "Gemini", "deepseek": "DeepSeek"}
RUNTIME_FILES = [*compact.RUNTIME_FILES, Path(__file__).resolve()]
_TIMED_CALLS = compact.original.TimedCalls
_KEY_FOR = transport.key_for
_BUCKET = transport.bucket
TRANSPORTS = {"grok": {"endpoint": "https://openrouter.ai/api/v1", "credential_slot": "gpt",
    "account": "openrouter_usd", "provider": "together", "allow_fallbacks": False,
    "reasoning": {"effort": "low", "exclude": True}, "reasoning_mandatory": True},
    "qwen": {"endpoint": transport.MODELS["qwen"][0], "credential_slot": "qwen",
    "account": "aliyun_cny", "enable_thinking": False}}


def lightweight_body(body, seat):
    result = deepcopy(body)
    result["model"] = MODELS[seat]
    if seat == "grok":
        result.pop("reasoning_effort", None)
        result["reasoning"] = {"effort": "low", "exclude": True}
        result["provider"] = {"only": ["together"], "order": ["together"],
            "allow_fallbacks": False, "require_parameters": True}
    elif seat == "qwen":
        result["enable_thinking"] = False
    return result


def review_wire(seat, base, selected, pool):
    return lightweight_body(compact.review_wire(seat, base, selected, pool), seat)


def phase_wire(seat, selected):
    return lightweight_body(compact.phase_wire(seat, selected), seat)


def verify_plan(plan):
    if (plan.get("profile") != PROFILE or plan.get("repair_version") != VERSION
            or plan.get("verifier_prompt_profile") != compact.PROMPT_PROFILE
            or plan.get("reviewer_families") != FAMILIES or plan.get("reviewer_transports") != TRANSPORTS):
        raise ValueError("lightweight repair plan/version/transport mismatch")
    for path in RUNTIME_FILES:
        if plan["source_sha256"].get(path.relative_to(ROOT).as_posix()) != sha(path):
            raise ValueError("lightweight runtime/template missing or changed")
    compact._VERIFY_PLAN(plan)


class GLMCalls(_TIMED_CALLS):
    def __init__(self, *args, **kwargs):
        # GLM requires reasoning. Only this seat is additionally permitted.
        kwargs["reasoning_seats"] = tuple(dict.fromkeys((*kwargs.get("reasoning_seats", ()), "grok")))
        super().__init__(*args, **kwargs)


@contextmanager
def lightweight_protocol():
    with compact.compact_protocol():
        names = ("PROFILE", "MODELS", "PROVIDERS", "RATES_V2", "review_wire", "phase_wire", "verify_plan", "TimedCalls")
        bindings = {n: getattr(compact.original, n) for n in names}
        scorer_verify = compact.original_score.verify_plan
        transport_bindings = {n: getattr(transport, n) for n in ("MODELS", "key_for", "bucket")}
        try:
            compact.original.PROFILE = PROFILE
            compact.original.MODELS = MODELS
            compact.original.PROVIDERS = PROVIDERS
            compact.original.RATES_V2 = RATES
            compact.original.review_wire = review_wire
            compact.original.phase_wire = phase_wire
            compact.original.verify_plan = verify_plan
            compact.original.TimedCalls = GLMCalls
            compact.original_score.verify_plan = verify_plan
            transport.MODELS = {**transport.MODELS, "grok": (
                TRANSPORTS["grok"]["endpoint"], MODELS["grok"], "docs/API.txt", "together")}
            transport.key_for = lambda seat: _KEY_FOR("gpt" if seat == "grok" else seat)
            transport.bucket = lambda seat: "openrouter_usd" if seat == "grok" else _BUCKET(seat)
            yield
        finally:
            for name, value in transport_bindings.items():
                setattr(transport, name, value)
            for name, value in bindings.items():
                setattr(compact.original, name, value)
            compact.original_score.verify_plan = scorer_verify


def finalize_plan(output):
    plan = read(output / "plan.json")
    if plan["profile"] != compact.PROFILE:
        raise ValueError("expected newly prepared compact plan")
    plan.update(profile=PROFILE, repair_version=VERSION, base_repair_version=previous.VERSION,
        models=MODELS, providers=PROVIDERS, rates=RATES, reviewer_families=FAMILIES, reviewer_transports=TRANSPORTS)
    for path in RUNTIME_FILES:
        plan["source_sha256"][path.relative_to(ROOT).as_posix()] = sha(path)
        destination = output / "frozen_source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for key, selected in plan["phase_inputs"].items():
        plan["wire_fingerprints"][key]["phase"] = {seat: fingerprint(phase_wire(seat, selected)) for seat in MODELS}
    plan["policy"] = (
        "Cached original H0 and original graph retrieval/proposal. Compact four-head five-seat mean review and "
        "compact blind Phase five-seat majority review overlap; original selectors and merge unchanged. "
        "User-selected roster: GLM-5.3-Flash low reasoning, Qwen3.5-35B-A3B, GPT-5.6-luna, Gemini-3.5-flash-lite, "
        "DeepSeek-v4-flash-vision-exp. Legacy seat key grok identifies GLM, charged to OpenRouter with its "
        "assigned key and strict Together route; mandatory reasoning set to low, excluded from response but billed. No xAI requests. No extra rounds, GT input or new H0 calls.")
    plan["limitations"] += [
        "GLM selected by user, not established as semantic best. Interface smoke does not establish repair gains or full-pipeline latency. GLM reasoning cannot be disabled; exclude only hides its text.",
        "Default Phase remains single-choice majority3, not the separate seven-rating experiment. Existing prompt and voting behavior unchanged.",
        "Qwen cost is a conservative rate envelope, not native billing. Old seat identifiers retained for archive compatibility; reviewer_families and models are authoritative."]
    save(output / "plan.json", plan)
    with lightweight_protocol():
        verify_plan(plan)
    return plan


def dispatch(command, output, adapter, *, source=compact.original.SOURCE):
    if command == "prepare":
        compact.prepare(output, source, adapter)
        plan = finalize_plan(output)
        print(json.dumps({"default_repair_version": VERSION, "models": MODELS,
            "plan_sha256": sha(output / "plan.json"), "targets": len(plan["selection"])}), flush=True)
        return plan
    if command not in ("execute", "score"):
        raise ValueError("unknown repair command")
    if read(output / "plan.json")["profile"] != PROFILE:
        # Do not reinterpret any already prepared historical experiment.
        return previous.dispatch(command, output, adapter, source=source)
    with lightweight_protocol():
        action = compact.original.execute if command == "execute" else compact.original_score.score
        return action(output, adapter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "score"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=compact.original.SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    args = parser.parse_args()
    dispatch(args.command, args.output, CholecTrack20DatasetAdapter(args.dataset_root, causal_window_size=3), source=args.source)
