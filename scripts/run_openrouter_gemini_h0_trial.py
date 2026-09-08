"""Versioned Gemini/OpenRouter H0 -> Gemini candidates -> five visual reviewers.

Preserves the historical Qwen baseline and its exact prompt/input recipe.
Prepare freezes requests before paid inference; execute never reuses Qwen H0.
"""
import argparse
import json
import shutil
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.check_candidate_panel_providers import MODELS, redact_images
from scripts.run_candidate_panel_trial import RATES, sha
from scripts.run_complete_gt_semantic_trial import (
    choose_complete_samples,
    normalize_review_wire,
    review_body_v3,
)
from scripts.run_presence_review_trial import MODEL as QWEN_MODEL
from scripts.run_presence_review_trial import wire_body
from scripts.run_prior_panel_trial import build_base, now, read, save
from scripts.run_semantic_candidate_trial import PROPOSER, execute
from surgical_agent.api.contracts import thaw_json
from surgical_agent.api.request_hash import canonical_request_metadata
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter

PROFILE = "openrouter_gemini38_h0_final_only_v1_20260908"
ALLOWANCE = {"openrouter_usd": Decimal("0.60"), "xai_usd": Decimal("0.25"),
             "aliyun_cny": Decimal("0.30")}


def gemini_request(template):
    # Validate the frozen recipe before changing exactly model and upstream.
    wire_body(template)
    payload = thaw_json(template.payload)
    payload["openrouter_routing_profile"] = "strict_google_ai_studio"
    return replace(template, model_identifier=PROPOSER, payload=payload)


def build_gemini_base(adapter, selected):
    return gemini_request(build_base(adapter, selected))


def gemini_h0_wire(request):
    if (request.model_identifier != PROPOSER
            or request.payload.get("openrouter_routing_profile") != "strict_google_ai_studio"):
        raise ValueError("Gemini H0 model and fixed Google route required")
    payload = thaw_json(request.payload)
    payload["openrouter_routing_profile"] = "strict_alibaba"
    template = replace(request, model_identifier=QWEN_MODEL, payload=payload)
    body = wire_body(template)  # Retain the exact schema and generation envelope.
    body["model"] = request.model_identifier
    body["provider"]["only"] = ["google-ai-studio"]
    return body


def prepare(output, previous, adapter):
    if output.exists():
        raise ValueError("new output directory required")
    prior = read(previous / "budget.json")
    if not prior["stopped"]:
        raise ValueError("previous inference still running")
    selected = choose_complete_samples(adapter)
    bodies = []
    for item in selected:
        request = build_gemini_base(adapter, item)
        item["request_metadata"] = canonical_request_metadata(request).to_mapping()
        bodies.append(gemini_h0_wire(request))
    url = f"https://openrouter.ai/api/v1/models/{PROPOSER}/endpoints"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    metadata = response.json()
    endpoint = next(e for e in metadata["data"]["endpoints"] if e["tag"] == "google-ai-studio")
    if endpoint["status"] != 0 or not {"reasoning", "response_format", "structured_outputs", "temperature"} <= set(endpoint["supported_parameters"]):
        raise ValueError("Google route unavailable or incompatible")
    if (Decimal(endpoint["pricing"]["prompt"]) > Decimal(RATES["base"][0])
            or Decimal(endpoint["pricing"]["completion"]) > Decimal(RATES["base"][1])):
        raise ValueError("current price exceeds reserved token envelope")
    limits = {k: str(Decimal(prior["occupied"][k]) + v) for k, v in ALLOWANCE.items()}
    paths = [Path(__file__), ROOT / "scripts/run_complete_gt_semantic_trial.py",
             ROOT / "scripts/run_semantic_candidate_trial.py", ROOT / "scripts/replay_semantic_review.py",
             ROOT / "scripts/run_candidate_panel_trial.py", ROOT / "scripts/check_candidate_panel_providers.py",
             ROOT / "scripts/run_prior_panel_trial.py", ROOT / "scripts/run_presence_review_trial.py",
             ROOT / "scripts/run_grounded_api_pipeline.py", ROOT / "configs/perception/joint_openrouter_h0.yaml"]
    for package in ("api", "perception", "data", "config", "evaluation", "artifacts", "research/verification"):
        paths.extend((ROOT / "src/surgical_agent" / package).rglob("*.py"))
    paths.extend((ROOT / "src/surgical_agent/perception/prompts").glob("*"))
    paths = sorted({p for p in paths if p.is_file()})
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    plan = {"created_utc": now(), "profile": PROFILE, "selection": selected,
            "h0": PROPOSER, "h0_provider": "openrouter", "h0_upstream": "google-ai-studio",
            "h0_cache_reused": False, "proposer": PROPOSER, "reviewers": MODELS,
            "compare_old": False, "max_calls": 28, "round_cap": 1,
            "previous_budget": str(previous), "carried_occupied": prior["occupied"], "limits": limits,
            "incremental_allowances": {k: str(v) for k, v in ALLOWANCE.items()},
            "source_sha256": hashes, "pricing_url": url,
            "comparison": "new Gemini H0 vs its evidence-panel repair; cached Qwen as historical comparison only",
            "scope": "same four previously analyzed Training targets; development diagnostic, not held-out evidence",
            "gt_policy": "availability masks only before inference; score GT labels after all calls stop",
            "changed_from_frozen_h0": ["model_identifier", "openrouter_routing_profile"],
            "reviewer_note": "Only the Qwen reviewer uses a direct Aliyun endpoint; neither base call does."}
    save(output / "plan.json", plan)
    save(output / "model_endpoints.json", metadata)
    for item, body in zip(selected, bodies, strict=True):
        save(output / "h0_preflight" / f"{item['key']}.json", redact_images(body))
    for path in paths:
        dest = output / "frozen_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    print(json.dumps({"profile": PROFILE, "targets": [s["key"] for s in selected],
                      "max_calls": 28, "allowances": plan["incremental_allowances"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    adapter = CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3)
    if args.command == "prepare":
        if args.previous is None:
            raise ValueError("previous budget required")
        prepare(args.output, args.previous, adapter)
    else:
        if read(args.output / "plan.json")["profile"] != PROFILE:
            raise ValueError("wrong experiment profile")
        execute(args.output, adapter, review_builder=review_body_v3, review_normalizer=normalize_review_wire,
                base_builder=build_gemini_base, h0_wire_builder=gemini_h0_wire)


if __name__ == "__main__":
    main()
