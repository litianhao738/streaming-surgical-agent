"""Extend recorded identity exclusions and freeze 24 time/mask-selected targets."""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import requests

from scripts.prepare_new_training_verb_selection import prepare as select_targets
from scripts.run_candidate_panel_trial import sha
from scripts.run_prior_panel_trial import now, read, save
from scripts.run_recent_mean_panel_trial import MODELS, RATES_V2, ROUTES
from scripts.run_semantic_candidate_trial import PROPOSER


def prepare(output):
    inventory_source = ROOT / "artifacts/preflight/verb_addition_confirmation_20260909_inventory/historical_inventory.json"
    previous = ROOT / "artifacts/preflight/new_training_verb_guard_20260909_v1"
    if output.exists():
        raise ValueError("new input directory required")
    inventory_path = output.with_name(output.name + "_history.json")
    if inventory_path.exists():
        raise ValueError("new inventory path required")
    old = read(inventory_source)
    excluded = {video: set(frames) for video, frames in old["excluded_targets_by_video"].items()}
    plan, closed = read(previous / "plan.json"), read(previous / "completion.json")
    if closed["fatal_error"] is not None:
        raise ValueError("prior experiment closure required")
    for row in plan["selection"]:
        excluded.setdefault(row["video_id"], set()).update([row["frame_id"], *row["causal_frame_ids"]])
    save(inventory_path, {"created_utc": now(), "parent_inventory": str(inventory_source),
        "parent_inventory_sha256": sha(inventory_source), "previous_plan_sha256": sha(previous / "plan.json"),
        "previous_completion_sha256": sha(previous / "completion.json"),
        "scope": "Original comprehensive history inventory plus all 24 just-completed targets and their 72 image identities.",
        "excluded_targets_by_video": {v: sorted(frames) for v, frames in excluded.items()}})
    select_targets(inventory_path, output, Path("D:/cholec_dataset"))

    def endpoint(item):
        seat, route = item
        model = PROPOSER if seat == "base" else MODELS[seat]
        url = f"https://openrouter.ai/api/v1/models/{model}/endpoints"
        result = requests.get(url, timeout=30, proxies={"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"})
        result.raise_for_status()
        data = result.json()
        route_data = next(e for e in data["data"]["endpoints"] if e["tag"] == route)
        if route_data["status"] != 0 or not {"response_format", "reasoning"} <= set(route_data["supported_parameters"]):
            raise ValueError("fixed endpoint unavailable")
        if any(Decimal(route_data["pricing"][k]) > Decimal(rate)
               for k, rate in zip(("prompt", "completion"), RATES_V2[seat], strict=True)):
            raise ValueError("current price exceeds predeclared envelope")
        return seat, {"url": url, "route": route, "response": data}
    with ThreadPoolExecutor(max_workers=4) as workers:
        metadata = dict(workers.map(endpoint, ROUTES.items()))
    save(output / "model_metadata.json", {"queried_utc": now(), "paid_calls": 0,
        "openrouter_endpoints": metadata, "direct_provider_note": "Original Grok/Qwen settings retained; no extra paid smoke calls."})
    save(output / "budget.json", {"limits": {"openrouter_usd": "10", "xai_usd": "5", "aliyun_cny": "5"},
                                 "rates": RATES_V2})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)
