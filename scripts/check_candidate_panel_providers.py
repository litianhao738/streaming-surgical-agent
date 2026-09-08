"""Five-family visual contract preflight only; no repair or H0 API calls.

One request per seat, no retries or substitutions. Existing outputs prevent
accidental paid replay. Secrets are loaded only from their assigned local file.
"""
import argparse
import base64
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_prior_panel_trial import build_base, now, read, save
from surgical_agent.api.credentials import (
    SecretValue,
    assert_secret_absent,
    load_api_key_file,
)
from surgical_agent.data.dataset import CholecTrack20DatasetAdapter
from surgical_agent.perception.main_h0 import _LABEL_BOUNDARY
from surgical_agent.research.verification.candidate_coordinator import (
    make_pool,
    score_schema,
    validate_scores,
)

ALIYUN = "https://llm-osbw8dppqg2h8sx0.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODELS = {
    "qwen": (ALIYUN, "qwen3-vl-flash", "docs/aliyun_API KEY", None),
    "deepseek": ("https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash-vision-exp", "docs/API.txt", "fireworks"),
    "grok": ("https://api.x.ai/v1", "grok-4.20-0309-non-reasoning", "docs/grok_API.txt", None),
    "gpt": ("https://openrouter.ai/api/v1", "openai/gpt-4.1-mini", "docs/API.txt", "openai"),
    "gemini": ("https://openrouter.ai/api/v1", "google/gemini-2.5-flash-lite", "docs/API.txt", "google-ai-studio"),
}
ACADEMIC = ("This is an academic analysis of a laparoscopic medical training video. "
            "The task is dataset annotation of surgical instruments, actions and anatomy.")


def key_for(seat):
    path = ROOT / MODELS[seat][2]
    if seat == "qwen":
        value = path.read_text(encoding="utf-8-sig").strip()
        if not value.startswith("sk-ws-") or any(x.isspace() for x in value):
            raise ValueError("expected one workspace credential in assigned local file")
        return SecretValue(value)
    return load_api_key_file(path)


def body_for(seat, base, selection, pool, *, gemini_json_object=False):
    packet = {
        "academic_context": ACADEMIC,
        "task": "Rate each candidate independently for presence anywhere in the TARGET frame. Multiple IVTs may coexist.",
        "rating_scale": {"1": "clearly absent/refuted across the target frame", "2": "likely absent",
                         "3": "uncertain, obscured or insufficient evidence", "4": "likely visually present",
                         "5": "clearly visually present"},
        "instructions": ("History only helps judge motion; do not label history-only events as current. "
                         "A tool doing something else does not rule out another tool doing this action. "
                         "Not seeing clearly means 3, not absence. Candidate membership is not evidence. "
                         "Return only a JSON object with scores mapping every candidate ID to an integer 1..5; no prose."),
        "target_frame_id": selection["frame_id"],
        "images": [{"index": i, "frame_id": f, "seconds_relative_to_target": (f-selection["frame_id"])/25}
                   for i, f in enumerate(selection["causal_frame_ids"])],
        "propositions": pool["propositions"],
        "label_boundaries": _LABEL_BOUNDARY,
        "response_schema": score_schema(pool),
    }
    content = [{"type": "text", "text": json.dumps(packet, ensure_ascii=False)}]
    for im, detail in zip(base.images, base.payload["image_details"], strict=True):
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{im.mime_type};base64," + base64.b64encode(im.content).decode(), "detail": detail}})
    body = {"model": MODELS[seat][1], "messages": [{"role": "user", "content": content}],
            "temperature": 0, "max_tokens": 512, "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "candidate_scores_v1", "strict": True, "schema": score_schema(pool)}}}
    if seat == "qwen":
        # Aliyun JSON object mode plus the identical explicit local numeric contract.
        body["response_format"] = {"type": "json_object"}
        body["enable_thinking"] = False
    if MODELS[seat][3]:
        tag = MODELS[seat][3]
        body["provider"] = {"only": [tag], "order": [tag], "allow_fallbacks": False, "require_parameters": True}
    if seat == "gemini":
        body["reasoning"] = {"enabled": False}
        if gemini_json_object:
            body["response_format"] = {"type": "json_object"}
    if seat == "deepseek":
        body["reasoning"] = {"enabled": False}
        body["response_format"] = {"type": "json_object"}
    return body


def redact_images(body):
    safe = deepcopy(body)
    for message in safe["messages"]:
        if not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if block.get("type") != "image_url":
                continue
            url = block["image_url"]["url"]
            block["image_url"]["url"] = "[image bytes omitted]"
            block["image_url"]["data_url_sha256"] = hashlib.sha256(url.encode()).hexdigest()
    return safe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="dispatch exactly one request per seat")
    p.add_argument("--seats", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--gemini-json-object", action="store_true")
    args = p.parse_args()
    if len(set(args.seats)) != len(args.seats):
        raise ValueError("duplicate seats")
    if args.output.exists():
        raise ValueError("use a new output directory; paid preflight cannot overwrite/replay")
    fixture = ROOT / "artifacts/preflight/h0_prior_panel_openrouter_20260907_v7/smoke_input.json"
    source = read(fixture)
    base = build_base(CholecTrack20DatasetAdapter(Path("D:/cholec_dataset"), causal_window_size=3), source["selection"])
    pool = make_pool(source["h0"])
    plan = {"created_utc": now(), "scope": "visual_interface_compatibility_only",
            "selection": source["selection"], "h0": source["h0"], "pool": pool,
            "max_post_calls": len(args.seats), "max_output_tokens_each": 512, "automatic_retries": 0,
            "selected_seats": args.seats, "gemini_json_object": args.gemini_json_object,
            "no_h0_generation": True, "no_gt_in_requests": True, "models": MODELS,
            "grok_caveat": "Account-available non-reasoning model; not established to be a small model.",
            "budget": {"openrouter_usd_planning_allowance": 0.10, "xai_usd_planning_allowance": 0.10,
                       "aliyun_cny_planning_allowance": 1.0, "type": "planning allowances, not provider billing guarantees"},
            "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in [Path(__file__), ROOT / "src/surgical_agent/research/verification/candidate_coordinator.py", fixture]}}
    save(args.output / "plan.json", plan)
    bodies = {seat: body_for(seat, base, source["selection"], pool,
                            gemini_json_object=args.gemini_json_object) for seat in args.seats}
    for seat, body in bodies.items():
        save(args.output / f"{seat}_request.json", redact_images(body))
    if not args.execute:
        print("Prepared only; no API requests.")
        return
    results = {}
    for seat, body in bodies.items():
        secret = key_for(seat)
        entry = {"seat": seat, "model": MODELS[seat][1], "started_utc": now(), "status": "DISPATCHED"}
        save(args.output / f"{seat}_response.json", entry)
        try:
            r = requests.post(MODELS[seat][0] + "/chat/completions",
                              headers={"Authorization": "Bearer " + secret.reveal()}, json=body, timeout=(15, 90))
            entry["http_status"] = r.status_code
            try:
                raw = r.json()
            except ValueError:
                raw = {"non_json_response": r.text}
            raw = json.loads(json.dumps(raw).replace(secret.reveal(), "[REDACTED]"))
            entry["body"] = raw
            # Billing must survive numeric validation failures too.
            entry["usage"] = raw.get("usage")
            entry["status"] = "API_FAILED"
            if r.ok and not raw.get("error"):
                choice = raw["choices"][0]
                if choice.get("finish_reason") != "stop" or choice["message"].get("refusal"):
                    raise ValueError("incomplete/refused output")
                if raw.get("model") != MODELS[seat][1] and seat not in {"gpt", "gemini"}:
                    raise ValueError("unexpected returned model")
                entry["numeric_validation"] = validate_scores(json.loads(choice["message"]["content"]), pool)
                entry["status"] = "VALID_NUMERIC_SCORES"
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            entry["status"] = "INVALID_OR_TRANSPORT_FAILED"
            entry["exception_type"] = type(exc).__name__
        entry["finished_utc"] = now()
        save(args.output / f"{seat}_response.json", entry)
        assert_secret_absent(secret, args.output.glob("*.json"))
        results[seat] = {k: entry.get(k) for k in ("model", "http_status", "status", "usage")}
        print(json.dumps({seat: results[seat]}), flush=True)
    save(args.output / "summary.json", {"results": results,
         "five_seat_ready": set(results) == set(MODELS)
         and all(r["status"] == "VALID_NUMERIC_SCORES" for r in results.values()),
         "semantic_repair_effect_tested": False, "actual_post_calls": len(results)})


if __name__ == "__main__":
    main()
