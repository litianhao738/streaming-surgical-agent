"""One-request Kimi H0 smoke. Read the key from env, hidden prompt, or stdin."""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgical_agent.api.errors import ApiSchemaError
from surgical_agent.perception.final_only import final_only_schema, validate_final_only
from surgical_agent.perception.main_h0 import load_main_h0_prompt, main_h0_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://speed.toter.me")
    parser.add_argument("--model", default="kimi-k3")
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/cholec_dataset"))
    parser.add_argument("--video", default="VID103")
    parser.add_argument("--frame", type=int, default=25101)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--key-stdin", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if not base.startswith("https://"):
        parser.error("An HTTPS endpoint is required")
    if not base.endswith("/v1"):
        base += "/v1"
    frame_ids = (args.frame - 50, args.frame - 25, args.frame)
    content = [{"type": "text", "text": json.dumps(main_h0_input(
        video_id=args.video, target_frame_id=args.frame, frame_ids=frame_ids))}]
    for index, frame in enumerate(frame_ids):
        path = args.dataset_root / "Training" / args.video / "Frames" / f"{frame:06d}.png"
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(),
            "detail": "high" if index == 2 else "low",
        }})
    if args.key_stdin:
        print("Waiting for API key on stdin (not logged)", flush=True)
        key = sys.stdin.readline().strip()
    else:
        key = os.environ.get("KIMI_API_KEY") or getpass.getpass("API key: ")
    if not key:
        parser.error("Missing API key")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)

    def request(url, payload=None, timeout=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json",
        })
        with opener.open(req, timeout=timeout or args.timeout) as response:
            return json.loads(response.read().decode())

    model = args.model
    report = {"endpoint": base, "target": f"{args.video}_{args.frame}",
              "automatic_retries": 0, "inference_attempts": 0}
    try:
        catalog = request(base + "/models", timeout=15)
        matches = [row["id"] for row in catalog.get("data", [])
                   if "kimi" in row.get("id", "").lower() and "k3" in row["id"].lower()]
        report["available_k3_models"] = matches
        exact = next((m for m in matches if m.lower() == model.lower()), None)
        if exact:
            model = exact
        elif len(matches) == 1:
            model = matches[0]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["model_discovery_error"] = str(exc).replace(key, "[REDACTED]")
    report["requested_model"] = model
    payload = {"model": model, "stream": False, "max_tokens": 4096,
               "reasoning_effort": "low", "messages": [
                   {"role": "system", "content": load_main_h0_prompt()},
                   {"role": "user", "content": content}],
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": "h0", "strict": True, "schema": final_only_schema()}}}
    print(f"Sending ONE H0 request: {model}; target={report['target']}", flush=True)
    started = time.perf_counter()
    report["inference_attempts"] = 1
    try:
        response = request(base + "/chat/completions", payload)
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        report["response"] = response
        choice = response["choices"][0]
        output = choice["message"]["content"]
        prediction = json.loads(output)
        validate_final_only(prediction)
        if choice.get("finish_reason") != "stop":
            raise ValueError("Response did not finish normally")
        report["status"] = "H0_SCHEMA_VALID"
        report["h0"] = prediction
    except urllib.error.HTTPError as exc:
        report.update(status="HTTP_ERROR", http_status=exc.code,
                      error=exc.read().decode(errors="replace")[:8000])
    except (OSError, ValueError, KeyError, TypeError, IndexError, ApiSchemaError) as exc:
        report.update(status="FAILED", error=str(exc))
    report.setdefault("elapsed_seconds", round(time.perf_counter() - started, 3))
    output_dir = ROOT / "artifacts" / "preflight" / (
        "kimi_endpoint_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2).replace(key, "[REDACTED]"), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "response"}
    summary["usage"] = report.get("response", {}).get("usage")
    summary["artifact"] = str(output_dir / "result.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2).replace(key, "[REDACTED]"))
    return 0 if report["status"] == "H0_SCHEMA_VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
