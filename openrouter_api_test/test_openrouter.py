"""Minimal OpenRouter image-input regression test for Windows CMD."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import shutil
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import requests
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
CHAT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"
DEFAULT_PROMPT = (
    "请分析这张图片。先说明你是否能看到图片，再用中文简要描述主要内容，"
    "并准确抄写至少一段清晰可见的文字。"
)
SUPPORTED_IMAGE_FORMATS = {
    "GIF": ("image/gif", ".gif"),
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}


class TestFailure(RuntimeError):
    """A user-facing failure with a stable process exit code."""

    def __init__(self, message: str, *, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send one local image to an OpenRouter vision model and save the "
            "response beside this script."
        )
    )
    parser.add_argument("--model", required=True, help="OpenRouter model ID")
    credentials = parser.add_mutually_exclusive_group(required=True)
    credentials.add_argument(
        "--api-key",
        help="OpenRouter API key supplied directly in CMD",
    )
    credentials.add_argument(
        "--api-key-file",
        type=Path,
        help="Text file whose only non-empty line is the OpenRouter API key",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=SCRIPT_DIR / "input.png",
        help="PNG, JPEG, WebP, or GIF input image (default: input.png here)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Question to ask about the image",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Where response files are written (default: this directory)",
    )
    parser.add_argument(
        "--proxy",
        help="Optional proxy URL, for example http://127.0.0.1:7890",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum completion tokens (default: 256)",
    )
    return parser


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_api_key(args: argparse.Namespace) -> tuple[str, str]:
    if args.api_key is not None:
        key = args.api_key.strip()
        source = "command_line"
    else:
        assert args.api_key_file is not None
        try:
            lines = [
                line.strip()
                for line in args.api_key_file.expanduser().read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.strip()
            ]
        except OSError as exc:
            raise TestFailure(
                f"无法读取 API key 文件: {exc}", exit_code=2
            ) from None
        if len(lines) != 1:
            raise TestFailure(
                "API key 文件必须只有一个非空行。", exit_code=2
            )
        key = lines[0]
        source = "file"
    if not key:
        raise TestFailure("API key 不能为空。", exit_code=2)
    return key, source


def inspect_image(path: Path) -> tuple[str, str, int, int, int, str]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise TestFailure(f"找不到输入图片: {exc}", exit_code=2) from None
    if not resolved.is_file():
        raise TestFailure(f"输入图片不是文件: {resolved}", exit_code=2)
    try:
        with Image.open(resolved) as image:
            detected_format = image.format
            width, height = image.size
            image.verify()
    except (OSError, ValueError) as exc:
        raise TestFailure(f"无法解析输入图片: {exc}", exit_code=2) from None
    if detected_format not in SUPPORTED_IMAGE_FORMATS:
        raise TestFailure(
            "图片必须是 PNG、JPEG、WebP 或 GIF。", exit_code=2
        )
    mime_type, canonical_suffix = SUPPORTED_IMAGE_FORMATS[detected_format]
    content = resolved.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return mime_type, canonical_suffix, width, height, len(content), digest


def check_model(
    session: requests.Session,
    headers: dict[str, str],
    model: str,
    timeout: float,
) -> dict[str, Any]:
    """Record catalog evidence, but never block aliases or a transient catalog error."""

    started = perf_counter()
    try:
        response = session.get(
            MODELS_ENDPOINT,
            headers=headers,
            timeout=(15.0, timeout),
        )
        response.raise_for_status()
        body = response.json()
        models = body.get("data", []) if isinstance(body, dict) else []
        match = next(
            (
                item
                for item in models
                if isinstance(item, dict) and item.get("id") == model
            ),
            None,
        )
        if match is None:
            return {
                "status": "not_found_in_catalog",
                "http_status": response.status_code,
                "elapsed_ms": round((perf_counter() - started) * 1000, 1),
                "requested_model": model,
                "note": "The chat request will still be attempted because aliases may be valid.",
            }
        architecture = match.get("architecture")
        if not isinstance(architecture, dict):
            architecture = {}
        input_modalities = architecture.get("input_modalities", [])
        output_modalities = architecture.get("output_modalities", [])
        return {
            "status": "found",
            "http_status": response.status_code,
            "elapsed_ms": round((perf_counter() - started) * 1000, 1),
            "requested_model": model,
            "name": match.get("name"),
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
            "accepts_image": "image" in input_modalities,
        }
    except (requests.RequestException, ValueError) as exc:
        return {
            "status": "catalog_check_failed",
            "requested_model": model,
            "elapsed_ms": round((perf_counter() - started) * 1000, 1),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "note": "The chat request will still be attempted.",
        }


def extract_text(response_body: object) -> str:
    if not isinstance(response_body, dict):
        return ""
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part)


def iter_data_image_urls(value: object) -> Iterator[str]:
    if isinstance(value, str):
        if value.startswith("data:image/") and ";base64," in value:
            yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_data_image_urls(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_data_image_urls(item)


def save_returned_images(response_body: object, output_dir: Path) -> list[str]:
    saved: list[str] = []
    seen: set[str] = set()
    extensions = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    for data_url in iter_data_image_urls(response_body):
        if data_url in seen:
            continue
        seen.add(data_url)
        header, encoded = data_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0].lower()
        suffix = extensions.get(mime_type, ".bin")
        try:
            binary = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            continue
        destination = output_dir / f"generated_output_{len(saved) + 1:02d}{suffix}"
        destination.write_bytes(binary)
        saved.append(destination.name)
    return saved


def error_message(response_body: object, fallback: str) -> str:
    if isinstance(response_body, dict):
        error = response_body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
    return fallback


def semantic_response_error(response_body: object) -> str | None:
    """Detect provider errors that can be returned inside a successful HTTP response."""

    if not isinstance(response_body, dict):
        return "响应 JSON 不是对象。"
    if response_body.get("error") is not None:
        return error_message(response_body, "响应包含顶层 error。")
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return "响应中没有 choices。"
    first = choices[0]
    if not isinstance(first, dict):
        return "响应中的第一个 choice 格式无效。"
    choice_error = first.get("error")
    if isinstance(choice_error, dict):
        message = choice_error.get("message")
        return message if isinstance(message, str) else json.dumps(
            choice_error, ensure_ascii=False
        )
    if choice_error is not None:
        return str(choice_error)
    if first.get("finish_reason") == "error":
        return "模型以 finish_reason=error 结束。"
    return None


def run(args: argparse.Namespace) -> int:
    if args.timeout <= 0:
        raise TestFailure("--timeout 必须大于 0。", exit_code=2)
    if args.max_tokens <= 0:
        raise TestFailure("--max-tokens 必须大于 0。", exit_code=2)
    model = args.model.strip()
    prompt = args.prompt.strip()
    if not model:
        raise TestFailure("--model 不能为空。", exit_code=2)
    if not prompt:
        raise TestFailure("--prompt 不能为空。", exit_code=2)

    api_key, key_source = load_api_key(args)
    image_path = args.image
    mime_type, suffix, width, height, byte_count, digest = inspect_image(image_path)
    resolved_image = image_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_copy = output_dir / f"input_used{suffix}"
    if resolved_image != input_copy.resolve():
        shutil.copy2(resolved_image, input_copy)

    response_path = output_dir / "response.json"
    answer_path = output_dir / "response.txt"
    summary_path = output_dir / "test_summary.json"
    model_info_path = output_dir / "model_info.json"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": "http://localhost/openrouter-api-test",
        "X-OpenRouter-Title": "Local OpenRouter Image API Test",
    }
    session = requests.Session()
    if args.proxy:
        session.proxies.update({"http": args.proxy, "https": args.proxy})

    model_info = check_model(session, headers, model, args.timeout)
    write_json(model_info_path, model_info)

    image_bytes = resolved_image.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    request_body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_image}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    del encoded_image

    base_summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "endpoint": CHAT_ENDPOINT,
        "requested_model": model,
        "api_key_source": key_source,
        "api_key_saved_to_output": False,
        "proxy_argument_supplied": bool(args.proxy),
        "input": {
            "source": str(resolved_image),
            "saved_copy": str(input_copy),
            "mime_type": mime_type,
            "width": width,
            "height": height,
            "bytes": byte_count,
            "sha256": digest,
        },
        "model_catalog": model_info,
    }
    started = perf_counter()
    try:
        response = session.post(
            CHAT_ENDPOINT,
            headers=headers,
            json=request_body,
            timeout=(15.0, args.timeout),
        )
    except requests.RequestException as exc:
        elapsed_ms = round((perf_counter() - started) * 1000, 1)
        failure_body = {
            "transport_error": type(exc).__name__,
            "message": str(exc),
        }
        write_json(response_path, failure_body)
        answer_path.write_text(str(exc) + "\n", encoding="utf-8")
        write_json(
            summary_path,
            {
                **base_summary,
                "status": "FAIL",
                "stage": "network",
                "elapsed_ms": elapsed_ms,
                "error": str(exc),
            },
        )
        print(f"FAIL: 网络请求失败：{exc}", file=sys.stderr)
        print(f"结果目录: {output_dir}", file=sys.stderr)
        return 3
    finally:
        del request_body
        del api_key

    elapsed_ms = round((perf_counter() - started) * 1000, 1)
    try:
        response_body: object = response.json()
    except ValueError:
        response_body = {"non_json_response": response.text}
    write_json(response_path, response_body)

    if not response.ok:
        message = error_message(response_body, response.reason)
        answer_path.write_text(message + "\n", encoding="utf-8")
        write_json(
            summary_path,
            {
                **base_summary,
                "status": "FAIL",
                "stage": "http",
                "http_status": response.status_code,
                "elapsed_ms": elapsed_ms,
                "error": message,
            },
        )
        print(
            f"FAIL: OpenRouter 返回 HTTP {response.status_code}: {message}",
            file=sys.stderr,
        )
        print(f"结果目录: {output_dir}", file=sys.stderr)
        return 4

    semantic_error = semantic_response_error(response_body)
    if semantic_error is not None:
        answer_path.write_text(semantic_error + "\n", encoding="utf-8")
        write_json(
            summary_path,
            {
                **base_summary,
                "status": "FAIL",
                "stage": "response",
                "http_status": response.status_code,
                "elapsed_ms": elapsed_ms,
                "error": semantic_error,
            },
        )
        print(f"FAIL: OpenRouter 响应无有效结果：{semantic_error}", file=sys.stderr)
        print(f"结果目录: {output_dir}", file=sys.stderr)
        return 5

    answer = extract_text(response_body)
    if not answer:
        answer = "请求成功，但响应中没有可提取的文本；请查看 response.json。"
    answer_path.write_text(answer.rstrip() + "\n", encoding="utf-8")
    generated_images = save_returned_images(response_body, output_dir)

    response_id = None
    returned_model = None
    finish_reason = None
    usage = None
    if isinstance(response_body, dict):
        response_id = response_body.get("id")
        returned_model = response_body.get("model")
        usage = response_body.get("usage")
        choices = response_body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")

    write_json(
        summary_path,
        {
            **base_summary,
            "status": "PASS",
            "stage": "complete",
            "http_status": response.status_code,
            "elapsed_ms": elapsed_ms,
            "response_id": response_id,
            "returned_model": returned_model,
            "finish_reason": finish_reason,
            "usage": usage,
            "saved_files": [
                response_path.name,
                answer_path.name,
                summary_path.name,
                model_info_path.name,
                input_copy.name,
                *generated_images,
            ],
        },
    )
    print("PASS: OpenRouter 图片输入请求成功。")
    print(f"请求模型: {model}")
    print(f"返回模型: {returned_model}")
    print(f"耗时: {elapsed_ms / 1000:.2f} 秒")
    print(f"结果目录: {output_dir}")
    print(f"文本结果: {answer_path}")
    return 0


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    try:
        exit_code = run(build_parser().parse_args())
    except TestFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from None
    except OSError as exc:
        print(f"FAIL: 文件操作失败：{exc}", file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
