"""Continuation preserves streamed evidence and prevents a second dispatch."""

import json
from typing import ClassVar
from unittest.mock import patch

import pytest

from scripts.resume_h0_frame_strategy_study import (
    DurableAuditSender,
    reconcile_accounting,
)
from surgical_agent.api.credentials import SecretValue


def test_sender_persists_generation_before_stream_interruption(tmp_path):
    class InterruptedResponse:
        status = 200
        headers: ClassVar[dict] = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            yield b'data: {"id":"gen-test","choices":[]}\n'
            raise OSError("connection interrupted")

    sender = DurableAuditSender(tmp_path, SecretValue("test-secret"))
    with patch("urllib.request.urlopen", return_value=InterruptedResponse()) as network:
        with pytest.raises(OSError, match="connection interrupted"):
            sender("https://example.invalid", {}, b'{"messages":[]}', 1)
        assert json.loads((tmp_path / "generation.json").read_text())["id"] == "gen-test"
        assert b"gen-test" in (tmp_path / "response_stream.sse").read_bytes()
        with pytest.raises(FileExistsError):
            sender("https://example.invalid", {}, b'{"messages":[]}', 1)
        assert network.call_count == 1


def test_sender_redacts_images_and_secrets(tmp_path):
    class Response:
        status = 200
        headers: ClassVar[dict] = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            yield b'data: {"id":"gen-test","text":"test-secret"}\n'

    sender = DurableAuditSender(tmp_path, SecretValue("test-secret"))
    body = json.dumps({"messages": [{"content": [{"type": "image_url", "image_url": {
        "url": "data:image/png;base64,c2VjcmV0"}}]}]}).encode()
    with patch("urllib.request.urlopen", return_value=Response()):
        result = sender("https://example.invalid", {}, body, 1)
    assert result.status_code == 200
    assert result.headers["content-type"] == "text/event-stream"
    for path in tmp_path.iterdir():
        content = path.read_bytes()
        assert b"test-secret" not in content
        assert b"base64,c2VjcmV0" not in content


def test_billing_reconciliation_keeps_failed_prediction_without_network(tmp_path):
    (tmp_path / "generation.json").write_text(json.dumps({"id": "gen-test"}))
    (tmp_path / "generation_accounting.json").write_text(json.dumps({"data": {
        "id": "gen-test", "total_cost": 0.012926}}))
    old = {"provider_call_count": 1, "provider_cost": None, "cache_hit": False,
           "total_latency_ms": 100, "error": {"code": "response_envelope_invalid"}}
    (tmp_path / "api_usage.jsonl").write_text(json.dumps(old) + "\n")
    row = {"status": "API_FAILURE", "error": "response_envelope_invalid", "usage": {"unpriced_calls": 1}}
    with patch("urllib.request.urlopen", side_effect=AssertionError("no network allowed")):
        recovered = reconcile_accounting(row, tmp_path, SecretValue("test-secret"))
    assert recovered["status"] == "API_FAILURE"
    assert "selected_ids" not in recovered
    assert recovered["usage"]["reported_cost_usd"] == 0.012926
    assert recovered["usage"]["provider_calls"] == 1
    assert recovered["usage"]["unpriced_calls"] == 0
    assert json.loads((tmp_path / "usage_before_billing_reconciliation.json").read_text()) == old
