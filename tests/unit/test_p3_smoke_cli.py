import argparse

import pytest

from scripts import smoke_api
from scripts.smoke_api import _safe_error_category
from surgical_agent.api.errors import ApiCallFailure, ApiTransportError


def test_safe_error_category_preserves_terminal_transport_cause() -> None:
    cause = ApiTransportError(
        "provider detail must not escape",
        code="authentication",
        retryable=False,
        status_code=403,
    )
    failure = ApiCallFailure(cause, attempt_count=1, retry_count=0)

    assert _safe_error_category(failure) == "authentication"


def test_main_reports_only_safe_terminal_transport_category(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    provider_detail = "provider-detail-sentinel"
    response_body = "response-body-sentinel"
    key_material = "key-material-sentinel"
    cause = ApiTransportError(
        provider_detail,
        code="authentication",
        retryable=False,
        status_code=403,
    )
    failure = ApiCallFailure(cause, attempt_count=1, retry_count=0)

    class StubParser:
        def parse_args(self) -> argparse.Namespace:
            return argparse.Namespace()

    def raise_failure(_: argparse.Namespace) -> None:
        raise failure

    monkeypatch.setattr(smoke_api, "build_parser", lambda: StubParser())
    monkeypatch.setattr(smoke_api, "run", raise_failure)

    with pytest.raises(SystemExit) as caught:
        smoke_api.main()

    captured = capsys.readouterr()
    assert caught.value.code == 2
    assert captured.out == ""
    assert captured.err == "P3 smoke failed: authentication\n"
    for sensitive_text in (provider_detail, response_body, key_material, "Traceback"):
        assert sensitive_text not in captured.out
        assert sensitive_text not in captured.err
