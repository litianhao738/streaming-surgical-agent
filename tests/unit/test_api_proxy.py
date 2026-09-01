from __future__ import annotations

import os

import pytest

from surgical_agent.api.proxy import configure_local_proxy


def test_configure_local_proxy_sets_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    configure_local_proxy("http://127.0.0.1:7897")
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7897"


@pytest.mark.parametrize(
    "value",
    (
        "http://example.com:7897",
        "socks5://127.0.0.1:7897",
        "http://user:pass@127.0.0.1:7897",
        "http://127.0.0.1",
    ),
)
def test_configure_local_proxy_rejects_non_loopback_or_ambiguous_values(value: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        configure_local_proxy(value)
