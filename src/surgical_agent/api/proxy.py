"""Explicit process-local proxy setup for urllib provider transports."""

from __future__ import annotations

import os
import urllib.request
from urllib.parse import urlsplit


def configure_local_proxy(proxy_url: str | None) -> None:
    """Route HTTP(S) API traffic through an explicit loopback proxy."""

    if proxy_url is None:
        return
    if not isinstance(proxy_url, str) or not proxy_url.strip():
        raise ValueError("proxy_url must be non-empty text")
    value = proxy_url.strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "proxy_url must be an unauthenticated loopback HTTP(S) proxy"
        )
    os.environ["HTTP_PROXY"] = value
    os.environ["HTTPS_PROXY"] = value
    urllib.request.install_opener(
        urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": value, "https": value})
        )
    )


__all__ = ["configure_local_proxy"]
