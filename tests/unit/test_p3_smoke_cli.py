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
