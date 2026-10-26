from app.models.schemas import (
    AIResponse,
    LockedResponse,
    OutOfScopeResponse,
    TokenUsage,
    WebSearchConfirmRequest,
)


def test_ai_response_defaults_source_to_document() -> None:
    response = AIResponse(
        request_id="r1", model="m", text="hi", latency_ms=1.0, token_usage=TokenUsage()
    )
    assert response.source == "document"


def test_ai_response_accepts_web_source() -> None:
    response = AIResponse(
        request_id="r1",
        model="m",
        text="hi",
        latency_ms=1.0,
        token_usage=TokenUsage(),
        source="web",
    )
    assert response.source == "web"


def test_locked_response_defaults_source_to_document() -> None:
    locked = LockedResponse(request_id="r1", message="locked", attempts_remaining=2)
    assert locked.source == "document"


def test_out_of_scope_response_shape() -> None:
    body = OutOfScopeResponse(request_id="r1", message="nothing found, search the web?")
    assert body.status == "out_of_scope"


def test_web_search_confirm_request_shape() -> None:
    body = WebSearchConfirmRequest(request_id="r1", confirm=True)
    assert body.confirm is True
