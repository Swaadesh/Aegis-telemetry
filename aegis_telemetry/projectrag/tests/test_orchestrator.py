import asyncio

import pytest

from app.config.settings import Settings
from app.context.base import ContextResult, ContextRetriever, RetrievedDocument
from app.models.schemas import AIRequest
from app.orchestrator.model_router import ModelRouter
from app.orchestrator.orchestrator import (
    AIOrchestrationError,
    AIOrchestrator,
    OutOfScopeError,
)
from app.providers.base import (
    AIProvider,
    ProviderAPIError,
    ProviderResponse,
    ProviderTimeoutError,
)
from app.services.pending_search_store import PendingSearchStore


class _StubProvider(AIProvider):
    def __init__(self, responses=None, exceptions=None) -> None:
        self._responses = list(responses or [])
        self._exceptions = list(exceptions or [])
        self.calls = 0

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        self.calls += 1
        if self._exceptions:
            raise self._exceptions.pop(0)
        return self._responses.pop(0)


def _settings(**overrides) -> Settings:
    return Settings(gemini_api_key="test-key", **overrides)


def _orchestrator(provider: AIProvider, settings: Settings) -> AIOrchestrator:
    router = ModelRouter({settings.model_name: provider}, settings)
    return AIOrchestrator(router, settings)


@pytest.mark.asyncio
async def test_handle_returns_formatted_response_on_success() -> None:
    provider = _StubProvider(
        responses=[ProviderResponse(text="42", model="gemini-3.6-flash", total_tokens=10)]
    )
    orchestrator = _orchestrator(provider, _settings())

    response = await orchestrator.handle(AIRequest(prompt="what is the answer?"))

    assert response.text == "42"
    assert response.status == "success"
    assert response.token_usage.total_tokens == 10


@pytest.mark.asyncio
async def test_handle_retries_on_timeout_then_succeeds() -> None:
    provider = _StubProvider(
        responses=[ProviderResponse(text="ok", model="gemini-3.6-flash")],
        exceptions=[ProviderTimeoutError("slow")],
    )
    orchestrator = _orchestrator(provider, _settings(max_retries=1))

    response = await orchestrator.handle(AIRequest(prompt="hello"))

    assert response.text == "ok"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_handle_does_not_retry_api_error() -> None:
    provider = _StubProvider(exceptions=[ProviderAPIError("bad key")])
    orchestrator = _orchestrator(provider, _settings(max_retries=2))

    with pytest.raises(AIOrchestrationError) as exc_info:
        await orchestrator.handle(AIRequest(prompt="hello"))

    assert provider.calls == 1
    assert exc_info.value.error_type == "ProviderAPIError"


@pytest.mark.asyncio
async def test_handle_exhausts_retries_on_repeated_timeout() -> None:
    provider = _StubProvider(
        exceptions=[ProviderTimeoutError("slow"), ProviderTimeoutError("slow again")]
    )
    orchestrator = _orchestrator(provider, _settings(max_retries=1))

    with pytest.raises(AIOrchestrationError):
        await orchestrator.handle(AIRequest(prompt="hello"))

    assert provider.calls == 2


@pytest.mark.asyncio
async def test_concurrent_requests_are_independent() -> None:
    provider = _StubProvider(
        responses=[
            ProviderResponse(text="a", model="m"),
            ProviderResponse(text="b", model="m"),
            ProviderResponse(text="c", model="m"),
        ]
    )
    orchestrator = _orchestrator(provider, _settings())

    results = await asyncio.gather(
        orchestrator.handle(AIRequest(prompt="1")),
        orchestrator.handle(AIRequest(prompt="2")),
        orchestrator.handle(AIRequest(prompt="3")),
    )

    assert {r.request_id for r in results}.__len__() == 3
    assert provider.calls == 3


class _EmptyRetriever(ContextRetriever):
    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult:
        return ContextResult(documents=[], citations=[], confidence=None)


class _WebRetriever(ContextRetriever):
    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult:
        return ContextResult(
            documents=[
                RetrievedDocument(text="web answer text", source="https://example.com", score=0.7)
            ],
            citations=["https://example.com"],
            confidence=0.7,
        )


@pytest.mark.asyncio
async def test_empty_document_context_raises_out_of_scope_and_stores_pending() -> None:
    provider = _StubProvider(responses=[ProviderResponse(text="unused", model="m")])
    settings = _settings()
    pending_store = PendingSearchStore()
    orchestrator = AIOrchestrator(
        ModelRouter({settings.model_name: provider}, settings),
        settings,
        context_retriever=_EmptyRetriever(),
        pending_search_store=pending_store,
    )

    with pytest.raises(OutOfScopeError) as exc_info:
        await orchestrator.handle(AIRequest(prompt="something not in the doc"))

    request_id = exc_info.value.request_id
    entry = pending_store.peek(request_id)
    assert entry is not None
    assert entry.prompt == "something not in the doc"


@pytest.mark.asyncio
async def test_confirm_web_search_false_evicts_pending_and_returns_none() -> None:
    provider = _StubProvider(responses=[ProviderResponse(text="unused", model="m")])
    settings = _settings()
    pending_store = PendingSearchStore()
    orchestrator = AIOrchestrator(
        ModelRouter({settings.model_name: provider}, settings),
        settings,
        context_retriever=_EmptyRetriever(),
        pending_search_store=pending_store,
    )
    with pytest.raises(OutOfScopeError) as exc_info:
        await orchestrator.handle(AIRequest(prompt="q"))
    request_id = exc_info.value.request_id

    result = await orchestrator.confirm_web_search(request_id, confirm=False)

    assert result is None
    assert pending_store.peek(request_id) is None


@pytest.mark.asyncio
async def test_confirm_web_search_true_uses_web_retriever_and_marks_source_web() -> None:
    provider = _StubProvider(responses=[ProviderResponse(text="web-backed answer", model="m")])
    settings = _settings()
    pending_store = PendingSearchStore()
    orchestrator = AIOrchestrator(
        ModelRouter({settings.model_name: provider}, settings),
        settings,
        context_retriever=_EmptyRetriever(),
        pending_search_store=pending_store,
        web_retriever=_WebRetriever(),
    )
    with pytest.raises(OutOfScopeError) as exc_info:
        await orchestrator.handle(AIRequest(prompt="q"))
    request_id = exc_info.value.request_id

    response = await orchestrator.confirm_web_search(request_id, confirm=True)

    assert response is not None
    assert response.source == "web"
    assert response.citations == ["https://example.com"]
