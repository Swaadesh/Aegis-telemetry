import logging

import pytest

from app.config.settings import Settings
from app.context.base import ContextResult, ContextRetriever, ContextRetrieverError
from app.context.mock_retriever import MockContextRetriever
from app.models.schemas import AIRequest
from app.orchestrator.model_router import ModelRouter
from app.orchestrator.orchestrator import AIOrchestrator, OutOfScopeError
from app.providers.base import AIProvider, ProviderResponse


class _StubProvider(AIProvider):
    def __init__(self) -> None:
        self.last_system_prompt: str | None = None

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        self.last_system_prompt = system_prompt
        return ProviderResponse(text="the answer", model="gemini-3.6-flash", total_tokens=10)


class _FailingRetriever(ContextRetriever):
    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult:
        raise ContextRetrieverError("backend unavailable")


class _EmptyRetriever(ContextRetriever):
    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult:
        return ContextResult(documents=[], citations=[], confidence=None)


def _settings(**overrides) -> Settings:
    return Settings(gemini_api_key="test-key", **overrides)


def _orchestrator(provider, settings, context_retriever=None) -> AIOrchestrator:
    router = ModelRouter({settings.model_name: provider}, settings)
    return AIOrchestrator(router, settings, context_retriever=context_retriever)


@pytest.mark.asyncio
async def test_retrieval_flows_into_prompt_response_and_logs(caplog) -> None:
    provider = _StubProvider()
    orchestrator = _orchestrator(provider, _settings(), context_retriever=MockContextRetriever())

    with caplog.at_level(logging.INFO, logger="aegis.ai"):
        response = await orchestrator.handle(AIRequest(prompt="what is Project Aegis?"))

    assert response.documents_used == 3
    assert len(response.citations) == 3
    assert provider.last_system_prompt is not None
    assert "aegis-master-spec.pdf" in provider.last_system_prompt

    log_line = caplog.records[-1].message
    assert '"documents_retrieved": 3' in log_line
    assert '"citation_ids"' in log_line
    assert '"search_score": 0.93' in log_line


@pytest.mark.asyncio
async def test_no_retriever_matches_phase1_behavior() -> None:
    provider = _StubProvider()
    orchestrator = _orchestrator(provider, _settings(), context_retriever=None)

    response = await orchestrator.handle(AIRequest(prompt="hello"))

    assert response.documents_used == 0
    assert response.citations == []
    assert provider.last_system_prompt is not None
    assert "Source:" not in provider.last_system_prompt


@pytest.mark.asyncio
async def test_retriever_failure_degrades_gracefully() -> None:
    provider = _StubProvider()
    orchestrator = _orchestrator(provider, _settings(), context_retriever=_FailingRetriever())

    response = await orchestrator.handle(AIRequest(prompt="hello"))

    assert response.status == "success"
    assert response.documents_used == 0


@pytest.mark.asyncio
async def test_empty_retrieval_result_raises_out_of_scope() -> None:
    provider = _StubProvider()
    orchestrator = _orchestrator(provider, _settings(), context_retriever=_EmptyRetriever())

    with pytest.raises(OutOfScopeError):
        await orchestrator.handle(AIRequest(prompt="hello"))


@pytest.mark.asyncio
async def test_identity_headers_threaded_into_logs(caplog) -> None:
    provider = _StubProvider()
    orchestrator = _orchestrator(provider, _settings())

    with caplog.at_level(logging.INFO, logger="aegis.ai"):
        await orchestrator.handle(AIRequest(prompt="hello"), user_id="u-123", role="analyst")

    log_line = caplog.records[-1].message
    assert '"user_id": "u-123"' in log_line
    assert '"role": "analyst"' in log_line
    assert '"prompt_tokens"' in log_line
