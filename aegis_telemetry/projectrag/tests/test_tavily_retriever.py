import httpx
import pytest

from app.context.base import ContextRetrieverError
from app.context.tavily_retriever import TavilyContextRetriever

_SEARCH_URL = "https://api.tavily.com/search"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_retrieve_maps_results_to_context_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == _SEARCH_URL
        assert request.headers["authorization"] == "Bearer tvly-test-key"
        body = request.read()
        assert b'"query"' in body
        return httpx.Response(
            200,
            json={
                "query": "what is aegis",
                "results": [
                    {
                        "title": "Project Aegis",
                        "url": "https://example.com/aegis",
                        "content": "Project Aegis is an AI gateway.",
                        "score": 0.82,
                    }
                ],
            },
        )

    retriever = TavilyContextRetriever(
        api_key="tvly-test-key", max_results=5, client=_client(handler)
    )

    result = await retriever.retrieve("what is aegis", top_k=5)

    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.text == "Project Aegis is an AI gateway."
    assert doc.source == "https://example.com/aegis"
    assert doc.score == 0.82
    assert result.citations == ["https://example.com/aegis"]
    assert result.confidence == 0.82


@pytest.mark.asyncio
async def test_retrieve_with_no_results_returns_empty_context() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": "xyzzy123", "results": []})

    retriever = TavilyContextRetriever(api_key="tvly-test-key", client=_client(handler))

    result = await retriever.retrieve("xyzzy123", top_k=5)

    assert result.documents == []
    assert result.citations == []
    assert result.confidence is None


@pytest.mark.asyncio
async def test_retrieve_raises_context_retriever_error_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid api key"})

    retriever = TavilyContextRetriever(api_key="bad-key", client=_client(handler))

    with pytest.raises(ContextRetrieverError):
        await retriever.retrieve("anything", top_k=5)


@pytest.mark.asyncio
async def test_retrieve_raises_context_retriever_error_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    retriever = TavilyContextRetriever(api_key="tvly-test-key", client=_client(handler))

    with pytest.raises(ContextRetrieverError):
        await retriever.retrieve("anything", top_k=5)
