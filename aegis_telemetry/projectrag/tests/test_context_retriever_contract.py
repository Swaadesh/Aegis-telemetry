"""Reusable contract test for any ContextRetriever implementation.

Run this against the mock today, and against Atif's real implementation
before it is wired into app/dependencies.py (see phase2.txt Step 10) — a
silent shape mismatch gets caught here, not in production.
"""

from pathlib import Path

import httpx
import pytest

from app.context.base import ContextResult, ContextRetriever, RetrievedDocument
from app.context.mock_retriever import MockContextRetriever
from app.context.pdf_retriever import PDFContextRetriever
from app.context.tavily_retriever import TavilyContextRetriever


class ContextRetrieverContract:
    """Base class: subclass and set `retriever` to test a new implementation."""

    retriever: ContextRetriever

    @pytest.mark.asyncio
    async def test_retrieve_returns_context_result(self) -> None:
        result = await self.retriever.retrieve("any query", top_k=5)
        assert isinstance(result, ContextResult)

    @pytest.mark.asyncio
    async def test_documents_have_required_shape(self) -> None:
        result = await self.retriever.retrieve("any query", top_k=5)
        for doc in result.documents:
            assert isinstance(doc, RetrievedDocument)
            assert isinstance(doc.text, str) and doc.text
            assert isinstance(doc.source, str) and doc.source
            assert isinstance(doc.score, int | float)
            assert 0.0 <= doc.score <= 1.0
            assert isinstance(doc.metadata, dict)

    @pytest.mark.asyncio
    async def test_top_k_zero_does_not_raise(self) -> None:
        result = await self.retriever.retrieve("any query", top_k=0)
        assert isinstance(result, ContextResult)
        assert result.documents == []

    @pytest.mark.asyncio
    async def test_no_match_case_does_not_raise(self) -> None:
        result = await self.retriever.retrieve(
            "a query guaranteed to match nothing xyzzy123", top_k=5
        )
        assert isinstance(result, ContextResult)
        assert isinstance(result.documents, list)


class TestMockContextRetrieverContract(ContextRetrieverContract):
    retriever = MockContextRetriever()


_PDF_PATH = Path(__file__).resolve().parents[2] / "TokenLens_Telemetry_and_SpendGuard.pdf"
_pdf_retriever = PDFContextRetriever(str(_PDF_PATH)) if _PDF_PATH.is_file() else None


@pytest.mark.skipif(_pdf_retriever is None, reason="master spec PDF not present in this checkout")
class TestPDFContextRetrieverContract(ContextRetrieverContract):
    retriever = _pdf_retriever  # type: ignore[assignment]


def _tavily_contract_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "query": "any query",
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com/result",
                    "content": "Some relevant content.",
                    "score": 0.75,
                }
            ],
        },
    )


class TestTavilyContextRetrieverContract(ContextRetrieverContract):
    retriever = TavilyContextRetriever(
        api_key="test-key",
        client=httpx.AsyncClient(transport=httpx.MockTransport(_tavily_contract_handler)),
    )
