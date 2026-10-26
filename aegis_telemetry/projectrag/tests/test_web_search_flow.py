import json

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.context.base import ContextResult, ContextRetriever, RetrievedDocument
from app.dependencies import (
    get_orchestrator,
    get_pending_search_store,
    get_response_lock_store,
)
from app.main import app
from app.orchestrator.model_router import ModelRouter
from app.orchestrator.orchestrator import AIOrchestrator
from app.providers.base import AIProvider, ProviderResponse
from app.services.pending_search_store import PendingSearchStore
from app.services.response_lock_store import ResponseLockStore


class _StubProvider(AIProvider):
    def __init__(self, total_tokens: int = 10, text: str = "web-backed answer") -> None:
        self._total_tokens = total_tokens
        self._text = text

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        return ProviderResponse(text=self._text, model="m", total_tokens=self._total_tokens)


class _EmptyRetriever(ContextRetriever):
    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult:
        return ContextResult(documents=[], citations=[], confidence=None)


class _WebRetriever(ContextRetriever):
    async def retrieve(self, query: str, top_k: int = 5) -> ContextResult:
        return ContextResult(
            documents=[RetrievedDocument(text="web text", source="https://example.com", score=0.6)],
            citations=["https://example.com"],
            confidence=0.6,
        )


def _client(
    total_tokens: int = 10, text: str = "web-backed answer"
) -> tuple[TestClient, PendingSearchStore, ResponseLockStore]:
    settings = Settings(gemini_api_key="k", tavily_api_key="tvly-k")
    provider = _StubProvider(total_tokens=total_tokens, text=text)
    router = ModelRouter({settings.model_name: provider}, settings)
    pending_store = PendingSearchStore()
    lock_store = ResponseLockStore()
    orchestrator = AIOrchestrator(
        router,
        settings,
        context_retriever=_EmptyRetriever(),
        pending_search_store=pending_store,
        web_retriever=_WebRetriever(),
        lock_store=lock_store,
    )
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_pending_search_store] = lambda: pending_store
    app.dependency_overrides[get_response_lock_store] = lambda: lock_store
    return TestClient(app), pending_store, lock_store


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_generate_with_no_document_matches_returns_out_of_scope() -> None:
    client, _, _ = _client()
    resp = client.post("/api/v1/generate", json={"prompt": "not in the doc"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "out_of_scope"
    assert "request_id" in body


def test_websearch_confirm_false_returns_declined_and_evicts_pending() -> None:
    client, pending_store, _ = _client()
    request_id = client.post("/api/v1/generate", json={"prompt": "q"}).json()["request_id"]

    resp = client.post("/api/v1/websearch-confirm", json={"request_id": request_id, "confirm": False})

    assert resp.status_code == 200
    assert resp.json()["status"] == "declined"
    assert pending_store.peek(request_id) is None


def test_websearch_confirm_true_returns_web_sourced_response() -> None:
    client, _, _ = _client(total_tokens=10)
    request_id = client.post("/api/v1/generate", json={"prompt": "q"}).json()["request_id"]

    resp = client.post("/api/v1/websearch-confirm", json={"request_id": request_id, "confirm": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["source"] == "web"
    assert body["citations"] == ["https://example.com"]


def test_websearch_confirm_true_with_long_response_locks_with_web_source() -> None:
    client, _, lock_store = _client(total_tokens=1500)
    request_id = client.post("/api/v1/generate", json={"prompt": "q"}).json()["request_id"]

    resp = client.post("/api/v1/websearch-confirm", json={"request_id": request_id, "confirm": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "locked"
    entry = lock_store.peek(request_id)
    assert entry is not None
    assert entry.response.source == "web"

    unlock_resp = client.post("/api/v1/unlock", json={"request_id": request_id, "admin_key": "7894"})
    assert unlock_resp.status_code == 200
    assert unlock_resp.json()["source"] == "web"


def test_websearch_confirm_unknown_request_id_returns_404() -> None:
    client, _, _ = _client()
    resp = client.post(
        "/api/v1/websearch-confirm", json={"request_id": "does-not-exist", "confirm": True}
    )
    assert resp.status_code == 404


def test_websearch_confirm_twice_second_call_404s() -> None:
    client, _, _ = _client()
    request_id = client.post("/api/v1/generate", json={"prompt": "q"}).json()["request_id"]

    first = client.post("/api/v1/websearch-confirm", json={"request_id": request_id, "confirm": True})
    assert first.status_code == 200

    second = client.post("/api/v1/websearch-confirm", json={"request_id": request_id, "confirm": True})
    assert second.status_code == 404


def test_out_of_scope_and_confirm_events_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="aegis.ai")
    client, _, _ = _client()
    request_id = client.post("/api/v1/generate", json={"prompt": "q"}).json()["request_id"]
    client.post("/api/v1/websearch-confirm", json={"request_id": request_id, "confirm": True})

    lines = [json.loads(r.message) for r in caplog.records if r.name == "aegis.ai"]
    assert any(line.get("status") == "out_of_scope" for line in lines)
    assert any(line.get("event") == "confirmed" for line in lines)
