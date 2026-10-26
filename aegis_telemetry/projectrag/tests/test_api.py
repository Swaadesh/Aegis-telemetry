import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_orchestrator
from app.main import app
from app.orchestrator.model_router import ModelRouter
from app.orchestrator.orchestrator import AIOrchestrator
from app.providers.base import AIProvider, ProviderResponse, ProviderTimeoutError


class _StubProvider(AIProvider):
    def __init__(self, response=None, exception=None) -> None:
        self._response = response
        self._exception = exception

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        if self._exception:
            raise self._exception
        return self._response


def _client_with_provider(provider: AIProvider, max_retries: int = 0) -> TestClient:
    from app.config.settings import Settings

    settings = Settings(
        gemini_api_key="test-key", tavily_api_key="tvly-test-key", max_retries=max_retries
    )
    router = ModelRouter({settings.model_name: provider}, settings)
    orchestrator = AIOrchestrator(router, settings)
    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_generate_normal_prompt_returns_200() -> None:
    client = _client_with_provider(
        _StubProvider(response=ProviderResponse(text="hi there", model="gemini-3.6-flash"))
    )
    resp = client.post("/api/v1/generate", json={"prompt": "hello"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "hi there"
    assert body["status"] == "success"
    assert "request_id" in body


def test_generate_empty_prompt_returns_422() -> None:
    client = _client_with_provider(_StubProvider(response=ProviderResponse(text="x", model="m")))
    resp = client.post("/api/v1/generate", json={"prompt": ""})

    assert resp.status_code == 422


def test_generate_long_prompt_returns_200() -> None:
    long_prompt = "word " * 5000
    client = _client_with_provider(
        _StubProvider(response=ProviderResponse(text="summarized", model="m"))
    )
    resp = client.post("/api/v1/generate", json={"prompt": long_prompt})

    assert resp.status_code == 200


def test_generate_provider_timeout_returns_502() -> None:
    client = _client_with_provider(_StubProvider(exception=ProviderTimeoutError("slow")))
    resp = client.post("/api/v1/generate", json={"prompt": "hello"})

    assert resp.status_code == 502
    body = resp.json()
    assert body["status"] == "error"


def test_health_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_concurrent_requests_via_api() -> None:
    client = _client_with_provider(
        _StubProvider(response=ProviderResponse(text="concurrent-ok", model="m"))
    )
    responses = [client.post("/api/v1/generate", json={"prompt": f"req-{i}"}) for i in range(5)]

    assert all(r.status_code == 200 for r in responses)
    request_ids = {r.json()["request_id"] for r in responses}
    assert len(request_ids) == 5
