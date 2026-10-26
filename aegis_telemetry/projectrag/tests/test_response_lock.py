import json

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings, get_settings
from app.dependencies import get_orchestrator, get_response_lock_store
from app.main import app
from app.orchestrator.model_router import ModelRouter
from app.orchestrator.orchestrator import AIOrchestrator
from app.providers.base import AIProvider, ProviderResponse
from app.services.response_lock_store import ResponseLockStore


class _StubProvider(AIProvider):
    def __init__(self, response: ProviderResponse) -> None:
        self._response = response

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        return self._response


def _client_with_locking(total_tokens: int) -> TestClient:
    settings = Settings(
        gemini_api_key="test-key",
        tavily_api_key="tvly-test-key",
        response_token_limit=1000,
        admin_unlock_key="7894",
        unlock_max_attempts=2,
    )
    provider = _StubProvider(
        ProviderResponse(text="a very long answer", model="m", total_tokens=total_tokens)
    )
    router = ModelRouter({settings.model_name: provider}, settings)
    store = ResponseLockStore()
    orchestrator = AIOrchestrator(router, settings, lock_store=store)

    app.dependency_overrides[get_orchestrator] = lambda: orchestrator
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_response_lock_store] = lambda: store
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_short_response_is_not_locked() -> None:
    client = _client_with_locking(total_tokens=50)
    resp = client.post("/api/v1/generate", json={"prompt": "hello"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_long_response_returns_locked_message() -> None:
    client = _client_with_locking(total_tokens=1500)
    resp = client.post("/api/v1/generate", json={"prompt": "give me a long answer"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "locked"
    assert body["attempts_remaining"] == 2
    assert "request_id" in body


def test_unlock_with_correct_key_returns_original_response() -> None:
    client = _client_with_locking(total_tokens=1500)
    locked = client.post("/api/v1/generate", json={"prompt": "long"}).json()

    resp = client.post(
        "/api/v1/unlock", json={"request_id": locked["request_id"], "admin_key": "7894"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["text"] == "a very long answer"


def test_unlock_wrong_key_once_then_correct_key() -> None:
    client = _client_with_locking(total_tokens=1500)
    request_id = client.post("/api/v1/generate", json={"prompt": "long"}).json()["request_id"]

    fail = client.post("/api/v1/unlock", json={"request_id": request_id, "admin_key": "0000"})
    assert fail.status_code == 401
    assert "1 attempt" in fail.json()["message"]

    success = client.post("/api/v1/unlock", json={"request_id": request_id, "admin_key": "7894"})
    assert success.status_code == 200
    assert success.json()["status"] == "success"


def test_unlock_exhausts_attempts_then_locks_out() -> None:
    client = _client_with_locking(total_tokens=1500)
    request_id = client.post("/api/v1/generate", json={"prompt": "long"}).json()["request_id"]

    first = client.post("/api/v1/unlock", json={"request_id": request_id, "admin_key": "0000"})
    assert first.status_code == 401

    second = client.post("/api/v1/unlock", json={"request_id": request_id, "admin_key": "0000"})
    assert second.status_code == 403
    assert "contact admin" in second.json()["message"].lower()

    # entry evicted — even the correct key now 404s
    after = client.post("/api/v1/unlock", json={"request_id": request_id, "admin_key": "7894"})
    assert after.status_code == 404


def test_unlock_unknown_request_id_returns_404() -> None:
    client = _client_with_locking(total_tokens=50)
    resp = client.post(
        "/api/v1/unlock", json={"request_id": "does-not-exist", "admin_key": "7894"}
    )

    assert resp.status_code == 404


def _log_lines(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [json.loads(r.message) for r in caplog.records if r.name == "aegis.ai"]


def test_locked_log_line_includes_threshold(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="aegis.ai")
    client = _client_with_locking(total_tokens=1500)
    client.post(
        "/api/v1/generate",
        json={"prompt": "long"},
        headers={"X-User-Id": "u1", "X-User-Role": "analyst"},
    )

    locked_lines = [line for line in _log_lines(caplog) if line.get("status") == "locked"]
    assert len(locked_lines) == 1
    assert locked_lines[0]["threshold"] == 1000
    assert locked_lines[0]["user_id"] == "u1"
    assert locked_lines[0]["role"] == "analyst"


def test_unlock_resolution_events_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="aegis.ai")
    client = _client_with_locking(total_tokens=1500)
    request_id = client.post("/api/v1/generate", json={"prompt": "long"}).json()["request_id"]

    client.post(
        "/api/v1/unlock",
        json={"request_id": request_id, "admin_key": "0000"},
        headers={"X-User-Id": "u1", "X-User-Role": "analyst"},
    )
    client.post(
        "/api/v1/unlock",
        json={"request_id": request_id, "admin_key": "7894"},
        headers={"X-User-Id": "u1", "X-User-Role": "analyst"},
    )

    events = [line["event"] for line in _log_lines(caplog) if "event" in line]
    assert events == ["attempt_failed", "unlocked"]
    unlocked_line = next(line for line in _log_lines(caplog) if line.get("event") == "unlocked")
    assert unlocked_line["user_id"] == "u1"
    assert unlocked_line["role"] == "analyst"


def test_unlock_blocked_event_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="aegis.ai")
    client = _client_with_locking(total_tokens=1500)
    request_id = client.post("/api/v1/generate", json={"prompt": "long"}).json()["request_id"]

    client.post("/api/v1/unlock", json={"request_id": request_id, "admin_key": "0000"})
    client.post("/api/v1/unlock", json={"request_id": request_id, "admin_key": "0000"})

    events = [line["event"] for line in _log_lines(caplog) if "event" in line]
    assert events == ["attempt_failed", "blocked"]
