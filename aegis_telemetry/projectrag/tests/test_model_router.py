import pytest

from app.config.settings import Settings
from app.orchestrator.model_router import ModelRouter, ModelRouterError
from app.providers.base import AIProvider, ProviderResponse


class _StubProvider(AIProvider):
    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        return ProviderResponse(text="ok", model="stub")


def _settings(**overrides) -> Settings:
    return Settings(gemini_api_key="test-key", **overrides)


@pytest.mark.asyncio
async def test_route_resolves_configured_provider() -> None:
    settings = _settings()
    provider = _StubProvider()
    router = ModelRouter({settings.model_name: provider}, settings)

    resolved = await router.route()

    assert resolved is provider


@pytest.mark.asyncio
async def test_route_resolves_explicit_model_name() -> None:
    settings = _settings()
    gemini = _StubProvider()
    router = ModelRouter({settings.model_name: gemini}, settings)

    resolved = await router.route(model=settings.model_name)

    assert resolved is gemini


@pytest.mark.asyncio
async def test_route_raises_for_unregistered_model() -> None:
    settings = _settings()
    router = ModelRouter({settings.model_name: _StubProvider()}, settings)

    with pytest.raises(ModelRouterError):
        await router.route(model="claude-unregistered")
