from app.config.settings import Settings
from app.providers.base import AIProvider, ProviderError


class ModelRouterError(ProviderError):
    """Raised when the router cannot resolve a provider for the request."""


class ModelRouter:
    """Resolves which AIProvider handles a request.

    Currently a single-entry {name: provider} map that always resolves to
    the configured Gemini provider. Adding Claude/OpenAI later means
    registering another entry here, not rewriting the router or the
    orchestrator.
    """

    def __init__(self, providers: dict[str, AIProvider], settings: Settings) -> None:
        self._providers = providers
        self._settings = settings

    async def route(self, model: str | None = None) -> AIProvider:
        name = model or self._settings.model_name
        try:
            return self._providers[name]
        except KeyError:
            raise ModelRouterError(f"No provider registered for model '{name}'") from None
