from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResponse:
    """Raw result from a provider, before formatting/validation."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ProviderError(Exception):
    """Base class for provider-level failures (timeout, quota, API error, ...)."""


class ProviderTimeoutError(ProviderError):
    pass


class ProviderQuotaExceededError(ProviderError):
    pass


class ProviderAPIError(ProviderError):
    pass


class AIProvider(ABC):
    """Common interface every LLM provider (Gemini, Claude, OpenAI, ...) implements.

    The orchestrator only ever talks to this interface, so new providers can
    be added without touching orchestration logic.
    """

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        """Call the underlying model and return a raw ProviderResponse.

        Raises a ProviderError subclass on failure.
        """
        raise NotImplementedError
