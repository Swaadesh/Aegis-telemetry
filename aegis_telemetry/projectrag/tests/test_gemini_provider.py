from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from google.genai import errors as genai_errors

from app.config.settings import Settings
from app.providers.base import (
    ProviderAPIError,
    ProviderQuotaExceededError,
    ProviderTimeoutError,
)
from app.providers.gemini.gemini_provider import GeminiProvider


def _settings() -> Settings:
    return Settings(gemini_api_key="test-key", tavily_api_key="tvly-test-key", max_retries=0)


def _fake_response(text: str = "hello there") -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=5, candidates_token_count=3, total_token_count=8
        ),
    )


@pytest.mark.asyncio
async def test_generate_returns_provider_response() -> None:
    provider = GeminiProvider(_settings())
    with patch.object(
        provider._client.aio.models, "generate_content", new=AsyncMock(return_value=_fake_response())
    ):
        result = await provider.generate("system", "user prompt")

    assert result.text == "hello there"
    assert result.total_tokens == 8


@pytest.mark.asyncio
async def test_generate_raises_timeout_error() -> None:
    provider = GeminiProvider(_settings())
    with patch.object(
        provider._client.aio.models,
        "generate_content",
        new=AsyncMock(side_effect=httpx.TimeoutException("timed out")),
    ), pytest.raises(ProviderTimeoutError):
        await provider.generate("system", "user prompt")


@pytest.mark.asyncio
async def test_generate_raises_quota_exceeded_error() -> None:
    provider = GeminiProvider(_settings())
    api_error = genai_errors.APIError(429, {"message": "quota exceeded"})
    with (
        patch.object(provider._client.aio.models, "generate_content", new=AsyncMock(side_effect=api_error)),
        pytest.raises(ProviderQuotaExceededError),
    ):
        await provider.generate("system", "user prompt")


@pytest.mark.asyncio
async def test_generate_raises_provider_api_error_for_invalid_key() -> None:
    provider = GeminiProvider(_settings())
    api_error = genai_errors.APIError(401, {"message": "invalid API key"})
    with (
        patch.object(provider._client.aio.models, "generate_content", new=AsyncMock(side_effect=api_error)),
        pytest.raises(ProviderAPIError),
    ):
        await provider.generate("system", "user prompt")
