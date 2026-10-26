
import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.config.settings import Settings
from app.providers.base import (
    AIProvider,
    ProviderAPIError,
    ProviderQuotaExceededError,
    ProviderResponse,
    ProviderTimeoutError,
)

# HTTP status codes that indicate the caller exhausted its rate/quota budget.
_QUOTA_EXCEEDED_CODES = {429}


class GeminiProvider(AIProvider):
    """Single-responsibility Gemini integration: prompt in, response out.

    No routing, RAG, or business logic lives here — just the call to
    Google's official google-genai SDK and translation of its errors into
    the provider-agnostic error types the orchestrator understands.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)

    async def generate(self, system_prompt: str, user_prompt: str) -> ProviderResponse:
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=genai_types.ThinkingConfig(
                thinking_level=genai_types.ThinkingLevel(self._settings.thinking_level)
            ),
            max_output_tokens=self._settings.max_output_tokens,
            safety_settings=[
                genai_types.SafetySetting(
                    category=category,
                    threshold=genai_types.HarmBlockThreshold(self._settings.safety_block_threshold),
                )
                for category in (
                    genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                )
            ],
            http_options=genai_types.HttpOptions(
                timeout=int(self._settings.timeout_seconds * 1000)
            ),
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self._settings.model_name,
                contents=user_prompt,
                config=config,
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except genai_errors.APIError as exc:
            if exc.code in _QUOTA_EXCEEDED_CODES:
                raise ProviderQuotaExceededError(exc.message or str(exc)) from exc
            raise ProviderAPIError(f"{exc.code}: {exc.message or exc}") from exc

        usage = response.usage_metadata
        return ProviderResponse(
            text=response.text or "",
            model=self._settings.model_name,
            prompt_tokens=usage.prompt_token_count if usage else None,
            completion_tokens=usage.candidates_token_count if usage else None,
            total_tokens=usage.total_token_count if usage else None,
        )
