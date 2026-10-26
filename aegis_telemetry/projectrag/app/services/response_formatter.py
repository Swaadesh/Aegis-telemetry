from typing import Literal

from app.context.base import ContextResult
from app.models.schemas import AIResponse, TokenUsage
from app.providers.base import ProviderResponse


def format_response(
    request_id: str,
    latency_ms: float,
    raw: ProviderResponse,
    context: ContextResult | None = None,
    source: Literal["document", "web"] = "document",
) -> AIResponse:
    """Validate and normalize a raw provider response into the stable envelope.

    This is what lets the API contract stay identical when a new provider
    (Claude, OpenAI, ...) or retrieval backend is added behind the
    orchestrator later. `context` is optional and defaults to no
    citations/documents_used, preserving Phase 1 behavior for non-RAG calls.
    """
    text = (raw.text or "").strip()
    if not text:
        raise ValueError("Provider returned an empty response")

    return AIResponse(
        request_id=request_id,
        model=raw.model,
        text=text,
        latency_ms=latency_ms,
        token_usage=TokenUsage(
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
            total_tokens=raw.total_tokens,
        ),
        citations=context.citations if context else [],
        documents_used=len(context.documents) if context else 0,
        source=source,
    )
