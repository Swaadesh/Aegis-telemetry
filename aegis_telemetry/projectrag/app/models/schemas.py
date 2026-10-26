from typing import Literal

from pydantic import BaseModel, Field


class AIRequest(BaseModel):
    """Request the API layer accepts from the gateway/client."""

    prompt: str = Field(..., min_length=1, max_length=100_000)
    system_prompt: str | None = Field(
        default=None,
        description="Optional override for the default system prompt.",
    )


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class AIResponse(BaseModel):
    """Standardized response envelope returned to the gateway/client.

    Shape stays identical regardless of which provider generated it.
    """

    request_id: str
    status: Literal["success"] = "success"
    model: str
    text: str
    latency_ms: float
    token_usage: TokenUsage
    citations: list[str] = Field(default_factory=list)
    documents_used: int = 0
    source: Literal["document", "web"] = "document"


class ErrorResponse(BaseModel):
    request_id: str
    status: Literal["error"] = "error"
    error_type: str
    message: str
    latency_ms: float


class LockedResponse(BaseModel):
    """Returned instead of AIResponse when the generated text exceeds
    settings.response_token_limit — the real response is held server-side
    until unlocked via POST /api/v1/unlock."""

    request_id: str
    status: Literal["locked"] = "locked"
    message: str
    attempts_remaining: int
    source: Literal["document", "web"] = "document"


class UnlockRequest(BaseModel):
    request_id: str
    admin_key: str


class OutOfScopeResponse(BaseModel):
    """Returned instead of AIResponse when no relevant document context was
    found — offers the caller a web search instead via
    POST /api/v1/websearch-confirm."""

    request_id: str
    status: Literal["out_of_scope"] = "out_of_scope"
    message: str


class WebSearchConfirmRequest(BaseModel):
    request_id: str
    confirm: bool
